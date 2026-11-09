"""Codabench submission -- CaEmgMamba, vel head, config ca_emgmamba_vel_R2a (Track 4, EMG-to-pose).

Architecture (neuraltrain/models/ca_emgmamba.py in UCF-ESCAPE-TEAM/neuroai_jf,
branch emg-pose-cnn, commit b17f885; training config
neuralbench/models/ca_emgmamba_vel_R2a.yaml):

    EMG (B, 16, T) --> WindowNorm (per-window per-electrode z-score, no
    parameters, no filters) --> ConvStem (3 strided Conv1d stages, 2 kHz -> 50 Hz,
    SE channel attention after each stage) --> MambaEncoder (2 bidirectional
    Mamba layers, d 192) --> VelocityHead (position + velocity readouts per frame, integrated with complementary fusion, fuse_gain 0.1)
    --> x40 linear upsample --> (B, T, 20) radians, transposed to (B, 20, T) by
    the wrapper.

Every class is a verbatim copy from the fork so a NeuralBench checkpoint loads
with ``strict=True``. The ``mamba_ssm`` CUDA kernel is NOT imported here:
``_fast_scan = None`` makes every MambaBlock use the pure-PyTorch selective scan
(same weights, same output up to float error; about 0.1 s per 64 windows on a
GPU). A ``config`` entry inside weights.pt (written by
prepare_submission/extract_weights_rev1.py from the training YAML) overrides
``CONFIG`` key by key.

Folder layout to zip (flat): submission.py, weights.pt.

Contract (tracks/emg_pose/objective.py): ``predict(X)`` takes ``(B, C, T)`` float
EMG and returns ``(B, n_joints, T)`` joint angles in radians; windows are
independent. Bidirectional processing inside a window is allowed.

Local test, from the 2026-competition repo:
    COMPET_SUBMISSION_DIR=/path/to/ca_emgmamba_vel_R2a \\
        benchopt run tracks/emg_pose -d Simulated -s CaEmgMambaVelR2a --no-cache
or through shell/run_submissions.sh.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from benchmark_utils.base_solver import CompetSolver

_fast_scan = None   # never use the mamba_ssm kernel in the submission

# Mirror of ca_emgmamba_vel_R2a.yaml's brain_model_config (minus ``name``). Must match
# the weights; the embedded checkpoint config wins where the two differ.
CONFIG = dict(
    sfreq=2000.0,
    norm_eps=1e-12,
    stem_channels=(64, 128, 192),
    stem_kernels=(31, 9, 5),
    stem_strides=(5, 4, 2),
    stem_groups=8,
    stem_dropout=0.1,
    ca_kind='se',
    ca_stages=(1, 2, 3),
    ca_reduction=8,
    ca_eca_k=0,
    d_model=192,
    n_layers=2,
    d_state=16,
    expand=2,
    d_conv=4,
    dt_rank=0,
    bidirectional=True,
    enc_dropout=0.1,
    scan_impl='auto',
    scan_checkpoint=True,
    head='vel',
    position_steps=12,
    vel_scale=0.02,
    fuse_gain=0.1,
)
LEGACY_KEYS = ()
N_CHANS_TRAINED = 16
N_JOINTS_TRAINED = 20


# =============================================================================
# Per-window normalisation
# =============================================================================
class WindowNorm(nn.Module):
    """Z-score of every electrode over the time axis of its own window.

    This is ``LayerNorm`` over time without the affine weights (which would be
    one weight per sample and would fix the window length). The epsilon is added
    to the standard deviation, not the variance, and is far below any real signal
    (1e-12 vs a std of ~1e-6 for EMG in volts), so the result does not depend on
    the amplitude unit the server feeds (uV, V, or NeuralBench's scale). A dead
    electrode (std 0) comes out as all zeros.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=-1, keepdim=True) + self.eps)


# =============================================================================
# Channel attention
# =============================================================================
def eca_kernel_size(n_ch: int, gamma: int = 2, b: int = 1) -> int:
    """ECA Eq. 12: k = |log2(C)/gamma + b/gamma|_odd."""
    k = int(abs((math.log2(n_ch) + b) / gamma))
    return k if k % 2 else k + 1


class ChannelAttention(nn.Module):
    """One gain in (0, 2) per feature channel and per window; identity at init."""

    def __init__(self, n_ch: int, kind: str = "se", reduction: int = 8, eca_k: int = 0):
        super().__init__()
        if kind not in ("se", "eca"):
            raise ValueError(f"ca_kind must be 'se', 'eca' or 'none', got {kind!r}")
        self.kind = kind
        if kind == "se":
            hidden = max(n_ch // reduction, 4)
            self.excite = nn.Sequential(nn.Linear(n_ch, hidden), nn.SiLU(), nn.Linear(hidden, n_ch))
            last = self.excite[-1]
        else:
            k = eca_k if eca_k else eca_kernel_size(n_ch)
            if k % 2 == 0:
                raise ValueError("ca_eca_k must be odd (or 0 for adaptive)")
            self.excite = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
            last = self.excite
        with torch.no_grad():
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()

    def gains(self, x: torch.Tensor) -> torch.Tensor:            # (B, C, T) -> (B, C)
        s = x.mean(dim=-1)                                       # squeeze
        s = self.excite(s) if self.kind == "se" else self.excite(s.unsqueeze(1)).squeeze(1)
        return 2.0 * torch.sigmoid(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gains(x).unsqueeze(-1)


# =============================================================================
# 1-D CNN downsampler
# =============================================================================
class StemStage(nn.Module):
    """Conv1d (strided) -> GroupNorm -> GELU -> Dropout -> [channel attention]."""

    def __init__(self, c_in: int, c_out: int, k: int, stride: int, groups: int, dropout: float,
                 ca_kind: str, ca_reduction: int, ca_eca_k: int):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("stem_kernels must be odd")
        self.conv = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
        self.norm = nn.GroupNorm(groups, c_out)
        self.drop = nn.Dropout(dropout)
        self.ca = (nn.Identity() if ca_kind == "none"
                   else ChannelAttention(c_out, ca_kind, ca_reduction, ca_eca_k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.drop(F.gelu(self.norm(self.conv(x)))))


class ConvStem(nn.Module):
    def __init__(self, n_in: int, channels, kernels, strides, groups, dropout,
                 ca_kind, ca_stages, ca_reduction, ca_eca_k):
        super().__init__()
        if not len(channels) == len(kernels) == len(strides):
            raise ValueError("stem_channels, stem_kernels and stem_strides must have the same length")
        bad = [s for s in ca_stages if not 1 <= s <= len(channels)]
        if bad:
            raise ValueError(f"ca_stages {bad} outside 1..{len(channels)}")
        dims = (n_in, *channels)
        self.stages = nn.ModuleList(
            StemStage(dims[i], dims[i + 1], kernels[i], strides[i], groups, dropout,
                      ca_kind if (i + 1) in ca_stages else "none", ca_reduction, ca_eca_k)
            for i in range(len(channels))
        )
        self.out_channels = channels[-1]
        self.downsample = math.prod(strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T) -> (B, F, T')
        for stage in self.stages:
            x = stage(x)
        return x


# =============================================================================
# Mamba
# =============================================================================
def selective_scan_torch(u, delta, A, B, C, D):
    """Reference selective scan.

    u, delta : (N, L, d_inner)      input and (already softplus-ed) step size
    A        : (d_inner, d_state)   negative decay rates
    B, C     : (N, L, d_state)      input-dependent input / output maps
    D        : (d_inner,)           skip gain
    returns  : (N, L, d_inner)

        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = <h_t, C_t> + D * u_t
    """
    N, L, d_inner = u.shape
    h = u.new_zeros(N, d_inner, A.shape[1])
    ys = []
    for t in range(L):
        dt = delta[:, t, :, None]                                 # (N, d_inner, 1)
        h = torch.exp(dt * A) * h + dt * B[:, t, None, :] * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))                 # (N, d_inner)
    return torch.stack(ys, dim=1) + u * D


class MambaBlock(nn.Module):
    """Mamba mixer (same parameter names and shapes as ``mamba_ssm.Mamba``)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int = 0, dt_min: float = 1e-3, dt_max: float = 0.1,
                 scan_impl: str = "auto", scan_checkpoint: bool = False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank else math.ceil(d_model / 16)
        self.scan_impl = scan_impl
        self.scan_checkpoint = scan_checkpoint

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # step size: softplus(bias) log-uniform in [dt_min, dt_max] at init
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))     # inverse softplus
        # decay rates 1..d_state for every inner channel (stored as log, used as -exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N, L, d_model)
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)                  # (N, L, d_inner) each
        u = self.conv1d(u.transpose(1, 2))[..., :L]              # causal depthwise conv
        u = F.silu(u).transpose(1, 2)                            # (N, L, d_inner)
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        if self.scan_impl == "auto" and _fast_scan is not None and x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                y = _fast_scan(
                    u.float().transpose(1, 2).contiguous(),                       # (N, d_inner, L)
                    (dt.float() @ self.dt_proj.weight.float().t()).transpose(1, 2).contiguous(),
                    A,
                    B.float().transpose(1, 2).contiguous(),                       # (N, d_state, L)
                    C.float().transpose(1, 2).contiguous(),
                    self.D.float(),
                    z=z.float().transpose(1, 2).contiguous(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                ).transpose(1, 2)                                                 # gated by SiLU(z)
            return self.out_proj(y.to(x.dtype))
        delta = F.softplus(self.dt_proj(dt))
        args = (u.float(), delta.float(), A, B.float(), C.float(), self.D.float())
        if self.scan_checkpoint and self.training and torch.is_grad_enabled():
            y = checkpoint(selective_scan_torch, *args, use_reentrant=False)
        else:
            y = selective_scan_torch(*args)
        return self.out_proj((y * F.silu(z.float())).to(x.dtype))


class MambaLayer(nn.Module):
    """Pre-norm residual layer; bidirectional = mean of a forward and a reversed block."""

    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, bidirectional, dropout, scan_impl,
                 scan_checkpoint=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        block = dict(scan_impl=scan_impl, scan_checkpoint=scan_checkpoint)
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, **block)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, **block) if bidirectional else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, L, d)
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = 0.5 * (y + self.bwd(h.flip(1)).flip(1))
        return x + self.drop(y)


class MambaEncoder(nn.Module):
    """Frame features ``(B, F, T')`` -> contextual features ``(B, d_model, T')``."""

    def __init__(self, in_ch, d_model, n_layers, d_state, d_conv, expand, dt_rank,
                 bidirectional, dropout, scan_impl, scan_checkpoint=False):
        super().__init__()
        self.proj = nn.Identity() if in_ch == d_model else nn.Conv1d(in_ch, d_model, 1)
        self.layers = nn.ModuleList(
            MambaLayer(d_model, d_state, d_conv, expand, dt_rank, bidirectional, dropout, scan_impl,
                       scan_checkpoint)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).transpose(1, 2)                         # (B, T', d)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).transpose(1, 2)                      # (B, d, T')


# =============================================================================
# Time-stretch augmentation (training only; applied to EMG AND targets)
# =============================================================================
class TimeStretch(nn.Module):
    """Resample a window in time about its centre by a per-window factor ``f``.

    ``out[t] = in[c + (t - c) * f]`` for EMG and targets alike (linear
    interpolation), so the same movement is seen faster (``f > 1``) or slower
    (``f < 1``) while the window keeps its length. Source indices outside the
    window are clamped for the EMG (edge hold) and the target frames there are
    set to NaN, which the training step already masks out of the loss and the
    metrics. Purpose: break the fixed tempo of each stage so the model cannot
    recall a stage's trajectory from timing alone (R0 failed on unseen stages).

    Hooked into training through ``CaEmgMambaModel.train_batch_transform``,
    the same hook C-Mixup uses, so no change to the Lightning module is needed.
    Band rotation (the task's augmentation) is applied afterwards as usual.
    """

    def __init__(self, prob: float = 1.0, lo: float = 0.8, hi: float = 1.2):
        super().__init__()
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"time_stretch_prob must be in [0, 1], got {prob}")
        if not 0.0 < lo <= hi:
            raise ValueError(f"time_stretch_range must satisfy 0 < lo <= hi, got {(lo, hi)}")
        self.prob, self.lo, self.hi = prob, lo, hi
        self.last_stats: dict[str, float] = {}

    @staticmethod
    def _resample(x: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:  # (B, C, T), (B, T) in [-1, 1]
        grid = torch.stack([gx, torch.zeros_like(gx)], dim=-1).unsqueeze(1)   # (B, 1, T, 2)
        y = F.grid_sample(x.unsqueeze(2).float(), grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        return y.squeeze(2).to(x.dtype)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` EMG (B, C, T), ``y`` targets (B, J, T) with NaN for unlabelled frames."""
        B, _, T = x.shape
        if y.shape[-1] != T:
            raise ValueError(f"EMG and target must share the time axis, got {T} vs {y.shape[-1]}")
        dev = x.device
        f = torch.empty(B, device=dev).uniform_(self.lo, self.hi)
        f = torch.where(torch.rand(B, device=dev) < self.prob, f, torch.ones_like(f))
        c = (T - 1) / 2.0
        t = torch.arange(T, device=dev, dtype=torch.float32)
        src = c + (t[None, :] - c) * f[:, None]                                # (B, T)
        inside = (src >= 0) & (src <= T - 1)
        gx = src / (T - 1) * 2.0 - 1.0
        keep = (f == 1.0)[:, None, None]          # untouched windows: exact pass-through, no interpolation
        xs = torch.where(keep, x, self._resample(x, gx))
        ys = torch.where(keep, y, self._resample(y, gx).masked_fill(~inside.unsqueeze(1), torch.nan))
        self.last_stats = {"ts_factor_mean": float(f.mean()),
                           "ts_frac_masked": float((~inside).float().mean())}
        return xs, ys


# =============================================================================
# Heads
# =============================================================================
class PoseHead(nn.Module):
    """One absolute joint-angle vector per frame (no continuity prior, no drift)."""

    def __init__(self, d_model: int, n_outputs: int):
        super().__init__()
        self.pos = nn.Conv1d(d_model, n_outputs, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, d, T') -> (B, J, T')
        return self.pos(z)


class VelocityHead(nn.Module):
    """Position + velocity readouts, integrated and fused (VEMG2Pose-style).

        pose_t = position_t                                          t <  position_steps
        pose_t = (1-g) * (pose_{t-1} + vel_scale * velocity_t) + g * position_t   otherwise

    g = ``fuse_gain``: 0 is pure integration (smooth, drifts), 1 is the pose
    head, values in between let the velocity shape fast motion while the
    position readout removes drift. The velocity readout starts at zero, so the
    trajectory begins as "hold the position".
    """

    def __init__(self, d_model: int, n_outputs: int, position_steps: int = 12,
                 vel_scale: float = 0.02, fuse_gain: float = 0.1):
        super().__init__()
        if not 0.0 <= fuse_gain <= 1.0:
            raise ValueError(f"fuse_gain must be in [0, 1], got {fuse_gain}")
        self.position_steps, self.vel_scale, self.fuse_gain = position_steps, vel_scale, fuse_gain
        self.pos = nn.Conv1d(d_model, n_outputs, 1)
        self.vel = nn.Conv1d(d_model, n_outputs, 1)
        with torch.no_grad():
            self.vel.weight.zero_()
            self.vel.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, d, T') -> (B, J, T')
        pos, vel = self.pos(z), self.vel(z)
        T = pos.shape[-1]
        k = min(self.position_steps, T)
        if k == T:
            return pos
        step = vel[..., k:] * self.vel_scale
        if self.fuse_gain == 0.0:
            return torch.cat([pos[..., :k], pos[..., k - 1:k] + torch.cumsum(step, dim=-1)], dim=-1)
        g = self.fuse_gain
        prev, out = pos[..., k - 1], []
        for t in range(T - k):
            prev = (1.0 - g) * (prev + step[..., t]) + g * pos[..., k + t]
            out.append(prev)
        return torch.cat([pos[..., :k], torch.stack(out, dim=-1)], dim=-1)


# =============================================================================
# Full model
# =============================================================================
class CaEmgMambaModel(nn.Module):
    MODEL_FIELDS = (
        "norm_eps",
        "stem_channels", "stem_kernels", "stem_strides", "stem_groups", "stem_dropout",
        "ca_kind", "ca_stages", "ca_reduction", "ca_eca_k",
        "d_model", "n_layers", "d_state", "expand", "d_conv", "dt_rank",
        "bidirectional", "enc_dropout", "scan_impl", "scan_checkpoint",
        "head", "position_steps", "vel_scale", "fuse_gain",
        "time_stretch_prob", "time_stretch_range",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        norm_eps: float = 1e-12,
        stem_channels=(64, 128, 192), stem_kernels=(31, 9, 5), stem_strides=(5, 4, 2),
        stem_groups: int = 8, stem_dropout: float = 0.1,
        ca_kind: str = "se", ca_stages=(1, 2, 3), ca_reduction: int = 8, ca_eca_k: int = 0,
        d_model: int = 192, n_layers: int = 4, d_state: int = 16, expand: int = 2,
        d_conv: int = 4, dt_rank: int = 0, bidirectional: bool = True,
        enc_dropout: float = 0.1, scan_impl: str = "auto", scan_checkpoint: bool = True,
        head: str = "pose", position_steps: int = 12, vel_scale: float = 0.02,
        fuse_gain: float = 0.1,
        time_stretch_prob: float = 0.0, time_stretch_range=(0.8, 1.2),
    ):
        super().__init__()
        if head not in ("pose", "vel"):
            raise ValueError(f"head must be 'pose' or 'vel', got {head!r}")
        self.norm = WindowNorm(norm_eps)
        self.stem = ConvStem(n_in_channels, tuple(stem_channels), tuple(stem_kernels),
                             tuple(stem_strides), stem_groups, stem_dropout,
                             ca_kind, tuple(ca_stages), ca_reduction, ca_eca_k)
        self.encoder = MambaEncoder(self.stem.out_channels, d_model, n_layers, d_state, d_conv,
                                    expand, dt_rank, bidirectional, enc_dropout, scan_impl,
                                    scan_checkpoint)
        self.head = (PoseHead(d_model, n_outputs) if head == "pose"
                     else VelocityHead(d_model, n_outputs, position_steps, vel_scale, fuse_gain))
        self.feature_rate = sfreq / self.stem.downsample
        # training-only joint augmentation; the Lightning module calls
        # ``train_batch_transform`` when the attribute exists (see pl_module._run_step)
        self.time_stretch = (TimeStretch(time_stretch_prob, *time_stretch_range)
                             if time_stretch_prob > 0 else None)
        self.last_mix_stats: dict[str, float] = {}
        if self.time_stretch is not None:
            self.train_batch_transform = self._time_stretch_batch

    def _time_stretch_batch(self, neuro, target, subject_ids=None):
        """Hook for pl_module: stretch EMG and targets together (training only)."""
        if not self.training:
            return neuro, target
        neuro, target = self.time_stretch(neuro, target)
        self.last_mix_stats = self.time_stretch.last_stats
        return neuro, target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        z = self.stem(self.norm(x))        # (B, F, T')  features at ~50 Hz
        z = self.encoder(z)                # (B, d, T')  contextual features
        y = self.head(z)                   # (B, J, T')  one pose per frame
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)           # (B, T, J)


# =============================================================================
# Submission wrapper
# =============================================================================
TUPLE_KEYS = ("stem_channels", "stem_kernels", "stem_strides", "ca_stages")


def build_model(n_chans, n_joints, config):
    cfg = dict(config)
    for k in TUPLE_KEYS:
        if cfg.get(k) is not None:
            cfg[k] = tuple(cfg[k])
    cfg["scan_impl"] = "torch"           # belt and braces: _fast_scan is None anyway
    cfg["time_stretch_prob"] = 0.0       # training-only augmentation; inert at inference anyway
    # keys the YAML left at their defaults are taken from the class defaults
    model_kwargs = {k: cfg[k] for k in CaEmgMambaModel.MODEL_FIELDS if k in cfg}
    return CaEmgMambaModel(n_in_channels=n_chans, n_outputs=n_joints,
                           sfreq=cfg["sfreq"], **model_kwargs)


def _read_checkpoint(path, device):
    """-> (state_dict with model-level keys, config dict or None)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state, config = ckpt["state_dict"], ckpt.get("config")
    else:
        state = ckpt
    if any(k.startswith("model.") for k in state):  # Lightning / NeuralBench
        state = {k.removeprefix("model."): v for k, v in state.items()
                 if k.startswith("model.")}
    state = {k: v for k, v in state.items() if not k.startswith("loss.")}
    if config is not None:
        config = {k: v for k, v in dict(config).items() if k in CONFIG or k in LEGACY_KEYS}
    return state, config


class CaEmgMambaRegressor:
    """``fit`` / ``predict`` wrapper owning its device."""

    def __init__(self, n_chans, n_joints, config, device, lr=3e-4, n_epochs=20):
        self.device = device
        self.config = dict(config)
        self.n_chans, self.n_joints = n_chans, n_joints
        self.lr, self.n_epochs = lr, n_epochs
        self.net = build_model(n_chans, n_joints, self.config).to(device)

    def load(self, path):
        state, config = _read_checkpoint(path, self.device)
        merged = {**self.config, **(config or {})}
        if merged != self.config:
            self.config = merged
            self.net = build_model(self.n_chans, self.n_joints, self.config).to(self.device)
        self.net.load_state_dict(state, strict=True)
        return self

    # ---- optional training through benchopt (``training=True``): plain L1 on
    # finite targets, AdamW. The server never takes this path.
    def fit(self, train_loader):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.05)
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                X = X.float(); y = y.float()                 # (B,C,T), (B,J,T)
                opt.zero_grad()
                pred = self.net(X).transpose(1, 2)           # (B, J, T) radians
                finite = torch.isfinite(y)
                loss = (pred - torch.nan_to_num(y)).abs()[finite].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if not getattr(self, "_logged_input", False):
            # one-off diagnostic (visible in the Codabench log): the units the
            # server delivers. WindowNorm makes the model invariant to them.
            print(f"[input] shape {tuple(X.shape)}, |X| mean {X.abs().mean():.4g}, "
                  f"std {X.std():.4g}, max {X.abs().max():.4g}")
            self._logged_input = True
        return self.net(X).transpose(1, 2)                   # (B, J, T) radians


class Solver(CompetSolver):

    name = "CaEmgMambaVelR2a"
    requirements = []  # torch comes with the environment; no mamba_ssm needed

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        inference = self.train_loader is None
        if inference and weights.exists():
            model = CaEmgMambaRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED,
                                        CONFIG, self.device).load(weights)
            n_params = sum(p.numel() for p in model.net.parameters())
            print(f"[loading] {weights} into {self.name} ({n_params / 1e6:.2f} M params, "
                  f"head={model.config['head']}, scan=torch)")
            if meta.get("n_chans") != N_CHANS_TRAINED:
                raise RuntimeError(
                    f"weights expect {N_CHANS_TRAINED} EMG channels, data has "
                    f"{meta.get('n_chans')}")
            if float(meta.get("sfreq", 0)) != float(model.config["sfreq"]):
                warnings.warn(
                    f"data at {meta.get('sfreq')} Hz, weights trained at "
                    f"{model.config['sfreq']} Hz; kernels and strides stay in "
                    "samples (fine for a smoke test, not for scoring).")
            return model
        if inference:
            raise FileNotFoundError(f"{weights} not found (inference-only run)")
        # Training run: build from the data's metadata, train from scratch.
        config = {**CONFIG, "sfreq": float(meta["sfreq"])}
        return CaEmgMambaRegressor(meta["n_chans"], meta["n_joints"], config, self.device)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save({"state_dict": model.net.state_dict(), "config": model.config},
                   path / "weights.pt")
