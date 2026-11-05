"""Codabench submission -- StVelTransformer trained with the OHEM loss (hardest 30 % of frames per batch at full weight, rest at 0.1) (Track 4, EMG-to-pose).

Architecture (neuraltrain/models/st_vel_transformer.py in UCF-ESCAPE-TEAM/neuroai_jf,
branch emg-pose-cnn, commit 77a9e18; training config neuralbench/models/st_vel_transformer.yaml):

    EMG (B, 16, T) --> EmgPreproc (FIR 20-450 Hz + 60/120/180 Hz notches, per-window
    bad-channel rejection + ring interpolation, per-window per-channel z-score; no
    parameters) --> ChannelGate (SE gate over the 16 electrodes) --> MsSpatialBlock
    (multi-scale temporal bank 15/31/63/127 taps, ring convs, depthwise collapse,
    pool /5, separable conv, pool /8 -> 50 Hz) --> SequenceEncoder (4 pre-norm
    Conformer blocks, d 128, 4 heads, RoPE) --> VelocityDecoder (position + velocity readouts, hybrid integration with complementary fusion) --> x40 upsample
    --> (B, T, 20) radians, transposed to (B, 20, T) by the wrapper.

The loss (OhemRegressionLoss) changes training only and ships nothing in the
checkpoint, so the three loss variants of this model share this file apart from
``Solver.name``. Every class is a verbatim copy from the fork so a NeuralBench
checkpoint loads with ``strict=True``; the FIR kernel is a non-persistent buffer
rebuilt from ``CONFIG``. A ``config`` entry inside weights.pt (written by
prepare_submission/extract_weights_rev1.py from the training YAML) overrides
``CONFIG`` key by key.

Folder layout to zip (flat): submission.py, weights.pt.

Contract (tracks/emg_pose/objective.py): ``predict(X)`` takes ``(B, C, T)`` float
EMG and returns ``(B, n_joints, T)`` joint angles in radians; windows are
independent. Bidirectional processing inside a window is allowed (causal=False).

Local test, from the 2026-competition repo:
    COMPET_SUBMISSION_DIR=/path/to/st_vel_transformer \\
        benchopt run tracks/emg_pose -d Simulated -s StVelTransformer --no-cache
or through shell/run_submissions.sh.
"""

import math
import typing as tp
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# Mirror of st_vel_transformer.yaml's brain_model_config (minus ``name``). Must match the
# weights; the embedded checkpoint config wins where the two differ.
CONFIG = dict(
    sfreq=2000.0,
    l_freq=20.0,
    h_freq=450.0,
    notch_freqs=(60.0, 120.0, 180.0),
    notch_width=6.0,
    bp_taps=None,
    notch_taps=None,
    norm='zscore',
    scope='channel',
    signed=True,
    robust_q=0.005,
    eps=1e-06,
    clip=None,
    reject_channels=True,
    reject_rms_nstd=3.0,
    reject_mains=60.0,
    reject_mains_nstd=3.0,
    reject_mains_bw=2.0,
    reject_segment=400,
    reject_max_frac=0.5,
    reject_method='std',
    channel_attention=True,
    ca_hidden=32,
    f1=32,
    k_e=65,
    ms_kernels=(15, 31, 63, 127),
    spatial_mode='conv2d',
    spatial_layers=2,
    k_s=9,
    depth_mult=2,
    f2=128,
    k_sep=16,
    pool1=5,
    pool2=8,
    spatial_dropout=0.2,
    d_model=128,
    n_heads=4,
    n_layers=4,
    ff_mult=4,
    enc_dropout=0.1,
    attn_dropout=0.1,
    block_style='conformer',
    conv_kernel=15,
    pos_enc='rope',
    attn_window=None,
    position_steps=12,
    vel_scale=0.02,
    fuse_gain=0.1,
    vel_gru_hidden=0,
    vel_gru_layers=1,
    causal=False,
)
N_CHANS_TRAINED = 16
N_JOINTS_TRAINED = 20


# =============================================================================
# EmgPreproc (verbatim copy of neuraltrain/models/emg_frontend.py)
# =============================================================================
def design_fir(sfreq: float, l_freq: float | None, h_freq: float | None,
               notch_freqs: tuple[float, ...] | None = None, notch_width: float = 6.0,
               bp_taps: int | None = None, notch_taps: int | None = None,
               window: str = "hamming") -> np.ndarray | None:
    """Band-pass + notch as ONE symmetric (zero-phase) FIR, designed with scipy.

    Tap counts follow MNE's rule of thumb for a Hamming window
    (``3.3 * sfreq / transition_bandwidth``, rounded to odd); pass ``bp_taps`` /
    ``notch_taps`` to override. The cascade is obtained by convolving the kernels.
    """
    nyq = sfreq / 2.0

    def odd(n: float) -> int:
        n = int(np.ceil(n))
        return n if n % 2 else n + 1

    h = None
    if l_freq is not None or h_freq is not None:
        lo = l_freq if l_freq is not None else 0.0
        hi = h_freq if h_freq is not None else nyq
        trans = min(max(lo * 0.25, 2.0), max(0.25 * (nyq - hi), 2.0))  # MNE-like auto transition
        n = bp_taps or odd(3.3 * sfreq / trans)
        if lo > 0 and hi < nyq:
            h = signal.firwin(n, [lo, hi], pass_zero=False, window=window, fs=sfreq)
        elif lo > 0:
            h = signal.firwin(n, lo, pass_zero=False, window=window, fs=sfreq)
        else:
            h = signal.firwin(n, hi, pass_zero=True, window=window, fs=sfreq)
    for f0 in (notch_freqs or ()):
        if f0 + notch_width / 2 >= nyq:
            continue
        n = notch_taps or odd(3.3 * sfreq / (notch_width / 2))
        hn = signal.firwin(n, [f0 - notch_width / 2, f0 + notch_width / 2],
                           pass_zero=True, window=window, fs=sfreq)  # band-stop
        h = hn if h is None else np.convolve(h, hn)
    return None if h is None else h.astype(np.float32)


class EmgPreproc(nn.Module):
    NORMS = ("maxabs", "minmax", "robust", "zscore", "none")
    SCOPES = ("channel", "window", "batch")

    def __init__(self, n_channels: int = 16, sfreq: float = 2000.0,
                 l_freq: float | None = 20.0, h_freq: float | None = 450.0,
                 notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0),
                 notch_width: float = 6.0, bp_taps: int | None = None,
                 notch_taps: int | None = None,
                 norm: str = "maxabs", scope: str = "channel", signed: bool = True,
                 robust_q: float = 0.005, eps: float = 1e-6, clip: float | None = None,
                 causal: bool = False,
                 reject_channels: bool = False, reject_rms_nstd: float = 3.0,
                 reject_mains: float | None = 60.0, reject_mains_nstd: float = 3.0,
                 reject_mains_bw: float = 2.0, reject_segment: int = 400,
                 reject_max_frac: float = 0.5, reject_method: str = "std",
                 kernel: np.ndarray | None = None):
        super().__init__()
        if norm not in self.NORMS:
            raise ValueError(f"norm must be maxabs|minmax|robust|zscore|none, got {norm!r}")
        if scope not in self.SCOPES:
            raise ValueError(f"scope must be channel|window|batch, got {scope!r}")
        if causal and norm == "robust":
            raise ValueError("norm='robust' has no running form; use maxabs/minmax/zscore when causal")
        if causal and scope == "batch":
            raise ValueError("scope='batch' is not causal")
        if reject_method not in ("std", "mad"):
            raise ValueError(f"reject_method must be 'std' or 'mad', got {reject_method!r}")
        self.n_channels, self.sfreq = n_channels, float(sfreq)
        self.norm, self.scope, self.signed = norm, scope, signed
        self.robust_q, self.eps, self.clip = robust_q, eps, clip
        self.causal = causal
        self.reject_channels = reject_channels
        self.reject_rms_nstd, self.reject_mains = reject_rms_nstd, reject_mains
        self.reject_mains_nstd, self.reject_mains_bw = reject_mains_nstd, reject_mains_bw
        self.reject_segment, self.reject_max_frac = reject_segment, reject_max_frac
        self.reject_method = reject_method
        self.last_reject_fraction: float = 0.0
        if kernel is None:
            kernel = design_fir(self.sfreq, l_freq, h_freq, notch_freqs, notch_width, bp_taps, notch_taps)
        if kernel is None:
            self.kernel = None
        else:
            # not a parameter, not persisted: rebuilt from the config, never from weights.pt
            self.register_buffer("kernel", torch.as_tensor(np.asarray(kernel, dtype=np.float32)),
                                 persistent=False)
        self.n_taps = 0 if kernel is None else int(len(kernel))

    # ------------------------------------------------------------ rejection
    def _outliers(self, v: torch.Tensor, n: float) -> torch.Tensor:
        """Outliers across channels, per window. (B, C) -> bool (B, C).

        "std": v > mean + n*std.  "mad": v > median + n*1.4826*MAD (robust: two
        loud channels cannot hide each other; use n ~ 5).
        """
        if self.reject_method == "mad":
            med = v.median(-1, keepdim=True).values
            mad = (v - med).abs().median(-1, keepdim=True).values
            return v > med + n * 1.4826 * mad
        return v > v.mean(-1, keepdim=True) + n * v.std(-1, keepdim=True, unbiased=False)

    def bad_channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Bool mask (B, C): channels whose RMS or mains-band energy (±reject_mains_bw
        Hz around reject_mains and its 2nd harmonic) is an outlier in that window.
        Causal: statistics from the first ``reject_segment`` samples. Windows with
        more than ``reject_max_frac`` of the channels flagged are left untouched."""
        seg = (x[..., : self.reject_segment] if self.causal else x).float()
        bad = self._outliers(seg.pow(2).mean(-1).sqrt(), self.reject_rms_nstd)
        if self.reject_mains is not None:
            spec = torch.fft.rfft(seg, dim=-1).abs()
            freqs = torch.fft.rfftfreq(seg.shape[-1], d=1.0 / self.sfreq).to(seg.device)
            band = torch.zeros_like(freqs, dtype=torch.bool)
            for harmonic in (1, 2):
                f0 = self.reject_mains * harmonic
                if f0 < self.sfreq / 2:
                    band |= (freqs - f0).abs() <= self.reject_mains_bw
            bad |= self._outliers(spec[..., band].sum(-1), self.reject_mains_nstd)
        too_many = bad.float().mean(-1, keepdim=True) > self.reject_max_frac
        return bad & ~too_many

    def reject(self, x: torch.Tensor) -> torch.Tensor:
        """Replace flagged channels by the mean of their two ring neighbours (circular)."""
        bad = self.bad_channel_mask(x)
        self.last_reject_fraction = bad.float().mean().item()
        if not bool(bad.any()):
            return x
        neighbours = (x.roll(1, dims=1) + x.roll(-1, dims=1)) / 2
        return torch.where(bad.unsqueeze(-1), neighbours, x)

    # ------------------------------------------------------------ filtering
    @staticmethod
    def _pad_reflect(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
        """Odd reflection about the end samples (MNE 'reflect_limited'), zeros beyond."""
        T = x.shape[-1]
        ml, mr = min(left, T - 1), min(right, T - 1)
        parts = []
        if left > ml:
            parts.append(x.new_zeros(x.shape[:-1] + (left - ml,)))
        if ml:
            parts.append(2 * x[..., :1] - x[..., 1:ml + 1].flip(-1))
        parts.append(x)
        if mr:
            parts.append(2 * x[..., -1:] - x[..., -mr - 1:-1].flip(-1))
        if right > mr:
            parts.append(x.new_zeros(x.shape[:-1] + (right - mr,)))
        return torch.cat(parts, dim=-1)

    def filter(self, x: torch.Tensor) -> torch.Tensor:
        """'Same'-length FIR via FFT: O(T log T) whatever the tap count.

        Centred / zero phase (output t uses x[t-k//2 .. t+k//2]) or, when causal,
        left-aligned (output t uses x[t-k+1 .. t]).
        """
        k = self.n_taps
        T = x.shape[-1]
        left, right = (k - 1, 0) if self.causal else (k // 2, k // 2)
        xp = self._pad_reflect(x.float(), left, right)
        n = xp.shape[-1] + k - 1
        y = torch.fft.irfft(torch.fft.rfft(xp, n=n) * torch.fft.rfft(self.kernel, n=n), n=n)
        return y[..., k - 1: k - 1 + T].to(x.dtype)

    # ---------------------------------------------------------- normalising
    def _reduce_dims(self) -> tuple[int, ...]:
        return {"channel": (-1,), "window": (-2, -1), "batch": (0, -2, -1)}[self.scope]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "none":
            return x
        if self.causal:
            return self._normalize_causal(x)
        dims = self._reduce_dims()
        if self.norm == "zscore":
            mean = x.mean(dims, keepdim=True)
            std = x.std(dims, keepdim=True, unbiased=False)
            return (x - mean) / (std + self.eps)
        if self.norm == "maxabs":
            return x / (x.abs().amax(dims, keepdim=True) + self.eps)
        if self.norm == "minmax":
            lo, hi = x.amin(dims, keepdim=True), x.amax(dims, keepdim=True)
        else:  # robust: quantiles instead of extremes, via kthvalue (no size limit)
            B, C, _ = x.shape
            flat = {"channel": x, "window": x.reshape(B, 1, -1),
                    "batch": x.reshape(1, 1, -1)}[self.scope]
            n = flat.shape[-1]
            k_lo = max(1, int(round(self.robust_q * n)))
            k_hi = max(1, int(round((1 - self.robust_q) * n)))
            lo = flat.kthvalue(k_lo, dim=-1, keepdim=True).values   # (B,C,1) / (B,1,1) / (1,1,1)
            hi = flat.kthvalue(k_hi, dim=-1, keepdim=True).values
        y = (x - lo) / (hi - lo + self.eps)
        if self.norm == "robust":
            y = y.clamp(0.0, 1.0)
        return y * 2 - 1 if self.signed else y

    def _normalize_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Same scalings with running statistics over x[.. t] (per channel, or over
        all channels of the window so far when scope='window')."""
        xd = x.double()
        if self.scope == "window":                       # running stats over (C, t)
            B, C, T = xd.shape
            flat = xd.transpose(1, 2).reshape(B, 1, T * C)   # time-major: all channels of t before t+1
            y = self._running(flat)
            return y.reshape(B, T, C).transpose(1, 2).to(x.dtype)
        return self._running(xd).to(x.dtype)

    def _running(self, xd: torch.Tensor) -> torch.Tensor:
        if self.norm == "zscore":
            n = torch.arange(1, xd.shape[-1] + 1, device=xd.device, dtype=torch.float64)
            mean = xd.cumsum(-1) / n
            var = (xd * xd).cumsum(-1) / n - mean * mean
            return (xd - mean) / (var.clamp_min(0).sqrt() + self.eps)
        if self.norm == "maxabs":
            return xd / (xd.abs().cummax(-1).values + self.eps)
        lo, hi = xd.cummin(-1).values, xd.cummax(-1).values   # minmax
        y = (xd - lo) / (hi - lo + self.eps)
        return y * 2 - 1 if self.signed else y

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        if self.reject_channels:
            x = self.reject(x)         # on the raw signal, before the filter
        if self.kernel is not None:
            x = self.filter(x)         # filter FIRST: removes the DC offset min-max would otherwise keep
        x = self.normalize(x)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x

    def extra_repr(self) -> str:
        return (f"sfreq={self.sfreq}, taps={self.n_taps}, norm={self.norm}, "
                f"scope={self.scope}, signed={self.signed}, causal={self.causal}, "
                f"reject={self.reject_channels}")


EmgFrontEnd = EmgPreproc   # name used by older code


# ---------------------------------------------------------------------------
# Building from a config dict, with support for checkpoints trained before this
# front end (raw / R0 / R1): their configs carry highpass / lowpass / notch /
# notch_width / fir_taps / normalize and used one multi-band firwin kernel plus
# a per-channel z-score (eps 1e-5). ``frontend_from_config`` rebuilds that
# exactly so those weights evaluate as trained.
# ---------------------------------------------------------------------------
FILTER_KEYS = ("l_freq", "h_freq", "notch_freqs", "notch_width", "bp_taps", "notch_taps")
SCALE_KEYS = ("norm", "scope", "signed", "robust_q", "eps", "clip")
COMMON_KEYS = ("sfreq", "causal", "reject_channels", "reject_rms_nstd", "reject_mains",
               "reject_mains_nstd", "reject_mains_bw", "reject_segment", "reject_max_frac",
               "reject_method")
LEGACY_KEYS = ("highpass", "lowpass", "notch", "fir_taps", "normalize")
FRONTEND_KEYS = FILTER_KEYS + SCALE_KEYS + COMMON_KEYS


def legacy_fir_kernel(sfreq: float, highpass: float | None, lowpass: float | None,
                      notch: float | None, notch_width: float, numtaps: int) -> np.ndarray | None:
    nyq = sfreq / 2.0
    lo = highpass if highpass is not None else 0.0
    hi = lowpass if lowpass is not None else nyq
    edges: list[float] = []
    if lo > 0:
        edges.append(lo)
    if notch is not None:
        f0 = notch
        while f0 + notch_width / 2 < hi:
            if f0 - notch_width / 2 > lo:
                edges += [f0 - notch_width / 2, f0 + notch_width / 2]
            f0 += notch
    if hi < nyq:
        edges.append(hi)
    if not edges:
        return None
    return signal.firwin(numtaps, edges, pass_zero=(lo <= 0), fs=sfreq).astype(np.float32)


def merge_config(base: dict, ckpt: dict | None) -> dict:
    """``base`` (submission CONFIG) updated by a checkpoint's config. A legacy
    checkpoint config replaces the new filter/scaling keys instead of mixing."""
    if not ckpt:
        return dict(base)
    merged = {**base, **ckpt}
    if "fir_taps" in ckpt:
        for k in FILTER_KEYS + SCALE_KEYS:
            if k not in ckpt:
                merged.pop(k, None)
    return merged


def frontend_from_config(n_channels: int, cfg: dict) -> EmgPreproc:
    common = {k: cfg[k] for k in COMMON_KEYS if k in cfg}
    if "fir_taps" in cfg:                      # legacy checkpoint
        warnings.warn("legacy front end (multi-band firwin + z-score) rebuilt from the checkpoint config")
        kernel = legacy_fir_kernel(cfg["sfreq"], cfg.get("highpass"), cfg.get("lowpass"),
                                   cfg.get("notch"), cfg.get("notch_width", 6.0), cfg["fir_taps"])
        return EmgPreproc(n_channels, l_freq=None, h_freq=None, notch_freqs=None, kernel=kernel,
                          norm="zscore" if cfg.get("normalize", True) else "none", scope="channel",
                          eps=1e-5, clip=cfg.get("clip"), **common)
    kw = {k: cfg[k] for k in FILTER_KEYS + SCALE_KEYS if k in cfg}
    if kw.get("notch_freqs") is not None:
        kw["notch_freqs"] = tuple(kw["notch_freqs"])
    return EmgPreproc(n_channels, **kw, **common)


# =============================================================================
# Model (verbatim copies of neuraltrain/models/st_tcn_pose.py pieces that
# st_pose_transformer.py imports, then st_pose_transformer.py, then st_vel_transformer.py)
# =============================================================================
# =============================================================================
# padding helpers
# =============================================================================
def _same_pad(k: int, causal: bool) -> tuple[int, int]:
    """(left, right) zero padding so a stride-1 conv of kernel ``k`` keeps length.

    Non-causal: centred (``k // 2`` each side for odd ``k``; for even ``k`` the
    extra sample goes left, matching PyTorch's symmetric ``padding=k//2`` on
    the kept samples). Causal: all ``k - 1`` on the left.
    """
    if causal:
        return k - 1, 0
    left = k // 2
    return left, k - 1 - left


# =============================================================================
# Spatial block: 2-D CNN over (channels x time)
# =============================================================================
class _RingConv2d(nn.Module):
    """Conv2d over (C, T) with circular padding along C and 'same' padding along T."""

    def __init__(self, ch: int, k_c: int, k_t: int, causal: bool):
        super().__init__()
        self.pad_c = k_c // 2
        self.pad_t = _same_pad(k_t, causal)
        self.conv = nn.Conv2d(ch, ch, (k_c, k_t), padding=0, bias=False)

    def forward(self, x):                                   # (B, F, C, T)
        x = F.pad(x, (0, 0, self.pad_c, self.pad_c), mode="circular")  # electrode ring
        x = F.pad(x, (*self.pad_t, 0, 0))                                # time (zeros)
        return self.conv(x)


class SpatialBlock(nn.Module):
    """EEGNet-style front end, optionally with 2-D spatial convs over the electrode ring."""

    def __init__(self, n_ch, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
                 f2, k_sep, pool1, pool2, dropout, causal):
        super().__init__()
        self.pad_e = _same_pad(k_e, causal)
        self.pad_sep = _same_pad(k_sep, causal)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, k_e), padding=0, bias=False),
            nn.BatchNorm2d(f1),
        )
        layers = []
        if spatial_mode == "conv2d":
            for _ in range(spatial_layers):
                layers += [_RingConv2d(f1, 3, k_s, causal), nn.BatchNorm2d(f1), nn.ELU()]
        self.spatial2d = nn.Sequential(*layers)
        # depthwise (C, 1): one spatial filter per feature map (x depth_mult), as in EEGNet
        self.collapse = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_mult, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_mult),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),   # bin j = samples [j*p, (j+1)*p): end-aligned, causal
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * depth_mult, f1 * depth_mult, (1, k_sep), padding=0,
                      groups=f1 * depth_mult, bias=False),
            nn.Conv2d(f1 * depth_mult, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        self.out_channels = f2
        self.downsample = pool1 * pool2

    def forward(self, x):                      # (B, C, T)
        x = F.pad(x.unsqueeze(1), (*self.pad_e, 0, 0))   # (B, 1, C, T + k_e - 1)
        x = self.temporal(x)                   # (B, F1, C, T)
        x = self.spatial2d(x)                  # (B, F1, C, T)
        x = self.collapse(x)                   # (B, F1*D, 1, T/p1)
        x = F.pad(x, (*self.pad_sep, 0, 0))
        x = self.separable(x)                  # (B, F2, 1, T/(p1*p2))
        return x.squeeze(2)                    # (B, F2, T')


# =============================================================================
# Output head
# =============================================================================
class PosePost(nn.Module):
    """Turns the 1x1 ``head`` readout into a pose trajectory ``(B, J, T')``.

    kind="linear"   : identity -- one pose per frame, no continuity prior.
    kind="smooth"   : learnable depthwise ``smooth_k``-tap filter per joint,
                      initialised to a moving average (centred, or left-aligned
                      when causal). Adds ``J * smooth_k`` parameters.
    kind="velocity" : the readout has 2J channels, a position AND a velocity per
                      joint. The pose is the position for the first
                      ``position_steps`` frames and from there on the previous
                      pose plus ``vel_scale`` x velocity -- VEMG2Pose's hybrid
                      parameterisation without the LSTM. The cumulative sum is
                      causal by construction. No parameters.
    """

    def __init__(self, n_outputs: int, kind: str = "linear", smooth_k: int = 5,
                 position_steps: int = 12, vel_scale: float = 0.02, causal: bool = False):
        super().__init__()
        if kind not in ("linear", "smooth", "velocity"):
            raise ValueError(f"head_kind must be linear | smooth | velocity, got {kind!r}")
        self.kind = kind
        self.position_steps = position_steps
        self.vel_scale = vel_scale
        if kind == "smooth":
            self.pad = _same_pad(smooth_k, causal)
            self.smooth = nn.Conv1d(n_outputs, n_outputs, smooth_k, groups=n_outputs, bias=False)
            with torch.no_grad():
                self.smooth.weight.fill_(1.0 / smooth_k)   # moving average at init

    @staticmethod
    def readout_channels(n_outputs: int, kind: str) -> int:
        return 2 * n_outputs if kind == "velocity" else n_outputs

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return y
        if self.kind == "smooth":
            return self.smooth(F.pad(y, self.pad))
        pos, vel = y.chunk(2, dim=1)                       # (B, J, T') each
        k = min(self.position_steps, pos.shape[-1])
        if k == pos.shape[-1]:
            return pos
        integrated = pos[..., k - 1:k] + torch.cumsum(vel[..., k:] * self.vel_scale, dim=-1)
        return torch.cat([pos[..., :k], integrated], dim=-1)


# =============================================================================
# Channel attention: a gate over the electrodes from per-window statistics
# =============================================================================
class ChannelGate(nn.Module):
    """Squeeze-and-excitation over the C input electrodes.

    Band placement and rotation differ between users, so the informative
    electrodes differ from window to window. The gate reads two scale-free
    statistics per channel (log RMS and log mean |x|, both of the already
    normalised signal), passes them through a tiny MLP and multiplies every
    channel by ``2 * sigmoid(.)``. The last layer starts at zero, so the gate is
    exactly 1 (identity) at initialisation and the model begins as if the gate
    were absent.

    Non-causal: statistics over the whole window -> one gate per window.
    Causal: expanding-window statistics -> one gate per sample; the gate at
    sample ``t`` uses ``x[.. t]`` only.
    """

    def __init__(self, n_ch: int, hidden: int = 32, causal: bool = False, eps: float = 1e-6):
        super().__init__()
        self.causal = causal
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(2 * n_ch, hidden), nn.SiLU(), nn.Linear(hidden, n_ch))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        if not self.causal:
            rms = x.pow(2).mean(-1).add(self.eps).sqrt()          # (B, C)
            mav = x.abs().mean(-1).add(self.eps)
            stats = torch.cat([rms.log(), mav.log()], dim=-1)    # (B, 2C)
            gate = 2.0 * torch.sigmoid(self.mlp(stats))          # (B, C)
            return x * gate.unsqueeze(-1)
        n = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        rms = (x.pow(2).cumsum(-1) / n).add(self.eps).sqrt()     # (B, C, T)
        mav = (x.abs().cumsum(-1) / n).add(self.eps)
        stats = torch.cat([rms.log(), mav.log()], dim=1).transpose(1, 2)   # (B, T, 2C)
        gate = 2.0 * torch.sigmoid(self.mlp(stats)).transpose(1, 2)        # (B, C, T)
        return x * gate


# =============================================================================
# Multi-scale temporal filter bank inside the spatial block
# =============================================================================
class _MultiScaleTemporal(nn.Module):
    """Parallel per-channel temporal convs with different kernel lengths.

    The ``f1`` filters are split evenly over the kernel sizes; each branch pads
    its own input ("same" length, left-only when causal) and the outputs are
    concatenated along the filter axis, followed by one BatchNorm. This is the
    MSST idea (Lin et al. 2026, Measurement 257:118953) with kernel lengths
    chosen for 2 kHz: 15/31/63/127 samples = 7.5/15/31/63 ms. A single kernel
    (``k_e = 65``) sees one window length; the bank sees short bursts and slow
    envelopes at once.
    """

    def __init__(self, f1: int, kernels: tp.Sequence[int], causal: bool):
        super().__init__()
        if f1 % len(kernels):
            raise ValueError(f"f1={f1} must be divisible by the number of ms_kernels ({len(kernels)})")
        per = f1 // len(kernels)
        self.pads = [_same_pad(k, causal) for k in kernels]
        self.branches = nn.ModuleList(nn.Conv2d(1, per, (1, k), bias=False) for k in kernels)
        self.bn = nn.BatchNorm2d(f1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, C, T)
        outs = [conv(F.pad(x, (*pad, 0, 0))) for conv, pad in zip(self.branches, self.pads)]
        return self.bn(torch.cat(outs, dim=1))                   # (B, f1, C, T)


class MsSpatialBlock(SpatialBlock):
    """:class:`SpatialBlock` whose first temporal conv is a multi-scale bank."""

    def __init__(self, n_ch, f1, kernels, spatial_mode, spatial_layers, k_s, depth_mult,
                 f2, k_sep, pool1, pool2, dropout, causal):
        super().__init__(n_ch, f1, max(kernels), spatial_mode, spatial_layers, k_s, depth_mult,
                         f2, k_sep, pool1, pool2, dropout, causal)
        self.temporal = _MultiScaleTemporal(f1, kernels, causal)
        self.pad_e = (0, 0)                      # each branch pads itself


def make_spatial_block(n_ch, f1, k_e, ms_kernels, spatial_mode, spatial_layers, k_s, depth_mult,
                       f2, k_sep, pool1, pool2, dropout, causal) -> SpatialBlock:
    if ms_kernels:
        return MsSpatialBlock(n_ch, f1, tuple(ms_kernels), spatial_mode, spatial_layers, k_s,
                              depth_mult, f2, k_sep, pool1, pool2, dropout, causal)
    return SpatialBlock(n_ch, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
                        f2, k_sep, pool1, pool2, dropout, causal)


# =============================================================================
# Sequence encoder: pre-norm Conformer / transformer blocks at the frame rate
# =============================================================================
def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on the last axis of ``x`` (B, H, T, D)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


def _rotary_tables(T: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None]   # (T, dim/2)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _sinusoid(T: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pe = torch.zeros(T, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe.to(dtype)


def attention_mask(T: int, causal: bool, window: int | None, device) -> torch.Tensor | None:
    """Boolean (T, T) mask, True = may attend; None when unrestricted."""
    if not causal and window is None:
        return None
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    if causal:
        allowed &= j <= i
    if window is not None:
        allowed &= (i - j).abs() <= window
    return allowed


class _SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, rope: bool):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model={d} must be divisible by n_heads={n_heads}")
        self.h = n_heads
        self.hd = d // n_heads
        if rope and self.hd % 2:
            raise ValueError("rotary positions need an even head dimension")
        self.rope = rope
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:   # (B, T, d)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)   # 3 x (B, H, T, hd)
        if self.rope:
            cos, sin = _rotary_tables(T, self.hd, x.device, q.dtype)
            q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class _FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(mult * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _ConvModule(nn.Module):
    """Conformer convolution module: LN -> 1x1 (GLU) -> depthwise k -> BN -> SiLU -> 1x1."""

    def __init__(self, d: int, k: int, dropout: float, causal: bool):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.pad = _same_pad(k, causal)
        self.dw = nn.Conv1d(d, d, k, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                        # (B, T, d)
        y = self.ln(x).transpose(1, 2)                           # (B, d, T)
        y = F.glu(self.pw1(y), dim=1)
        y = self.dw(F.pad(y, self.pad))
        y = self.pw2(F.silu(self.bn(y)))
        return self.drop(y).transpose(1, 2)


class EncoderBlock(nn.Module):
    """Pre-norm block.

    conformer  : x + 1/2 FFN -> x + MHSA -> x + Conv -> x + 1/2 FFN -> LN  (Gulati et al. 2020)
    transformer: x + MHSA -> x + FFN                               (Vaswani et al. 2017, pre-norm)
    """

    def __init__(self, d, n_heads, ff_mult, dropout, attn_dropout, style, conv_kernel, rope, causal):
        super().__init__()
        self.style = style
        if style == "conformer":
            self.ff1 = _FeedForward(d, ff_mult, dropout)
            self.conv = _ConvModule(d, conv_kernel, dropout, causal)
            self.ln_out = nn.LayerNorm(d)
        self.ln_attn = nn.LayerNorm(d)
        self.attn = _SelfAttention(d, n_heads, attn_dropout, rope)
        self.attn_drop = nn.Dropout(dropout)
        self.ff2 = _FeedForward(d, ff_mult, dropout)

    def forward(self, x, mask):
        if self.style == "conformer":
            x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.ln_attn(x), mask))
        if self.style == "conformer":
            x = x + self.conv(x)
            x = x + 0.5 * self.ff2(x)
            return self.ln_out(x)
        return x + self.ff2(x)


class SequenceEncoder(nn.Module):
    """Frame features ``(B, F, T')`` -> contextual features ``(B, d_model, T')``."""

    def __init__(self, in_ch, d_model, n_heads, n_layers, ff_mult, dropout, attn_dropout,
                 block_style, conv_kernel, pos_enc, attn_window, causal):
        super().__init__()
        self.pos_enc = pos_enc
        self.attn_window = attn_window
        self.causal = causal
        self.d_model = d_model
        self.proj = nn.Conv1d(in_ch, d_model, 1)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ff_mult, dropout, attn_dropout, block_style,
                         conv_kernel, pos_enc == "rope", causal)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, F, T')
        x = self.proj(z).transpose(1, 2)                         # (B, T', d)
        if self.pos_enc == "sinusoid":
            x = x + _sinusoid(x.shape[1], self.d_model, x.device, x.dtype)
        x = self.in_drop(x)
        mask = attention_mask(x.shape[1], self.causal, self.attn_window, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x).transpose(1, 2)                      # (B, d, T')


# =============================================================================
# Shared helpers
# =============================================================================
def upsample_frames(y: torch.Tensor, n_samples: int, causal: bool, P: int) -> torch.Tensor:
    """(B, J, T') at the frame rate -> (B, J, T) at the input rate (as StTcnPose)."""
    if not causal:
        return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
    t = torch.arange(n_samples, device=y.device)
    idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)         # last completed frame
    return y[..., idx]


SPATIAL_FIELDS = (
    "f1", "k_e", "ms_kernels", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
    "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
)
ENCODER_FIELDS = (
    "d_model", "n_heads", "n_layers", "ff_mult", "enc_dropout", "attn_dropout",
    "block_style", "conv_kernel", "pos_enc", "attn_window",
)


# =============================================================================
# Full model
# =============================================================================
class StPoseTransformerModel(nn.Module):
    MODEL_FIELDS = (
        "channel_attention", "ca_hidden",
        *SPATIAL_FIELDS, *ENCODER_FIELDS,
        "head_kind", "head_smooth_k",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # channel attention
        channel_attention=False, ca_hidden=32,
        # spatial
        f1=32, k_e=65, ms_kernels=None, spatial_mode="conv2d", spatial_layers=2, k_s=9,
        depth_mult=2, f2=128, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # encoder
        d_model=128, n_heads=4, n_layers=4, ff_mult=4, enc_dropout=0.1, attn_dropout=0.1,
        block_style="conformer", conv_kernel=15, pos_enc="rope", attn_window=None,
        # head
        head_kind="linear", head_smooth_k=5,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, norm="zscore", causal=causal)
        self.gate = ChannelGate(n_in_channels, ca_hidden, causal) if channel_attention else nn.Identity()
        self.spatial = make_spatial_block(
            n_in_channels, f1, k_e, ms_kernels, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        self.encoder = SequenceEncoder(
            self.spatial.out_channels, d_model, n_heads, n_layers, ff_mult, enc_dropout,
            attn_dropout, block_style, conv_kernel, pos_enc, attn_window, causal,
        )
        self.head = nn.Conv1d(d_model, n_outputs, 1)
        self.head_post = PosePost(n_outputs, head_kind, head_smooth_k, causal=causal)
        self.feature_rate = sfreq / self.spatial.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.gate(self.frontend(x))    # (B, C, T)   filtered, scaled, gated
        z = self.spatial(x)                # (B, f2, T') spatial features at ~50 Hz
        z = self.encoder(z)                # (B, d, T')  contextual features
        y = self.head_post(self.head(z))   # (B, J, T')  one angle vector per frame
        y = upsample_frames(y, n_samples, self.causal, self.spatial.downsample)
        return y.transpose(1, 2)           # (B, T, J)


# =============================================================================
# Velocity decoder
# =============================================================================
class VelocityDecoder(nn.Module):
    """Encoder frames ``(B, d, T')`` -> pose trajectory ``(B, J, T')``.

    Two readouts: ``pos`` (1x1 conv, an absolute angle per frame) and ``vel``
    (1x1 conv on the encoder frames, or a linear map on a GRU run over them).
    The velocity readout starts at zero so the trajectory begins as "hold the
    position" instead of a random walk. ``fuse_gain`` blends the integrated
    pose with the position readout at every step (see module docstring).
    """

    def __init__(self, d_model: int, n_outputs: int, position_steps: int = 12,
                 vel_scale: float = 0.02, fuse_gain: float = 0.0,
                 gru_hidden: int = 0, gru_layers: int = 1):
        super().__init__()
        if not 0.0 <= fuse_gain <= 1.0:
            raise ValueError(f"fuse_gain must be in [0, 1], got {fuse_gain}")
        self.n_outputs = n_outputs
        self.position_steps = position_steps
        self.vel_scale = vel_scale
        self.fuse_gain = fuse_gain
        self.pos = nn.Conv1d(d_model, n_outputs, 1)
        if gru_hidden > 0:
            self.gru = nn.GRU(d_model, gru_hidden, num_layers=gru_layers, batch_first=True)
            self.vel = nn.Linear(gru_hidden, n_outputs)
            for name, p in self.gru.named_parameters():
                nn.init.normal_(p, std=0.02) if "weight" in name else nn.init.zeros_(p)
        else:
            self.gru = None
            self.vel = nn.Conv1d(d_model, n_outputs, 1)
        with torch.no_grad():                                   # "hold the position" at init
            self.vel.weight.zero_()
            self.vel.bias.zero_()

    def readouts(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``z`` (B, d, T') -> position (B, J, T'), velocity (B, J, T')."""
        pos = self.pos(z)
        if self.gru is None:
            return pos, self.vel(z)
        h, _ = self.gru(z.transpose(1, 2))                      # (B, T', H), h0 = 0
        return pos, self.vel(h).transpose(1, 2)

    def integrate(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        T = pos.shape[-1]
        k = min(self.position_steps, T)
        if k == T:
            return pos
        step = vel[..., k:] * self.vel_scale                    # (B, J, T'-k)
        if self.fuse_gain == 0.0:                               # pure integration: one cumsum
            integrated = pos[..., k - 1:k] + torch.cumsum(step, dim=-1)
            return torch.cat([pos[..., :k], integrated], dim=-1)
        g = self.fuse_gain
        prev, out = pos[..., k - 1], []
        for t in range(T - k):                                  # first-order IIR, T'-k cheap steps
            prev = (1.0 - g) * (prev + step[..., t]) + g * pos[..., k + t]
            out.append(prev)
        return torch.cat([pos[..., :k], torch.stack(out, dim=-1)], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pos, vel = self.readouts(z)
        return self.integrate(pos, vel)


# =============================================================================
# Full model
# =============================================================================
class StVelTransformerModel(nn.Module):
    MODEL_FIELDS = (
        "channel_attention", "ca_hidden",
        *SPATIAL_FIELDS, *ENCODER_FIELDS,
        "position_steps", "vel_scale", "fuse_gain", "vel_gru_hidden", "vel_gru_layers",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # channel attention
        channel_attention=False, ca_hidden=32,
        # spatial
        f1=32, k_e=65, ms_kernels=None, spatial_mode="conv2d", spatial_layers=2, k_s=9,
        depth_mult=2, f2=128, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # encoder
        d_model=128, n_heads=4, n_layers=4, ff_mult=4, enc_dropout=0.1, attn_dropout=0.1,
        block_style="conformer", conv_kernel=15, pos_enc="rope", attn_window=None,
        # velocity decoder
        position_steps=12, vel_scale=0.02, fuse_gain=0.0, vel_gru_hidden=0, vel_gru_layers=1,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, norm="zscore", causal=causal)
        self.gate = ChannelGate(n_in_channels, ca_hidden, causal) if channel_attention else nn.Identity()
        self.spatial = make_spatial_block(
            n_in_channels, f1, k_e, ms_kernels, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        self.encoder = SequenceEncoder(
            self.spatial.out_channels, d_model, n_heads, n_layers, ff_mult, enc_dropout,
            attn_dropout, block_style, conv_kernel, pos_enc, attn_window, causal,
        )
        self.decoder = VelocityDecoder(d_model, n_outputs, position_steps, vel_scale,
                                       fuse_gain, vel_gru_hidden, vel_gru_layers)
        self.feature_rate = sfreq / self.spatial.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.gate(self.frontend(x))    # (B, C, T)
        z = self.spatial(x)                # (B, f2, T')  ~50 Hz
        z = self.encoder(z)                # (B, d, T')
        y = self.decoder(z)                # (B, J, T')   integrated trajectory
        y = upsample_frames(y, n_samples, self.causal, self.spatial.downsample)
        return y.transpose(1, 2)           # (B, T, J)


# =============================================================================
# Submission wrapper
# =============================================================================
def build_model(n_chans, n_joints, config):
    cfg = dict(config)
    if cfg.get("ms_kernels") is not None:
        cfg["ms_kernels"] = tuple(cfg["ms_kernels"])
    frontend = frontend_from_config(n_chans, cfg)
    model_kwargs = {k: cfg[k] for k in StVelTransformerModel.MODEL_FIELDS}
    return StVelTransformerModel(n_in_channels=n_chans, n_outputs=n_joints,
                       sfreq=cfg["sfreq"], frontend=frontend, **model_kwargs)


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
    # the front end keeps nothing persistent; the loss keeps nothing at all
    state = {k: v for k, v in state.items()
             if not k.startswith(("frontend.", "loss."))}
    if config is not None:
        config = {k: v for k, v in dict(config).items() if k in CONFIG or k in LEGACY_KEYS}
    return state, config


class StVelTransformerRegressor:
    """``fit`` / ``predict`` wrapper owning its device."""

    def __init__(self, n_chans, n_joints, config, device, lr=3e-4, n_epochs=20):
        self.device = device
        self.config = dict(config)
        self.n_chans, self.n_joints = n_chans, n_joints
        self.lr, self.n_epochs = lr, n_epochs
        self.net = build_model(n_chans, n_joints, self.config).to(device)

    def load(self, path):
        state, config = _read_checkpoint(path, self.device)
        merged = merge_config(self.config, config)
        if merged != self.config:
            self.config = merged
            self.net = build_model(self.n_chans, self.n_joints, self.config).to(self.device)
        self.net.load_state_dict(state, strict=True)
        return self

    # ---- optional training through benchopt (``training=True``): plain L1,
    # AdamW. The server never takes this path.
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
            # server delivers. NeuralBench training batches had std ~1e5-1e6.
            print(f"[input] shape {tuple(X.shape)}, |X| mean {X.abs().mean():.4g}, "
                  f"std {X.std():.4g}, max {X.abs().max():.4g}")
            self._logged_input = True
        return self.net(X).transpose(1, 2)                   # (B, J, T) radians


class Solver(CompetSolver):

    name = "StVelTransformer"
    requirements = []  # torch + scipy (via scikit-learn) come with the environment

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        inference = self.train_loader is None
        if inference and weights.exists():
            model = StVelTransformerRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED,
                                      CONFIG, self.device).load(weights)
            n_params = sum(p.numel() for p in model.net.parameters())
            print(f"[loading] {weights} into {self.name} ({n_params / 1e6:.2f} M params, "
                  f"causal={model.config['causal']}, norm={model.config.get('norm')})")
            if meta.get("n_chans") != N_CHANS_TRAINED:
                raise RuntimeError(
                    f"weights expect {N_CHANS_TRAINED} EMG channels, data has "
                    f"{meta.get('n_chans')}")
            if float(meta.get("sfreq", 0)) != float(model.config["sfreq"]):
                warnings.warn(
                    f"data at {meta.get('sfreq')} Hz, weights trained at "
                    f"{model.config['sfreq']} Hz; filter and kernels stay in "
                    "samples (fine for a smoke test, not for scoring).")
            return model
        if inference:
            raise FileNotFoundError(f"{weights} not found (inference-only run)")
        # Training run: build from the data's metadata, train from scratch.
        sfreq = float(meta["sfreq"]); nyq = sfreq / 2
        config = {**CONFIG, "sfreq": sfreq}
        if config["h_freq"] is not None and config["h_freq"] >= nyq:
            config["h_freq"] = None
        if config["l_freq"] is not None and config["l_freq"] >= nyq:
            config["l_freq"] = None
        return StVelTransformerRegressor(meta["n_chans"], meta["n_joints"], config, self.device)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save({"state_dict": model.net.state_dict(), "config": model.config},
                   path / "weights.pt")
