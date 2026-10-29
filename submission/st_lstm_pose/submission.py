"""Codabench submission -- StLstmPose (Track 4, EMG-to-pose).

Self-contained: ``EmgPreproc`` (emg_frontend.py), the spatial block
(st_tcn_pose.py) and the LSTM decoder + model (st_lstm_pose.py) are copied
verbatim from neuraltrain (fork commit a5b6633), so a NeuralBench checkpoint
loads with ``strict=True``. Needs torch + scipy (scipy ships with the
benchmark environment through scikit-learn).

Folder layout to zip (flat):
    submission.py        this file
    weights.pt           output of extract_weights.py from the run's best.ckpt
                         (with the st_lstm_pose.yaml config embedded), or the
                         raw best.ckpt, or a bare state dict

The front end has no trainable parameters and its FIR kernel is a
non-persistent buffer, so ``weights.pt`` holds only ``spatial.*``,
``project.*`` and ``decoder.*``; the filter is rebuilt from ``CONFIG`` at
load time.

Contract (tracks/emg_pose/objective.py): ``predict(X)`` takes a float batch
``(B, C, T)`` and returns ``(B, n_joints, T)`` joint angles in **radians**;
windows are independent (no state across calls). The LSTM rollout is a
Python loop over the 250 frames of a window; batch the windows generously.

Local test, from the 2026-competition repo (inference only, like the server):
    COMPET_SUBMISSION_DIR=/path/to/st_lstm_pose \\
        benchopt run tracks/emg_pose -d Simulated -s StLstmPose --no-cache
or through ``run_submissions.sh``.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# Architecture used for training (mirror of st_lstm_pose.yaml). Must match the
# weights; a ``config`` entry inside weights.pt (written by extract_weights.py
# from the training YAML) overrides it.
CONFIG = dict(
    # spatial 2-D CNN (R1 widths)
    f1=32, k_e=65, spatial_mode="conv2d", spatial_layers=2, k_s=9, depth_mult=2,
    f2=128, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
    # LSTM decoder (VEMG2Pose-style)
    feature_dim=64, hidden_size=256, lstm_layers=2, condition_on_pose=True,
    bidirectional=False, parameterization="hybrid", position_steps=12,
    output_scalar=0.01, decoder_dropout=0.0,
    # causality of the front end / spatial block (the LSTM rollout is always causal)
    causal=False,
    # front end (EmgPreproc): scipy FIR band-pass + notch, per-window scaling.
    # sfreq is the TRAINING rate: it fixes the FIR design and is kept even if
    # the evaluation data is at another rate. A per-window scaling is REQUIRED:
    # the server feeds EMG at a different scale than the NeuralBench cache.
    sfreq=2000.0, l_freq=20.0, h_freq=450.0, notch_freqs=(60.0, 120.0, 180.0),
    notch_width=6.0, bp_taps=None, notch_taps=801,
    norm="zscore", scope="channel", signed=True, robust_q=0.005, eps=1e-6, clip=None,
    # per-window bad-channel rejection + ring interpolation (off unless the
    # weights were trained with it; the embedded config carries the setting)
    reject_channels=False, reject_rms_nstd=3.0, reject_mains=60.0,
    reject_mains_nstd=3.0, reject_mains_bw=2.0, reject_segment=400,
    reject_max_frac=0.5, reject_method="std",
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
# Spatial block + TCN classes (verbatim copy of neuraltrain/models/st_tcn_pose.py,
# minus the config); then the LSTM decoder and model from st_lstm_pose.py
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
# Temporal block: EEG-TCNet style TCN
# =============================================================================
class _DilatedConv1d(nn.Module):
    """Dilated conv with 'same' length output; causal (left pad) or centred."""

    def __init__(self, c_in, c_out, k, dilation, causal):
        super().__init__()
        total = (k - 1) * dilation
        self.pad = (total, 0) if causal else (total // 2, total - total // 2)
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class TcnResidualBlock(nn.Module):
    """Two [dilated conv -> BN -> ELU -> dropout] layers, optional residual (EEG-TCNet Fig. 2)."""

    def __init__(self, c_in, c_out, k, dilation, dropout, residual, causal):
        super().__init__()
        self.net = nn.Sequential(
            _DilatedConv1d(c_in, c_out, k, dilation, causal), nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout),
            _DilatedConv1d(c_out, c_out, k, dilation, causal), nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout),
        )
        self.residual = residual
        if residual:
            # 1x1 projection only when the width changes (the "(Optional)" box in the paper)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ELU()

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y = self.act(y + self.skip(x))
        return y


class TcnModule(nn.Module):
    """``n_blocks`` residual blocks with dilations 1, 2, 4, ... (one EEG-TCNet TCN)."""

    def __init__(self, c_in, filters, k, n_blocks, dropout, residual, causal):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(TcnResidualBlock(c_in if i == 0 else filters, filters, k, 2 ** i,
                                           dropout, residual, causal))
        self.blocks = nn.Sequential(*blocks)
        # RFS = 1 + 2 (K_T - 1)(2^L - 1)   (paper, eq. 1)
        self.receptive_field = 1 + 2 * (k - 1) * (2 ** n_blocks - 1)

    def forward(self, x):
        return self.blocks(x)


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
# Full model
# =============================================================================
class StTcnPoseModel(nn.Module):
    # architecture kwargs the config forwards (keeps build() and __init__ in sync)
    MODEL_FIELDS = (
        "f1", "k_e", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
        "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
        "tcn_modules", "tcn_blocks", "tcn_filters", "tcn_kernel",
        "tcn_residual", "tcn_dropout",
        "head_kind", "head_smooth_k", "head_position_steps", "head_vel_scale",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # spatial
        f1=16, k_e=65, spatial_mode="conv2d", spatial_layers=2, k_s=9, depth_mult=2,
        f2=64, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # tcn
        tcn_modules=2, tcn_blocks=3, tcn_filters=64, tcn_kernel=4,
        tcn_residual=True, tcn_dropout=0.2,
        # head
        head_kind="linear", head_smooth_k=5, head_position_steps=12, head_vel_scale=0.02,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = causal
        # parameter-free preprocessing applied to every EMG window, train and test
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, causal=causal)
        self.spatial = SpatialBlock(
            n_in_channels, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        modules, c_in = [], self.spatial.out_channels
        for _ in range(tcn_modules):
            modules.append(TcnModule(c_in, tcn_filters, tcn_kernel, tcn_blocks,
                                     tcn_dropout, tcn_residual, causal))
            c_in = tcn_filters
        self.tcn = nn.Sequential(*modules)
        # ``head`` keeps its name (and state-dict key) for every head_kind; only
        # its width changes for the velocity head (2J channels).
        self.head = nn.Conv1d(tcn_filters, PosePost.readout_channels(n_outputs, head_kind), 1)
        self.head_post = PosePost(n_outputs, head_kind, head_smooth_k,
                                  head_position_steps, head_vel_scale, causal)
        if head_kind == "velocity":
            # start from "hold the position": zero velocity readout at init, so the
            # integrated trajectory does not random-walk before training
            with torch.no_grad():
                self.head.weight[n_outputs:].zero_()
                self.head.bias[n_outputs:].zero_()

        # bookkeeping for the docstring / logs
        self.feature_rate = sfreq / self.spatial.downsample
        self.tcn_receptive_field_s = (
            sum(m.receptive_field - 1 for m in modules) + 1
        ) / self.feature_rate

    def _upsample(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """(B, J, T') at the feature rate -> (B, J, T) at the input rate."""
        if not self.causal:
            return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        # Zero-order hold from the last completed frame. Frame j summarises
        # samples [j*P, (j+1)*P) and is complete at sample (j+1)*P - 1, so
        # sample t may use frame (t+1)//P - 1; samples before the first
        # completed frame fall back to frame 0.
        P = self.spatial.downsample
        t = torch.arange(n_samples, device=y.device)
        idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)
        return y[..., idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.frontend(x)            # (B, C, T)   filtered + normalised, no parameters
        z = self.spatial(x)             # (B, F2, T')  spatial features at ~50 Hz
        z = self.tcn(z)                 # (B, F_T, T') temporal context
        y = self.head_post(self.head(z))   # (B, n_outputs, T')  readout -> trajectory
        y = self._upsample(y, n_samples)
        return y.transpose(1, 2)        # (B, T, n_outputs)


# =============================================================================
# LSTM decoder
# =============================================================================
class LstmPoseDecoder(nn.Module):
    """Features ``(B, F, T')`` at the frame rate -> joint angles ``(B, J, T')``.

    A re-implementation of VEMG2Pose's recurrent decoder (Salter et al. 2024;
    braindecode ``VEMG2Pose``), without the TDS encoder in front of it.

    Two ways to run it
    ------------------
    ``condition_on_pose=True`` (VEMG2Pose, default)
        A Python loop over the T' frames. At step t the LSTM input is
        ``[feature_t ; pose_{t-1}]`` -- the current EMG feature *and the pose
        the decoder itself predicted one frame earlier*. The LSTM updates its
        state, a linear head turns the new hidden vector into a position and
        a velocity for every joint, and the pose is advanced from them (see
        ``parameterization``). Because each step feeds on the previous
        output, the steps are sequential: 250 LSTM calls per 5 s window.
    ``condition_on_pose=False``
        No feedback. The whole feature sequence is passed to ``nn.LSTM`` in
        one call (cuDNN runs the recurrence internally, ~20x faster), then
        the head is applied to every frame at once. May be ``bidirectional``
        (a second LSTM reads the sequence backwards; outputs are
        concatenated) when the model is not meant to be causal.

    Parameterization (how the head output becomes a pose; VEMG2Pose names)
    ---------------------------------------------------------------------
    ``"position"``  pose_t = head_t                      (read the angle out)
    ``"velocity"``  pose_t = pose_{t-1} + head_t          (integrate a velocity)
    ``"hybrid"``    position for the first ``position_steps`` frames, velocity
                    integration afterwards -- what ``regression_vemg2pose``
                    was trained as. The head emits 2J numbers in this mode.
    All head outputs are multiplied by ``output_scalar`` (0.01 in VEMG2Pose)
    so that the trajectory starts near zero and moves in small steps.
    """

    def __init__(self, in_features: int, n_outputs: int, hidden_size: int = 256,
                 num_layers: int = 2, condition_on_pose: bool = True,
                 bidirectional: bool = False, parameterization: str = "hybrid",
                 position_steps: int = 12, output_scalar: float = 0.01,
                 dropout: float = 0.0):
        super().__init__()
        if parameterization not in ("hybrid", "velocity", "position"):
            raise ValueError(f"parameterization must be hybrid|velocity|position, got {parameterization!r}")
        if bidirectional and condition_on_pose:
            raise ValueError("bidirectional=True needs condition_on_pose=False (no feedback loop to run backwards)")
        self.n_outputs = n_outputs
        self.condition_on_pose = condition_on_pose
        self.bidirectional = bidirectional
        self.parameterization = parameterization
        self.position_steps = position_steps
        self.output_scalar = output_scalar

        # The recurrent core. Input = feature (+ previous pose when conditioning).
        #   input_size  : length of the vector fed in at every step
        #   hidden_size : length of h and c; the "width" of the decoder
        #   num_layers  : stacked LSTMs -- layer 2 reads layer 1's h sequence
        #   batch_first : tensors are (B, L, features) rather than (L, B, features)
        #   dropout     : between stacked layers only (PyTorch applies none after the last)
        lstm_in = in_features + (n_outputs if condition_on_pose else 0)
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.activation = nn.LeakyReLU()
        head_in = hidden_size * (2 if bidirectional else 1)
        head_out = n_outputs * (2 if parameterization == "hybrid" else 1)
        self.head = nn.Linear(head_in, head_out)

        # VEMG2Pose / braindecode initialisation: small truncated-normal weights,
        # zero biases. Small LSTM weights keep the gates near 0.5 at the start,
        # so gradients flow through the 250-step recurrence instead of saturating.
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------ one step
    def _advance(self, output: torch.Tensor, prev_pose: torch.Tensor, step: int) -> torch.Tensor:
        """Head output (B, head_out) + previous pose (B, J) -> pose at this step (B, J)."""
        output = output * self.output_scalar
        if self.parameterization == "hybrid":
            position, velocity = torch.split(output, self.n_outputs, dim=-1)
            return position if step < self.position_steps else prev_pose + velocity
        if self.parameterization == "velocity":
            return prev_pose + output
        return output                                              # position

    # ------------------------------------------------------------ forward
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` (B, F, T') -> poses (B, J, T')."""
        B, _, n_steps = z.shape
        seq = z.transpose(1, 2)                                     # (B, T', F): time-major for the LSTM

        if not self.condition_on_pose:
            # -------- fast path: whole sequence through the LSTM at once.
            # h0/c0 default to zeros. ``out`` holds the top layer's h at every step.
            out, _ = self.lstm(seq)                                 # (B, T', hidden * dirs)
            head = self.head(self.activation(out))                  # (B, T', head_out)
            poses, pose = [], z.new_zeros(B, self.n_outputs)
            for t in range(n_steps):                                # integration only (cheap)
                pose = self._advance(head[:, t], pose, t)
                poses.append(pose)
            return torch.stack(poses, dim=-1)                       # (B, J, T')

        # -------- VEMG2Pose path: step-by-step rollout with pose feedback.
        # State for every layer: (num_layers, B, hidden). Zeros = "nothing seen yet".
        n_layers = self.lstm.num_layers
        h = z.new_zeros(n_layers, B, self.lstm.hidden_size)
        c = torch.zeros_like(h)
        pose = z.new_zeros(B, self.n_outputs)                       # regression mode: start from zero pose
        poses = []
        for t in range(n_steps):
            # 1. build this step's input: current EMG feature + the pose we just predicted
            x_t = torch.cat([seq[:, t], pose], dim=-1).unsqueeze(1)  # (B, 1, F + J)
            # 2. one recurrent step: the LSTM reads x_t and its own memory (h, c),
            #    decides what to forget / store / expose, and returns the new (h, c).
            out, (h, c) = self.lstm(x_t, (h, c))                    # out: (B, 1, hidden)
            # 3. read the pose out of the exposed state and advance the trajectory
            pose = self._advance(self.head(self.activation(out[:, 0])), pose, t)
            poses.append(pose)
        return torch.stack(poses, dim=-1)                           # (B, J, T')


# =============================================================================
# Full model
# =============================================================================
class StLstmPoseModel(nn.Module):
    MODEL_FIELDS = (
        "f1", "k_e", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
        "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
        "feature_dim", "hidden_size", "lstm_layers", "condition_on_pose", "bidirectional",
        "parameterization", "position_steps", "output_scalar", "decoder_dropout",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # spatial (as StTcnPose)
        f1=16, k_e=65, spatial_mode="conv2d", spatial_layers=2, k_s=9, depth_mult=2,
        f2=64, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # LSTM decoder
        feature_dim=64, hidden_size=256, lstm_layers=2, condition_on_pose=True,
        bidirectional=False, parameterization="hybrid", position_steps=12,
        output_scalar=0.01, decoder_dropout=0.0,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        if causal and bidirectional:
            raise ValueError("a bidirectional LSTM reads the future: not allowed with causal=True")
        self.causal = causal
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, norm="zscore", causal=causal)
        # Same spatial block as StTcnPose -> (B, f2, T') at sfreq / (pool1 * pool2)
        self.spatial = SpatialBlock(
            n_in_channels, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        # VEMG2Pose projects its encoder output to a small feature vector before
        # the LSTM; a 1x1 conv is that linear map applied at every frame.
        self.project = nn.Conv1d(f2, feature_dim, 1)
        self.decoder = LstmPoseDecoder(
            feature_dim, n_outputs, hidden_size, lstm_layers, condition_on_pose,
            bidirectional, parameterization, position_steps, output_scalar, decoder_dropout,
        )
        self.feature_rate = sfreq / self.spatial.downsample

    def _upsample(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """(B, J, T') at the frame rate -> (B, J, T) at the input rate (as StTcnPose)."""
        if not self.causal:
            return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        P = self.spatial.downsample
        t = torch.arange(n_samples, device=y.device)
        idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)         # last completed frame
        return y[..., idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.frontend(x)              # (B, C, T)    filtered + scaled, no parameters
        z = self.spatial(x)               # (B, f2, T')  spatial features at ~50 Hz
        z = self.project(z)               # (B, feature_dim, T')
        y = self.decoder(z)               # (B, J, T')   LSTM rollout, one pose per frame
        y = self._upsample(y, n_samples)  # (B, J, T)
        return y.transpose(1, 2)          # (B, T, J)


# =============================================================================
# Submission wrapper
# =============================================================================
def build_model(n_chans, n_joints, config):
    cfg = dict(config)
    frontend = frontend_from_config(n_chans, cfg)
    model_kwargs = {k: cfg[k] for k in StLstmPoseModel.MODEL_FIELDS}
    return StLstmPoseModel(n_in_channels=n_chans, n_outputs=n_joints,
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
    # the front end keeps nothing persistent; drop any stray buffer (and any
    # loss-side state a training run might have saved)
    state = {k: v for k, v in state.items()
             if not k.startswith(("frontend.", "loss."))}
    if config is not None:
        config = dict(config)
        if "causal" not in config and "tcn_causal" in config:
            if config["tcn_causal"]:
                warnings.warn("checkpoint was trained with tcn_causal=True but a "
                              "non-causal front end / spatial block; loading it as "
                              "causal=False (the TCN padding differs: expect a "
                              "small accuracy change).")
            config["causal"] = False
        config = {k: v for k, v in config.items() if k in CONFIG or k in LEGACY_KEYS}
    return state, config


class StLstmPoseRegressor:
    """``fit`` / ``predict`` wrapper owning its device."""

    def __init__(self, n_chans, n_joints, config, device, lr=1e-3, n_epochs=20):
        self.device = device
        self.config = dict(config)
        self.n_chans, self.n_joints = n_chans, n_joints
        self.lr, self.n_epochs = lr, n_epochs
        self.net = build_model(n_chans, n_joints, self.config).to(device)

    def load(self, path):
        state, config = _read_checkpoint(path, self.device)
        merged = merge_config(self.config, config)      # legacy configs replace the new keys
        if merged != self.config:
            self.config = merged
            self.net = build_model(self.n_chans, self.n_joints, self.config).to(self.device)
        self.net.load_state_dict(state, strict=True)
        return self

    # ---- optional training through benchopt (``training=True``) ----------
    def fit(self, train_loader):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                X = X.float(); y = y.float()                 # (B,C,T), (B,J,T)
                opt.zero_grad()
                pred = self.net(X).transpose(1, 2)           # (B, J, T) radians
                finite = torch.isfinite(y)
                loss = (pred - torch.nan_to_num(y)).abs()[finite].mean()
                loss.backward()
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

    name = "StLstmPose"
    requirements = []  # torch + scipy (via scikit-learn) come with the environment

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        inference = self.train_loader is None
        if inference and weights.exists():
            model = StLstmPoseRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED,
                                       CONFIG, self.device).load(weights)
            print(f"[loading] {weights} into {self.name} "
                  f"(causal={model.config['causal']}, norm={model.config.get('norm', 'legacy zscore')})")
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
        return StLstmPoseRegressor(meta["n_chans"], meta["n_joints"], config, self.device)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save({"state_dict": model.net.state_dict(), "config": model.config},
                   path / "weights.pt")
