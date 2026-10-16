"""Hybrid spatial-2D-CNN + TCN model for dense EMG -> joint-angle regression.

Pipeline (input ``(B, C=16, T)`` at 2 kHz, output ``(B, T, 20)`` radians)::

    EmgPreprocess            band-pass / notch / per-channel z-score (fixed, no params)
    SpatialBlock (2-D CNN)   learn spatial patterns across the 16 electrodes
        temporal Conv2d (1, k_e)         F1 "frequency" filters           (B, F1, C, T)
        [conv2d mode]  n x Conv2d (3, k_s) with circular padding along C  (B, F1, C, T)
                       -> electrodes form a ring around the wrist, so ch15 neighbours ch0
        depthwise Conv2d (C, 1)          collapse the channel axis, D maps per filter
                                                                          (B, F1*D, 1, T)
        AvgPool (1, p1) -> separable conv (1, k_sep) -> AvgPool (1, p2)   (B, F2, T/(p1*p2))
    TemporalBlock (TCN)      EEG-TCNet residual blocks (Ingolfsson et al. 2020)
        n_modules x n_blocks x [dilated conv - BN - ELU - drop] x 2, dilation 2**i
        residual connection optional; causal or centred padding
    Head                     1x1 conv -> n_outputs, linear upsample to T

The TCN follows EEG-TCNet's residual block (BN instead of weight norm, ELU,
ordinary dropout). Differences for this task: several TCN modules can be
stacked (each restarts the dilation schedule at 1), the residual is a switch,
padding can be centred instead of causal (offline windows: the model may use
the whole 5-s window), and the read-out is a per-frame regression rather than
the last frame.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn



# =============================================================================
# Config (NeuralBench entry point)
# =============================================================================
class StTcnPose(nn.Module):
    """Config for :class:`StTcnPoseModel`. All fields can be set in the YAML."""

    # --- spatial 2-D CNN ---------------------------------------------------
    f1: int = 16                 # temporal filters of the first Conv2d
    k_e: int = 65                # its kernel length (samples); 65 @ 2 kHz = 32 ms
    spatial_mode: tp.Literal["depthwise", "conv2d"] = "conv2d"
    spatial_layers: int = 2      # conv2d mode: number of (3, k_s) layers over the channel ring
    k_s: int = 9                 # conv2d mode: temporal extent of those spatial kernels
    depth_mult: int = 2          # D: spatial maps per temporal filter in the collapsing conv
    f2: int = 64                 # filters after the separable conv
    k_sep: int = 16              # separable conv kernel (at rate sfreq / pool1)
    pool1: int = 5               # 2000 -> 400 Hz
    pool2: int = 8               # 400  -> 50 Hz
    spatial_dropout: float = 0.2
    # --- TCN --------------------------------------------------------------
    tcn_modules: int = 2         # how many TCN modules to stack
    tcn_blocks: int = 3          # residual blocks per module (dilations 1, 2, 4, ...)
    tcn_filters: int = 64        # F_T
    tcn_kernel: int = 4          # K_T
    tcn_residual: bool = True    # the optional skip connection
    tcn_causal: bool = False     # True = EEG-TCNet causal padding; False = centred
    tcn_dropout: float = 0.2
    # --- preprocessing (see EmgPreprocess) ------------------------------------
    bandpass: tuple[float, float] | None = (20.0, 450.0)
    bandpass_taps: int = 401
    notch_hz: float | None = 60.0
    notch_harmonics: int = 3
    notch_bw: float = 6.0
    notch_taps: int = 1501
    norm: tp.Literal["zscore", "rms", "none"] = "zscore"

    def build(
        self,
        n_spatial_locations: int,
        n_outputs: int | None,
        frequency: float | None = None,
    ) -> nn.Module:
        assert n_outputs is not None, "StTcnPose requires n_outputs"
        cfg = self.model_dump() if hasattr(self, "model_dump") else dict(self)
        cfg = {k: v for k, v in cfg.items() if k != "name"}
        return StTcnPoseModel(
            n_in_channels=n_spatial_locations,
            n_outputs=n_outputs,
            sfreq=frequency if frequency is not None else 2000.0,
            **cfg,
        )


# =============================================================================
# Spatial block: 2-D CNN over (channels x time)
# =============================================================================
class _RingConv2d(nn.Module):
    """Conv2d over (C, T) with circular padding along C and zero padding along T."""

    def __init__(self, ch: int, k_c: int, k_t: int):
        super().__init__()
        self.pad_c = k_c // 2
        self.conv = nn.Conv2d(ch, ch, (k_c, k_t), padding=(0, k_t // 2), bias=False)

    def forward(self, x):                                   # (B, F, C, T)
        x = F.pad(x, (0, 0, self.pad_c, self.pad_c), mode="circular")
        return self.conv(x)


class SpatialBlock(nn.Module):
    """EEGNet-style front end, optionally with 2-D spatial convs over the electrode ring."""

    def __init__(self, n_ch, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
                 f2, k_sep, pool1, pool2, dropout):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, k_e), padding=(0, k_e // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        layers = []
        if spatial_mode == "conv2d":
            for _ in range(spatial_layers):
                layers += [_RingConv2d(f1, 3, k_s), nn.BatchNorm2d(f1), nn.ELU()]
        self.spatial2d = nn.Sequential(*layers)
        # depthwise (C, 1): one spatial filter per feature map (x depth_mult), as in EEGNet
        self.collapse = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_mult, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_mult),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * depth_mult, f1 * depth_mult, (1, k_sep), padding=(0, k_sep // 2),
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
        x = self.temporal(x.unsqueeze(1))      # (B, F1, C, T)
        x = self.spatial2d(x)                  # (B, F1, C, T)
        x = self.collapse(x)                   # (B, F1*D, 1, T/p1)
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
# Full model
# =============================================================================
class StTcnPoseModel(nn.Module):
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
        tcn_residual=True, tcn_causal=False, tcn_dropout=0.2,
        # preprocessing
        bandpass=(20.0, 450.0), bandpass_taps=401, notch_hz=60.0, notch_harmonics=3,
        notch_bw=6.0, notch_taps=1501, norm="zscore",
    ):
        super().__init__()
        self.pre = EmgPreprocess(
            n_channels=n_in_channels, sfreq=sfreq, bandpass=bandpass, bandpass_taps=bandpass_taps,
            notch_hz=notch_hz, notch_harmonics=notch_harmonics, notch_bw=notch_bw,
            notch_taps=notch_taps, norm=norm,
        )
        self.spatial = SpatialBlock(
            n_in_channels, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout,
        )
        modules, c_in = [], self.spatial.out_channels
        for _ in range(tcn_modules):
            modules.append(TcnModule(c_in, tcn_filters, tcn_kernel, tcn_blocks,
                                     tcn_dropout, tcn_residual, tcn_causal))
            c_in = tcn_filters
        self.tcn = nn.Sequential(*modules)
        self.head = nn.Conv1d(tcn_filters, n_outputs, 1)

        # bookkeeping for the docstring / logs
        self.feature_rate = sfreq / self.spatial.downsample
        self.tcn_receptive_field_s = (
            sum(m.receptive_field - 1 for m in modules) + 1
        ) / self.feature_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.pre(x)                 # (B, C, T)   filtered + normalised
        z = self.spatial(x)             # (B, F2, T')  spatial features at ~50 Hz
        z = self.tcn(z)                 # (B, F_T, T') temporal context
        y = self.head(z)                # (B, n_outputs, T')
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)        # (B, T, n_outputs)