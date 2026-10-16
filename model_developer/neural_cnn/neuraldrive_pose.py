"""Neural-drive-style EMG-to-pose model: an unrolled HD-EMG decomposition pipeline.

The encoder mimics FastICA/CKC decomposition stage by stage (extension ->
whitening -> separation -> spike detection -> population firing rate) and is
trained end to end on the task loss. No BSS/ICA output is used as a teacher:
the ICA *objective* (non-Gaussianity + decorrelation + sparsity of the sources)
is returned by :meth:`auxiliary_losses` as a self-supervised regulariser, which
keeps the source layer from collapsing into an amplitude/envelope detector.

    stage                      layer                                   rate
    -------------------------  --------------------------------------  ------
    band-pass (Negro 2016)     fixed zero-phase FIR, depthwise conv     2 kHz
    mean / gain removal        InstanceNorm1d(C)                       2 kHz
    extension (R delays)       fixed shift-and-stack, C -> C*R         2 kHz
    whitening                  1x1 conv CR -> CR   (+ decorr. loss)    2 kHz
    separation vectors         1x1 conv CR -> K    (+ ICA loss)        2 kHz
    spike detection            power -> per-source norm -> threshold   2 kHz
    population firing rate     avg-pool 100 ms / stride 20 ms          50 Hz
    decoder                    2 conv blocks + 1x1 head -> n_outputs   50 Hz
    upsample                   linear interpolation to input rate      2 kHz

``forward``'s output goes to the task loss as usual; :meth:`auxiliary_losses`
returns the decomposition losses for the patched ``BrainModule.training_step``
to add, exactly as :class:`UlhmPoseModel` does.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig


class NeuralDrivePose(BaseBrainModelConfig):
    """Config for the neural-drive-style EMG-to-pose model.

    Parameters
    ----------
    band_low_hz, band_high_hz : float
        Band-pass edges of the EMG front-end (Negro et al. 2016 use 10-900 Hz
        at 10 kHz; 20-500 Hz is the usual sEMG decomposition band at 2 kHz).
        Set ``band_high_hz`` <= 0 to disable the filter.
    filter_taps : int
        Length of the zero-phase FIR (odd). 201 taps = 100 ms at 2 kHz.
    learnable_filter : bool
        Fine-tune the FIR from its band-pass initialisation (EEGNet-style).
    n_delays : int
        Extension factor R. Negro et al. use R ~ 1000 / n_channels (64 for
        16 channels); 32 (16 ms at 2 kHz) covers a MUAP at a quarter of the
        memory.
    n_sources : int
        Number of separation vectors K (sources at the input rate).
    hidden : int
        Channels in the 50 Hz decoder blocks.
    dropout : float
        Dropout probability in the decoder blocks.
    sample_rate : int
        Input sampling rate in Hz (2000 for emg2pose).
    rate_window_ms, rate_stride_ms : float
        Window and stride of the population firing-rate pooling.
    threshold_init : float
        Initial spike-detection threshold on mean-normalised source power.
    ica_weight : float
        Weight of the non-Gaussianity (kurtosis) contrast on the sources.
    decorr_weight : float
        Weight of the between-source decorrelation penalty.
    sparsity_weight : float
        Weight of the L1 penalty on the event trains.
    augment : bool
        Per-channel gain jitter and +-1 channel rotation during training.
    gain_jitter : float
        Half-range of the multiplicative gain jitter.
    """

    band_low_hz: float = 20.0
    band_high_hz: float = 500.0
    filter_taps: int = 201
    learnable_filter: bool = False
    n_delays: int = 32
    n_sources: int = 32
    hidden: int = 128
    dropout: float = 0.1
    sample_rate: int = 2000
    rate_window_ms: float = 100.0
    rate_stride_ms: float = 20.0
    threshold_init: float = 2.0
    ica_weight: float = 0.1
    decorr_weight: float = 0.1
    sparsity_weight: float = 0.01
    augment: bool = True
    gain_jitter: float = 0.3

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "NeuralDrivePose requires n_outputs"
        return NeuralDrivePoseModel(
            n_in_channels=n_spatial_locations,
            n_outputs=n_outputs,
            band_low_hz=self.band_low_hz,
            band_high_hz=self.band_high_hz,
            filter_taps=self.filter_taps,
            learnable_filter=self.learnable_filter,
            n_delays=self.n_delays,
            n_sources=self.n_sources,
            hidden=self.hidden,
            dropout=self.dropout,
            sample_rate=self.sample_rate,
            rate_window_ms=self.rate_window_ms,
            rate_stride_ms=self.rate_stride_ms,
            threshold_init=self.threshold_init,
            ica_weight=self.ica_weight,
            decorr_weight=self.decorr_weight,
            sparsity_weight=self.sparsity_weight,
            augment=self.augment,
            gain_jitter=self.gain_jitter,
        )


def _bandpass_fir(low_hz: float, high_hz: float, sample_rate: float, n_taps: int) -> torch.Tensor:
    """Windowed-sinc (Hamming) band-pass FIR, symmetric -> zero phase with centred padding."""
    assert n_taps % 2 == 1, "filter_taps must be odd for zero phase"
    n = torch.arange(n_taps, dtype=torch.float64) - (n_taps - 1) / 2
    window = torch.hamming_window(n_taps, periodic=False, dtype=torch.float64)

    def lowpass(fc_hz: float) -> torch.Tensor:
        fc = fc_hz / sample_rate  # cycles / sample
        h = 2 * fc * torch.sinc(2 * fc * n) * window
        return h / h.sum()  # unit DC gain

    high_hz = min(high_hz, 0.499 * sample_rate)
    h = lowpass(high_hz)
    if low_hz > 0:
        h = h - lowpass(low_hz)
    return h.to(torch.float32)


class EmgFrontEnd(nn.Module):
    """Negro et al. (2016) conditioning: band-pass + mean subtraction.

    The band-pass is a depthwise zero-phase FIR; the mean subtraction (and
    per-channel gain removal) is done by the InstanceNorm that follows.
    """

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        low_hz: float = 20.0,
        high_hz: float = 500.0,
        n_taps: int = 201,
        learnable: bool = False,
    ):
        super().__init__()
        self.enabled = high_hz > 0
        self.norm = nn.InstanceNorm1d(n_channels, affine=False)
        if not self.enabled:
            return
        h = _bandpass_fir(low_hz, high_hz, sample_rate, n_taps)
        self.filt = nn.Conv1d(n_channels, n_channels, n_taps, padding=n_taps // 2,
                              groups=n_channels, bias=False)
        with torch.no_grad():
            self.filt.weight.copy_(h.expand(n_channels, 1, n_taps))
        self.filt.weight.requires_grad_(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            x = self.filt(x)
        return self.norm(x)


def _extend(x: torch.Tensor, n_delays: int) -> torch.Tensor:
    """Extension step of CKC/FastICA: (B, C, T) -> (B, C*R, T), delay r in 0..R-1."""
    T = x.shape[-1]
    shifted = [F.pad(x, (r, 0))[..., :T] for r in range(n_delays)]
    return torch.cat(shifted, dim=1)


def _block(c_in: int, c_out: int, kernel: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class NeuralDrivePoseModel(nn.Module):
    """``nn.Module`` implementation of :class:`NeuralDrivePose`."""

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        band_low_hz: float = 20.0,
        band_high_hz: float = 500.0,
        filter_taps: int = 201,
        learnable_filter: bool = False,
        n_delays: int = 32,
        n_sources: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        sample_rate: int = 2000,
        rate_window_ms: float = 100.0,
        rate_stride_ms: float = 20.0,
        threshold_init: float = 2.0,
        ica_weight: float = 0.1,
        decorr_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        augment: bool = True,
        gain_jitter: float = 0.3,
    ):
        super().__init__()
        self.n_delays = n_delays
        n_ext = n_in_channels * n_delays

        # 1. EMG front-end: band-pass FIR + mean subtraction / gain removal
        self.front_end = EmgFrontEnd(n_in_channels, sample_rate, band_low_hz,
                                     band_high_hz, filter_taps, learnable_filter)
        # 2. whitening: 1x1 conv, identity init; see fit_whitening() for ZCA init
        self.whiten = nn.Conv1d(n_ext, n_ext, 1, bias=False)
        nn.init.eye_(self.whiten.weight[:, :, 0])
        # 3. separation vectors
        self.separate = nn.Conv1d(n_ext, n_sources, 1, bias=False)
        nn.init.orthogonal_(self.separate.weight[:, :, 0])
        # 4. spike-detection threshold, per source, on mean-normalised power
        self.threshold = nn.Parameter(torch.full((1, n_sources, 1), threshold_init))
        # 5. population firing rate
        self.rate_kernel = int(sample_rate * rate_window_ms / 1000)
        self.rate_stride = int(sample_rate * rate_stride_ms / 1000)
        self.rate_norm = nn.BatchNorm1d(n_sources)
        # 6. decoder at the pooled rate
        self.decoder = nn.Sequential(
            _block(n_sources, hidden, 9, dropout),
            _block(hidden, hidden, 9, dropout),
        )
        self.head = nn.Conv1d(hidden, n_outputs, kernel_size=1)

        self.sample_rate = sample_rate
        self.ica_weight = ica_weight
        self.decorr_weight = decorr_weight
        self.sparsity_weight = sparsity_weight
        self.augment = augment
        self.gain_jitter = gain_jitter
        self._reg: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------ helpers
    @torch.no_grad()
    def fit_whitening(self, x: torch.Tensor, eps: float | None = None) -> None:
        """Initialise the whitening layer by ZCA on a batch of raw EMG (B, C, T).

        ``eps`` regularises the eigenvalues; ``None`` uses Negro et al.'s rule
        (the mean of the smallest half of the eigenvalues of the extended
        correlation matrix, i.e. an estimate of the noise variance).
        """
        xe = _extend(self.front_end(x), self.n_delays)
        xe = xe.transpose(0, 1).reshape(xe.shape[1], -1)  # (C*R, B*T)
        cov = (xe @ xe.T / xe.shape[1]).double()
        evals, evecs = torch.linalg.eigh(cov)  # ascending
        if eps is None:
            eps = evals[: evals.numel() // 2].mean().clamp_min(1e-8).item()
        w = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
        self.whiten.weight.copy_(w.float().unsqueeze(-1))

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.augment):
            return x
        B, C, _ = x.shape
        gain = 1.0 + self.gain_jitter * (2 * torch.rand(B, C, 1, device=x.device) - 1)
        x = x * gain
        shift = int(torch.randint(-1, 2, (1,)).item())  # wristband electrode shift
        if shift:
            x = torch.roll(x, shifts=shift, dims=1)
        return x

    def sources(self, x: torch.Tensor) -> torch.Tensor:
        """Separated sources (B, K, T) at the input rate, before spike detection."""
        x = self.front_end(self._augment(x))
        return self.separate(self.whiten(_extend(x, self.n_delays)))

    def events(self, s: torch.Tensor) -> torch.Tensor:
        """Sparse non-negative event trains (B, K, T): sign-free, amplitude-invariant."""
        p = s.pow(2)
        p = p / (p.mean(dim=-1, keepdim=True) + 1e-6)
        return F.relu(p - F.softplus(self.threshold))

    def firing_rate(self, e: torch.Tensor) -> torch.Tensor:
        """(B, K, T) events -> (B, K, T/stride) population rate."""
        r = F.avg_pool1d(e, self.rate_kernel, self.rate_stride,
                         padding=self.rate_kernel // 2, count_include_pad=False)
        return self.rate_norm(torch.log1p(r))

    # ------------------------------------------------------------ forward
    def forward(self, x, channel_positions=None):
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)`` (time-major)."""
        n_samples = x.shape[-1]
        s = self.sources(x)
        e = self.events(s)
        self._reg = self._regularisers(s, e) if self.training else {}
        y = self.head(self.decoder(self.firing_rate(e)))
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)

    def _regularisers(self, s: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        z = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
        # ICA contrast: maximise excess kurtosis (spiky sources are super-Gaussian)
        kurt = z.pow(4).mean(-1) - 3.0  # (B, K)
        ica = -torch.log1p(F.relu(kurt)).mean()
        # deflation / decorrelation between sources
        gram = z @ z.transpose(1, 2) / z.shape[-1]  # (B, K, K)
        off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        decorr = off.pow(2).mean()
        # sparsity of the event trains
        sparsity = e.mean()
        return {
            "ica": self.ica_weight * ica,
            "decorr": self.decorr_weight * decorr,
            "sparsity": self.sparsity_weight * sparsity,
        }

    def auxiliary_losses(self, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Self-supervised decomposition losses from the last training forward().

        Same signature as ``UlhmPoseModel.auxiliary_losses``; ``target`` is
        accepted for compatibility and ignored (these losses need no labels).
        """
        reg, self._reg = self._reg, {}
        return reg
