"""Codabench submission -- NeuralDrivePose (Track 4, EMG-to-pose).

Self-contained: the model code below is copied verbatim from
``neuraltrain/models/neuraldrive_pose.py`` (fork commit a5b6633; same
module/parameter names, so a NeuralBench checkpoint loads without renaming).
Only ``torch`` is needed.

Folder layout to zip (flat):
    submission.py        this file
    weights.pt           see below

``weights.pt`` may be any of:
  * a Lightning ``best.ckpt`` written by NeuralBench (``state_dict`` with
    ``model.`` prefixes; architecture taken from ``CONFIG`` below unless the
    checkpoint carries a ``config`` entry),
  * the dict written by ``extract_weights.py`` / ``Solver.save_model``
    (``{"state_dict", "config"}``),
  * a bare ``NeuralDrivePoseModel.state_dict()``.

Contract (tracks/emg_pose/objective.py): ``predict(X)`` takes a float batch
``(B, C, T)`` and returns ``(B, n_joints, T)`` joint angles in **radians**;
windows are independent (no state across calls).

Local test, from the 2026-competition repo (inference only, like the server):
    COMPET_SUBMISSION_DIR=/path/to/neuraldrive_pose \\
        benchopt run tracks/emg_pose -d Simulated -s NeuralDrivePose
or through ``run_submissions.sh``.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# Architecture used for training (mirror of neuraldrive_pose.yaml). Must match
# the weights; a ``config`` entry inside weights.pt overrides it. ``sample_rate``
# is the TRAINING rate: it fixes the kernel sizes in samples and is kept even
# when the evaluation data is at another rate (e.g. the 200 Hz Simulated test).
CONFIG = dict(
    sample_rate=2000,
    band_low_hz=20.0, band_high_hz=500.0, filter_taps=201, learnable_filter=False,
    n_delays=32, n_sources=32, threshold_init=2.0,
    ica_weight=0.1, decorr_weight=0.1, sparsity_weight=0.01, freeze_drive=False,
    twitch_ms=100.0, rate_stride_ms=20.0,
    feature_hidden=128, feature_depth=2, feature_kernel=9,
    decoder_kind="conv", decoder_hidden=128, decoder_depth=2, decoder_kernel=9,
    dropout=0.1, augment=True, gain_jitter=0.3,
)
N_CHANS_TRAINED = 16
N_JOINTS_TRAINED = 20


# =============================================================================
# Model (verbatim copy of neuraldrive_pose.py, minus the pydantic config)
# =============================================================================
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


# ============================================================ 1. neural drive
class NeuralDriveDecoder(nn.Module):
    """EMG ``(B, C, T)`` -> sparse event trains ``(B, K, T)`` at the input rate.

    Unrolled convolutive BSS: front-end -> extension -> whitening -> separation
    -> spike detection. Carries the self-supervised decomposition losses, so it
    can be trained (or pretrained) without any labels.
    """

    def __init__(
        self,
        n_in_channels: int,
        sample_rate: int,
        band_low_hz: float,
        band_high_hz: float,
        filter_taps: int,
        learnable_filter: bool,
        n_delays: int,
        n_sources: int,
        threshold_init: float,
        ica_weight: float,
        decorr_weight: float,
        sparsity_weight: float,
    ):
        super().__init__()
        self.n_delays = n_delays
        n_ext = n_in_channels * n_delays
        self.front_end = EmgFrontEnd(n_in_channels, sample_rate, band_low_hz,
                                     band_high_hz, filter_taps, learnable_filter)
        self.whiten = nn.Conv1d(n_ext, n_ext, 1, bias=False)
        nn.init.eye_(self.whiten.weight[:, :, 0])
        self.separate = nn.Conv1d(n_ext, n_sources, 1, bias=False)
        nn.init.orthogonal_(self.separate.weight[:, :, 0])
        self.threshold = nn.Parameter(torch.full((1, n_sources, 1), threshold_init))
        self.ica_weight = ica_weight
        self.decorr_weight = decorr_weight
        self.sparsity_weight = sparsity_weight

    @torch.no_grad()
    def fit_whitening(self, x: torch.Tensor, eps: float | None = None) -> None:
        """ZCA init of the whitening layer from a batch of raw EMG (B, C, T).

        ``eps`` regularises the eigenvalues; ``None`` uses Negro et al.'s rule
        (mean of the smallest half of the eigenvalues, a noise-variance estimate).
        """
        xe = _extend(self.front_end(x), self.n_delays)
        xe = xe.transpose(0, 1).reshape(xe.shape[1], -1)
        cov = (xe @ xe.T / xe.shape[1]).double()
        evals, evecs = torch.linalg.eigh(cov)
        if eps is None:
            eps = evals[: evals.numel() // 2].mean().clamp_min(1e-8).item()
        w = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
        self.whiten.weight.copy_(w.float().unsqueeze(-1))

    def sources(self, x: torch.Tensor) -> torch.Tensor:
        """Separated sources (B, K, T) before spike detection."""
        x = self.front_end(x)
        return self.separate(self.whiten(_extend(x, self.n_delays)))

    def events(self, s: torch.Tensor) -> torch.Tensor:
        """Sign-free, amplitude-invariant event trains (B, K, T)."""
        p = s.pow(2)
        p = p / (p.mean(dim=-1, keepdim=True) + 1e-6)
        return F.relu(p - F.softplus(self.threshold))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.sources(x)
        return s, self.events(s)

    def losses(self, s: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        z = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
        kurt = z.pow(4).mean(-1) - 3.0  # (B, K): super-Gaussian = spiky
        ica = -torch.log1p(F.relu(kurt)).mean()
        gram = z @ z.transpose(1, 2) / z.shape[-1]  # (B, K, K)
        off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        decorr = off.pow(2).mean()
        sparsity = e.mean()
        return {
            "ica": self.ica_weight * ica,
            "decorr": self.decorr_weight * decorr,
            "sparsity": self.sparsity_weight * sparsity,
        }


# ======================================================= 2. drive -> features
class DriveFeatureExtractor(nn.Module):
    """Event trains ``(B, K, T)`` -> features ``(B, F, T / stride)``.

    Stage A is a depthwise "twitch" filter per source (kernel ``twitch_ms``,
    stride ``rate_stride_ms``), initialised to a Hann window so that at init it
    equals the population firing-rate pooling; training turns it into a learned
    spike-to-force kernel per source. Stage B mixes across sources with conv
    blocks (synergy-level features) at the pooled rate.
    """

    def __init__(
        self,
        n_sources: int,
        sample_rate: int,
        twitch_ms: float,
        rate_stride_ms: float,
        hidden: int,
        depth: int,
        kernel: int,
        dropout: float,
    ):
        super().__init__()
        taps = int(sample_rate * twitch_ms / 1000) | 1  # odd
        stride = int(sample_rate * rate_stride_ms / 1000)
        self.twitch = nn.Conv1d(n_sources, n_sources, taps, stride=stride,
                                padding=taps // 2, groups=n_sources, bias=False)
        with torch.no_grad():
            w = torch.hann_window(taps, periodic=False)
            self.twitch.weight.copy_((w / w.sum()).expand(n_sources, 1, taps))
        self.norm = nn.BatchNorm1d(n_sources)
        layers, c_in = [], n_sources
        for _ in range(depth):
            layers.append(_block(c_in, hidden, kernel, dropout))
            c_in = hidden
        self.mix = nn.Sequential(*layers)
        self.n_features = c_in

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        r = self.norm(torch.log1p(F.relu(self.twitch(e))))
        return self.mix(r)


# ====================================================== 3. features -> pose
class KinematicsDecoder(nn.Module):
    """Features ``(B, F, T')`` -> joint angles ``(B, n_outputs, T')``.

    ``kind="conv"``: temporal conv blocks + 1x1 head.
    ``kind="gru"``: bidirectional GRU + linear head (smoother dynamics).
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden: int,
        depth: int,
        kernel: int,
        dropout: float,
        kind: str = "conv",
    ):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(n_features, hidden // 2, num_layers=max(depth, 1),
                              batch_first=True, bidirectional=True,
                              dropout=dropout if depth > 1 else 0.0)
            self.head = nn.Linear(hidden, n_outputs)
        else:
            layers, c_in = [], n_features
            for _ in range(depth):
                layers.append(_block(c_in, hidden, kernel, dropout))
                c_in = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(c_in, n_outputs, kernel_size=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            h, _ = self.rnn(f.transpose(1, 2))
            return self.head(h).transpose(1, 2)
        return self.head(self.body(f))


# ================================================================ full model
class NeuralDrivePoseModel(nn.Module):
    """``nn.Module`` implementation of :class:`NeuralDrivePose`."""

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sample_rate: int = 2000,
        # neural-drive decoder
        band_low_hz: float = 20.0,
        band_high_hz: float = 500.0,
        filter_taps: int = 201,
        learnable_filter: bool = False,
        n_delays: int = 32,
        n_sources: int = 32,
        threshold_init: float = 2.0,
        ica_weight: float = 0.1,
        decorr_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        freeze_drive: bool = False,
        # feature extractor
        twitch_ms: float = 100.0,
        rate_stride_ms: float = 20.0,
        feature_hidden: int = 128,
        feature_depth: int = 2,
        feature_kernel: int = 9,
        # kinematics decoder
        decoder_kind: str = "conv",
        decoder_hidden: int = 128,
        decoder_depth: int = 2,
        decoder_kernel: int = 9,
        dropout: float = 0.1,
        # augmentation
        augment: bool = True,
        gain_jitter: float = 0.3,
    ):
        super().__init__()
        self.drive = NeuralDriveDecoder(
            n_in_channels, sample_rate, band_low_hz, band_high_hz, filter_taps,
            learnable_filter, n_delays, n_sources, threshold_init,
            ica_weight, decorr_weight, sparsity_weight,
        )
        self.features = DriveFeatureExtractor(
            n_sources, sample_rate, twitch_ms, rate_stride_ms,
            feature_hidden, feature_depth, feature_kernel, dropout,
        )
        self.decoder = KinematicsDecoder(
            self.features.n_features, n_outputs, decoder_hidden, decoder_depth,
            decoder_kernel, dropout, decoder_kind,
        )
        self.freeze_drive = freeze_drive
        if freeze_drive:
            for p in self.drive.parameters():
                p.requires_grad_(False)
        self.augment = augment
        self.gain_jitter = gain_jitter
        self._reg: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------ helpers
    def fit_whitening(self, x: torch.Tensor, eps: float | None = None) -> None:
        self.drive.fit_whitening(x, eps)

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

    def neural_drive(self, x: torch.Tensor) -> torch.Tensor:
        """Event trains (B, K, T): the decoded neural drive, for inspection."""
        return self.drive(x)[1]

    # ------------------------------------------------------------ forward
    def forward(self, x, channel_positions=None):
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)`` (time-major)."""
        n_samples = x.shape[-1]
        x = self._augment(x)
        if self.freeze_drive:
            with torch.no_grad():
                s, e = self.drive(x)
            self._reg = {}
        else:
            s, e = self.drive(x)
            self._reg = self.drive.losses(s, e) if self.training else {}
        y = self.decoder(self.features(e))
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)

    def auxiliary_losses(self, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Self-supervised decomposition losses from the last training forward().

        Same signature as ``UlhmPoseModel.auxiliary_losses``; ``target`` is
        accepted for compatibility and ignored (these losses need no labels).
        """
        reg, self._reg = self._reg, {}
        return reg


# =============================================================================
# Submission wrapper
# =============================================================================
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
    return state, config


class NeuralDrivePoseRegressor:
    """``fit`` / ``predict`` wrapper owning its device."""

    def __init__(self, n_chans, n_joints, config, device, lr=1e-3, n_epochs=20):
        self.device = device
        self.config = dict(config)
        self.n_chans, self.n_joints = n_chans, n_joints
        self.lr, self.n_epochs = lr, n_epochs
        self.net = NeuralDrivePoseModel(n_in_channels=n_chans, n_outputs=n_joints,
                                        **self.config).to(device)

    def load(self, path):
        state, config = _read_checkpoint(path, self.device)
        if config is not None:  # keep only model arguments (drop name etc.)
            config = {k: v for k, v in dict(config).items() if k in CONFIG}
        if config and {**self.config, **config} != self.config:
            self.config = {**self.config, **config}
            self.net = NeuralDrivePoseModel(n_in_channels=self.n_chans,
                                            n_outputs=self.n_joints,
                                            **self.config).to(self.device)
        self.net.load_state_dict(state, strict=True)
        return self

    # ---- optional training through benchopt (``training=True``) ----------
    def fit(self, train_loader):
        first = True
        opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],
                               lr=self.lr)
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                X = X.float(); y = y.float()                 # (B,C,T), (B,J,T)
                if first:                                    # ZCA init once
                    self.net.fit_whitening(X); first = False
                opt.zero_grad()
                pred = self.net(X).transpose(1, 2)           # (B, J, T) radians
                finite = torch.isfinite(y)
                loss = (pred - torch.nan_to_num(y)).abs()[finite].mean()
                for v in self.net.auxiliary_losses(y).values():
                    loss = loss + v
                loss.backward()
                opt.step()
        return self

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if not getattr(self, "_logged_input", False):
            print(f"[input] shape {tuple(X.shape)}, |X| mean {X.abs().mean():.4g}, "
                  f"std {X.std():.4g}, max {X.abs().max():.4g}")
            self._logged_input = True
        return self.net(X).transpose(1, 2)                   # (B, J, T) radians


class Solver(CompetSolver):

    name = "NeuralDrivePose"
    requirements = []  # torch comes with the benchmark environment

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        inference = self.train_loader is None
        if inference and weights.exists():
            # Architecture fixed by training: 16 ch / 20 joints / 2 kHz.
            model = NeuralDrivePoseRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED,
                                             CONFIG, self.device).load(weights)
            print(f"[loading] {weights} into {self.name}")
            if meta.get("n_chans") != N_CHANS_TRAINED:
                raise RuntimeError(
                    f"weights expect {N_CHANS_TRAINED} EMG channels, data has "
                    f"{meta.get('n_chans')}")
            if float(meta.get("sfreq", 0)) != float(model.config["sample_rate"]):
                warnings.warn(
                    f"data at {meta.get('sfreq')} Hz, weights trained at "
                    f"{model.config['sample_rate']} Hz; kernel sizes stay in "
                    "samples (fine for a smoke test, not for scoring).")
            return model
        if inference:
            raise FileNotFoundError(f"{weights} not found (inference-only run)")
        # Training run: build from the data's metadata, train from scratch.
        config = {**CONFIG, "sample_rate": int(meta["sfreq"])}
        return NeuralDrivePoseRegressor(meta["n_chans"], meta["n_joints"],
                                        config, self.device)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save({"state_dict": model.net.state_dict(), "config": model.config},
                   path / "weights.pt")
