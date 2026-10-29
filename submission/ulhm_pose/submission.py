"""Codabench submission for the ULHM-style EMG-to-pose model (Track 04).

Self-contained: the Codabench worker extracts ``submission.py`` + ``weights.pt``
and imports nothing else from the ZIP, so the inference half of ``ulhm_pose.py``
(front end, EMG encoder, shared decoder) is copied here. ``weights.pt`` is
written by ``export_ulhm_submission.py`` and holds::

    {"config": {...model kwargs...},
     "fir_kernel": FloatTensor (K,) or None,   # the front-end FIR taps
     "state_dict": {encoder_x.*, decoder.*}}   # encoder_s (pose branch) dropped

The FIR taps are shipped as data so inference needs torch only (no scipy).
Contract (tracks/emg_pose/objective.py): ``predict(X)`` takes ``(B, 16, T)``
microvolt EMG at 2 kHz and returns ``(B, 20, T)`` joint angles in radians. The
front end filters and normalizes each window independently, so no state is
carried across calls.
"""

import torch
import torch.nn.functional as F
from torch import nn

from benchmark_utils.base_solver import CompetSolver


# --------------------------------------------------------------- front end
class EmgFrontEnd(nn.Module):
    """FIR (band-pass / notch) + per-window, per-channel z-score + clamp.

    Parameter-free. The FIR kernel is given as a tensor (designed offline);
    ``None`` skips filtering. Identical to models/emg_frontend.py.
    """

    def __init__(self, kernel: torch.Tensor | None, normalize: bool = True,
                 clip: float | None = 20.0):
        super().__init__()
        self.normalize = normalize
        self.clip = clip
        if kernel is None:
            self.kernel = None
        else:
            self.register_buffer("kernel", kernel.float().clone(), persistent=False)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel.shape[-1]
        pad = k // 2
        mode = "reflect" if x.shape[-1] > pad else "replicate"
        xp = F.pad(x, (pad, pad), mode=mode)
        n = xp.shape[-1] + k - 1
        y = torch.fft.irfft(torch.fft.rfft(xp.float(), n=n)
                            * torch.fft.rfft(self.kernel, n=n), n=n)
        return y[..., k - 1: k - 1 + x.shape[-1]].to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is not None:
            x = self._filter(x)
        if self.normalize:
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, unbiased=False) + 1e-5)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x


# ------------------------------------------------------------- architecture
def _block(c_in, c_out, kernel, stride, dropout):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _encoder(c_in, hidden, latent, dropout, input_norm=True):
    layers = [nn.BatchNorm1d(c_in)] if input_norm else []
    layers += [
        _block(c_in, hidden, 11, 5, dropout),
        _block(hidden, hidden, 9, 4, dropout),
        _block(hidden, hidden, 9, 2, dropout),
        _block(hidden, hidden, 9, 1, dropout),
        nn.Conv1d(hidden, latent, kernel_size=1),
    ]
    return nn.Sequential(*layers)


class UlhmPoseInference(nn.Module):
    """Inference half of UlhmPoseModel: frontend -> encoder_x -> decoder.

    Module names match the training model so its state_dict loads directly.
    """

    def __init__(self, n_in_channels, n_outputs, hidden=128, latent=32, dropout=0.1,
                 fir_kernel=None, normalize=True, clip=20.0):
        super().__init__()
        self.frontend = EmgFrontEnd(fir_kernel, normalize=normalize, clip=clip)
        self.encoder_x = _encoder(n_in_channels, hidden, latent, dropout, input_norm=False)
        self.decoder = nn.Sequential(
            _block(latent, hidden, 9, 1, dropout),
            _block(hidden, hidden, 9, 1, dropout),
            nn.Conv1d(hidden, n_outputs, kernel_size=1),
        )

    def forward(self, x):  # (B, C, T) -> (B, n_outputs, T), radians
        x = self.frontend(x)
        z = self.encoder_x(x)
        y = self.decoder(z)
        return F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = X.to(next(self.parameters()).device, dtype=torch.float32)
        return self.forward(X)  # (B, 20, T)


# ------------------------------------------------------------------ solver
class Solver(CompetSolver):

    name = "ULHM-Pose"

    def load_model(self, meta):
        ckpt = torch.load(meta["submission_dir"] / "weights.pt",
                          map_location=meta["device"], weights_only=True)
        cfg = ckpt["config"]
        model = UlhmPoseInference(
            n_in_channels=meta["n_chans"],
            n_outputs=meta["n_joints"],
            hidden=cfg["hidden"],
            latent=cfg["latent"],
            dropout=cfg.get("dropout", 0.0),
            fir_kernel=ckpt.get("fir_kernel"),
            normalize=cfg.get("normalize", True),
            clip=cfg.get("clip", 20.0),
        )
        # strict: every encoder_x / decoder weight must be present and nothing else
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model.to(meta["device"]).eval()
