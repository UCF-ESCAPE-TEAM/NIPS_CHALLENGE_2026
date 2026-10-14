"""ULHM-style EMG-to-pose model: two encoders, one shared pose decoder.

Port of the MNIST "case 1" recipe (sparse image / full image / shared decoder):

    MNIST                      EMG challenge
    -----------------------    ---------------------------------
    sparse image  x            EMG window            (B, 16, T)
    full image    s            joint angles          (B, 20, T)   training only
    encoder_x, encoder_s       EMG encoder, pose encoder -> shared latent
    shared decoder -> image    shared decoder -> joint angles

    loss = recon_x + recon_s + 0.1 * consistency(z_x, z_s)

``recon_x`` is the task loss neuralbench already applies to ``forward``'s output.
``recon_s`` and ``consistency`` are returned by :meth:`auxiliary_losses`, which the
patched ``BrainModule.training_step`` adds during training. At inference only
``encoder_x`` and ``decoder`` are used.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig


class UlhmPose(BaseBrainModelConfig):
    """Config for the ULHM-style EMG-to-pose model.

    Parameters
    ----------
    hidden : int
        Channels in the convolutional blocks.
    latent : int
        Channels of the shared latent (at 1/40 of the input rate, 50 Hz for 2 kHz).
    dropout : float
        Dropout probability after each block.
    recon_s_weight : float
        Weight of the pose -> latent -> pose reconstruction loss.
    consistency_weight : float
        Weight of the latent consistency loss between EMG and pose encoders.
    consistency : {"mse", "cosine"}
        Form of the consistency loss ("mse" as in MNIST case 1).
    """

    hidden: int = 128
    latent: int = 32
    dropout: float = 0.1
    recon_s_weight: float = 1.0
    consistency_weight: float = 0.1
    consistency: tp.Literal["mse", "cosine"] = "mse"

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "UlhmPose requires n_outputs"
        return UlhmPoseModel(
            n_in_channels=n_spatial_locations,
            n_outputs=n_outputs,
            hidden=self.hidden,
            latent=self.latent,
            dropout=self.dropout,
            recon_s_weight=self.recon_s_weight,
            consistency_weight=self.consistency_weight,
            consistency=self.consistency,
        )


def _block(c_in: int, c_out: int, kernel: int, stride: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _encoder(c_in: int, hidden: int, latent: int, dropout: float) -> nn.Module:
    """(B, c_in, T) -> (B, latent, T / 40). Same strides for both modalities."""
    return nn.Sequential(
        nn.BatchNorm1d(c_in),
        _block(c_in, hidden, 11, 5, dropout),
        _block(hidden, hidden, 9, 4, dropout),
        _block(hidden, hidden, 9, 2, dropout),
        _block(hidden, hidden, 9, 1, dropout),
        nn.Conv1d(hidden, latent, kernel_size=1),
    )


class UlhmPoseModel(nn.Module):
    """``nn.Module`` implementation of :class:`UlhmPose`."""

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        hidden: int = 128,
        latent: int = 32,
        dropout: float = 0.1,
        recon_s_weight: float = 1.0,
        consistency_weight: float = 0.1,
        consistency: str = "mse",
    ):
        super().__init__()
        self.encoder_x = _encoder(n_in_channels, hidden, latent, dropout)  # EMG
        self.encoder_s = _encoder(n_outputs, hidden, latent, dropout)  # pose
        self.decoder = nn.Sequential(  # shared: latent -> pose at 50 Hz
            _block(latent, hidden, 9, 1, dropout),
            _block(hidden, hidden, 9, 1, dropout),
            nn.Conv1d(hidden, n_outputs, kernel_size=1),
        )
        self.recon_s_weight = recon_s_weight
        self.consistency_weight = consistency_weight
        self.consistency = consistency
        self._z_x: torch.Tensor | None = None

    def _decode(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        y = self.decoder(z)  # (B, n_outputs, T / 40)
        return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)

    def forward(self, x):
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)`` (time-major)."""
        z_x = self.encoder_x(x)
        self._z_x = z_x  # kept for auxiliary_losses; overwritten every call
        return self._decode(z_x, x.shape[-1]).transpose(1, 2)

    def auxiliary_losses(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extra training losses from the pose branch.

        Call right after ``forward`` on the same batch.

        Parameters
        ----------
        target : Tensor
            Joint angles ``(B, n_outputs, T)`` as extracted by neuralbench
            (channel-major), aligned with the EMG window.
        """
        assert self._z_x is not None, "call forward() before auxiliary_losses()"
        z_x, self._z_x = self._z_x, None
        finite = torch.isfinite(target)
        s = torch.nan_to_num(target)

        z_s = self.encoder_s(s)
        recon_s = self._decode(z_s, s.shape[-1])
        loss_recon_s = (recon_s - s).abs()[finite].mean()

        if self.consistency == "cosine":
            cos = F.cosine_similarity(z_x.flatten(1), z_s.flatten(1), dim=1)
            loss_consistency = (1.0 - cos).mean()
        else:
            loss_consistency = F.mse_loss(z_x, z_s)

        return {
            "recon_s": self.recon_s_weight * loss_recon_s,
            "consistency": self.consistency_weight * loss_consistency,
        }
