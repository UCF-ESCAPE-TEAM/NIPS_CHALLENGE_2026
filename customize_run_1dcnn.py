"""Run the 1D CNN on the emg/pose task through neuralbench.evaluate_model().

Usage:
    python eval_cnn1d_pose.py            # shape check + debug run (2 epochs, 5 batches)
    python eval_cnn1d_pose.py --full     # shape check + full training (3 seeds)

The model is defined here, not registered in neuraltrain: evaluate_model() takes
the built nn.Module, so no config class, no __init__.py import, and no YAML.
"""

import torch
from torch import nn
import torch.nn.functional as F
import neuralbench.evaluate as nb_evaluate

from neuralbench import check_model, evaluate_model

N_EMG_CHANNELS = 16  # emg2pose wristband
N_JOINT_ANGLES = 20


def _block(c_in: int, c_out: int, kernel: int, stride: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Cnn1dPoseModel(nn.Module):
    """EMG (B, 16, T) -> joint angles (B, T, 20)."""

    def __init__(
        self,
        n_in_channels: int = N_EMG_CHANNELS,
        n_outputs: int = N_JOINT_ANGLES,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_in_channels)
        self.encoder = nn.Sequential(  # 2 kHz -> 50 Hz
            _block(n_in_channels, hidden, 11, 5, dropout),
            _block(hidden, hidden, 9, 4, dropout),
            _block(hidden, hidden, 9, 2, dropout),
        )
        self.body = nn.Sequential(
            _block(hidden, hidden, 9, 1, dropout),
            _block(hidden, hidden, 9, 1, dropout),
        )
        # The model keeps its own 20-output head: it is run without a probe.
        self.head = nn.Conv1d(hidden, n_outputs, kernel_size=1)

    # CHANGE vs. the registered model: forward must accept `channel_positions`
    # (neuralbench passes it by keyword as (B, C, 3)). It is unused here.
    def forward(self, x, channel_positions=None):
        n_samples = x.shape[-1]
        z = self.body(self.encoder(self.input_norm(x)))
        y = self.head(z)
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)  # (B, T, 20): time-major, as neuralbench expects

def main():
    torch.manual_seed(0)
    model = Cnn1dPoseModel()
    nb_evaluate._PROBE_CHANNELS = (16,)
    print(check_model(model, "emg", "pose"))
    scores = evaluate_model(model, "emg", "pose", name="cnn1d-pose",
                            downstream_wrapper=None, debug=True)
    print(scores)


if __name__ == "__main__":
    main()