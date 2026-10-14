from torch import nn


class TinyFm(nn.Module):
    """A small position-conditioned encoder, adaptive in channels and time."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.width = width
        # Channel identity comes from the position, so an unseen montage needs
        # no new parameters -- which is what lets one instance span datasets.
        self.pos_embedding = nn.Linear(3, width)
        self.encoder = nn.Conv1d(1, width, kernel_size=17, stride=8, padding=8)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, channel_positions):
        # x: (batch, channels, samples)
        # channel_positions: (batch, channels, 3)
        batch, channels, samples = x.shape

        # Encode every channel with the same temporal filters.
        tokens = self.encoder(x.reshape(batch * channels, 1, samples))
        tokens = tokens.reshape(batch, channels, self.width, -1)
        tokens = tokens.permute(0, 1, 3, 2)  # (batch, channels, tokens, width)

        tokens = tokens + self.pos_embedding(channel_positions).unsqueeze(2)
        return self.norm(tokens.mean(1))  # (batch, tokens, width)


from neuralbench import check_model

print(check_model(TinyFm(), "eeg", "motor_imagery"))

from neuralbench import evaluate_model

scores = evaluate_model(
    TinyFm(),
    "eeg",
    "motor_imagery",
    name="tiny-fm",
    dataset=["schalk2004bci2000", "leeb2007"],
    debug=True,
)
print(scores[["dataset_name", "test/bal_acc", "n_total_params",
              "n_trainable_params"]])