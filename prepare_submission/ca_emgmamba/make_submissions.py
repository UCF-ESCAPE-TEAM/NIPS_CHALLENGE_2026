"""Generate submission/ca_emgmamba_{pose,vel}/submission.py from the fork's model file.

The model classes are copied verbatim (from ``class WindowNorm`` to the end of
ca_emgmamba.py) so a NeuralBench checkpoint loads with strict=True. Only the
module header changes: no pydantic config class, no ``mamba_ssm`` import
(``_fast_scan = None`` forces the pure-PyTorch scan, which the Codabench
environment can always run).
"""
import pathlib
import re
import sys

import yaml

FORK = pathlib.Path(sys.argv[1])            # neuroai_jf checkout
OUT = pathlib.Path(sys.argv[2])             # submission root
MODEL = FORK / "neuraltrain-repo/neuraltrain/models/ca_emgmamba.py"
COMMIT = sys.argv[3] if len(sys.argv) > 3 else "unknown"

src = MODEL.read_text()
start = src.index("# =============================================================================\n# Per-window normalisation")
classes = src[start:]
assert "class CaEmgMambaModel" in classes and "import mamba_ssm" not in classes
assert "from mamba_ssm" not in classes and "BaseBrainModelConfig" not in classes

HEAD_DOC = {
    "pose": "PoseHead (one absolute joint-angle vector per frame)",
    "vel": "VelocityHead (position + velocity readouts per frame, integrated with complementary fusion, fuse_gain 0.1)",
}

# (submission folder, training YAML stem in the fork, Solver.name)
VARIANTS = [
    ("ca_emgmamba_pose", "ca_emgmamba_pose", "CaEmgMambaPose"),
    ("ca_emgmamba_vel", "ca_emgmamba_vel", "CaEmgMambaVel"),
    ("ca_emgmamba_vel_R2a", "ca_emgmamba_vel_R2a", "CaEmgMambaVelR2a"),   # R0 with n_layers 2
    ("ca_emgmamba_vel_R4", "ca_emgmamba_vel_R4", "CaEmgMambaVelR4"),   # R0 + time-stretch (training only)
    ("ca_emgmamba_vel_R5", "ca_emgmamba_vel_newton_R5", "CaEmgMambaVelR5"),   # R0 trained with C-Mixup (CaEmgMambaCMixup)
]

for folder, stem, solver in VARIANTS:
    yml = yaml.safe_load((FORK / f"neuralbench-repo/neuralbench/models/{stem}.yaml").read_text())
    cfg = {k: v for k, v in yml["brain_model_config"].items() if k not in ("name", "=replace=")}
    head = cfg["head"]
    cfg_lines = "\n".join(f"    {k}={tuple(v) if isinstance(v, list) else v!r}," for k, v in cfg.items())

    header = f'''"""Codabench submission -- CaEmgMamba, {head} head, config {stem} (Track 4, EMG-to-pose).

Architecture (neuraltrain/models/ca_emgmamba.py in UCF-ESCAPE-TEAM/neuroai_jf,
branch emg-pose-cnn, commit {COMMIT}; training config
neuralbench/models/{stem}.yaml):

    EMG (B, 16, T) --> WindowNorm (per-window per-electrode z-score, no
    parameters, no filters) --> ConvStem (3 strided Conv1d stages, 2 kHz -> 50 Hz,
    SE channel attention after each stage) --> MambaEncoder ({cfg['n_layers']} {'bidirectional' if cfg['bidirectional'] else 'forward-only'}
    Mamba layers, d {cfg['d_model']}) --> {HEAD_DOC[head]}
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
    COMPET_SUBMISSION_DIR=/path/to/{folder} \\\\
        benchopt run tracks/emg_pose -d Simulated -s {solver} --no-cache
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

# Mirror of {stem}.yaml's brain_model_config (minus ``name``). Must match
# the weights; the embedded checkpoint config wins where the two differ.
# Training-only keys (time_stretch_*, mix_*) are carried for the audit trail and
# ignored by build_model: they add no parameters and do nothing at inference.
CONFIG = dict(
{cfg_lines}
)
LEGACY_KEYS = ()
N_CHANS_TRAINED = 16
N_JOINTS_TRAINED = 20


'''

    wrapper = f'''

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
    model_kwargs = {{k: cfg[k] for k in CaEmgMambaModel.MODEL_FIELDS if k in cfg}}
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
        state = {{k.removeprefix("model."): v for k, v in state.items()
                 if k.startswith("model.")}}
    state = {{k: v for k, v in state.items() if not k.startswith("loss.")}}
    if config is not None:
        config = {{k: v for k, v in dict(config).items() if k in CONFIG or k in LEGACY_KEYS}}
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
        merged = {{**self.config, **(config or {{}})}}
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
            print(f"[input] shape {{tuple(X.shape)}}, |X| mean {{X.abs().mean():.4g}}, "
                  f"std {{X.std():.4g}}, max {{X.abs().max():.4g}}")
            self._logged_input = True
        return self.net(X).transpose(1, 2)                   # (B, J, T) radians


class Solver(CompetSolver):

    name = "{solver}"
    requirements = []  # torch comes with the environment; no mamba_ssm needed

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        inference = self.train_loader is None
        if inference and weights.exists():
            model = CaEmgMambaRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED,
                                        CONFIG, self.device).load(weights)
            n_params = sum(p.numel() for p in model.net.parameters())
            print(f"[loading] {{weights}} into {{self.name}} ({{n_params / 1e6:.2f}} M params, "
                  f"head={{model.config['head']}}, scan=torch)")
            if meta.get("n_chans") != N_CHANS_TRAINED:
                raise RuntimeError(
                    f"weights expect {{N_CHANS_TRAINED}} EMG channels, data has "
                    f"{{meta.get('n_chans')}}")
            if float(meta.get("sfreq", 0)) != float(model.config["sfreq"]):
                warnings.warn(
                    f"data at {{meta.get('sfreq')}} Hz, weights trained at "
                    f"{{model.config['sfreq']}} Hz; kernels and strides stay in "
                    "samples (fine for a smoke test, not for scoring).")
            return model
        if inference:
            raise FileNotFoundError(f"{{weights}} not found (inference-only run)")
        # Training run: build from the data's metadata, train from scratch.
        config = {{**CONFIG, "sfreq": float(meta["sfreq"])}}
        return CaEmgMambaRegressor(meta["n_chans"], meta["n_joints"], config, self.device)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save({{"state_dict": model.net.state_dict(), "config": model.config}},
                   path / "weights.pt")
'''
    out = OUT / folder / "submission.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + classes.rstrip() + "\n" + wrapper)
    print(out, len((header + classes + wrapper).splitlines()), "lines")
