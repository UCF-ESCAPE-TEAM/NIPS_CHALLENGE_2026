"""Package a trained UlhmPose checkpoint into weights.pt for submission.py.

Usage:
    python export_ulhm_submission.py --ckpt path/to/best.ckpt --config ulhm_pose.yaml \
        --out my_submission/
    # then: cd my_submission && zip -j ../ulhm_submission.zip submission.py weights.pt

``--ckpt`` accepts a Lightning ``best.ckpt`` (keys ``model.*`` under
``state_dict``), a NeuralBench-exported state dict, or a bare model state dict.
``--config`` is the model YAML (its ``brain_model_config`` block) so the FIR
kernel and architecture sizes are rebuilt exactly as trained. The pose branch
``encoder_s`` is dropped; the FIR taps are stored as a tensor so the worker
needs no scipy.
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml

from neuraltrain.models.ulhm_pose import UlhmPose # your fork: models/ulhm_pose.py

KEEP = ("encoder_x.", "decoder.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True, help="ulhm_pose.yaml used for training")
    p.add_argument("--out", default="submission_ulhm")
    p.add_argument("--n-chans", type=int, default=16)
    p.add_argument("--n-joints", type=int, default=20)
    a = p.parse_args()

    cfg_yaml = yaml.safe_load(open(a.config))["brain_model_config"]
    cfg_yaml = {k: v for k, v in cfg_yaml.items() if k not in ("name", "=replace=")}
    cfg = UlhmPose(**cfg_yaml)
    model = cfg.build(a.n_chans, a.n_joints)  # rebuilds the FIR from the config

    raw = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(k.startswith("model.") for k in state):            # Lightning / BrainModule
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)                  # full model incl. encoder_s

    kept = {k: v for k, v in model.state_dict().items() if k.startswith(KEEP)}
    kernel = model.frontend.kernel.detach().cpu() if model.frontend.kernel is not None else None
    payload = {
        "config": {"hidden": cfg.hidden, "latent": cfg.latent, "dropout": cfg.dropout,
                   "normalize": cfg.normalize, "clip": cfg.clip,
                   "front_end": {k: getattr(cfg, k) for k in
                                 ("sfreq", "highpass", "lowpass", "notch", "notch_width", "fir_taps")}},
        "fir_kernel": kernel,
        "state_dict": kept,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "weights.pt")
    shutil.copyfile(Path(__file__).with_name("submission.py"), out / "submission.py")
    n = sum(v.numel() for v in kept.values())
    print(f"wrote {out/'weights.pt'} ({n/1e6:.2f} M params, "
          f"FIR {None if kernel is None else kernel.numel()} taps) and {out/'submission.py'}")
    print(f"zip -j {out.name}.zip {out/'submission.py'} {out/'weights.pt'}")


if __name__ == "__main__":
    main()
