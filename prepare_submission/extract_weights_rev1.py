#!/usr/bin/env python3
"""Extract the brain model's weights from a NeuralBench / Lightning best.ckpt
into weights.pt for a Codabench submission. Works for any model.

NeuralBench saves ``best.ckpt`` with ``save_weights_only=True``, so the file
carries NO hyper-parameters: the architecture/front-end config has to come
from somewhere else. Point ``CONFIG_YAML`` at the training YAML (its
``brain_model_config`` block is embedded into weights.pt) so that
``submission.py`` rebuilds exactly the trained model and its own ``CONFIG``
never has to match by hand. If the checkpoint does carry hyper-parameters
(a non-NeuralBench run), those are used instead.

Edit the paths below and run:  python extract_weights.py
"""

from pathlib import Path

import torch

CKPT_PATH = Path("~/Challenge_2026/result/.../best.ckpt").expanduser()
OUT_PATH = Path("~/submissions/st_tcn_pose/weights.pt").expanduser()
# Training YAML whose ``brain_model_config`` describes these weights; None to skip.
CONFIG_YAML = Path("~/Challenge_2026/configs/st_tcn_pose.yaml").expanduser()

# Prefix under which the brain model lives in the Lightning state_dict;
# None = auto-detect ("model." for neuraltrain's BrainModule).
PREFIX = None
# Sub-trees that belong to training or the downstream wrapper, not the model.
DROP = ("probe", "loss", "criterion", "metric", "downstream", "ema.", "teacher.")
KNOWN_PREFIXES = ("model.brain_model.", "brain_model.", "model.model.", "model.", "net.", "")
# Config keys that are not model arguments.
META_KEYS = ("name", "=replace=")


def _config_from_checkpoint(ckpt) -> dict | None:
    hp = ckpt.get("hyper_parameters") or {}
    if hasattr(hp, "model_dump"):
        hp = hp.model_dump()
    for key in ("brain_model_config", "model_config", "config"):
        if isinstance(hp.get(key), dict):
            return {k: v for k, v in hp[key].items() if k not in META_KEYS}
    return None


def _config_from_yaml(path: Path) -> dict | None:
    if path is None or not path.exists():
        return None
    import yaml  # PyYAML ships with lightning / neuralbench

    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    block = doc.get("brain_model_config", doc)
    return {k: v for k, v in block.items() if k not in META_KEYS}


def main() -> None:
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    prefix = PREFIX
    if prefix is None:
        prefix = next(p for p in KNOWN_PREFIXES
                      if sum(k.startswith(p) for k in sd) >= 0.5 * len(sd))

    state, dropped = {}, []
    for k, v in sd.items():
        if not k.startswith(prefix):
            dropped.append(k)
            continue
        nk = k[len(prefix):]
        if any(d in nk for d in DROP):
            dropped.append(k)
        else:
            state[nk] = v

    config = _config_from_checkpoint(ckpt)
    source = "checkpoint hyper_parameters"
    if config is None:
        config = _config_from_yaml(CONFIG_YAML)
        source = f"yaml {CONFIG_YAML}" if config else "none"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": config, "source": str(CKPT_PATH),
                "epoch": ckpt.get("epoch")}, OUT_PATH)

    print(f"prefix  : {prefix!r}")
    print(f"kept    : {len(state)} tensors, {sum(v.numel() for v in state.values()):,} elements")
    if dropped:
        print(f"dropped : {len(dropped)} tensors, e.g. {dropped[:3]}")
    print(f"epoch   : {ckpt.get('epoch')}")
    print(f"config  : from {source}")
    if config:
        for k, v in config.items():
            print(f"          {k} = {v}")
    else:
        print("          WARNING: no config embedded -- submission.py will use its own "
              "CONFIG, which must match the training YAML exactly.")
    print(f"written : {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
