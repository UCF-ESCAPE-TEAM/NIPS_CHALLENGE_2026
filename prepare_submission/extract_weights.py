#!/usr/bin/env python3
"""Extract the brain model's weights from a NeuralBench / Lightning best.ckpt
into weights.pt for a Codabench submission. Works for any model.

Edit the two paths below and run:  python extract_weights.py
"""

from pathlib import Path

import torch

CKPT_PATH = Path("/home/ji045311/NIPS_2026_Challenge/Results/neuralbench.main.Experiment.run,1/seed=33,task_name=pose,loss.name=L1Loss,brain_model_output_size=20,delete_checkpoints_on_exit=False,augmentation={num_bands=1,pr...543-2b405cf3/best.ckpt").expanduser()
OUT_PATH = Path("/home/ji045311/NIPS_2026_Challenge/submission/stcn/weights.pt").expanduser()

# Prefix under which the brain model lives in the Lightning state_dict;
# None = auto-detect ("model." for neuraltrain's BrainModule).
PREFIX = None
# Sub-trees that belong to training or the downstream wrapper, not the model.
DROP = ("probe", "loss", "criterion", "metric", "downstream", "ema.", "teacher.")
KNOWN_PREFIXES = ("model.brain_model.", "brain_model.", "model.model.", "model.", "net.", "")


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

    # architecture config, if the checkpoint carries it (neuraltrain hparams)
    config = None
    hp = ckpt.get("hyper_parameters") or {}
    if hasattr(hp, "model_dump"):
        hp = hp.model_dump()
    for key in ("brain_model_config", "model_config", "config"):
        if isinstance(hp.get(key), dict):
            config = {k: v for k, v in hp[key].items() if k not in ("name", "=replace=")}
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": config, "source": str(CKPT_PATH),
                "epoch": ckpt.get("epoch")}, OUT_PATH)

    print(f"prefix  : {prefix!r}")
    print(f"kept    : {len(state)} tensors, {sum(v.numel() for v in state.values()):,} elements")
    if dropped:
        print(f"dropped : {len(dropped)} tensors, e.g. {dropped[:3]}")
    print(f"config  : {config if config else 'none (submission uses its CONFIG)'}")
    print(f"written : {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
