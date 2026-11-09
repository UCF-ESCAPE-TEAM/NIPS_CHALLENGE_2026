"""Strict-load + output-parity check for the two CaEmgMamba submissions.

For each head: build the fork model from its YAML, save a NeuralBench-style
checkpoint ("model." prefix) and a rev1-style weights.pt ({"state_dict",
"config"}), load both through submission.py (benchmark_utils stubbed), compare
predictions against the fork model, check shape / finiteness / time variation,
and time predict on 64 windows.

Usage: python check_submissions.py <fork_root> <submission_root>
"""
import importlib.util
import pathlib
import sys
import tempfile
import time
import types

import torch
import yaml

FORK, SUB = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(FORK / "neuraltrain-repo"))

# stub the harness
bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
class CompetSolver:  # noqa: D101
    pass
bs.CompetSolver = CompetSolver; bu.base_solver = bs
sys.modules["benchmark_utils"], sys.modules["benchmark_utils.base_solver"] = bu, bs

from neuraltrain.models.ca_emgmamba import CaEmgMamba  # noqa: E402

torch.manual_seed(0)
X = torch.randn(8, 16, 10000) * 100
ok = True
VARIANTS = [("ca_emgmamba_pose", "ca_emgmamba_pose", "CaEmgMambaPose"),
            ("ca_emgmamba_vel", "ca_emgmamba_vel", "CaEmgMambaVel"),
            ("ca_emgmamba_vel_R2a", "ca_emgmamba_vel_R2a", "CaEmgMambaVelR2a")]
for folder, stem, solver in VARIANTS:
    head = folder
    yml = yaml.safe_load((FORK / f"neuralbench-repo/neuralbench/models/{stem}.yaml").read_text())
    cfg = {k: v for k, v in yml["brain_model_config"].items() if k not in ("name", "=replace=")}
    torch.manual_seed(1)
    ref = CaEmgMamba(**cfg).build(16, 20).eval()
    for p in ref.parameters():                       # non-trivial weights (gates not identity)
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        y_ref = ref(X).transpose(1, 2)               # (B, J, T)

    spec = importlib.util.spec_from_file_location(f"sub_{head}", SUB / folder / "submission.py")
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    assert sub.Solver.name == solver, sub.Solver.name
    assert set(sub.CONFIG) == set(cfg) and all(
        (tuple(v) if isinstance(v, list) else v) == sub.CONFIG[k] for k, v in cfg.items()), "CONFIG != YAML"

    with tempfile.TemporaryDirectory() as d:
        raw = pathlib.Path(d) / "best.ckpt"; rev1 = pathlib.Path(d) / "weights.pt"
        torch.save({"state_dict": {"model." + k: v for k, v in ref.state_dict().items()}}, raw)
        torch.save({"state_dict": ref.state_dict(), "config": cfg, "source": "test", "epoch": 0}, rev1)
        for path in (raw, rev1):
            reg = sub.CaEmgMambaRegressor(16, 20, sub.CONFIG, "cpu").load(path)   # strict=True inside
            t0 = time.perf_counter(); y = reg.predict(X); dt = time.perf_counter() - t0
            same = torch.allclose(y, y_ref, atol=1e-4)
            n_par = sum(p.numel() for p in reg.net.parameters())
            print(f"{head:20s} {path.name:10s} shape {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} "
                  f"std_t>0 {bool(y.std(-1).min() > 0)} match_fork {same} params {n_par:,} "
                  f"predict(8 win) {dt:.2f}s")
            ok &= same and y.shape == (8, 20, 10000)
        t0 = time.perf_counter(); reg.predict(torch.randn(64, 16, 10000)); dt = time.perf_counter() - t0
        print(f"{head:20s} predict(64 win) {dt:.2f}s CPU -> ~{dt * 4000 / 64 / 60:.0f} min per 4000 windows on CPU")
print("ALL OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
