# From `best.ckpt` to a Codabench zip: the transformer submissions

*How to turn a finished `st_pose_transformer*` / `st_vel_transformer*` training run into a tested, zipped Codabench submission. Written 2026-11-05 for the UCF ESCAPE team. The same steps work for every folder under `submission/`; the transformer folders are used as the example.*

## 0. What is where

| Thing | Location |
|---|---|
| Training code and YAMLs | `UCF-ESCAPE-TEAM/neuroai_jf`, branch `emg-pose-cnn` (`neuralbench-repo/neuralbench/models/st_{pose,vel}_transformer{,_ghm,_l1}{,_newton}.yaml`) |
| Submission scripts (this repo) | `submission/<stem>/submission.py` for the six stems `st_pose_transformer`, `st_pose_transformer_ghm`, `st_pose_transformer_l1`, `st_vel_transformer`, `st_vel_transformer_ghm`, `st_vel_transformer_l1` |
| Weight extraction | `prepare_submission/extract_weights_rev1.py` |
| Test + zip | `shell/run_submissions.sh` |
| Official harness | the `2026-competition` repo, cloned locally, with `benchopt` installed in its environment |
| Run output | `SAVE_DIR/neuralbench.main.Experiment.run,1/<long hash dir>/best.ckpt` (SAVE_DIR from `~/.neuralbench/config.json`; on Newton `/lustre/fs1/home/ji045311/challenge_2026/Result`) |

`Solver.name` per folder: `StPoseTransformer`, `StPoseTransformerGhm`, `StPoseTransformerL1`, `StVelTransformer`, `StVelTransformerGhm`, `StVelTransformerL1`. The three loss variants of a model are the same architecture; the folders exist so the three sets of weights can be tested side by side and uploaded under different names.

The whole process is four steps: find the checkpoint, extract `weights.pt`, run the test-and-zip script, upload. Steps 2 and 3 are the only ones with commands.

## 1. Find the checkpoint of the run you want to ship

A NeuralBench run writes one directory per config hash. Its name starts with the config summary, so you can find it by the YAML stem's distinguishing values:

```bash
cd $SAVE_DIR/neuralbench.main.Experiment.run,1/
ls -d */ | grep -i "StPoseTransformer" | grep -i "OhemRegressionLoss"     # OHEM pose model
ls -d */ | grep -i "StVelTransformer"  | grep -i "L1Loss"                 # L1 velocity model
```

Inside the directory you need `best.ckpt`. Confirm it is the run you think it is:

```bash
grep -h "Metric val/mae improved" <dir>/*.err | tail -3     # last improvements and the epoch
tail -n 20 <dir>/*.out                                       # test table at the end
```

Write down the run directory, the epoch, the fork commit (`git -C neuroai_jf rev-parse --short HEAD` at training time) and the YAML stem. Codabench's top-3 audit asks for exactly these.

**If the run was launched with a `_newton.yaml`:** that is fine, the `brain_model_config` is identical to the non-`_newton` file; just point `CONFIG_YAML` (step 2) at the file you actually trained with.

**If `best.ckpt` is missing:** the YAML lacked `delete_checkpoints_on_exit: false`, or the run was stopped with Ctrl+C (graceful stop deletes the checkpoint). All six transformer YAMLs have the flag; `kill -TERM` / `scancel` keeps the file.

## 2. Extract `weights.pt`

`best.ckpt` is a Lightning checkpoint with `save_weights_only=True`: it holds the `model.*` state dict and **no hyper-parameters**. `extract_weights_rev1.py` strips the `model.` prefix, drops training-only sub-trees, and embeds the YAML's `brain_model_config` so `submission.py` rebuilds the exact architecture.

Edit the three paths at the top of `prepare_submission/extract_weights_rev1.py`:

```python
CKPT_PATH   = Path("/lustre/fs1/home/ji045311/challenge_2026/Result/neuralbench.main.Experiment.run,1/<run dir>/best.ckpt")
OUT_PATH    = Path("/path/to/NIPS_CHALLENGE_2026/submission/st_pose_transformer/weights.pt")
CONFIG_YAML = Path("/path/to/neuroai_jf/neuralbench-repo/neuralbench/models/st_pose_transformer_newton.yaml")
```

Then, in an environment with torch and PyYAML (the `challenge_2026` conda env is fine):

```bash
python prepare_submission/extract_weights_rev1.py
```

Read the printed summary before moving on:

```
prefix  : 'model.'
kept    : 2xx tensors, 1,62x,xxx elements      <- ~1.62 M for the shipped YAMLs
dropped : ... (loss.*, metric.*)
epoch   : <best epoch>
config  : from yaml /.../st_pose_transformer_newton.yaml
written : .../submission/st_pose_transformer/weights.pt (6.5 MB)
```

Three things must be true: `kept` is about 1.62 M elements (the transformer models; a TCN checkpoint would show ~0.4 M, which means you pointed at the wrong run); `config : from yaml`, not the "no config embedded" warning; and `OUT_PATH` is inside the folder whose `Solver.name` matches the loss you trained. Putting OHEM weights in the `_l1` folder will run and score, but the upload name will lie about what it is.

Repeat for each run you want to ship; one `weights.pt` per folder. `weights.pt` is git-ignored, so it never gets committed.

## 3. Test and zip with `run_submissions.sh`

The script runs the official harness on each folder exactly as Codabench will (inference only, `-d Simulated`), then zips the folder flat. It needs the `2026-competition` repo and `benchopt` on `PATH`, so activate that environment first.

```bash
conda activate <competition env>          # the one where `benchopt --version` works
cd /path/to/NIPS_CHALLENGE_2026
bash shell/run_submissions.sh submission /path/to/2026-competition --only st_pose_transformer
```

Positional arguments: the submission root (the folder holding one sub-folder per model) and the competition repo. Options you will use:

| Option | When |
|---|---|
| `--only a,b,c` | test only these folders (comma-separated); without it every folder with a `weights.pt` is processed |
| `--dry-run` | print the commands without running anything; do this once to see what it will do |
| `--no-zip` | test only, no zip (e.g. while iterating on a weights file) |
| `--zip-only` | zip without re-running the harness (after a run already passed) |
| `--keep-solvers` | leave the copied `tracks/emg_pose/solvers/<folder>.py` in place for manual `benchopt` debugging |
| `--dataset NAME` | a different benchopt dataset; `Simulated` is the default and what the server uses for the smoke test |

What happens per folder, and where to look:

1. Checks `submission.py` and `weights.pt` exist (an empty `weights.pt` counts as missing).
2. Reads `Solver.name` from the file.
3. Copies `submission.py` to `2026-competition/tracks/emg_pose/solvers/<folder>.py`.
4. Runs `COMPET_SUBMISSION_DIR=<folder> benchopt run tracks/emg_pose -d Simulated -s <Solver.name>`, log in `submission/logs/<folder>.log`.
5. On success zips the folder **flat** (no top-level directory inside the zip) into `submission/zips/<folder>.zip`.
6. Prints a summary table; exit code 0 when every processed folder passed.

The console shows `benchopt : PASSED (NN s)` or `FAILED`. For a failure, open `submission/logs/<folder>.log`; the last traceback is the cause. In the passing case also skim the log for two lines the submission prints itself:

```
[loading] .../weights.pt into StPoseTransformer (1.62 M params, causal=False, norm=zscore)
[input] shape (B, 16, 10000), |X| mean ..., std ..., max ...
```

`norm=zscore` confirms the embedded config was picked up (the YAML's value). The `[input]` line shows the amplitude the harness delivers; it is only a diagnostic here, but on the real Codabench log it is how we caught the R0 scale bug.

To run all six transformer folders at once:

```bash
bash shell/run_submissions.sh submission /path/to/2026-competition \
     --only st_pose_transformer,st_pose_transformer_ghm,st_pose_transformer_l1,st_vel_transformer,st_vel_transformer_ghm,st_vel_transformer_l1
```

Folders without a `weights.pt` are reported as `NO weights.pt` and skipped; that is expected until their runs finish.

## 4. Upload

Upload `submission/zips/<folder>.zip` on Codabench. Sealed phase: **one upload per day until Nov 21**, so choose by local test MAE (`test/mae` in the run's `.out`, or `--dataset heldout_user` numbers), not by curiosity. With the zip, record in `tutorial/TRANSFORMER_PROGRESS.md` (fork) and `tutorial/PROGRESS.md`: folder, fork commit, YAML stem, seed (33), best epoch, local val/test MAE, and later the Codabench score.

## 5. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `load_state_dict` strict error: missing / unexpected keys | model code in the fork changed after the submission scripts were generated, or weights from a different model were put in the folder | check `kept` count in step 2; if the fork model changed, regenerate the six scripts (they are verbatim copies) and re-verify |
| `unexpected keyword argument` in the model `__init__` | embedded config carries a key the submission's `CONFIG` does not know | `_read_checkpoint` filters to `CONFIG` keys; this means the fork added a config field, regenerate the script |
| `benchopt: command not found` | wrong environment | activate the competition env, check `benchopt --version` |
| `NO weights.pt` for a folder you extracted | `OUT_PATH` in step 2 pointed elsewhere, or the file is 0 bytes | re-run step 2 and check `written : ...` |
| harness passes locally, Codabench score ≈ 19° or a constant pose | input scale mismatch | the transformer front end z-scores per window, so this should not happen; if it does, check `norm` in the `[loading]` line is not `none` |
| zip has a top-level folder inside | zipped by hand | use the script's `--zip-only`, which zips flat |
| `MemoryError` importing torch on Newton | you are on the login node | `srun -p short --pty bash` first; the harness test can also run on your own machine, it needs no GPU for `Simulated` |

## 6. Quick check without the harness

If `benchopt` is not available where you are, this stand-alone check does the part that matters for the weights (strict load + output shape + non-constant output):

```bash
cd submission/st_pose_transformer
python - <<'EOF'
import sys, types, torch
bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
class CompetSolver: pass
bs.CompetSolver = CompetSolver; bu.base_solver = bs
sys.modules["benchmark_utils"] = bu; sys.modules["benchmark_utils.base_solver"] = bs
import submission as s
reg = s.StPoseTransformerRegressor(16, 20, s.CONFIG, "cpu").load("weights.pt")   # StVelTransformerRegressor for the vel folders
y = reg.predict(torch.randn(4, 16, 10000) * 100)
print(tuple(y.shape), torch.isfinite(y).all().item(), (y.std(dim=-1) > 0).all().item())
EOF
```

Expected: `(4, 20, 10000) True True`. It does not replace step 3; the harness also checks the solver API and timing.
