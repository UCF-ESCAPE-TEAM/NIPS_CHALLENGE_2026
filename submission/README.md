# Codabench submissions (Track 4, EMG-to-pose)

One folder per model, laid out for `shell/run_submissions.sh`:

```
submission/<model>/submission.py   # self-contained solver (this repo)
submission/<model>/weights.pt      # NOT committed: extract from the run's best.ckpt
```

| folder | `Solver.name` | training config (neuroai_jf, branch `emg-pose-cnn`) | weights from | warm-up MAE |
|---|---|---|---|---|
| `st_tcn_pose`     | `StTcnPose`      | `neuralbench/models/st_tcn_pose.yaml`     | `prepare_submission/extract_weights_rev1.py` | 16.60 deg (R1) |
| `st_tcn_pose_ghm` | `StTcnPoseGhm`   | `neuralbench/models/st_tcn_pose_ghm.yaml` (GHM-R loss) | same | pending |
| `st_tcn_pose_cmixup` | `StTcnPoseCMixup` | `neuralbench/models/st_tcn_pose_cmixup.yaml` (C-Mixup training) | same | pending |
| `st_tcn_pose_ohem` | `StTcnPoseOhem` | `neuralbench/models/st_tcn_pose_ohem.yaml` (OHEM loss) | same | pending |
| `st_lstm_pose`    | `StLstmPose`     | `neuralbench/models/st_lstm_pose.yaml`    | same | 17.06 deg |
| `st_pose_transformer` | `StPoseTransformer` | `neuralbench/models/st_pose_transformer.yaml` (Conformer encoder, position head, OHEM) | same | pending |
| `st_pose_transformer_ghm` | `StPoseTransformerGhm` | `neuralbench/models/st_pose_transformer_ghm.yaml` (same model, GHM-R loss) | same | pending |
| `st_pose_transformer_l1` | `StPoseTransformerL1` | `neuralbench/models/st_pose_transformer_l1.yaml` (same model, plain L1) | same | pending |
| `st_vel_transformer` | `StVelTransformer` | `neuralbench/models/st_vel_transformer.yaml` (Conformer encoder, velocity decoder, OHEM) | same | pending |
| `st_vel_transformer_ghm` | `StVelTransformerGhm` | `neuralbench/models/st_vel_transformer_ghm.yaml` (same model, GHM-R loss) | same | pending |
| `st_vel_transformer_l1` | `StVelTransformerL1` | `neuralbench/models/st_vel_transformer_l1.yaml` (same model, plain L1) | same | pending |
| `ca_emgmamba_pose` | `CaEmgMambaPose` | `neuralbench/models/ca_emgmamba_pose.yaml` (WindowNorm, CNN stem + SE, bidirectional Mamba, position head, OHEM) | same | pending |
| `ca_emgmamba_vel` | `CaEmgMambaVel` | `neuralbench/models/ca_emgmamba_vel.yaml` (same trunk, velocity head) | same | pending |
| `ca_emgmamba_vel_R2a` | `CaEmgMambaVelR2a` | `neuralbench/models/ca_emgmamba_vel_R2a.yaml` (R0 trunk with `n_layers 2`) | same | pending |
| `ca_emgmamba_vel_R4` | `CaEmgMambaVelR4` | `neuralbench/models/ca_emgmamba_vel_R4.yaml` (R0 + time-stretch augmentation; training-only, same weights layout) | same | pending |
| `ca_emgmamba_vel_R5` | `CaEmgMambaVelR5` | `neuralbench/models/ca_emgmamba_vel_newton_R5.yaml` (R0 trained with C-Mixup, model name `CaEmgMambaCMixup`; same weights layout) | same | pending |
| `neuraldrive_pose`| `NeuralDrivePose`| `neuralbench/models/neuraldrive_pose.yaml`| same | -- |
| `ulhm_pose`       | `ULHM-Pose`      | `ULHM/ulhm_pose.yaml`                     | `prepare_submission/ulhm_submission/export_ulhm_submission.py` | -- |

The model and front-end classes in the six neuraltrain-based scripts are
verbatim copies from the fork (checked against commit `8248dc7`); regenerate them if the
model files change, otherwise `load_state_dict(strict=True)` fails.
The six `st_{pose,vel}_transformer*` scripts were generated from the fork at commit `77a9e18` (front end from the `st_tcn_pose_ohem` script, model classes from `st_pose_transformer.py` / `st_vel_transformer.py`) and verified to load the fork models with `strict=True` and reproduce their outputs; the `_ghm` / `_l1` folders are architecturally identical to their base folder and differ only in `Solver.name`. On Newton the weights come from the `*_newton.yaml` runs, which share the `brain_model_config`; point `extract_weights_rev1.py` at whichever YAML was trained.

The `ca_emgmamba_*` scripts were generated from the fork at commit `b17f885` (adds `TimeStretch`; the R4 run uses the plain `ca_emgmamba_vel` script, the augmentation ships no weights) by
`prepare_submission/ca_emgmamba/make_submissions.py <fork_root> submission <commit>` (model classes
verbatim, no `mamba_ssm` import: the pure-PyTorch scan is forced) and verified with
`prepare_submission/ca_emgmamba/check_submissions.py` (strict load from `best.ckpt` and `weights.pt`,
outputs equal to the fork model). Re-run both if `ca_emgmamba.py` changes.

`st_tcn_pose_ghm`, `st_tcn_pose_ohem` and `st_tcn_pose_cmixup` are architecturally
identical to `st_tcn_pose` (the GHM-R / OHEM losses and C-Mixup change training only; the cmixup
checkpoint's `mix_*` config keys are dropped at load) and differ only in
`Solver.name`.

Step-by-step walkthrough (find `best.ckpt` → `extract_weights_rev1.py` → `run_submissions.sh` → upload), with the transformer folders as the example: `submission/TRANSFORMER_SUBMISSION_TUTORIAL.md`.

Test and zip everything:

```bash
bash shell/run_submissions.sh submission /path/to/2026-competition [--only st_tcn_pose_ghm]
```
