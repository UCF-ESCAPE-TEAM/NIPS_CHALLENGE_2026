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
| `st_lstm_pose`    | `StLstmPose`     | `neuralbench/models/st_lstm_pose.yaml`    | same | 17.06 deg |
| `neuraldrive_pose`| `NeuralDrivePose`| `neuralbench/models/neuraldrive_pose.yaml`| same | -- |
| `ulhm_pose`       | `ULHM-Pose`      | `ULHM/ulhm_pose.yaml`                     | `prepare_submission/ulhm_submission/export_ulhm_submission.py` | -- |

The model and front-end classes in the five neuraltrain-based scripts are
verbatim copies from the fork (checked against commit `ae48890`); regenerate them if the
model files change, otherwise `load_state_dict(strict=True)` fails.
`st_tcn_pose_ghm` and `st_tcn_pose_cmixup` are architecturally identical to
`st_tcn_pose` (the GHM-R loss and C-Mixup change training only; the cmixup
checkpoint's `mix_*` config keys are dropped at load) and differ only in
`Solver.name`.

Test and zip everything:

```bash
bash shell/run_submissions.sh submission /path/to/2026-competition [--only st_tcn_pose_ghm]
```
