# SO-101 Imitation Learning Dev Toolkit

End-to-end imitation-learning toolkit for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm: MuJoCo/URDF robot model, teleoperated data collection (sim and real), a QA layer for validating recorded datasets, ACT policy training/evaluation, MuJoCo-based inference, and a Streamlit dashboard that drives all four stages from the browser.

## Pipeline Overview

```
pipeline/
├── 1_data_collection/   leader→follower teleoperation + LeRobotDataset recording
├── 2_qa/                TeamRobotDataset: outlier/integrity checks, visualization
├── 3_training/          ACT (lerobot-train) checkpoint analysis
└── 4_inference/         MuJoCo rollout of trained ACT checkpoints
```

- **① Data collection** (`pipeline/1_data_collection/`)
  - `run_teleop_real.py` — real SO-101 leader arm → MuJoCo-simulated follower. Records `action`/`observation.state` and a wrist-camera stream into a [LeRobot](https://github.com/huggingface/lerobot) `LeRobotDataset` (v3.0, parquet-based). Includes an in-session episode object randomizer, start/discard keyboard controls, and a live camera preview.
  - `run_teleop_real_to_real.py` — real leader arm → real follower arm, no simulation involved. Used to collect a bottle grasp-and-pour demonstration set with a USB wrist camera, scaling leader/follower joint ranges to each physical unit's own calibration.
  - `run_teleop_follower.py` (repo root) — the inverse direction: a MuJoCo leader arm driven by mouse input in the viewer controls a real follower arm over serial. Teleoperation only, no recording.
- **② QA & validation** (`pipeline/2_qa/`)
  - `team_robot_dataset.py` defines **`TeamRobotDataset`**, a composition wrapper around `lerobot.datasets.LeRobotDataset` (the base class is never modified) that adds:
    - `check_integrity()` — scans every frame for NaN/Inf values (runs automatically on load).
    - `compute_custom_stats()`, `detect_outliers()` — per-joint z-score outlier detection on raw values.
    - `detect_outliers_delta()` — frame-to-frame delta z-score detection, which catches intermittent step-like transitions that raw-value z-scores miss.
    - `get_clean_indices()` — merges the above into a non-destructive clean/excluded index split, ready for `torch.utils.data.Subset`.
    - `split_episodes()` — episode-level train/val split.
    - `normalize()` / `unnormalize()`, `visualize_episode()`, `plot_action_distribution()`, `describe_preprocessing()` — QA-side diagnostics mirroring the normalization and preprocessing choices actually used by the ACT training path.
  - `run_outlier_detection.py`, `run_visualization.py`, `run_clean_subset_example.py` drive the above over a recorded dataset; sample output lives in `reports/` (`outlier_report.md`, trajectory plots, action-distribution histograms).
- **③ ACT training** (`pipeline/3_training/`) — training itself runs via the upstream `lerobot-train` CLI against YAML configs in `config/` (`train_main_run_config.yaml` for pick-and-place, `train_bottle_config.yaml` for the bottle task). This folder holds read-only analysis: `run_val_loss_check.py` reconstructs `lerobot-train`'s own forward/loss path to sweep held-out validation loss across checkpoints (`lerobot-train` has no built-in held-out eval for this setup), and `plot_checkpoint_sweep.py` renders the results.
- **④ Inference** (`pipeline/4_inference/`) — `run_inference_mujoco.py` loads a trained ACT checkpoint and runs closed-loop pick-and-place rollouts inside MuJoCo (viewer or headless smoke-test mode). `run_inference_mujoco_bottle.py` is the equivalent for the bottle checkpoint; since that policy was trained on real webcam frames, running it against MuJoCo's synthetic render only verifies the pipeline executes cleanly (no NaN/Inf, finite actions) — it does not validate grasp/pour success. Physical-robot execution of that checkpoint is handled in a separate project.

## Web Dashboard (`dashboard/`)

A Streamlit app (`streamlit run dashboard/app.py`) that wraps all four stages above so a teammate can run the full loop without touching a terminal:

1. **데이터 수집 (Data collection)** — launches/stops `run_teleop_real.py` as a subprocess with a local COM-port config form.
2. **QA 검증 (QA validation)** — imports `TeamRobotDataset` directly (pure computation, no subprocess) to run outlier/integrity checks on demand.
3. **ACT 학습 (Training)** — starts new or resumed `lerobot-train` runs as subprocesses, with a GPU resource lock so it can't collide with inference.
4. **추론 (Inference)** — runs `run_inference_mujoco.py` either as a synchronous headless health check or a subprocess-backed interactive viewer session.
5. **전처리 구조 (Preprocessing map)** — a read-only view of `TeamRobotDataset.describe_preprocessing()`, mapping each standard IL preprocessing step (resize, augmentation, action chunking, frame stacking, ...) to where it actually happens (or doesn't) across this QA layer, the ACT policy, and the `lerobot-train` CLI.

Cross-stage resource conflicts (serial port / GPU / MuJoCo viewer) are serialized by a lock system in `dashboard/lib/process_manager.py`. The dashboard only ever imports the pipeline modules read-only; it does not modify them.

## Robot Model (`robot_model/`)

URDF and MuJoCo (MJCF) description of the SO-101 arm.

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.
- `scene.xml` is the general-purpose pick-and-place scene (`pick_object`/`place_target`); `scene_bottle.xml` is the bottle grasp/pour variant; `scene_teleop.xml` attaches two SO-101 arms (`leader_`/`follower_` prefixes) for the mouse-driven teleoperation flow described below.

### Calibration Methods

The MuJoCo file `scene.xml` supports two differently calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use → `so101_new_calib.xml`.
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use → `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

### Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

### Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files.

## Leader/Follower Teleoperation (Mouse-Driven)

`run_teleop_follower.py` (repo root) loads `robot_model/scene_teleop.xml` — two SO-101 arms, a leader with no actuators and full gravity compensation (`gravcomp="1"`), and a follower — and every control step copies the leader's joint angles onto both a MuJoCo-visualized follower and a real follower arm over serial.

The leader arm floats in place and holds whatever pose you leave it in. Move it with the mouse in the MuJoCo viewer:

1. `Ctrl` + double-click a leader-arm link to select it.
2. `Ctrl` + right-drag to translate the selected body, `Ctrl` + `Shift` + right-drag to rotate it.

```bash
python run_teleop_follower.py --port COM7
```

On first connection to an uncalibrated real follower, an interactive calibration sequence runs (move the arm to its middle pose, then each joint to its range limits); subsequent runs reuse the calibration file named by `--id`. `--max-relative-target` (default 15) caps how far a single command step may move the real arm, for safety. This script is teleoperation only — it does not record a dataset; for that, see `pipeline/1_data_collection/`.

## Setup

See `docs/NEW_TEAMMATE_SETUP.md` for a full walkthrough (conda env from `environment.yml`, CUDA-enabled torch, hardware connection). Core stack: Python 3.11, `lerobot` (ACT policy + `LeRobotDataset`), `mujoco`, `streamlit`, `feetech-servo-sdk`/`pyserial` for the Feetech STS3215 servos, PyTorch (CUDA, installed separately from the conda env).

Other docs in `docs/`: `TELEOP_DATA_COLLECTION.md`, `REAL_LEADER_SETUP.md`, `CAMERA_TUNING_AND_TROUBLESHOOTING.md`, `GRIPPER_SLIP_TROUBLESHOOTING.md`, `TROUBLESHOOTING.md`.

---

Feel free to open an issue or contribute improvements!
