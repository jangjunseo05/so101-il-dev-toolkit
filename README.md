# SO101 Robot - URDF and MuJoCo Description

This repository contains the URDF and MuJoCo (MJCF) files for the SO101 robot.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

The MuJoCo file `scene.xml` supports two differenly calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

## Leader/Follower Teleoperation (Mouse-Driven)

`run_teleop.py` loads two SO101 arms into one scene — a **leader** (left, x=-0.3m)
and a **follower** (right, x=+0.3m) — and every step copies the leader's joint
angles onto the follower's position actuators.

The leader arm has no actuators and every body has `gravcomp="1"` (full gravity
compensation), so it floats in place and holds whatever pose you leave it in.
Move it with the mouse in the MuJoCo viewer:

1. `Ctrl` + double-click a leader-arm link to select it.
2. `Ctrl` + right-drag to translate the selected body, `Ctrl` + `Shift` +
   right-drag to rotate it.

The follower mirrors the leader's joint angles in real time.

```bash
python build_teleop_scene.py   # (re)generates scene_teleop.xml, only needed after editing so101_new_calib.xml
python run_teleop.py           # launches the interactive viewer
```

`scene_teleop.xml` is a generated file (built with `mujoco.MjSpec`, attaching
`so101_new_calib.xml` twice with `leader_`/`follower_` name prefixes) and is
checked in for convenience; `run_teleop.py` automatically rebuilds it if
`so101_new_calib.xml` or `build_teleop_scene.py` change.

---

Feel free to open an issue or contribute improvements!
