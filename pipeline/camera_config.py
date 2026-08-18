"""Single source of truth for the wrist camera's captured resolution.

Shared by `1_data_collection/run_teleop_real.py` (writes this resolution
into the dataset via `dataset.add_frame()`) and `4_inference/run_inference_mujoco.py`
(feeds a live render at this resolution straight into the policy, without
going through a LeRobotDataset at all).

Why this file exists (CLAUDE.md section 41): both scripts used to declare
their own `IMAGE_HEIGHT`/`IMAGE_WIDTH` = 480/640 independently. Nothing
enforced they stayed equal -- ACT itself doesn't require any fixed
resolution (its vision backbone and 2D positional embedding are computed
from whatever feature map shape the input actually produces, confirmed via
lerobot source inspection), so a silent mismatch between the two files
would not raise inside ACT. It would instead surface as a policy trained
on one resolution being fed a differently-shaped image at inference --
exactly the kind of bug that stays invisible until someone changes one
file and not the other. Both scripts now import from here instead of
declaring their own copy, and both assert the renderer's actual output
shape against these constants before using it (see each script's usage
site) so a future mismatch fails loudly at the render call instead of
silently at inference time.
"""

WRIST_IMAGE_HEIGHT = 480
WRIST_IMAGE_WIDTH = 640
