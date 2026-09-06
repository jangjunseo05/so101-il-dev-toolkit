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

# 브라우저 내 라이브 스트리밍(CLAUDE.md 44절: MuJoCo 네이티브 창 -> 브라우저
# 통합)용 3인칭 씬 카메라 해상도/주기. wrist_cam 녹화·정책 입력 경로와는
# 완전히 분리된 순수 모니터링용이라, 위 WRIST_IMAGE_*와 값이 같아야 할
# 이유는 없다 -- 우연히 같은 480x640을 재사용할 뿐, 별도 상수로 둔 이유는
# 이 둘이 독립적으로 바뀔 수 있어야 하기 때문(예: 스트리밍 대역폭을 줄이려고
# 이쪽만 해상도를 낮추는 경우).
SCENE_STREAM_HEIGHT = 480
SCENE_STREAM_WIDTH = 640
SCENE_STREAM_FPS = 15.0

# dashboard/lib/process_manager.py의 STAGE_RESOURCES가 이미 stage(data_collection/
# inference)당 동시 1개 프로세스만 실행되도록 자원 충돌로 보장하므로, 포트를
# 동적으로 탐색할 필요 없이 stage마다 고정값으로 둔다.
DATA_COLLECTION_STREAM_PORT = 8531
INFERENCE_STREAM_PORT = 8532

# 병 파지·따르기 데이터셋(so101_demo_slots, dataset_registry.py의
# bottle_pick_pour) 전용 -- 다른 프로젝트(so101-mobile-manipulation-main)의
# nl_pick_pour.py가 실제 웹캠 2대(front+wrist)를 이 해상도로 열어 수집한
# 데이터라, MuJoCo 추론(run_inference_mujoco_bottle.py)도 같은 해상도로
# 렌더해야 한다. 위 WRIST_IMAGE_*(블록 데이터셋 전용, 640x480)와는 독립된
# 상수 -- 두 데이터셋이 서로 다른 해상도를 쓰므로 공유하면 안 된다.
# INFERENCE_STREAM_PORT는 그대로 재사용(§ dashboard STAGE="inference" lock이
# 데이터셋과 무관하게 이미 동시 1개 세션만 허용하므로 포트 충돌 없음).
BOTTLE_FRONT_IMAGE_HEIGHT = 240
BOTTLE_FRONT_IMAGE_WIDTH = 320
BOTTLE_WRIST_IMAGE_HEIGHT = 240
BOTTLE_WRIST_IMAGE_WIDTH = 320
