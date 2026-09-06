"""파이프라인(QA/학습/추론)이 다룰 수 있는 데이터셋 프로필 레지스트리.

`data_sources.py`가 예전에는 `so101_teleop_real`(사각형 블록 pick&place) 하나만
모듈 최상단 상수로 하드코딩했다. 병 파지·따르기 데이터셋을 추가하면서, 각
페이지가 "지금 어떤 데이터셋을 보고 있는가"에 따라 경로/체크포인트/추론
스크립트를 전부 바꿔 낄 수 있도록 이 레지스트리로 뽑아냈다.

`block_pickplace` 항목의 경로 값은 기존 `data_sources.py`가 쓰던 값과
1:1 동일하다 -- 이 레지스트리를 도입해도 블록 파이프라인의 동작은 전혀
바뀌지 않는다(회귀 없음).

`bottle_pick_pour`는 다른 프로젝트(so101-mobile-manipulation-main)의
`nl_pick_pour.py`로 수집한 병 파지+따르기 데이터를 이 저장소로 옮겨온
것이다. 결정적인 차이:
  - 카메라 2대(front+wrist, 320x240) vs 블록 쪽 1대(wrist_cam, 640x480)
  - 실제 웹캠으로 찍은 사진(자연 이미지) vs 블록 쪽은 MuJoCo 렌더 자체를
    학습 데이터로 씀(그래서 정규화 방식이 다르다 -- VISUAL: MEAN_STD vs
    블록의 IDENTITY. team_robot_dataset.py의 normalize() 독스트링 참고)
  - 그래서 MuJoCo 추론 시 이미지 도메인이 학습 때와 완전히 다르다(실사진
    -> 합성 렌더) -- 블록 쪽보다 도메인 격차가 훨씬 크다. 이 프로젝트는
    이 사실을 알고 "파이프라인이 에러 없이 도는지"만 보는 스모크테스트로
    검증 범위를 명시적으로 좁혔다(사용자 결정).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 다른 프로젝트(so101-mobile-manipulation-main, so101_env)에서 이미 학습된
# 체크포인트를 재사용할 수 있게 해두는 폴백 -- so101_web 안에서 새로 학습한
# 체크포인트가 아직 없어도 즉시 추론 스모크테스트를 해볼 수 있다.
#
# 원본을 직접 가리키지 않고 checkpoints_bottle/ 아래 로컬 복사본을 쓴다 --
# 이 대시보드는 mujoco_env에서 돈다(README 참고)인데, mujoco_env의 lerobot이
# so101_env의 lerobot보다 오래된 버전이라 config.json의 `pretrained_revision`
# 필드를 모르는 필드로 거부한다(draccus.utils.DecodingError, 실제로 재현
# 확인함). 원본 체크포인트는 다른 프로젝트가 쓰는 파일이라 직접 고치지
# 않고, 이 필드만 뺀 복사본을 여기 둔다.
_EXTERNAL_BOTTLE_CKPT = (PROJECT_ROOT / "checkpoints_bottle"
                         / "from_nl_pick_pour_020000" / "pretrained_model")

DATASETS: dict[str, dict] = {
    "block_pickplace": {
        "label": "사각형 블록 Pick&Place",
        "repo_id": "so101_teleop_real",
        "dataset_root": PROJECT_ROOT / "data" / "so101_teleop_real",
        "checkpoints_dir": PROJECT_ROOT / "checkpoints",
        "train_config": PROJECT_ROOT / "config" / "train_main_run_config.yaml",
        "reports_dir": PROJECT_ROOT / "reports",
        "has_checkpoint_sweep_report": True,
        "scene_xml": PROJECT_ROOT / "robot_model" / "scene.xml",
        "inference_script": PROJECT_ROOT / "pipeline" / "4_inference" / "run_inference_mujoco.py",
        "default_checkpoint": PROJECT_ROOT / "checkpoints" / "010000" / "pretrained_model",
        "default_task": "teleoperate SO101 follower arm from real leader arm",
        "cameras": ["wrist_cam"],
    },
    "bottle_pick_pour": {
        "label": "병 파지·따르기 (자연어 슬롯)",
        "repo_id": "so101_demo_slots",
        "dataset_root": PROJECT_ROOT / "data" / "so101_demo_slots",
        "checkpoints_dir": PROJECT_ROOT / "checkpoints_bottle",
        "train_config": PROJECT_ROOT / "config" / "train_bottle_config.yaml",
        "reports_dir": PROJECT_ROOT / "reports_bottle",
        "has_checkpoint_sweep_report": False,
        "scene_xml": PROJECT_ROOT / "robot_model" / "scene_bottle.xml",
        "inference_script": PROJECT_ROOT / "pipeline" / "4_inference" / "run_inference_mujoco_bottle.py",
        "default_checkpoint": _EXTERNAL_BOTTLE_CKPT,
        "default_task": "Grasping bottle at slot A and pouring water",
        "cameras": ["front", "wrist"],
    },
}

DEFAULT_DATASET_ID = "block_pickplace"


def get_profile(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset_id: {dataset_id!r} (expected one of {sorted(DATASETS)})")
    return DATASETS[dataset_id]


def dataset_options() -> dict[str, str]:
    """{label: dataset_id} -- selectbox용."""
    return {profile["label"]: key for key, profile in DATASETS.items()}
