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

# 공개 데모(SO101_DEMO=1): 블록 데이터셋 하나만, 경로는 demo_assets/로 바꾼다.
# demo_assets/는 tools/export_demo_assets.py가 실제 결과에서 만든다(lib/demo_mode.py 참고).
from lib import demo_mode  # noqa: E402

if demo_mode.ENABLED:
    DATASETS = {
        "block_pickplace": {
            **DATASETS["block_pickplace"],
            "dataset_root": demo_mode.DEMO_ROOT / "dataset",
            "checkpoints_dir": demo_mode.DEMO_ROOT / "checkpoints",
            # export 전이라 demo_assets에 설정이 없으면 저장소의 기본 설정으로 대신한다.
            "train_config": (demo_mode.DEMO_ROOT / "train_config.yaml"
                             if (demo_mode.DEMO_ROOT / "train_config.yaml").is_file()
                             else DATASETS["block_pickplace"]["train_config"]),
            "reports_dir": demo_mode.DEMO_ROOT / "reports",
            "has_checkpoint_sweep_report": (demo_mode.DEMO_ROOT / "reports" / "checkpoint_sweep_results.json").is_file(),
        }
    }


def get_profile(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset_id: {dataset_id!r} (expected one of {sorted(DATASETS)})")
    return DATASETS[dataset_id]


def dataset_options() -> dict[str, str]:
    """{label: dataset_id} -- 이 컴퓨터에 데이터 폴더나 체크포인트가 실제로 있는 데이터셋만.
    ① 데이터 수집은 항상 block_pickplace(data/so101_teleop_real)에 쌓으므로 기본값은 늘 포함한다.
    병 데이터셋은 다른 프로젝트에서 옮겨 오는 것이라 폴더가 없으면 고를 이유가 없다."""
    return {
        profile["label"]: key for key, profile in DATASETS.items()
        if key == DEFAULT_DATASET_ID or profile["dataset_root"].exists() or profile["default_checkpoint"].exists()
    }


def pick_dataset() -> str:
    """②③④ 페이지 상단의 데이터셋 선택. 고를 데이터셋이 하나뿐이면 선택 상자 대신
    지금 어떤 데이터를 쓰는지만 표시한다. 선택은 st.session_state로 페이지 간 공유."""
    import streamlit as st

    options = dataset_options()
    ids = list(options.values())
    st.session_state.setdefault("selected_dataset_id", DEFAULT_DATASET_ID)
    if st.session_state["selected_dataset_id"] not in ids:
        st.session_state["selected_dataset_id"] = DEFAULT_DATASET_ID
    if len(ids) == 1:
        dataset_id = ids[0]
        p = DATASETS[dataset_id]
        try:
            root = p["dataset_root"].relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            root = str(p["dataset_root"])
        st.caption(f"데이터셋: **{p['label']}** (`{root}` — ① 데이터 수집이 녹화를 쌓는 곳)")
    else:
        labels = list(options.keys())
        current = next(l for l, d in options.items() if d == st.session_state["selected_dataset_id"])
        dataset_id = options[st.selectbox("데이터셋", options=labels, index=labels.index(current))]
    st.session_state["selected_dataset_id"] = dataset_id
    return dataset_id
