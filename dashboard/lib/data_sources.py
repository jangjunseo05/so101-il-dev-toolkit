"""읽기 전용 데이터 로더. 기존 파이프라인 파일은 전혀 수정하지 않고
import/read만 한다. 전부 st.cache_data/st.cache_resource로 캐싱해서
페이지 전환/위젯 상호작용마다 재계산되지 않게 한다.

모든 get_*()는 `dataset_id`를 받는다(기본값 `dataset_registry.DEFAULT_DATASET_ID`
= "block_pickplace") -- 인자를 안 주면 기존(병 데이터셋 추가 전)과 완전히
같은 경로/동작이라 회귀가 없다. 병 데이터셋을 보려면 페이지에서
`dataset_id="bottle_pick_pour"`를 명시적으로 넘긴다.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from lib import dataset_registry

PROJECT_ROOT = dataset_registry.PROJECT_ROOT
QA_PIPELINE_DIR = PROJECT_ROOT / "pipeline" / "2_qa"

if str(QA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(QA_PIPELINE_DIR))

# 하위 호환용 -- 이 상수들을 직접 참조하는 기존 코드(페이지 1 등)는 항상
# block_pickplace 프로필을 가리킨다(레지스트리 도입 전과 동일한 값).
_BLOCK = dataset_registry.get_profile(dataset_registry.DEFAULT_DATASET_ID)
DATASET_ROOT = _BLOCK["dataset_root"]
REPO_ID = _BLOCK["repo_id"]
CHECKPOINTS_DIR = _BLOCK["checkpoints_dir"]
REPORTS_DIR = _BLOCK["reports_dir"]


@st.cache_data
def get_dataset_info(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> dict:
    profile = dataset_registry.get_profile(dataset_id)
    with open(profile["dataset_root"] / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_episode_lengths_df(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> pd.DataFrame:
    profile = dataset_registry.get_profile(dataset_id)
    df = pd.read_parquet(profile["dataset_root"] / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return df[["episode_index", "length"]].sort_values("episode_index").reset_index(drop=True)


@st.cache_resource
def _load_team_robot_dataset(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID):
    """TeamRobotDataset을 지정한 데이터셋 전체에 대해 1회 로딩. 읽기 전용
    인스턴스 -- 이 세션 안에서 재사용(st.cache_resource)해 반복 로딩 비용을
    없앤다. dataset_id별로 별도 인스턴스가 캐싱된다(cache key에 포함됨)."""
    from team_robot_dataset import TeamRobotDataset

    profile = dataset_registry.get_profile(dataset_id)
    return TeamRobotDataset(profile["repo_id"], root=str(profile["dataset_root"]))


@st.cache_data
def get_qa_results(
    z_thresh_value: float = 3.0,
    z_thresh_delta: float = 3.0,
    dataset_id: str = dataset_registry.DEFAULT_DATASET_ID,
) -> dict:
    """team_robot_dataset.py의 읽기 전용 분석 메서드를 지정한 데이터셋에 대해
    실제로 호출해서 최신 수치를 낸다. 아무 것도 필터링/삭제하지 않는다 --
    순수 계산 결과만 반환.

    z_thresh_value/z_thresh_delta는 detect_outliers()/detect_outliers_delta()에
    그대로 전달됨 -- 다른 값으로 호출하면 st.cache_data가 별개 캐시 키로
    취급해 자동으로 재계산된다(CLAUDE.md 29-7절 QA 페이지 재실행 파라미터).
    """
    trd = _load_team_robot_dataset(dataset_id)

    value_outliers = trd.detect_outliers(z_thresh=z_thresh_value)
    delta_outliers = trd.detect_outliers_delta(z_thresh=z_thresh_delta)
    integrity = trd.check_integrity()
    clean = trd.get_clean_indices(
        outlier_results=value_outliers,
        delta_outlier_results=delta_outliers,
        integrity_results=integrity,
    )
    split = trd.split_episodes(val_ratio=0.2, seed=42)

    return {
        "num_frames": len(trd),
        "num_episodes": trd.meta.total_episodes,
        "value_outliers": value_outliers,
        "delta_outliers": delta_outliers,
        "integrity": integrity,
        "clean": clean,
        "split": split,
        "z_thresh_value": z_thresh_value,
        "z_thresh_delta": z_thresh_delta,
    }


@st.cache_data
def get_preprocessing_description(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> dict:
    """TeamRobotDataset.describe_preprocessing()의 결과를 그대로 반환.
    코드(정규화 통계는 실제 meta/stats.json, 6항목 표는 파이프라인 구조 설명)에서
    직접 읽어오기 위한 얇은 래퍼 -- 다른 get_* 함수와 동일한 캐싱 패턴.
    """
    trd = _load_team_robot_dataset(dataset_id)
    return trd.describe_preprocessing()


def invalidate_dataset_caches() -> None:
    """데이터 수집 job이 끝나 데이터셋(episode/frame)이 바뀌었을 가능성이 있을
    때 호출 -- 디스크의 최신 상태를 다시 읽도록 관련 캐시를 전부 지운다(모든
    dataset_id 변형 포함). `_load_team_robot_dataset()`은 st.cache_resource라
    인스턴스 자체를 지워야 다음 호출에서 새로 로딩한다.
    """
    get_dataset_info.clear()
    get_episode_lengths_df.clear()
    _load_team_robot_dataset.clear()
    get_qa_results.clear()
    get_preprocessing_description.clear()


@st.cache_data
def get_checkpoint_sweep_results(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> dict | None:
    """체크포인트 스윕(step별 train/val loss 재현 측정) 리포트. 아직 이 스윕을
    돌리지 않은 데이터셋(예: 병 데이터셋)은 리포트 파일 자체가 없으므로
    None을 반환한다 -- 호출부가 이 경우 차트 대신 안내 메시지를 보여줘야 한다.
    """
    profile = dataset_registry.get_profile(dataset_id)
    if not profile["has_checkpoint_sweep_report"]:
        return None
    path = profile["reports_dir"] / "checkpoint_sweep_results.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_checkpoint_sweep_df(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> pd.DataFrame | None:
    data = get_checkpoint_sweep_results(dataset_id)
    if data is None:
        return None
    rows = []
    for step in data["steps"]:
        r = data["results"][str(step)]
        rows.append(
            {
                "step": step,
                "train_loss": r["train_loss"],
                "val_loss": r["val_loss"],
                "ratio": r["ratio"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def get_train_config(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> dict:
    profile = dataset_registry.get_profile(dataset_id)
    with open(profile["train_config"], encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def get_checkpoint_list_df(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> pd.DataFrame:
    profile = dataset_registry.get_profile(dataset_id)
    checkpoints_dir = profile["checkpoints_dir"]
    rows = []
    if checkpoints_dir.exists():
        for entry in sorted(checkpoints_dir.iterdir()):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            size_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            mtime = entry.stat().st_mtime
            rows.append(
                {
                    "step": int(entry.name),
                    "size_MB": round(size_bytes / (1024 * 1024), 1),
                    "저장 시각": pd.to_datetime(mtime, unit="s"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["step", "size_MB", "저장 시각"])
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def get_asset_path(filename: str, dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> Path:
    """reports/ 아래 PNG 등 산출물 파일 경로."""
    profile = dataset_registry.get_profile(dataset_id)
    return profile["reports_dir"] / filename
