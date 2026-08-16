"""읽기 전용 데이터 로더. 기존 파이프라인 파일은 전혀 수정하지 않고
import/read만 한다. 전부 st.cache_data/st.cache_resource로 캐싱해서
페이지 전환/위젯 상호작용마다 재계산되지 않게 한다.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "so101_teleop_real"
REPO_ID = "so101_teleop_real"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
QA_PIPELINE_DIR = PROJECT_ROOT / "pipeline" / "2_qa"

if str(QA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(QA_PIPELINE_DIR))


@st.cache_data
def get_dataset_info() -> dict:
    with open(DATASET_ROOT / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_episode_lengths_df() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return df[["episode_index", "length"]].sort_values("episode_index").reset_index(drop=True)


@st.cache_resource
def _load_team_robot_dataset():
    """TeamRobotDataset을 현재 so101_teleop_real 전체(17 episodes)에 대해 1회 로딩.
    읽기 전용 인스턴스 -- 이 세션 안에서 재사용(st.cache_resource)해 반복 로딩 비용을 없앤다.
    """
    from team_robot_dataset import TeamRobotDataset

    return TeamRobotDataset(REPO_ID, root=str(DATASET_ROOT))


@st.cache_data
def get_qa_results(z_thresh_value: float = 3.0, z_thresh_delta: float = 3.0) -> dict:
    """team_robot_dataset.py의 읽기 전용 분석 메서드를 현재 데이터셋에 대해
    실제로 호출해서 최신 수치를 낸다(오래된 outlier_report.md 대신). 아무 것도
    필터링/삭제하지 않는다 -- 순수 계산 결과만 반환.

    z_thresh_value/z_thresh_delta는 detect_outliers()/detect_outliers_delta()에
    그대로 전달됨 -- 다른 값으로 호출하면 st.cache_data가 별개 캐시 키로
    취급해 자동으로 재계산된다(CLAUDE.md 29-7절 QA 페이지 재실행 파라미터).
    """
    ds = _load_team_robot_dataset()

    value_outliers = ds.detect_outliers(z_thresh=z_thresh_value)
    delta_outliers = ds.detect_outliers_delta(z_thresh=z_thresh_delta)
    integrity = ds.check_integrity()
    clean = ds.get_clean_indices(
        outlier_results=value_outliers,
        delta_outlier_results=delta_outliers,
        integrity_results=integrity,
    )
    split = ds.split_episodes(val_ratio=0.2, seed=42)

    return {
        "num_frames": len(ds),
        "num_episodes": ds.meta.total_episodes,
        "value_outliers": value_outliers,
        "delta_outliers": delta_outliers,
        "integrity": integrity,
        "clean": clean,
        "split": split,
        "z_thresh_value": z_thresh_value,
        "z_thresh_delta": z_thresh_delta,
    }


def invalidate_dataset_caches() -> None:
    """데이터 수집 job이 끝나 데이터셋(episode/frame)이 바뀌었을 가능성이 있을
    때 호출 -- 디스크의 최신 상태를 다시 읽도록 관련 캐시를 전부 지운다.
    `_load_team_robot_dataset()`은 st.cache_resource라 인스턴스 자체를 지워야
    다음 호출에서 새로 로딩한다.
    """
    get_dataset_info.clear()
    get_episode_lengths_df.clear()
    _load_team_robot_dataset.clear()
    get_qa_results.clear()


@st.cache_data
def get_checkpoint_sweep_results() -> dict:
    path = REPORTS_DIR / "checkpoint_sweep_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_checkpoint_sweep_df() -> pd.DataFrame:
    data = get_checkpoint_sweep_results()
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
def get_train_config() -> dict:
    path = PROJECT_ROOT / "config" / "train_main_run_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def get_checkpoint_list_df() -> pd.DataFrame:
    rows = []
    if CHECKPOINTS_DIR.exists():
        for entry in sorted(CHECKPOINTS_DIR.iterdir()):
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
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def get_asset_path(filename: str) -> Path:
    """reports/ 아래 PNG 등 산출물 파일 경로."""
    return REPORTS_DIR / filename
