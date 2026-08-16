"""QA 결과를 '학습용으로 확정'된 시점의 스냅샷으로 저장 (CLAUDE.md 29-3절).

페이지 로딩마다 자동 계산되는 라이브 QA 결과와 달리, 이건 사용자가 명시적으로
누른 시점의 train/val episode 구성을 JSON으로 남긴다 -- 데이터셋에 나중에
episode가 더 추가돼도 "그때 실제로 이 조건으로 학습했다"는 기록이 남게 하기
위함. 다음 세션에서 ③(학습) 페이지를 구현할 때 이 폴더의 최신 스냅샷을 읽어
`--dataset.episodes`를 자동으로 채우는 용도로 쓸 예정. 비파괴적 -- 데이터셋
자체는 전혀 건드리지 않고 인덱스 구성만 기록한다.
"""

import json
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "state" / "qa_snapshots"


def save_snapshot(qa_results: dict, repo_id: str, root: str) -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_repo_id = re.sub(r"[^A-Za-z0-9_.-]", "_", repo_id)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOTS_DIR / f"{safe_repo_id}_{timestamp}.json"

    snapshot = {
        "repo_id": repo_id,
        "root": root,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes": qa_results["num_episodes"],
        "total_frames": qa_results["num_frames"],
        "train_episodes": qa_results["split"]["train_episodes"],
        "val_episodes": qa_results["split"]["val_episodes"],
        "excluded_count": qa_results["clean"]["excluded_count"],
        "z_thresh_value": qa_results.get("z_thresh_value"),
        "z_thresh_delta": qa_results.get("z_thresh_delta"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return path


def list_snapshots() -> list[dict]:
    if not SNAPSHOTS_DIR.exists():
        return []
    rows = []
    for path in sorted(SNAPSHOTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                snapshot = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({"file": path.name, **snapshot})
    return rows
