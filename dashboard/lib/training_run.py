"""학습 실행(run) 기록 관리 (CLAUDE.md 31-5절 설계).

`state/training_runs/<run_id>.json`에 학습 세션 메타데이터를 남긴다.
`process_manager.py`의 job/lock과는 역할이 다르다 -- lock은 "지금 프로세스가
돌고 있는가"만 다루고 프로세스가 끝나면 사라지는 반면, 이 모듈은 "그 학습이
어떤 데이터셋/설정으로 시작됐고, 어디까지 진행됐고, 어떻게 끝났는가"를
프로세스 종료 이후에도 남겨서 "이어서 학습" 목록에 쓰기 위한 것이다.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS_DIR = PROJECT_ROOT / "state" / "training_runs"

# lerobot_train.py가 학습을 정상적으로 다 마쳤을 때만 찍는 로그 라인
# (lerobot/scripts/lerobot_train.py:529, `logging.info("End of training")`,
# CLAUDE.md 31-2절에서 소스 확인). SIGINT/강제종료로 죽으면 이 줄이 안
# 찍히므로, "정상 완료"와 "중단"을 가르는 신호로 재사용한다 -- 정교한 파싱
# 대신 단순 문자열 매칭(process_manager.looks_like_crash()와 같은 패턴).
_COMPLETION_MARKER = "end of training"

# lerobot_train.py:455 `logging.info(f"Checkpoint policy after step {step}")`
# 형식과 일치. 못 찾아도 예외 없이 조용히 None 반환(정교한 파싱 금지).
_CHECKPOINT_STEP_RE = re.compile(r"checkpoint policy after step (\d+)", re.IGNORECASE)


def generate_run_id(job_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_job_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_name)
    return f"{timestamp}_{safe_job_name}"


def _run_path(run_id: str) -> Path:
    return TRAINING_RUNS_DIR / f"{run_id}.json"


def create_run(
    run_id: str,
    output_dir: str,
    dataset_snapshot_path: str | None,
    cmd: list[str],
    dataset_id: str | None = None,
) -> dict:
    """새 학습 실행 기록을 만든다. "이어서 학습"은 새 레코드를 만들지 않고
    같은 run_id의 기존 레코드를 `mark_running()`으로 재사용한다 -- output_dir이
    그대로 이어지는 같은 run이기 때문(CLAUDE.md 31-1절: resume은 같은
    output_dir에 계속 저장됨).

    `dataset_id`는 병 데이터셋 추가(dataset_registry.py) 이후 생긴 필드 --
    이 필드가 없는(레지스트리 도입 전에 만들어진) 기존 레코드는
    `list_completed_runs(dataset_id=...)`가 "block_pickplace"로 취급한다.
    """
    TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "output_dir": output_dir,
        "dataset_snapshot_path": dataset_snapshot_path,
        "cmd": cmd,
        "dataset_id": dataset_id,
        "status": "running",
        "last_checkpoint_step": None,
        "started_at": time.time(),
        "ended_at": None,
    }
    with open(_run_path(run_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def mark_running(run_id: str) -> dict | None:
    """"이어서 학습" 시작 시 기존(interrupted) 레코드를 running으로 되돌린다.
    새 레코드를 만들지 않음 -- create_run() 독스트링 참고.
    """
    record = get_run(run_id)
    if record is None:
        return None
    record["status"] = "running"
    record["ended_at"] = None
    with open(_run_path(run_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def delete_run(run_id: str) -> None:
    """start_job() 실패 등으로 생성 직후 롤백해야 할 때만 쓰는 정리용."""
    _run_path(run_id).unlink(missing_ok=True)


def get_run(run_id: str) -> dict | None:
    path = _run_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def list_runs() -> list[dict]:
    if not TRAINING_RUNS_DIR.exists():
        return []
    runs = []
    for path in TRAINING_RUNS_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    runs.sort(key=lambda r: r.get("started_at", 0), reverse=True)
    return runs


def _matches_dataset(run: dict, dataset_id: str | None) -> bool:
    if dataset_id is None:
        return True
    # 레지스트리 도입 전 기록은 dataset_id 필드가 없음 -- 그 시절엔
    # block_pickplace 데이터셋밖에 없었으므로 그걸로 취급한다.
    return run.get("dataset_id", "block_pickplace") == dataset_id


def list_interrupted_runs(dataset_id: str | None = None) -> list[dict]:
    """'이어서 학습' 후보 목록 -- status=interrupted인 run만.

    dataset_id를 주면 그 데이터셋으로 시작된 run만 필터링한다(병/블록
    체크포인트가 서로 다른 씬/카메라를 기대하므로 섞이면 안 됨).
    """
    return [r for r in list_runs() if r.get("status") == "interrupted" and _matches_dataset(r, dataset_id)]


def list_completed_runs(dataset_id: str | None = None) -> list[dict]:
    """추론(④) 페이지의 체크포인트 선택 후보 목록 -- status=completed인 run만.

    dataset_id를 주면 그 데이터셋으로 시작된 run만 필터링한다."""
    return [r for r in list_runs() if r.get("status") == "completed" and _matches_dataset(r, dataset_id)]


def update_last_checkpoint_step(run_id: str, log_text: str) -> int | None:
    """로그에서 마지막으로 언급된 체크포인트 step을 뽑아 기록해둔다. 정교한
    파싱이 아니라 단순 정규식 -- 매칭 실패 시 조용히 None 반환하고 기존
    기록은 그대로 둔다(fallback: 호출부는 raw 로그만 보여주면 됨).
    """
    matches = _CHECKPOINT_STEP_RE.findall(log_text)
    if not matches:
        return None
    step = int(matches[-1])
    record = get_run(run_id)
    if record is None:
        return None
    record["last_checkpoint_step"] = step
    with open(_run_path(run_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return step


def finalize_run(run_id: str, log_text: str) -> str:
    """학습 프로세스 종료가 감지됐을 때 호출한다. 로그에 정상 완료
    마커(`_COMPLETION_MARKER`)가 있으면 completed, 없으면(크래시든 사용자의
    강제 종료든 구분하지 않고 전부) interrupted로 기록한다. 반환값은 최종
    status 문자열.
    """
    record = get_run(run_id)
    if record is None:
        return "unknown"
    status = "completed" if _COMPLETION_MARKER in log_text.lower() else "interrupted"
    record["status"] = status
    record["ended_at"] = time.time()
    with open(_run_path(run_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return status
