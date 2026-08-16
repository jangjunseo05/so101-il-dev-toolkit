"""팀원마다 다른 COM 포트/캘리브레이션 id/데이터셋 경로를 이 컴퓨터에만
저장하는 로컬 설정 (CLAUDE.md 29-4절). `config/local_settings.json`은 이
저장소가 git 저장소가 아니므로(2026-08-13 확인) 커밋 걱정 없이 자유롭게
쓴다 -- 다만 나중에 git이 도입되면 반드시 .gitignore 대상에 넣어야 한다.

기본값은 `run_teleop_real.py`의 DEFAULT_* 상수와 동일하게 맞춰서, 이 폼을
한 번도 저장하지 않아도 스크립트를 터미널에서 직접 실행했을 때와 같은
동작이 되게 한다.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SETTINGS_PATH = PROJECT_ROOT / "config" / "local_settings.json"

DEFAULTS = {
    "leader_port": "",
    "leader_id": "full_arm_calibration_leader",
    "calibration_dir": str(PROJECT_ROOT / "config"),
    "repo_id": "local/so101_teleop_real",
    "root": str(PROJECT_ROOT / "data" / "so101_teleop_real"),
    "task": "teleoperate SO101 follower arm from real leader arm",
}


def load_local_settings() -> dict:
    if not LOCAL_SETTINGS_PATH.exists():
        return dict(DEFAULTS)
    try:
        with open(LOCAL_SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    merged.update(saved)
    return merged


def save_local_settings(settings: dict) -> None:
    LOCAL_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = load_local_settings()
    merged.update(settings)
    with open(LOCAL_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
