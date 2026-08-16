"""로컬 subprocess job/lock 관리 (CLAUDE.md 29-2절, 30절 수정).

Streamlit의 `st.session_state`가 아니라 디스크 lock 파일(`state/locks/*.json`)에
상태를 둔다 -- 페이지 새로고침이나 Streamlit 프로세스 재시작이 있어도 "지금
데이터 수집/학습/추론이 돌고 있다"는 사실을 잃지 않기 위함. 데이터
수집(시리얼포트+MuJoCo뷰어)/학습(GPU)/추론(GPU+MuJoCo뷰어)이 서로 자원을
겹쳐 쓰지 않도록 resources 태그로 충돌을 미리 감지한다.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = PROJECT_ROOT / "state" / "locks"
JOB_LOGS_DIR = PROJECT_ROOT / "logs" / "jobs"

LOCKS_DIR.mkdir(parents=True, exist_ok=True)
JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# stage가 독점적으로 필요로 하는 자원 태그. ③④(학습/추론) 웹 트리거를 붙일 때
# 이 표만 채우면 process_manager 자체는 그대로 재사용 가능(CLAUDE.md 29-2절).
STAGE_RESOURCES: dict[str, list[str]] = {
    "data_collection": ["mujoco_viewer", "serial_port"],
    "training": ["gpu"],
    "inference": ["gpu", "mujoco_viewer"],
}

# start_job()이 lock 파일을 원자적으로 생성한 직후, 아직 실제 subprocess pid로
# 덮어쓰기 전인 "starting" placeholder 상태가 이보다 오래 지속되면(정상이라면
# 수 초 내로 끝남) 시작 도중 죽은 것으로 보고 stale 정리 대상에 포함한다.
STARTING_TIMEOUT_S = 60


class JobConflictError(Exception):
    """이미 실행 중이거나, 겹치는 자원을 다른 stage가 점유 중일 때."""


def _lock_path(stage: str) -> Path:
    return LOCKS_DIR / f"{stage}.json"


def _read_lock(stage: str) -> dict | None:
    path = _lock_path(stage)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _all_locks() -> dict[str, dict]:
    """살아있는(또는 막 시작 중인) lock만 모아 반환. 프로세스가 이미 죽었는데
    lock 파일만 남은 경우(stale lock)는 이 호출 중에 자동으로 지운다."""
    result = {}
    for path in LOCKS_DIR.glob("*.json"):
        stage = path.stem
        info = _read_lock(stage)
        if info is None:
            continue
        pid = info.get("pid")
        if pid is None:
            # start_job()이 원자적으로 lock을 막 생성한 직후 ~ 실제 subprocess
            # pid로 덮어쓰기 전까지의 아주 짧은 "starting" 구간. 죽은 lock으로
            # 오판하지 않되, 그 구간 자체가 비정상적으로 길어지면(부모 프로세스가
            # start_job() 도중 죽는 등) 정리한다.
            if time.time() - info.get("started_at", 0) > STARTING_TIMEOUT_S:
                path.unlink(missing_ok=True)
            else:
                result[stage] = info
            continue
        if psutil.pid_exists(pid):
            result[stage] = info
        else:
            path.unlink(missing_ok=True)
    return result


def get_job_status(stage: str) -> dict:
    """`{"running": bool, ...lock 정보}`. stale lock은 여기서도 정리됨."""
    info = _all_locks().get(stage)
    if info is None:
        return {"running": False}
    return {"running": True, **info}


def check_conflict(stage: str) -> str | None:
    """이 stage를 시작하려 할 때 자원이 겹치는 실행 중인 다른 stage 이름을
    반환(없으면 None)."""
    needed = set(STAGE_RESOURCES.get(stage, []))
    if not needed:
        return None
    for other_stage, info in _all_locks().items():
        if other_stage == stage:
            continue
        if needed & set(info.get("resources", [])):
            return other_stage
    return None


def start_job(stage: str, cmd: list[str], extra_meta: dict | None = None) -> dict:
    """subprocess를 새로 띄우고 lock + 로그 캡처를 시작한다.

    **원자성 (2026-08-14 수정)**: 이전 버전은 `get_job_status(stage)["running"]`로
    먼저 확인한 뒤 한참 뒤(Popen 호출 등을 거쳐)에야 lock 파일을 썼다
    (check-then-act) -- 그 사이 시간 동안 두 호출이 동시에 들어오면(예: 웹
    버튼 더블클릭) 둘 다 "실행 중 아님"으로 판단해 통과, MuJoCo 뷰어가 두 개
    뜨는 race condition이 있었다(실제 스레드 동시성 테스트로 재현/수정 확인
    완료 -- CLAUDE.md 30-1절). 지금은 lock 파일 생성 자체를
    `os.open(..., O_CREAT | O_EXCL)`로 원자적 연산으로 만들어, 두 호출이 동시에
    들어와도 운영체제 레벨에서 단 하나만 성공하도록 바꿨다 -- 진 쪽은
    FileExistsError를 받아 JobConflictError로 변환한다.

    이미 이 stage가 실행 중이거나 겹치는 자원을 쓰는 다른 stage가 실행
    중이면 JobConflictError를 낸다 -- 조용히 덮어쓰지 않고, 사용자가
    명시적으로 확인/종료한 뒤 재시도하게 한다.
    """
    conflict = check_conflict(stage)  # _all_locks() 호출로 stale lock 정리도 겸함
    if conflict is not None:
        raise JobConflictError(
            f"'{conflict}'가 겹치는 자원({', '.join(STAGE_RESOURCES.get(stage, []))})을 "
            f"사용 중이라 '{stage}'를 시작할 수 없습니다."
        )

    lock_path = _lock_path(stage)
    placeholder = {
        "pid": None,
        "starting": True,
        "started_at": time.time(),
        "cmd": cmd,
        "resources": STAGE_RESOURCES.get(stage, []),
        "log_path": None,
        **(extra_meta or {}),
    }
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise JobConflictError(f"'{stage}'가 이미 실행 중이거나 지금 막 시작되고 있습니다.") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(placeholder, f, ensure_ascii=False, indent=2)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise

    # 여기부터는 이 stage의 lock을 우리만 쥐고 있음이 원자적으로 보장된다 --
    # 이제 실제 subprocess를 띄우고, 완성된 정보(진짜 pid)로 lock을 덮어쓴다.
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = JOB_LOGS_DIR / f"{stage}_{timestamp}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        popen_kwargs = {}
        if sys.platform == "win32":
            # CREATE_NEW_PROCESS_GROUP: 이 child만 콘솔 신호(CTRL_BREAK_EVENT)로
            # 선택적으로 겨냥할 수 있게 해준다 -- graceful stop(30-2절)의 전제 조건.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        # 자식의 stdout/stderr 인코딩을 명시적으로 UTF-8로 강제한다. 이걸 안
        # 하면(실물 하드웨어 검증 중 실측 확인 -- CLAUDE.md 30-5절) 한국어
        # Windows에서 자식 프로세스가 콘솔 코드페이지(cp949)로 print()를
        # 인코딩해버려서, 우리가 encoding="utf-8"로 여는 로그 파일과 어긋나
        # 한글 로그가 전부 깨져서 저장됨(ASCII 텍스트라 우연히 안 깨졌던
        # crash 패턴 매칭은 영향 없었지만, 대시보드가 보여주는 실시간
        # 로그/크래시 스니펫의 한글이 전부 대체 문자로 나옴).
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env.setdefault("PYTHONUTF8", "1")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=child_env,
            **popen_kwargs,
        )
        # 자식 프로세스가 OS 레벨에서 파일 핸들을 복제해 갖고 있으므로, 부모(이
        # Streamlit 프로세스)가 파일 객체를 닫아도 자식의 로그 기록은 계속된다.
        log_file.close()
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise

    lock_info = {
        "pid": proc.pid,
        "started_at": placeholder["started_at"],
        "cmd": cmd,
        "resources": STAGE_RESOURCES.get(stage, []),
        "log_path": str(log_path),
        **(extra_meta or {}),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_info, f, ensure_ascii=False, indent=2)

    return {"running": True, **lock_info}


def _send_graceful_signal(pid: int) -> bool:
    """정상 종료 신호를 보낸다(성공적으로 '보냈다'는 뜻이지 실제로 죽었다는
    보장은 아니다 -- 그래서 stop_job()에서 timeout 후 강제 종료로 escalate함).

    Windows: CTRL_BREAK_EVENT는 CREATE_NEW_PROCESS_GROUP으로 띄운 특정 자식
    프로세스만 선택적으로 겨냥할 수 있는 유일한 콘솔 신호라 이걸 쓴다(CTRL_C_EVENT는
    호출자와 콘솔을 공유하는 전체 그룹에만 보낼 수 있어 특정 프로세스를 못
    고름). 자식 쪽(run_teleop_real.py)이 `signal.SIGBREAK` 핸들러를 등록해
    이 이벤트를 받는다 -- CLAUDE.md 30-2절.
    """
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False
    except SystemError:
        # 실물 하드웨어 검증 중 실측: 호출자와 대상이 서로 다른 프로세스에서
        # 만들어진 상태로 이 함수를 호출하면(콘솔 공유가 불확실한 경우)
        # os.kill(CTRL_BREAK_EVENT)가 OSError를 감싼 SystemError를 던지는
        # 경우가 있었음 -- 그런데 그 경우에도 신호 자체는 실제로 전달돼
        # 프로세스가 정상 종료되는 것까지 확인됨(CLAUDE.md 30-5절). 그래서
        # False로 포기하지 않고 True를 반환해 stop_job()의 폴링 단계로
        # 넘어가게 한다 -- 정말 안 죽었으면 어차피 timeout 후 강제 종료로
        # escalate되므로 안전.
        return True


def _force_kill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_job(stage: str, graceful: bool = True, graceful_timeout_s: float = 5.0) -> bool:
    """실행 중인 job을 종료한다.

    **Graceful-then-force (2026-08-14 수정, CLAUDE.md 30-2절)**: 이전
    버전은 곧바로 `taskkill /F`(하드 kill)를 써서, run_teleop_real.py의
    기존 `try/finally` 정리 로직(포트 닫기 `safe_disconnect()`,
    `dataset.finalize()`)이 전혀 실행되지 않았다(하드 kill은 OS가 프로세스를
    즉시 회수해 Python 인터프리터가 아예 스택을 풀 기회를 못 얻음 -- 확인
    완료). `graceful=True`(기본값, 데이터 수집이 씀)면 먼저 정상 종료
    신호(`_send_graceful_signal`)를 보내고 `graceful_timeout_s`초 동안
    프로세스가 스스로 끝나길 기다린 뒤, 그래도 살아있으면 강제 종료로
    escalate한다.

    **`graceful=False` (CLAUDE.md 31-2절, 학습이 씀)**: 신호 시도 자체를
    건너뛰고 곧바로 강제 종료한다. `lerobot-train`은 SIGINT/SIGBREAK
    핸들러가 전혀 없는 서드파티 코드(site-packages)라 -- 3절 원칙상
    거기에 신호 핸들러를 심어 넣을 수 없어 애초에 "정상 종료"라는 경로
    자체가 없다. 신호를 보내봐야 `graceful_timeout_s`만큼 그냥 대기만
    하다 결국 강제 종료로 끝나므로, 의미 없는 대기를 건너뛰는 것.
    """
    info = get_job_status(stage)
    if not info["running"]:
        return False
    pid = info["pid"]
    if pid is None:
        # 원자적 lock 생성 직후 ~ 실제 subprocess pid 확정 전의 극히 짧은
        # "starting" 구간 -- 죽일 대상 프로세스가 아직 없으므로 lock만 정리.
        _lock_path(stage).unlink(missing_ok=True)
        return True

    if graceful and _send_graceful_signal(pid):
        deadline = time.time() + graceful_timeout_s
        while time.time() < deadline:
            if not psutil.pid_exists(pid):
                _lock_path(stage).unlink(missing_ok=True)
                return True
            time.sleep(0.2)

    _force_kill(pid)
    _lock_path(stage).unlink(missing_ok=True)
    return True


def tail_log(log_path: str, n_lines: int = 200) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


# get_job_status()는 "돌고 있냐 아니냐"만 알려주고, 프로세스가 정상 종료됐는지
# 시작하자마자 죽었는지는 구분하지 않는다(실측 확인: 크래시 직후에도 그냥
# running=False로만 보임) -- 정교한 에러 파싱은 범위 밖이므로, 로그 tail에서
# 흔한 crash 신호만 문자열로 잡는 최소한의 휴리스틱만 둔다. 이 함수 하나만
# 페이지가 "세션이 예기치 않게 끝났다"는 걸 사용자에게 알리는 데 재사용한다
# (③④ 웹 트리거 구현 시에도 그대로 재사용 가능).
_CRASH_PATTERNS = (
    "traceback (most recent call last)",
    "serialexception",
    "modulenotfounderror",
    "filenotfounderror",
    "permissionerror",
    "connectionerror",
)


def looks_like_crash(log_text: str) -> bool:
    lowered = log_text.lower()
    return any(pattern in lowered for pattern in _CRASH_PATTERNS)
