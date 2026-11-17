"""① 데이터 수집 페이지의 '리더암 캘리브레이션' 카드.

`pipeline/1_data_collection/calibrate_leader_web.py`를 process_manager로 띄우고,
그 스크립트가 쓰는 `state/calibration/status.json`을 읽어 단계 안내와 관절 값을
보여 준다. Enter 키 대신 버튼이 신호 파일(cmd_middle / cmd_stop)을 만든다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from lib import process_manager as pm
from lib.ui_components import render_card_header

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE = "calibration"
SCRIPT_PATH = PROJECT_ROOT / "pipeline" / "1_data_collection" / "calibrate_leader_web.py"
STATE_DIR = PROJECT_ROOT / "state" / "calibration"
STATUS_PATH = STATE_DIR / "status.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# lerobot 모터 이름만으로는 처음 보는 사람이 어느 관절인지 알기 어렵다(예: base = shoulder_pan)
JOINT_KO = {
    "shoulder_pan": "베이스 좌우 회전",
    "shoulder_lift": "어깨 위아래",
    "elbow_flex": "팔꿈치",
    "wrist_flex": "손목 위아래",
    "wrist_roll": "손목 비틀기",
    "gripper": "집게",
}
STEPS_PER_TURN = 4096  # STS3215 raw position 해상도 (lerobot range_max 4095와 같은 기준)


def calibration_file(settings: dict) -> Path:
    return Path(settings["calibration_dir"]) / f"{settings['leader_id']}.json"


def calibration_exists(settings: dict) -> bool:
    return calibration_file(settings).is_file()


def _read_status() -> dict | None:
    # 한 번에 읽고 바로 닫는다 -- 파일을 열어 둔 시간이 길면 스크립트 쪽 교체가 막힌다(Windows)
    try:
        text = STATUS_PATH.read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None


def _send(cmd: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / cmd).touch()


def _deg(steps: int) -> float:
    return steps * 360.0 / STEPS_PER_TURN


def _saved_table(path: Path) -> pd.DataFrame | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rows = []
    for j in JOINTS:
        c = data.get(j)
        if not c:
            continue
        span = c["range_max"] - c["range_min"]
        rows.append({"관절": j, "부위": JOINT_KO[j], "range_min": c["range_min"], "range_max": c["range_max"],
                     "가동 범위(°)": round(_deg(span), 1), "homing_offset": c["homing_offset"]})
    return pd.DataFrame(rows) if rows else None


def render(settings: dict, port_missing: bool) -> None:
    job = pm.get_job_status(STAGE)
    status = _read_status()
    path = calibration_file(settings)
    exists = path.is_file()

    with st.container(border=True):
        render_card_header("🎛", "리더암 캘리브레이션", "action" if (job["running"] or not exists) else "settings")

        if job["running"]:
            st_autorefresh(interval=500, key="cal_autorefresh")
            step = (status or {}).get("step", "connecting")
            st.caption(f"포트 {settings['leader_port']} · 저장 위치 `{path}`")
            if step == "connecting":
                st.info("리더암에 연결하는 중입니다…", icon="🔌")
            elif step == "wait_middle":
                st.markdown(
                    "**1/2 단계 — 중앙 자세**  \n"
                    "리더암의 **모든 관절을 각 가동 범위의 가운데쯤**(중립 자세)에 두세요. "
                    "준비되면 아래 버튼을 누릅니다. (터미널의 첫 번째 Enter와 같은 단계)"
                )
                _positions_table(status, recording=False)
                if st.button("✅ 중앙 자세 완료", type="primary", key="cal_middle"):
                    _send("cmd_middle")
                    st.rerun()
            elif step == "recording":
                st.markdown(
                    "**2/2 단계 — 가동 범위 기록**  \n"
                    "`wrist_roll`(손목 비틀기)을 뺀 5개 관절을 **하나씩 양 끝까지 천천히** 움직이세요: "
                    "shoulder_pan(베이스 좌우 회전) → shoulder_lift(어깨 위아래) → elbow_flex(팔꿈치) → "
                    "wrist_flex(손목 위아래) → gripper(집게, 완전히 열었다가 닫기). "
                    "표의 MIN·MAX가 더 이상 바뀌지 않으면 아래 버튼을 누릅니다."
                )
                _positions_table(status, recording=True)
                if st.button("✅ 범위 기록 완료", type="primary", key="cal_stop"):
                    _send("cmd_stop")
                    st.rerun()
            elif step == "saving":
                st.info("모터에 쓰고 파일로 저장하는 중입니다…", icon="💾")
            if st.button("✖ 캘리브레이션 취소", key="cal_cancel"):
                pm.stop_job(STAGE, graceful=True)
                st.rerun()
            return

        # --- 실행 중이 아님 ---
        last_step = (status or {}).get("step")
        if last_step == "done" and st.session_state.get("cal_show_result"):
            st.success(f"캘리브레이션을 저장했습니다: `{status.get('calibration_path')}`", icon="✅")
            if status.get("backup"):
                st.caption(f"이전 파일 백업: `{status['backup']}`")
        elif last_step in ("error", "cancelled") and st.session_state.get("cal_show_result"):
            if last_step == "error":
                st.error(status.get("message", ""), icon="⚠️")
            else:
                st.info(status.get("message", ""))
            log_path = st.session_state.get("cal_log_path")
            if last_step == "error" and log_path:
                with st.expander("로그 보기"):
                    st.code(pm.tail_log(log_path, n_lines=40), language=None)
        elif last_step in ("connecting", "wait_middle", "recording", "saving") and st.session_state.get("cal_show_result"):
            st.warning("캘리브레이션 프로세스가 도중에 끝났습니다. 파일은 저장되지 않았습니다.", icon="⚠️")
            log_path = st.session_state.get("cal_log_path")
            if log_path:
                with st.expander("로그 보기", expanded=True):
                    st.code(pm.tail_log(log_path, n_lines=40), language=None)

        if exists:
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
            st.markdown(f"✅ 캘리브레이션 파일 있음 · `{path.name}` · {mtime}")
            table = _saved_table(path)
            if table is not None:
                with st.expander("저장된 값 보기"):
                    st.dataframe(table, hide_index=True, width="stretch")
                    st.caption(
                        "wrist_roll(손목 비틀기)의 range 0~4095는 lerobot 공식 캘리브레이션과 같은 고정값입니다 "
                        "(연속 회전 관절로 취급). homing_offset은 다른 관절처럼 실제로 측정한 값입니다."
                    )
        else:
            st.markdown(
                "⚠️ **이 리더암의 캘리브레이션 파일이 아직 없습니다.** 녹화 전에 한 번 진행하세요 "
                "(리더암마다 따로 필요하고 git에는 포함되지 않습니다)."
            )

        conflict = pm.check_conflict(STAGE)
        blocked = port_missing or conflict is not None
        if port_missing:
            st.caption("⚠️ 리더암 포트가 설정되지 않았습니다 (아래 '로컬 설정'에서 입력).")
        if conflict is not None:
            st.caption(f"⚠️ '{conflict}'가 시리얼 포트를 사용 중입니다. 녹화 세션을 먼저 종료하세요.")
        confirm_overwrite = True
        if exists:
            confirm_overwrite = st.checkbox(
                "기존 캘리브레이션을 새로 만듭니다 (이전 파일은 state/calibration/backups/에 백업)", key="cal_confirm"
            )
        if st.button("🎛 캘리브레이션 시작", disabled=blocked or not confirm_overwrite,
                     type="primary" if not exists else "secondary", key="cal_start"):
            cmd = [
                sys.executable, str(SCRIPT_PATH),
                "--port", settings["leader_port"],
                "--id", settings["leader_id"],
                "--calibration-dir", settings["calibration_dir"],
            ]
            try:
                STATUS_PATH.unlink(missing_ok=True)
                started = pm.start_job(STAGE, cmd)
                st.session_state["cal_log_path"] = started["log_path"]
                st.session_state["cal_show_result"] = True
                st.rerun()
            except pm.JobConflictError as e:
                st.error(str(e))


def _positions_table(status: dict | None, recording: bool) -> None:
    status = status or {}
    pos = status.get("positions", {})
    mins, maxes = status.get("mins", {}), status.get("maxes", {})
    rows = []
    for j in JOINTS:
        if j not in pos and j not in mins:
            continue
        row = {"관절": j, "부위": JOINT_KO[j], "현재": pos.get(j)}
        if recording:
            if j in mins:
                row.update({"MIN": mins[j], "MAX": maxes[j], "기록된 범위(°)": round(_deg(maxes[j] - mins[j]), 1)})
            else:
                row.update({"MIN": None, "MAX": None, "기록된 범위(°)": None})
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if recording:
            st.caption(
                "wrist_roll(손목 비틀기)은 **범위만** 기록하지 않습니다. lerobot 공식 캘리브레이션(`lerobot-calibrate`)도 "
                "wrist_roll을 연속 회전 관절로 보고 range를 0~4095(한 바퀴)로 고정합니다. 중앙 위치(homing)는 "
                "앞 단계에서 6개 관절 모두 맞췄습니다. 이 프로젝트도 2026-08부터 같은 방식으로 녹화해 wrist_roll이 "
                "끝에 걸리는 문제가 없음을 확인했습니다(CLAUDE.md 23~24절). 값은 모터 raw 단위(0~4095)."
            )
