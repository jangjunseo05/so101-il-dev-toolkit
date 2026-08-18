"""1단계: 데이터 수집. run_teleop_real.py를 참고용으로 설명만 하며,
import하거나 실행하지 않는다(모듈 top-level에 리더암 연결/로깅 설정 등
부작용이 있어 대시보드에서 그대로 import하면 안 됨 -- 이번 세션에서
run_inference_mujoco.py 작성 때도 같은 이유로 로직을 복제했었음).

정보 위계: 버튼(행동) → 설정 → 결과 → 메타정보(접힘) 순으로 배치한다
(직관 UX 개선 작업, 2026-08-16). 예외: "이미 벌어진 사건"에 대한 경고
(크래시 배너)는 새 시도를 시작하기 전에 가장 먼저 보여야 하므로 행동
카드보다도 위에 온다. "이 행동을 지금 할 수 있는가"에 대한 경고(자원
충돌/필수 입력 누락)는 별도 카드가 아니라 행동 카드 자체의 헤더 색을
조건부로 전환해서 표시한다(action → warning) -- 사유가 항상 버튼보다
위에 있게 되는 걸 헤더 위치 자체로 보장.
"""

import sys
import time
from pathlib import Path

import altair as alt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import local_config
from lib import process_manager as pm
from lib.theme import AQUA
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header

st.set_page_config(page_title="① 데이터 수집", page_icon=get_logo_icon(), layout="wide")

render_brand_header()

STAGE = "data_collection"
SCRIPT_PATH = ds.PROJECT_ROOT / "pipeline" / "1_data_collection" / "run_teleop_real.py"

job_status = pm.get_job_status(STAGE)

_was_running_key = f"{STAGE}_was_running"
_last_log_key = f"{STAGE}_last_log_path"
_crash_key = f"{STAGE}_crash_log_tail"

if job_status["running"] and job_status.get("log_path"):
    st.session_state[_last_log_key] = job_status["log_path"]

# job이 이전 rerun에서는 실행 중이었는데 지금은 아니면(=방금 세션이 끝났으면)
# 데이터셋 관련 캐시를 지워서 아래 지표/차트가 방금 녹화된 episode를 반영하게 한다.
# get_job_status()는 정상 종료와 크래시를 구분하지 않으므로(process_manager.py
# 참고), 마지막 로그 tail에서 흔한 crash 신호가 보이면 세션 시작 전까지 계속
# 보일 경고를 session_state에 남겨둔다(자동갱신이 running일 때만 돌아서, 크래시로
# running이 꺼진 뒤에는 사용자가 뭔가 조작하기 전까지 이 렌더가 그대로 유지됨).
if st.session_state.get(_was_running_key, False) and not job_status["running"]:
    ds.invalidate_dataset_caches()
    last_log_path = st.session_state.get(_last_log_key)
    if last_log_path:
        tail = pm.tail_log(last_log_path, n_lines=60)
        if pm.looks_like_crash(tail):
            st.session_state[_crash_key] = tail
st.session_state[_was_running_key] = job_status["running"]

st.title("① 데이터 수집")
st.caption("실물 SO101 리더암 → MuJoCo 팔로워암 텔레오퍼레이션 녹화 (`run_teleop_real.py`)")

settings = local_config.load_local_settings()

# 사전조건(자원 충돌/필수 입력 누락) 여부를 헤더 렌더 전에 먼저 계산 --
# "이 행동이 지금 실행 가능한가"에 대한 경고이므로 action 카드 헤더 색
# 자체를 조건부로 바꾸는 데 쓴다(카드를 분리하지 않음). check_conflict()
# 로직 자체는 건드리지 않고 호출 시점만 앞으로 당김.
conflict = None if job_status["running"] else pm.check_conflict(STAGE)
port_missing = (not job_status["running"]) and not settings["leader_port"].strip()
action_blocked = conflict is not None or port_missing

# "이미 벌어진 사건"에 대한 경고(크래시)는 action 카드보다도 위, 페이지에서
# 가장 먼저 눈에 띄는 위치에 독립 카드로 둔다 -- 새 시도를 시작하기 전에
# "지난번에 뭔가 잘못됐다"를 먼저 알아야 하기 때문.
crash_tail = st.session_state.get(_crash_key) if not job_status["running"] else None
if crash_tail:
    with st.container(border=True):
        render_card_header("❌", "이전 세션 크래시", "warning")
        st.write("세션이 예기치 않게 종료됐습니다 — 로그를 확인하세요.")
        with st.expander("마지막 로그 보기(크래시 시점 근처)", expanded=True):
            st.code(crash_tail, language=None)

# 1. 행동(action) -- 페이지에서 가장 먼저 해야 할 일
with st.container(border=True):
    render_card_header("🔴", "녹화 세션 제어", "warning" if action_blocked else "action")
    if action_blocked:
        _reasons = []
        if conflict is not None:
            _reasons.append(f"'{conflict}'가 겹치는 자원(MuJoCo 뷰어/GPU 등)을 사용 중")
        if port_missing:
            _reasons.append("리더암 포트가 설정되지 않음 (아래 '로컬 설정'에서 입력)")
        st.caption("⚠️ " + " · ".join(_reasons) + " — 해결 후 다시 시도하세요.")

    st.info(
        "이 버튼은 텔레오퍼레이션 세션(MuJoCo 뷰어 + 카메라 미리보기 창)을 이 컴퓨터에 새 프로세스로 띄우는 것까지만 합니다. \n\n "
        "**실제 episode 녹화 시작/종료(S 키)와 폐기(X 키)는 뜨는 MuJoCo 뷰어 창에서 직접 키보드로 조작해야 합니다** ",
        icon="⌨️",
    )

    if job_status["running"]:
        elapsed = time.time() - job_status["started_at"]
        c1, c2, c3 = st.columns(3)
        c1.metric("상태", "🔴 실행 중")
        c2.metric("PID", job_status["pid"])
        c3.metric("경과 시간", f"{int(elapsed // 60)}분 {int(elapsed % 60)}초")

        st_autorefresh(interval=2000, key="dc_log_autorefresh")
        log_text = pm.tail_log(job_status["log_path"], n_lines=200)
        st.text_area("실시간 로그 (2초마다 자동 갱신)", value=log_text, height=300, key="dc_log_area")

        st.markdown("**⏹ 세션 강제 종료**")
        st.caption(
            "정상 종료를 먼저 시도합니다(포트 닫기 등 정리 시간 최대 5초) — 그래도 안 끝나면 강제 종료로 전환합니다. "
            "MuJoCo 뷰어에서 S로 저장하지 않은, 녹화 중이던 episode는 이 종료로 유실됩니다."
        )
        confirm = st.checkbox("녹화 중인 episode가 유실될 수 있음을 이해했습니다", key="dc_confirm_stop")
        if st.button("⏹ 세션 강제 종료", disabled=not confirm, type="primary"):
            pm.stop_job(STAGE, graceful=True)
            ds.invalidate_dataset_caches()
            st.success("세션을 종료했습니다.")
            st.rerun()
    else:
        if st.button("🔴 녹화 세션 시작", disabled=action_blocked, type="primary"):
            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--port", settings["leader_port"],
                "--id", settings["leader_id"],
                "--calibration-dir", settings["calibration_dir"],
                "--repo-id", settings["repo_id"],
                "--root", settings["root"],
                "--task", settings["task"],
            ]
            try:
                started = pm.start_job(STAGE, cmd)
                st.session_state.pop(_crash_key, None)
                # 다음 rerun의 상단 전환 감지 로직(위 _was_running_key 블록)이 아주
                # 빠르게(수백ms 내) 죽는 프로세스까지 "실행 중이었다가 꺼짐"으로
                # 잡아낼 수 있도록, 지금 막 시작했다는 사실을 미리 기록해둔다.
                st.session_state[_was_running_key] = True
                st.session_state[_last_log_key] = started["log_path"]
                st.success("세션을 시작했습니다. 잠시 후 MuJoCo 뷰어 창이 뜹니다.")
                st.rerun()
            except pm.JobConflictError as e:
                st.error(str(e))

# 2. 설정(settings) -- 행동 바로 다음, 여전히 눈에 잘 띄어야 함
with st.container(border=True):
    render_card_header("⚙️", "로컬 설정 (이 컴퓨터에만 저장됨)", "settings")
    with st.expander("펼쳐서 편집", expanded=not job_status["running"]):
        with st.form("local_settings_form"):
            c1, c2 = st.columns(2)
            _port_example = "예: COM5" if sys.platform == "win32" else "예: /dev/ttyACM0"
            f_port = c1.text_input(f"리더암 시리얼 포트 ({_port_example})", value=settings["leader_port"])
            f_id = c2.text_input("캘리브레이션 id", value=settings["leader_id"])
            c3, c4 = st.columns(2)
            f_repo_id = c3.text_input("repo_id", value=settings["repo_id"])
            f_root = c4.text_input("데이터셋 저장 경로 (root)", value=settings["root"])
            f_calib_dir = st.text_input("캘리브레이션 파일 디렉터리", value=settings["calibration_dir"])
            f_task = st.text_input("task 설명", value=settings["task"])
            if st.form_submit_button("💾 저장"):
                local_config.save_local_settings(
                    {
                        "leader_port": f_port,
                        "leader_id": f_id,
                        "repo_id": f_repo_id,
                        "root": f_root,
                        "calibration_dir": f_calib_dir,
                        "task": f_task,
                    }
                )
                st.success("저장했습니다.")
                st.rerun()

# 3. 결과(results) -- 지금까지 쌓인 데이터의 현황
with st.container(border=True):
    render_card_header("📊", "현재 데이터셋 현황", "success")
    info = ds.get_dataset_info()
    c1, c2, c3 = st.columns(3)
    c1.metric("총 episode 수", info["total_episodes"])
    c2.metric("총 frame 수", info["total_frames"])
    c3.metric("fps", info["fps"])

    st.markdown("**episode별 길이**")
    ep_df = ds.get_episode_lengths_df()
    chart = (
        alt.Chart(ep_df)
        .mark_bar(color=AQUA, size=18)
        .encode(
            x=alt.X("episode_index:O", title="episode"),
            y=alt.Y("length:Q", title="frame 수"),
            tooltip=["episode_index", "length"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width='stretch')
    st.caption("episode 11은 길이가 짧고 wrist_roll 변화가 전혀 없어 S 키 오조작으로 무효 판정됨 (QA 페이지 참고)")

    if info["total_episodes"] > 0:
        st.page_link("pages/2_QA_검증.py", label="다음: ② QA 검증하기", icon="➡️")

# 4. 메타정보(접힘) -- 코드/스키마/파라미터 설명, 필요할 때만 펼쳐봄
with st.expander("(자세히 보기) 코드 구조 · 데이터 패킷 형태 · 실행 파라미터 "):
    st.subheader("코드 구조")
    st.markdown(
        """
`run_teleop_real.py`는 실물 리더암(`lerobot.teleoperators.so_leader.SO101Leader`)의 관절각을 매
제어 주기(50Hz)마다 읽어 MuJoCo 팔로워암에 그대로 전달하고, `S` 키로 episode 녹화를 시작/종료합니다.

| 함수 | 역할 |
|---|---|
| `load_or_create_dataset()` | 기존 `LeRobotDataset` 로드 또는 신규 생성 (손상 감지 시 사용자 확인 후에만 백업) |
| `leader_action_to_radians()` | 리더암 실측 캘리브레이션 range → follower `ctrlrange` 비례 스케일링 (23절: 1:1 매핑 시 clip 발생 문제 해결) |
| `randomize_pick_object()` | episode 종료(`S`) 시 물체를 접시 기준 범위 내 랜덤 위치로 재배치 |
| 메인 루프 | 리더 액션 읽기 → MuJoCo `ctrl` 적용 → 물리 스텝 → 30Hz로 카메라 프레임 캡처/녹화 |
"""
    )

    st.subheader("데이터 패킷 형태")
    st.caption("`build_dataset_features()`가 정의하는 LeRobotDataset 스키마")
    st.table(
        {
            "key": ["action", "observation.state", "observation.images.wrist_cam"],
            "dtype": ["float32", "float32", "image (480,640,3)"],
            "shape": ["(6,)", "(6,)", "(480, 640, 3)"],
            "설명": ["6개 관절 목표값(라디안)", "6개 관절 현재값(라디안)", "손목 카메라 RGB 프레임"],
        }
    )

    st.subheader("실행 파라미터")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("CONTROL_HZ", "50")
    p2.metric("RECORD_FPS", "30")
    p3.metric("관절 수", "6")
    p4.metric("task 문자열", "고정 1개")
    st.caption('JOINT_NAMES = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper')
