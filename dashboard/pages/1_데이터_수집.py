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
from lib import calibration_ui
from lib import data_sources as ds
from lib import dataset_registry
from lib import demo_mode
from lib import episode_manager
from lib import local_config
from lib import process_manager as pm
from lib.theme import AQUA
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header

# pipeline/camera_config.py의 스트리밍 포트 상수 재사용(§44) -- data_sources.py가
# 이미 pipeline/2_qa를 sys.path에 넣는 것과 같은 패턴.
sys.path.insert(0, str(ds.PROJECT_ROOT / "pipeline"))
from camera_config import DATA_COLLECTION_STREAM_PORT, WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH  # noqa: E402

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
port_missing = (not demo_mode.ENABLED) and (not job_status["running"]) and not settings["leader_port"].strip()
# 캘리브레이션 파일이 없으면 run_teleop_real.py가 콘솔 대화형 캘리브레이션(Enter 입력)으로
# 들어가는데, 대시보드가 띄운 프로세스에는 콘솔 입력을 줄 수 없어 멈춘다 -- 아래
# '리더암 캘리브레이션' 카드에서 먼저 만들게 한다.
calib_missing = (
    (not demo_mode.ENABLED) and (not job_status["running"]) and not calibration_ui.calibration_exists(settings)
)
action_blocked = conflict is not None or port_missing or calib_missing

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

# 0. 리더암 캘리브레이션 -- 녹화 전에 리더암마다 한 번. 웹 버튼으로 단계를 넘긴다.
if not demo_mode.ENABLED and not job_status["running"]:
    calibration_ui.render(settings, port_missing=not settings["leader_port"].strip())

# 1. 행동(action) -- 페이지에서 가장 먼저 해야 할 일
with st.container(border=True):
    render_card_header("🔴", "녹화 세션 제어", "warning" if action_blocked else "action")
    if action_blocked:
        _reasons = []
        if conflict is not None:
            _reasons.append(f"'{conflict}'가 겹치는 자원(MuJoCo 뷰어/GPU 등)을 사용 중")
        if port_missing:
            _reasons.append("리더암 포트가 설정되지 않음 (아래 '로컬 설정'에서 입력)")
        if calib_missing:
            _reasons.append("리더암 캘리브레이션 파일 없음 (위 '리더암 캘리브레이션'에서 진행)")
        st.caption("⚠️ " + " · ".join(_reasons) + " — 해결 후 다시 시도하세요.")

    if not demo_mode.ENABLED:
      st.info(
        "이 버튼은 텔레오퍼레이션 세션(MuJoCo 뷰어 + 카메라 미리보기 창)을 이 컴퓨터에 새 프로세스로 띄우는 것까지만 합니다. \n\n "
        "**실제 episode 녹화 시작/종료(S 키)와 폐기(X 키)는 뜨는 MuJoCo 뷰어 창에서 직접 키보드로 조작해야 합니다** ",
        icon="⌨️",
    )

    # 대표 프로젝트(HOLD THE FLOW)에서 확인한 시연 원칙 -- ACT는 현재 장면만 보고
    # 다음 1~2초 동작을 한꺼번에 예측하므로(과거·경과 시간 정보 없음), 멈춘 구간이나
    # 머뭇거림이 데이터에 섞이면 같은 장면에 "계속 멈춤"과 "진행" 정답이 섞여
    # 추론 때 제자리에서 떨거나 망설이는 원인이 된다.
    with st.expander("📋 녹화 가이드 — 성공률을 높이는 시연 방식", expanded=False):
        st.markdown(
            "- **같은 조건에서 반복**: 물체는 매 episode 같은 위치에서 시작합니다(녹화 종료 시 자동 복귀). "
            "팔도 매번 같은 시작 자세에서 출발하세요.\n"
            "- **멈추지 말고 한 번에**: 잡기 → 들기 → 옮기기 → 놓기를 머뭇거림 없이 이어서 시연하세요. "
            "중간에 멈춘 구간은 모델이 '계속 멈춤'을 배우는 원인이 됩니다.\n"
            "- **비슷한 경로와 속도**: 매번 비슷한 궤적·속도로 움직이면 적은 데이터로도 성공률이 올라갑니다.\n"
            "- **실수한 episode는 X로 버리기**: 놓치거나 다시 잡은 시연은 저장하지 마세요.\n"
            "- **목표 수량**: 최소 50개 이상(대표 프로젝트 기준 20개 0/5 → 80개 5/5)."
        )

    if demo_mode.ENABLED:
        demo_mode.render_locked("녹화는 실물 리더암을 USB로 연결한 컴퓨터에서만 할 수 있습니다.")
    elif job_status["running"]:
        elapsed = time.time() - job_status["started_at"]
        # 녹화 로그의 [REC] 줄에서 현재 상태와 저장 개수를 읽는다(2초마다 아래 자동 갱신)
        _rec = episode_manager.recording_status(job_status.get("log_path"))
        _state_label = {"recording": "🔴 녹화 중", "saving": "💾 저장 중", "idle": "⏸ 대기 (S로 시작)"}[_rec["state"]]
        if _rec["state"] == "recording" and _rec["current_ep"] is not None:
            _state_label += f" — episode {_rec['current_ep']}"
        _total = _rec["total"]
        if _total is None:  # 이번 세션에 아직 저장한 게 없으면 데이터셋 메타에서 읽는다
            _total = episode_manager.episode_table().shape[0] if episode_manager.dataset_exists() else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("지금", _state_label)
        c2.metric("저장된 episode", f"{_total} / 50", f"이번 세션 +{_rec['saved_this_session']}")
        c3.metric("경과 시간", f"{int(elapsed // 60)}분 {int(elapsed % 60)}초")
        if _rec["state"] == "recording" and _rec["slow_warning"]:
            st.warning("**녹화 루프가 느려지고 있습니다** — 이 episode는 실제보다 짧게 기록될 수 있습니다. "
                       "X로 버리고 다시 녹화하는 것을 권장합니다.\n\n`" + _rec["slow_warning"][:300] + "`", icon="🐢")
        if _rec["state"] != "recording" and _rec["short_saved_warning"]:
            st.error("**방금 저장한 episode가 실제 시연보다 짧게 기록됐습니다.** 아래 episode 목록에서 삭제하세요.\n\n`"
                     + _rec["short_saved_warning"][:300] + "`", icon="⚠️")
        st.caption(
            "**MuJoCo 창을 클릭한 상태**에서(손목 카메라 창이 선택돼 있으면 키가 안 먹음) "
            "**S** 녹화 시작(물체가 시작 위치로 돌아감) → 시연 → **S** 저장 · "
            "실수했으면 **X**(버리기 + 물체 복귀, 대기 중에도 복귀) 후 다시 **S** · 끝낼 때는 녹화 중이 아닐 때 MuJoCo 창을 닫거나 아래 세션 종료 "
            "(녹화 중에 닫으면 그때까지가 자동 저장됨)"
        )

        # 브라우저 내 라이브 스트리밍(§44) -- MuJoCo 네이티브 창을 대체하는 게
        # 아니라 추가하는 것. S/X 키 녹화 조작은 여전히 네이티브 창에서만 가능하다
        # (아래 안내문과 위 st.info 문구 참고). 서버가 아직 안 뜬 초기 수백ms
        # 구간에는 브라우저가 이미지 로드 실패를 보여줄 수 있는데, <img>는 스스로
        # 재시도하지 않으므로 몇 초 안에 안 뜨면 새로고침하라고 안내한다.
        st.markdown("**📺 실시간 화면 (3인칭 뷰)**")
        st.caption(
            "MuJoCo 뷰어 창과는 별개로, 브라우저에서도 씬을 볼 수 있습니다(약 15fps). "
            "S/X 키 녹화 조작은 여전히 뷰어 창에서 해야 합니다. 몇 초가 지나도 안 뜨면 새로고침하세요."
        )
        st.markdown(
            f'<img src="http://127.0.0.1:{DATA_COLLECTION_STREAM_PORT}/stream" '
            f'style="width:100%;max-width:640px;border-radius:4px;" />',
            unsafe_allow_html=True,
        )

        st_autorefresh(interval=2000, key="dc_log_autorefresh")
        log_text = pm.tail_log(job_status["log_path"], n_lines=200)
        st.text_area("실시간 로그 (2초마다 자동 갱신)", value=log_text, height=300, key="dc_log_area")

        st.markdown("**⏹ 세션 강제 종료**")
        st.caption(
            "정상 종료를 먼저 시도하고, 녹화 중이던 episode 저장과 데이터셋 마무리(finalize)가 끝날 때까지 최대 2분 기다립니다. "
            "그래도 안 끝날 때만 강제 종료합니다(강제 종료되면 이번 세션 데이터 파일이 깨질 수 있음). "
            "**녹화 중이던 episode는 그때까지의 프레임으로 자동 저장됩니다**(run_teleop_real.py 동작). "
            "중간에 멈춘 시연을 남기고 싶지 않으면 먼저 MuJoCo 창에서 **X**로 버린 뒤 종료하세요. "
            "가장 안전한 종료는 녹화 중이 아닐 때 MuJoCo 창을 닫는 것입니다."
        )
        confirm = st.checkbox("녹화 중이던 episode는 자동 저장되거나(정상 종료) 유실될 수 있음(강제 종료)을 이해했습니다",
                              key="dc_confirm_stop")
        if st.button("⏹ 세션 강제 종료", disabled=not confirm, type="primary"):
            # 정상 종료 신호 뒤 녹화 중이던 episode 저장(수 초~10초 이상) + finalize()가 끝날 때까지 기다린다.
            # 예전 5초 제한은 저장 도중 강제 종료해 data parquet가 footer 없이 깨졌다(2026-11 실측).
            with st.spinner("정상 종료 중… 녹화 중이던 episode 저장과 데이터셋 마무리를 기다립니다(최대 2분)"):
                pm.stop_job(STAGE, graceful=True, graceful_timeout_s=120)
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

if not demo_mode.ENABLED:
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
    _info_path = ds.DATASET_ROOT / "meta" / "info.json"
    _episodes_path = ds.DATASET_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    # 새 컴퓨터에서 처음 실행하면 아직 녹화된 데이터셋이 없어 info.json을 읽다가
    # 페이지 전체가 오류로 멈췄다 -- 첫 녹화 전에는 안내만 보여 준다.
    _edit_job = pm.get_job_status(episode_manager.DELETE_STAGE)
    # 삭제 job이 방금 끝났으면 캐시를 비우고 결과(로그 끝줄)를 알린다
    if st.session_state.get("dc_edit_was_running") and not _edit_job["running"]:
        ds.invalidate_dataset_caches()
        _tail = pm.tail_log(st.session_state.get("dc_edit_log", ""), n_lines=20)
        st.session_state["dc_edit_result"] = _tail
    st.session_state["dc_edit_was_running"] = _edit_job["running"]
    st.caption(f"저장 위치: `{ds.DATASET_ROOT}`")

    if _edit_job["running"]:
        st_autorefresh(interval=1500, key="dc_edit_autorefresh")
        st.info(f"episode {_edit_job.get('episodes')} 삭제 중입니다… (데이터셋을 다시 쓰는 중이라 잠시 걸립니다)", icon="🗑")
        st.code(pm.tail_log(_edit_job["log_path"], n_lines=15), language=None)
    elif not ds.has_episodes():
        st.info("아직 저장된 episode가 없습니다. 위에서 녹화 세션을 시작해 첫 episode를 저장하면 여기에 현황이 표시됩니다.")
    else:
        if st.session_state.get("dc_edit_result"):
            _res = st.session_state["dc_edit_result"]
            _msg = "episode 삭제 결과:\n\n```\n" + "\n".join(_res.strip().splitlines()[-4:]) + "\n```"
            if "완료:" in _res and "실패" not in _res:
                st.success(_msg)
            else:
                st.error(_msg)
            if st.button("확인", key="dc_edit_ack"):
                st.session_state.pop("dc_edit_result", None)
                st.rerun()
        info = episode_manager._info(dataset_registry.DEFAULT_DATASET_ID) if not demo_mode.ENABLED else ds.get_dataset_info()
        ep_table = episode_manager.episode_table()
        # 녹화 세션 중에는 lerobot이 episode 메타·관절 데이터를 메모리에 모았다가 10개마다(또는 세션 종료 시)
        # 파일에 쓴다. info.json의 개수는 바로 늘지만 목록 파일엔 아직 없는 episode가 생긴다(2026-11 확인).
        _listed = set(ep_table["episode_index"].tolist())
        _pending = [i for i in range(int(info["total_episodes"])) if i not in _listed]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 episode 수", info["total_episodes"], f"목표 50까지 {max(0, 50 - info['total_episodes'])}개")
        c2.metric("총 frame 수", info["total_frames"])
        c3.metric("평균 길이", f"{ep_table['seconds'].mean():.1f}초" if len(ep_table) else "-")
        c4.metric("fps", info["fps"])
        if _pending:
            st.info(
                f"episode {_pending[0]}~{_pending[-1]} ({len(_pending)}개)는 저장됐지만, lerobot이 녹화 세션 중에는 "
                "목록·관절 데이터를 모아 두었다가 **10개마다 또는 세션이 끝날 때** 파일에 씁니다. 그래서 아래 표·그래프·손목 카메라에는 "
                "아직 안 보이고, **MuJoCo 화면(3인칭) 영상은 바로 볼 수 있습니다**(둘러보기에서 선택).",
                icon="⏳",
            )

        st.markdown("**episode별 길이**")
        # 캐시된 get_episode_lengths_df() 대신 매번 디스크에서 읽은 표를 쓴다 -- 녹화 중에 새로 저장된
        # episode도 바로 목록·둘러보기에 나오게(세션이 끝나야 캐시가 비워지던 문제)
        ep_df = ep_table[["episode_index", "length"]]
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
        st.caption("길이가 유난히 짧거나 긴 episode는 오조작일 수 있으니 아래 목록과 둘러보기에서 확인하세요.")

        # episode 목록 + 삭제 -----------------------------------------------------------
        st.markdown("**episode 목록**")
        _view = ep_table.rename(columns={"episode_index": "episode", "length": "frame 수", "seconds": "길이(초)", "check": "확인 필요"})
        _sel = st.dataframe(
            _view, hide_index=True, width="stretch", height=min(38 + 35 * len(_view), 320),
            on_select="rerun", selection_mode="multi-row", key="dc_ep_table",
        )
        _picked = sorted(int(_view.iloc[i]["episode"]) for i in _sel.selection.rows)
        st.caption("행을 클릭해 고르면 아래 둘러보기에 표시되고, 여러 개를 골라 지울 수 있습니다.")

        if not demo_mode.ENABLED:
            _busy = pm.check_conflict(episode_manager.DELETE_STAGE)
            with st.popover(f"🗑 선택한 episode 삭제 ({len(_picked)}개)", disabled=not _picked):
                st.markdown(f"삭제할 episode: **{_picked}**")
                st.markdown(
                    "- 삭제 전에 데이터셋 전체를 `data/_backups/`로 옮겨 **백업**합니다(되돌리려면 그 폴더를 원래 이름으로 복사).\n"
                    "- lerobot 공식 기능(`delete_episodes`)으로 다시 쓰며, **남은 episode 번호가 0부터 다시 매겨집니다.** "
                    "그래서 이전에 저장한 QA 스냅샷의 episode 목록과 맞지 않게 되니 삭제 후 ② QA에서 스냅샷을 다시 저장하세요."
                )
                if _busy:
                    st.warning(f"'{_busy}'가 데이터셋을 사용 중이라 지금은 삭제할 수 없습니다.", icon="⚠️")
                _ok = st.checkbox("위 내용을 확인했습니다", key="dc_del_confirm")
                if st.button("삭제 실행", type="primary", disabled=not _ok or bool(_busy) or len(_picked) >= len(_view),
                             key="dc_del_run"):
                    try:
                        _job = episode_manager.start_delete(_picked)
                        st.session_state["dc_edit_log"] = _job["log_path"]
                        st.session_state["dc_edit_was_running"] = True
                        st.rerun()
                    except pm.JobConflictError as e:
                        st.error(str(e))
                if len(_picked) >= len(_view):
                    st.caption("모든 episode를 지울 수는 없습니다(lerobot 제약). 데이터셋을 새로 시작하려면 폴더를 직접 옮기세요.")

        st.markdown("**episode 둘러보기**")
        _ep_ids = ep_df["episode_index"].tolist() + _pending
        _default_ep = _picked[-1] if _picked else _ep_ids[-1]
        _ep = st.selectbox(
            "episode", options=_ep_ids, index=_ep_ids.index(_default_ep), key=f"dc_browse_episode_{_default_ep}",
            format_func=lambda e: f"{e} (세션 종료 후 데이터 반영)" if e in _pending else str(e),
        )
        _is_pending = _ep in _pending
        # 영상: 기본은 녹화 때 함께 저장한 3인칭 MuJoCo 화면, 없거나(기능 추가 전 녹화분) 원하면 손목 카메라
        _scene = None if demo_mode.ENABLED else episode_manager.scene_clip_path(int(_ep))
        _view_opts = (["MuJoCo 화면 (3인칭)"] if _scene else []) + ([] if _is_pending else ["손목 카메라 (학습 입력)"])
        if not _view_opts:
            st.info("이 episode는 아직 파일에 반영되지 않았고 3인칭 영상도 없습니다. 녹화 세션이 끝나면 표시됩니다.", icon="⏳")
            _view_opts = ["손목 카메라 (학습 입력)"]
        _view = st.radio("영상", _view_opts, horizontal=True, key="dc_browse_view")
        if _is_pending:
            st.caption("관절 그래프와 손목 카메라는 녹화 세션이 끝나면(또는 10개가 모이면) 표시됩니다.")
        elif not _scene and not demo_mode.ENABLED:
            st.caption("이 episode는 3인칭 화면 저장 기능을 넣기 전에 녹화돼 MuJoCo 화면 영상이 없습니다. "
                       "이후 녹화분부터 저장됩니다.")
        _preview, _preview_caption = None, ""
        if _view.startswith("MuJoCo"):
            _preview, _preview_caption = _scene, "MuJoCo 화면 (녹화 중 3인칭, 15fps — 학습에는 쓰지 않음)"
        elif _is_pending:
            _preview = None
        else:
            with st.spinner("손목 카메라 미리보기를 만드는 중…"):
                try:
                    _preview = (ds.get_episode_preview_path(int(_ep)) if demo_mode.ENABLED
                                else episode_manager.preview_path(int(_ep)))
                    _preview_caption = "손목 카메라 (학습 입력 이미지)"
                except Exception as e:  # 미리보기 실패가 페이지 전체를 막지 않게
                    _preview = None
                    st.caption(f"미리보기를 만들지 못했습니다: {e}")
        _left, _right = st.columns([2, 3]) if _preview else (None, st.container())
        if _preview:
            with _left:
                st.video(str(_preview), loop=True, autoplay=True, muted=True)
                st.caption(_preview_caption)
        with _right:
            if not _is_pending:
                _jdf = ds.get_episode_joint_df(int(_ep))
                _joint_chart = (
                    alt.Chart(_jdf)
                    .mark_line(strokeWidth=1.6)
                    .encode(
                        x=alt.X("frame:Q", title="frame (30fps)"),
                        y=alt.Y("state:Q", title="관절각 (rad)"),
                        color=alt.Color("joint:N", legend=alt.Legend(title=None, orient="bottom")),
                        tooltip=["frame", "joint", alt.Tooltip("state:Q", format=".3f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(_joint_chart, width='stretch')
                st.caption("6개 관절의 실제 위치(observation.state) — 매 episode 비슷한 궤적이면 시연이 일관된 것입니다.")

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
| `randomize_pick_object()` | `--randomize-object` 지정 시에만 episode 종료(`S`) 때 물체를 접시 기준 범위 내 랜덤 위치로 재배치. 기본값은 매 episode 고정 위치(추론과 동일)에서 시작 |
| 메인 루프 | 리더 액션 읽기 → MuJoCo `ctrl` 적용 → 물리 스텝 → 30Hz로 카메라 프레임 캡처/녹화 |
"""
    )

    st.subheader("데이터 패킷 형태")
    st.caption("`build_dataset_features()`가 정의하는 LeRobotDataset 스키마")
    st.table(
        {
            "key": ["action", "observation.state", "observation.images.wrist_cam"],
            "dtype": ["float32", "float32", f"image ({WRIST_IMAGE_HEIGHT},{WRIST_IMAGE_WIDTH},3)"],
            "shape": ["(6,)", "(6,)", f"({WRIST_IMAGE_HEIGHT}, {WRIST_IMAGE_WIDTH}, 3)"],
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
