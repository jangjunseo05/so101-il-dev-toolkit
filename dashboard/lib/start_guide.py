"""홈 화면의 단계별 가이드 -- 처음 쓰는 사람이 리더암 연결부터 추론 평가까지 순서대로 따라간다.

각 단계의 완료 여부는 실제 상태(로컬 설정, 캘리브레이션 파일, 데이터셋 episode 수,
QA 스냅샷, 학습 run, 평가 리포트)에서 읽는다 -- 진행 상태를 따로 저장하지 않는다.
간단한 일(포트 찾기·저장, 모터 응답 시험, 캘리브레이션)은 여기서 바로 하고,
화면이 큰 단계(녹화·QA·학습·추론)는 해야 할 일을 요약한 뒤 해당 페이지로 보낸다.
(2026-11: 별도 '시작 가이드' 페이지로 만들었다가 홈과 합침 -- 홈의 '다음 행동'·
'파이프라인 흐름'·'환경 상태'가 이 가이드와 겹쳤기 때문)
"""

import streamlit as st

from lib import calibration_ui
from lib import data_sources as ds
from lib import demo_mode
from lib import leader_check
from lib import local_config
from lib import process_manager as pm
from lib import qa_snapshot
from lib import training_run as tr
from lib.next_action import _total_episodes

TARGET_EPISODES = 50  # 녹화 가이드의 최소 권장 수량


def render(cuda_ok: bool, torch_version: str) -> None:

    st.subheader("단계별 가이드")
    st.caption("처음이라면 위에서부터 한 단계씩 따라가세요. 완료된 단계는 ✅, 지금 할 단계는 🔵로 표시됩니다.")

    if demo_mode.ENABLED:
        st.info("공개 데모에는 실제 장비가 없어 이 가이드의 1~3단계(리더암 연결·캘리브레이션)는 실행할 수 없습니다.", icon="ℹ️")

    settings = local_config.load_local_settings()
    port = settings["leader_port"].strip()
    episodes = _total_episodes()
    runs = tr.list_runs()
    _batches = [b for b in ds.list_eval_batches() if b.get("ended_at")]
    eval_report = _batches[0] if _batches else None

    done = [
        bool(port) and st.session_state.get("guide_ping_ok") is not False,  # 1·2. 연결 + 포트 (시험 실패 시 미완료)
        calibration_ui.calibration_exists(settings),                         # 3. 캘리브레이션
        episodes >= TARGET_EPISODES,                                         # 4. 데이터 수집
        bool(qa_snapshot.list_snapshots()),                                  # 5. QA
        any(r.get("status") == "completed" for r in runs),                   # 6. 학습
        eval_report is not None,                                             # 7. 추론 평가
    ]
    current = next((i for i, d in enumerate(done) if not d), None)


    def step_header(i: int, title: str) -> bool:
        """단계 제목을 그리고, 본문을 펼칠지 반환한다(완료된 단계는 접어 두고 토글로 열기)."""
        mark = "✅" if done[i] else ("🔵" if i == current else "⚪")
        st.markdown(f"### {mark} {i + 1}. {title}")
        if done[i]:
            return st.toggle("다시 보기", key=f"guide_show_{i}")
        return True


    # ---------------------------------------------------------------- 1·2. 연결 + 포트
    with st.container(border=True):
        if step_header(0, "리더암 연결하고 포트 정하기"):
            st.markdown(
                "1. SO-101 **리더암**에 **전원 어댑터**를 꽂고, USB 케이블로 이 컴퓨터에 연결합니다. "
                "(USB는 통신만 하고 모터 전원은 어댑터에서 들어옵니다)\n"
                "2. 아래 **포트 찾기**로 리더암이 연결된 포트를 고릅니다. 목록에 없으면 직접 입력하세요 "
                "(Windows 장치 관리자 → 포트(COM & LPT)에서도 확인할 수 있습니다).\n"
                "3. **저장**한 뒤 **연결 시험**으로 모터 6개가 응답하는지 확인합니다."
            )
            try:
                ports = leader_check.list_serial_ports()
            except Exception as e:  # pyserial 문제 등
                ports = []
                st.caption(f"포트 목록을 읽지 못했습니다: {e}")
            c1, c2 = st.columns([3, 2])
            with c1:
                if ports:
                    labels = [
                        f"{p['device']} — {p['description']}" + (f" ({p['chip']})" if p["chip"] else "") for p in ports
                    ]
                    devices = [p["device"] for p in ports]
                    idx = devices.index(port) if port in devices else 0
                    picked = st.radio("발견된 포트", labels, index=idx, key="guide_port_pick")
                    picked_dev = devices[labels.index(picked)]
                else:
                    st.warning("연결된 시리얼 포트가 없습니다. 케이블을 확인하고 '포트 다시 찾기'를 누르세요.", icon="🔌")
                    picked_dev = ""
                st.button("🔍 포트 다시 찾기", key="guide_rescan")
            with c2:
                typed = st.text_input("직접 입력 (예: COM3)", value=port or picked_dev, key="guide_port_typed")
                new_port = (typed or picked_dev).strip()
                if st.button("💾 이 포트로 저장", type="primary" if not port else "secondary", disabled=not new_port,
                             key="guide_save_port"):
                    local_config.save_local_settings({"leader_port": new_port})
                    st.session_state.pop("guide_ping_ok", None)
                    st.session_state.pop("guide_ping_msg", None)
                    st.rerun()
                st.caption(f"저장된 포트: **{port or '없음'}**")

            busy = pm.check_conflict("calibration")  # 녹화·캘리브레이션이 시리얼 포트를 쓰는 중이면 시험 불가
            if busy:
                st.caption(f"⚠️ '{busy}'가 시리얼 포트를 사용 중이라 지금은 연결 시험을 할 수 없습니다.")
            if st.button("🔌 연결 시험 (모터 응답 확인)", disabled=not port or bool(busy) or demo_mode.ENABLED,
                         key="guide_ping"):
                with st.spinner(f"{port}에서 모터 응답을 확인하는 중…"):
                    ok, msg = leader_check.ping_leader(port, settings["leader_id"], settings["calibration_dir"])
                st.session_state["guide_ping_ok"] = ok
                st.session_state["guide_ping_msg"] = msg
                st.rerun()
            if "guide_ping_msg" in st.session_state:
                if st.session_state["guide_ping_ok"]:
                    st.success(st.session_state["guide_ping_msg"])
                else:
                    st.error(st.session_state["guide_ping_msg"])

    # ---------------------------------------------------------------- 3. 캘리브레이션
    with st.container(border=True):
        if step_header(1, "리더암 캘리브레이션"):
            st.markdown(
                "리더암 각 관절의 **중앙 위치와 가동 범위**를 기록합니다. 이 값으로 리더암 각도를 시뮬레이션 팔(팔로워)의 "
                "각도로 바꿉니다.\n\n"
                "- **처음 쓰는 리더암**이면 반드시 한 번 합니다(리더암마다 다르고 git에는 포함되지 않습니다).\n"
                "- 팔로워가 리더암을 이상하게 따라가거나 한쪽 끝에서 멈추면 다시 합니다."
            )
            if demo_mode.ENABLED:
                st.caption("공개 데모에서는 실행할 수 없습니다.")
            else:
                calibration_ui.render(settings, port_missing=not port)

    # ---------------------------------------------------------------- 4. 데이터 수집
    with st.container(border=True):
        if step_header(2, f"데이터 수집 (시연 녹화, {TARGET_EPISODES}개 이상)"):
            st.markdown(
                "1. **① 데이터 수집** 페이지에서 **녹화 세션 시작**을 누르면 MuJoCo 창이 뜹니다.\n"
                "2. MuJoCo 창을 클릭한 뒤 **S** 키로 녹화를 시작하고, 리더암으로 물체를 집어 접시에 놓은 뒤 다시 **S**로 저장합니다. "
                "실수했으면 **X**로 버립니다.\n"
                "3. 물체는 매번 같은 위치로 돌아옵니다. 팔도 **같은 시작 자세**에서 **멈추지 않고 한 번에**, "
                "**비슷한 경로·속도**로 시연하세요. 머뭇거림이 섞이면 학습된 모델이 제자리에서 떠는 원인이 됩니다."
            )
            st.progress(min(episodes / TARGET_EPISODES, 1.0), text=f"저장된 episode: {episodes} / {TARGET_EPISODES}")
            st.page_link("pages/1_데이터_수집.py", label="① 데이터 수집으로 가기", icon="🔴")

    # ---------------------------------------------------------------- 5. QA
    with st.container(border=True):
        if step_header(3, "QA 검증"):
            st.markdown(
                "녹화한 데이터에 이상한 프레임(튀는 값·NaN 등)이나 지나치게 짧은/긴 episode가 없는지 확인합니다.\n\n"
                "1. **② QA 검증** 페이지에서 아웃라이어·무결성 결과를 봅니다.\n"
                "2. 학습에 쓸 episode를 확정하려면 **QA 스냅샷을 저장**합니다. 스냅샷이 있으면 학습이 그 episode 목록을 씁니다."
            )
            st.page_link("pages/2_QA_검증.py", label="② QA 검증으로 가기", icon="🔍")

    # ---------------------------------------------------------------- 6. 학습
    with st.container(border=True):
        if step_header(4, "ACT 학습"):
            st.markdown(
                "1. **③ ACT 학습** 페이지에서 설정(기본: 10000 step, batch 16)을 확인하고 학습을 시작합니다. GPU가 필요합니다.\n"
                "2. 진행 중에는 loss 곡선과 로그가 갱신됩니다. 체크포인트는 2000 step마다 저장됩니다.\n"
                "3. 학습 중에도 이 컴퓨터를 끄지 마세요(절전 포함)."
            )
            if cuda_ok:
                st.caption(f"✅ GPU 사용 가능 (torch {torch_version})")
            else:
                st.caption(f"⚠️ GPU를 쓸 수 없습니다 (torch {torch_version}) — 학습·추론(③④)에만 필요합니다.")
            st.page_link("pages/3_ACT_학습.py", label="③ ACT 학습으로 가기", icon="🧠")

    # ---------------------------------------------------------------- 7. 추론 평가
    with st.container(border=True):
        if step_header(5, "추론과 성공률 평가"):
            st.markdown(
                "1. **④ 추론** 페이지에서 체크포인트를 골라 MuJoCo에서 모델이 직접 팔을 움직이게 합니다.\n"
                "2. 같은 페이지의 **🎯 자동 성공률 평가**에서 체크포인트 2~3개를 고르고 '평가 시작'을 누르면, "
                "같은 조건(episode 수·물체 위치 변화)으로 차례로 돌려 물체가 접시 위에 1초 이상 놓인 비율을 비교합니다."
            )
            if eval_report:
                st.caption("최근 평가: " + " · ".join(
                    f"step {it['label'].rsplit(' ', 1)[-1]}: "
                    f"{it['successes']}/{eval_report['episodes']}"
                    for it in eval_report["items"] if it.get("status") == "done"))
            st.page_link("pages/4_추론.py", label="④ 추론으로 가기", icon="🤖")

    if current is None:
        st.success("모든 단계를 마쳤습니다. 데이터를 더 모으거나 설정을 바꿔 다시 학습해 볼 수 있습니다.", icon="🎉")
