"""SO-101 모방학습 파이프라인 대시보드 -- 엔트리포인트(라우터) + 홈 페이지.

4단계(데이터 수집 → QA → ACT 학습 → 추론) 전부 웹에서 직접 실행 가능
(process_manager.py의 lock 시스템으로 자원 충돌 방지).

`st.navigation()` 기반 라우터(2026-08-17, 42-5절 -- 기존 pages/ 자동 스캔
방식에서 전환): 홈 화면 본문은 아래 `home()` 함수, 나머지 페이지는 파일
그대로(`pages/*.py`) `st.Page()`로 등록해 사이드바에서 "모방학습 실행"
접이식 섹션으로 묶는다.

실행: `streamlit run dashboard/app.py` (mujoco_env, 프로젝트 루트에서)
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import data_sources as ds
from lib import local_config
from lib import next_action
from lib.theme import STAGE_NAMES, STAGE_PAGES
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header, render_splash_screen


def home() -> None:
    """홈 페이지 본문. `st.navigation()`의 콜러블 page-like 객체로 등록됨
    (42-5절) -- 이 함수가 실행 중일 때만 `st.set_page_config()`를 호출해도
    안전함을 실행 테스트로 확인 후 채택(각 페이지 파일이 자기 자신의
    `st.set_page_config()`를 그대로 유지하는 것과 동일한 패턴, 42-5절 참고)."""
    st.set_page_config(page_title="SO-101 파이프라인 대시보드", page_icon=get_logo_icon(), layout="wide")

    # 접속자(세션)당 최초 1회 스플래시(2026-08-17, 재수정).
    # 1차: config/local_settings.json에 저장 -> 서버를 껐다 켜도 안 뜸(너무 오래 지속).
    # 2차: st.cache_resource -> 서버 프로세스 전체가 공유해서, 한 사람이 이미
    #      봤으면 그 서버에 나중에 접속하는 "다른 사람"한테도 안 뜸(원하는 것과 반대).
    # 최종: st.session_state -> 세션(=브라우저 탭 접속)마다 독립이라, "서버를
    #      새로 켜면 뜸(새 세션이니까) / 같은 탭에서 페이지 이동해도 안 뜸(같은
    #      세션) / 이미 켜진 서버에 다른 사람이 새로 접속하면 그 사람 세션
    #      기준으로 뜸"이 전부 자연스럽게 성립한다 -- 서버가 하나뿐이어도 각
    #      접속(세션)은 독립적인 session_state를 받기 때문.
    if "splash_shown" not in st.session_state:
        render_splash_screen()
        time.sleep(1.0)
        st.session_state["splash_shown"] = True
        st.rerun()

    render_brand_header()
    st.title("🦾 SO-101 모방학습 파이프라인 대시보드")
    st.caption("DAPIER 내부용 모방학습 개발 키트: 텔레옵 데이터 수집 → QA/검증 → ACT 학습 → MuJoCo 추론")

    _action = next_action.get_next_action()
    with st.container(border=True):
        render_card_header("", "다음 행동", "action")
        st.page_link(_action["target_page"], label=_action["message"], icon=_action["icon"])

    st.divider()
    st.subheader("파이프라인 흐름")

    cols = st.columns([3, 1, 3, 1, 3, 1, 3])
    for i, name in enumerate(STAGE_NAMES):
        with cols[i * 2]:
            st.page_link(STAGE_PAGES[i], label=f"**{i + 1}. {name}**", width='stretch')
        if i < len(STAGE_NAMES) - 1:
            with cols[i * 2 + 1]:
                st.markdown("<h2 style='text-align:center; margin-top:0.5em;'>→</h2>", unsafe_allow_html=True)

    st.divider()
    st.subheader("환경 상태")

    _settings = local_config.load_local_settings()
    _calib_path = Path(_settings["calibration_dir"]) / f"{_settings['leader_id']}.json"
    _port_ok = bool(_settings["leader_port"].strip())
    _calib_ok = _calib_path.is_file()

    @st.cache_data(show_spinner=False)
    def _check_cuda() -> tuple[bool, str]:
        try:
            import torch

            return torch.cuda.is_available(), torch.__version__
        except ImportError:
            return False, "not installed"

    _cuda_ok, _torch_version = _check_cuda()

    e1, e2, e3 = st.columns(3)
    with e1:
        if _port_ok:
            st.success("로컬 설정 저장됨", icon="✅")
        else:
            st.warning("리더암 포트 미설정", icon="⚠️")
        st.caption("① 데이터 수집 페이지 → ⚙️ 로컬 설정에서 입력")
    with e2:
        if _calib_ok:
            st.success("캘리브레이션 파일 있음", icon="✅")
        else:
            st.warning(f"캘리브레이션 파일 없음: {_calib_path}", icon="⚠️")
        st.caption("docs/REAL_LEADER_SETUP.md 4단계 참고")
    with e3:
        if _cuda_ok:
            st.success(f"GPU 사용 가능 (torch {_torch_version})", icon="✅")
        else:
            st.info(f"GPU 미사용 가능 (torch {_torch_version}) — ③④단계에만 필요", icon="ℹ️")
        st.caption("③ ACT 학습 / ④ 추론 단계에서만 필요")

    if not (_port_ok and _calib_ok):
        st.info("처음 설정하는 경우 `docs/NEW_TEAMMATE_SETUP.md`를 참고하세요.", icon="🧭")

    st.divider()
    st.subheader("파이프라인 헬스 요약")
    st.caption("터미널 로그 대신, 이 파이프라인이 실제로 정상 작동했다는 근거가 되는 핵심 수치만 모았습니다.")

    info = ds.get_dataset_info()
    qa = ds.get_qa_results()
    sweep_df = ds.get_checkpoint_sweep_df()
    final_row = sweep_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("① 데이터 수집", f"{info['total_episodes']} episodes", f"{info['total_frames']} frames")
        st.caption(f"fps={info['fps']}")
    with c2:
        excluded_frames = qa["clean"]["excluded_count"]
        st.metric("② QA & 검증", f"{qa['num_episodes']} episodes 스캔", f"이상 프레임 {excluded_frames}건 탐지")
        st.caption("wrist_roll clip: 재캘리브레이션 후 0건 (23절)")
    with c3:
        st.metric("③ ACT 학습", f"final train loss {final_row['train_loss']:.3f}", f"val {final_row['val_loss']:.3f}")
        st.caption("RTX 5050 · 10000 steps · 3h39m 완주")
    with c4:
        st.metric("④ 추론", "헤드리스 검증 통과", "NaN/Inf 없음")
        st.caption("action dtype float32 유지 확인")

    st.divider()
    st.info(
        "4단계 전부 이 대시보드에서 직접 시작/종료할 수 있습니다. \n\n"
        "동시 실행 시 자원(시리얼 포트/GPU/MuJoCo 뷰어) 충돌은 `lib/process_manager.py`의 lock 시스템이 자동으로 막아줍니다.",
        icon="🛠️",
    )


# 사이드바 재구성(2026-08-17, 42-5절): pages/ 자동 스캔 -> st.navigation()
# 명시 방식으로 전환. app/전처리 구조는 섹션 밖 최상단("" 키)에 항상 노출,
# 4단계 파이프라인은 "모방학습 실행" 접이식 섹션으로 묶음. 실행 테스트로
# 확인된 사실(추측 아님, CLAUDE.md 42-5절):
# - sidebar position에서도 dict 섹션이 실제 클릭 토글로 동작(공식 문서는
#   position="top" 기준으로만 collapsible을 명시하지만 sidebar도 동일)
# - 펼침/접힘 상태는 네이티브 아이콘(⌄/›)으로 자동 구분, 직접 안 그려도 됨
# - 세션 안에서는 사용자가 접은 상태가 페이지 이동 후에도 유지되고, 새
#   세션(=다른 사람 접속)은 그 상태와 무관하게 항상 펼쳐진 기본값으로 시작
# STAGE_PAGES/STAGE_NAMES(lib/theme.py)를 그대로 재사용 -- 페이지 목록을
# 두 곳에 따로 유지하지 않음.
pages = {
    "": [
        st.Page(home, title="app", default=True),
        st.Page("pages/5_전처리_구조.py", title="전처리 구조"),
    ],
    "모방학습 실행": [st.Page(STAGE_PAGES[i], title=STAGE_NAMES[i]) for i in range(len(STAGE_NAMES))],
}

pg = st.navigation(pages)
pg.run()
