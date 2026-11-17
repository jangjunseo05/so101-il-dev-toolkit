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
from lib import start_guide
from lib.theme import STAGE_NAMES, STAGE_PAGES
from lib.ui_components import get_logo_icon, render_brand_header, render_splash_screen


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

    # 홈 = 헬스 요약(결과 한눈에) + 단계별 가이드(2026-11). 예전 홈의 '다음 행동'·'파이프라인
    # 흐름'·'환경 상태'(포트·캘리브레이션·GPU)는 가이드의 단계 표시가 같은 역할을 해서 합쳤다.
    _render_health_summary()
    st.divider()
    _cuda_ok, _torch_version = _check_cuda()
    start_guide.render(_cuda_ok, _torch_version)

    st.divider()
    st.caption(
        "4단계 전부 이 대시보드에서 직접 시작/종료할 수 있습니다. 동시 실행 시 자원(시리얼 포트/GPU/MuJoCo 뷰어) "
        "충돌은 `lib/process_manager.py`의 lock 시스템이 자동으로 막아줍니다."
    )


@st.cache_data(show_spinner=False)
def _check_cuda() -> tuple[bool, str]:
    try:
        import torch

        return torch.cuda.is_available(), torch.__version__
    except ImportError:
        return False, "not installed"


def _render_health_summary() -> None:
    st.subheader("파이프라인 헬스 요약")
    st.caption("터미널 로그 대신, 이 파이프라인이 실제로 정상 작동했다는 근거가 되는 핵심 수치만 모았습니다.")

    # 새 컴퓨터 첫 실행처럼 아직 녹화된 데이터셋이 없으면 info.json을 읽다가 홈 전체가
    # 오류로 멈췄다 -- 헬스 요약만 건너뛰고 안내를 보여 준다.
    if not ds.has_episodes():
        st.info("아직 녹화된 데이터셋이 없어 헬스 요약을 건너뜁니다. ① 데이터 수집에서 첫 episode를 저장하면 표시됩니다.", icon="📭")
        return

    info = ds.get_dataset_info()
    qa = ds.get_qa_results()
    summary = ds.get_pipeline_summary()
    sweep_df = ds.get_checkpoint_sweep_df()
    train_log = ds.get_train_log_df()
    eval_report = ds.get_latest_eval_report()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("① 데이터 수집", f"{info['total_episodes']} episodes", f"{info['total_frames']} frames")
        st.caption(f"fps={info['fps']}")
    with c2:
        excluded_frames = qa["clean"]["excluded_count"]
        st.metric("② QA & 검증", f"{qa['num_episodes']} episodes 스캔", f"이상 프레임 {excluded_frames}건 탐지")
    with c3:
        if sweep_df is not None:
            final_row = sweep_df.iloc[-1]
            st.metric("③ ACT 학습", f"final train loss {final_row['train_loss']:.3f}", f"val {final_row['val_loss']:.3f}")
        elif train_log is not None:
            st.metric("③ ACT 학습", f"final train loss {train_log['loss'].iloc[-1]:.3f}", f"step {int(train_log['step'].iloc[-1])}")
        else:
            st.metric("③ ACT 학습", "기록 없음")
        _train_bits = [summary.get(k) for k in ("gpu", "steps_label", "train_time_label") if summary.get(k)]
        if _train_bits:
            st.caption(" · ".join(_train_bits))
    with c4:
        if eval_report:
            st.metric("④ 추론 성공률", f"{eval_report['successes']} / {eval_report['episodes']}",
                      f"{eval_report['success_rate'] * 100:.0f}% (MuJoCo 자동 평가)")
        else:
            st.metric("④ 추론", "평가 기록 없음")


# 사이드바 재구성(2026-08-17, 42-5절): pages/ 자동 스캔 -> st.navigation()
# 명시 방식으로 전환. 홈(헬스 요약 + 단계별 가이드)은 섹션 밖 최상단("" 키)에 항상 노출
# (전처리 구조는 2026-11에 ② QA 검증 페이지의 접힌 섹션으로 옮김),
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
        st.Page(home, title="홈", icon="🏠", default=True),
    ],
    "모방학습 실행": [st.Page(STAGE_PAGES[i], title=STAGE_NAMES[i]) for i in range(len(STAGE_NAMES))],
}

pg = st.navigation(pages)
pg.run()
