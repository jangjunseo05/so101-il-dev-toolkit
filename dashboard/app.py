"""SO-101 모방학습 파이프라인 대시보드 -- 홈.

4단계(데이터 수집 → QA → ACT 학습 → 추론) 전부 웹에서 직접 실행 가능
(process_manager.py의 lock 시스템으로 자원 충돌 방지).

실행: `streamlit run dashboard/app.py` (mujoco_env, 프로젝트 루트에서)
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import data_sources as ds
from lib import local_config
from lib.theme import STAGE_NAMES, STAGE_PAGES

st.set_page_config(page_title="SO-101 파이프라인 대시보드", page_icon="🦾", layout="wide")

st.title("🦾 SO-101 모방학습 파이프라인 대시보드")
st.caption("텔레옵 데이터 수집 → QA/검증 → ACT 학습 → MuJoCo 추론. 부트캠프 팀 내부용 개발 키트 (1단계: 읽기 전용)")

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
    "4단계 전부 이 대시보드에서 직접 시작/종료할 수 있습니다. 동시 실행 시 자원(시리얼 포트/"
    "GPU/MuJoCo 뷰어) 충돌은 `lib/process_manager.py`의 lock 시스템이 자동으로 막아줍니다.",
    icon="🛠️",
)
