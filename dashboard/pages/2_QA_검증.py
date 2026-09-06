"""2단계: QA & 검증. team_robot_dataset.TeamRobotDataset의 읽기 전용
분석 메서드를 현재 so101_teleop_real 데이터셋에 대해 실제로 호출해서
최신 수치를 보여준다 (오래된 outlier_report.md는 시뮬레이션 2-episode
데이터 대상이라 참고용으로만 별도 표시). 아무 것도 필터링/삭제하지
않는다.

정보 위계(2026-08-16, 직관 UX 개선): 행동(스냅샷 저장) → 설정(z_thresh +
그 설정 기준 실시간 수치) → 결과(저장된 스냅샷 요약, 있을 때만) →
메타정보(접힘: 코드 구조/이상치 사례/시각화/스냅샷 전체 이력). 라이브 QA
수치는 결과가 아니라 "저장 전에 보고 판단하는" 설정 카드 소속으로 분류함
(페이지 1과 다른 점 -- QA는 자원 충돌 개념이 없어 action 카드 헤더의
조건부 전환도 없음).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import dataset_registry
from lib import qa_snapshot
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header


@st.cache_data
def _load_old_report_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


st.set_page_config(page_title="② QA & 검증", page_icon=get_logo_icon(), layout="wide")

render_brand_header()
st.title("② QA & 검증")
st.caption("`TeamRobotDataset` 커스텀 통계 · 아웃라이어 탐지 · 무결성 검증 (`team_robot_dataset.py`)")

# 데이터셋 선택 -- 다른 페이지(③④)와 st.session_state로 선택을 공유한다.
_dataset_options = dataset_registry.dataset_options()
st.session_state.setdefault("selected_dataset_id", dataset_registry.DEFAULT_DATASET_ID)
_current_id = st.session_state["selected_dataset_id"]
_labels = list(_dataset_options.keys())
_current_label = next(label for label, did in _dataset_options.items() if did == _current_id)
_selected_label = st.selectbox("데이터셋", options=_labels, index=_labels.index(_current_label))
dataset_id = _dataset_options[_selected_label]
st.session_state["selected_dataset_id"] = dataset_id
profile = dataset_registry.get_profile(dataset_id)

# z_thresh 기본값을 위젯 렌더보다 먼저 세션 상태에 심어둔다 -- 액션 카드
# (스냅샷 저장 버튼)가 설정 카드보다 위에 오지만, 저장에 쓰는 qa 결과는
# 설정 카드의 z_thresh 입력값에 의존한다. key로 바인딩된 위젯은 프레임워크가
# 스크립트 재실행 전에 session_state를 먼저 갱신해주므로, 여기서 읽는 값은
# 항상 "지금 사용자가 입력해둔" 값과 같다(위젯을 아직 렌더하기 전이어도).
st.session_state.setdefault("qa_z_thresh_value", 3.0)
st.session_state.setdefault("qa_z_thresh_delta", 3.0)

with st.spinner("TeamRobotDataset 분석 실행 중..."):
    qa = ds.get_qa_results(
        z_thresh_value=st.session_state["qa_z_thresh_value"],
        z_thresh_delta=st.session_state["qa_z_thresh_delta"],
        dataset_id=dataset_id,
    )

# 1. 행동(action) -- 이 페이지의 유일한 실행 버튼
with st.container(border=True):
    render_card_header("📌", "학습용 스냅샷 저장", "action")
    st.caption(
        "QA 검증은 페이지 접속 시 자동으로 실행되며, 결과를 확정된 시점의 train/val episode 구성으로 저장합니다 \n\n"
        "'③ 학습' 페이지가 이 스냅샷을 자동으로 읽어 `--dataset.episodes`를 채우는 데 씁니다.\n\n"
        "데이터셋 자체는 전혀 건드리지 않고 인덱스 구성만 JSON으로 남깁니다."
    )
    if st.button("📌 이 결과를 학습용으로 확정", type="primary"):
        saved_path = qa_snapshot.save_snapshot(qa, repo_id=profile["repo_id"], root=str(profile["dataset_root"]))
        st.success(f"저장했습니다: {saved_path.name}")
        st.rerun()

# 2. 설정(settings) -- z_thresh 파라미터 + "현재 설정 기준" 실시간 수치
# (저장 전에 보고 판단하는 용도이므로 result가 아니라 여기 소속)
with st.container(border=True):
    render_card_header("⚙️", "QA 파라미터 (z_thresh) — 현재 설정 기준 실시간 수치", "settings")

    zc1, zc2 = st.columns(2)
    zc1.number_input(
        "값 기준 z_thresh (detect_outliers)", min_value=0.5, max_value=10.0, step=0.5, key="qa_z_thresh_value"
    )
    zc2.number_input(
        "delta 기준 z_thresh (detect_outliers_delta)", min_value=0.5, max_value=10.0, step=0.5, key="qa_z_thresh_delta"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("스캔한 episode 수", qa["num_episodes"])
    c2.metric("스캔한 frame 수", qa["num_frames"])
    c3.metric("무결성 이상(NaN/Inf)", sum(qa["integrity"]["episode_problem_counts"].values()) or 0)
    c4.metric("clean frame 비율", f"{qa['clean']['total_count'] - qa['clean']['excluded_count']}/{qa['clean']['total_count']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**값 기준 아웃라이어 (`detect_outliers`)**")
        voc = qa["value_outliers"]["episode_outlier_counts"]
        st.table({"episode": list(voc.keys()) or ["(없음)"], "이상 frame 수": list(voc.values()) or [0]})
    with col2:
        st.markdown("**delta 기준 아웃라이어 (`detect_outliers_delta`)**")
        doc = qa["delta_outliers"]["episode_outlier_counts"]
        st.table({"episode": list(doc.keys()) or ["(없음)"], "이상 frame 수": list(doc.values()) or [0]})

    st.markdown("**`get_clean_indices()`** — timestamp 필드 제외, 값+delta 기준 합집합")
    clean = qa["clean"]
    st.write(
        f"전체 {clean['total_count']} frames 중 **{clean['excluded_count']}건 제외** "
        f"({clean['total_count'] - clean['excluded_count']} clean) — *비파괴적, 실제로 제거하지 않음*"
    )

    st.markdown("**`split_episodes(val_ratio=0.2, seed=42)`**")
    split = qa["split"]
    st.write(f"train_episodes = {split['train_episodes']}")
    st.write(f"val_episodes = {split['val_episodes']}")

# 3. 결과(results) -- 저장된 스냅샷이 있을 때만, "완료" 요약 하나만
# 다른 데이터셋의 스냅샷과 섞이면 "최신"의 의미가 깨지므로 repo_id로 필터링.
snapshots = [s for s in qa_snapshot.list_snapshots() if s.get("repo_id") == profile["repo_id"]]
if snapshots:
    _latest = snapshots[0]
    with st.container(border=True):
        render_card_header("✅", "스냅샷 저장 완료", "success")
        s1, s2, s3 = st.columns(3)
        s1.metric("train episodes", len(_latest["train_episodes"]))
        s2.metric("val episodes", len(_latest["val_episodes"]))
        s3.metric("제외된 frame 수", _latest["excluded_count"])
        st.caption(f"최신 스냅샷: `{_latest['file']}` ({_latest['created_at']})")
        st.page_link("pages/3_ACT_학습.py", label="다음: ③ 학습 시작하기 →", icon="➡️")

# 4. 메타정보(접힘) -- 코드 구조/이상치 사례/시각화/스냅샷 전체 이력
with st.expander("(자세히 보기) 코드 구조 · 이상치 필터링 사례 · 시각화 산출물 · 스냅샷 이력"):
    st.page_link(
        "pages/5_전처리_구조.py",
        label="🔍 이 파이프라인의 전처리 구조 보기 (이미지 리사이즈/증강, 액션 델타/청킹, 프레임 스태킹, 센서 동기화)",
        icon="🔍",
    )
    st.divider()

    st.subheader("코드 구조")
    st.table(
        {
            "메서드": [
                "compute_custom_stats()", "detect_outliers()", "detect_outliers_delta()",
                "check_integrity()", "get_clean_indices()", "split_episodes()",
                "visualize_episode()", "plot_action_distribution()",
            ],
            "역할": [
                "필드별 mean/std 계산 (다른 메서드들이 재사용)",
                "값 기준 z-score 아웃라이어 탐지 (|z|>3)",
                "프레임 간 delta 기준 z-score 아웃라이어 탐지 (값 기준과 상호보완적)",
                "NaN/Inf 스캔 (생성자에서 자동 실행)",
                "위 결과들의 합집합으로 '이상 없는' 인덱스 계산 (비파괴적)",
                "episode 단위 train/val 분할 (비파괴적, 인덱스만 계산)",
                "궤적 + 아웃라이어 오버레이 PNG 생성",
                "관절별 값 분포 히스토그램 PNG 생성",
            ],
        }
    )

    if dataset_id == "block_pickplace":
        st.subheader("실제로 있었던 이상치 필터링 사례")
        with st.container(border=True):
            st.markdown("#### wrist_roll clip 진단·수정 (CLAUDE.md 23절)")
            st.write(
                "20-episode 데이터에서 `wrist_roll` std가 다른 관절보다 28배 작다는 게 발견됨 → 진단 결과 "
                "task 특성이 아니라 **leader 실측 캘리브레이션 range가 follower MJCF 모델 range보다 넓어서 "
                "생기는 clip**으로 판명 (1:1 직접 매핑의 구조적 문제)."
            )
            b1, b2 = st.columns(2)
            b1.metric("수정 전 (재캘리브레이션 전)", "16 / 20 episodes", "wrist_roll 하한 고정", delta_color="inverse")
            b2.metric("수정 후 (비례 스케일링 적용)", "0 / 28 episodes", "clip 프레임 0%")
            img_path = ds.get_asset_path("wrist_roll_clip_diagnosis.png", dataset_id=dataset_id)
            if img_path.exists():
                st.image(str(img_path), caption="wrist_roll clip 진단 (수정 전)")

        with st.container(border=True):
            st.markdown("#### episode 11 제외 (CLAUDE.md 24-3절)")
            st.write(
                "값/delta 기준 아웃라이어 0건, 최단 프레임(225), wrist_roll range 0.00°로 완전 무변화 — "
                "사용자 확인 결과 **S 키 오조작으로 인한 무효 데모**로 판명되어 학습 시 `episodes` 파라미터로 제외."
            )

        st.subheader("시각화 산출물")
        img_cols = st.columns(2)
        for i, (fname, caption) in enumerate(
            [
                ("action_distribution.png", "관절별 값 분포 히스토그램"),
                ("episode_0_trajectory_combined.png", "episode 0 궤적 + 값/delta 아웃라이어 오버레이"),
                ("ep0_new17_trajectory.png", "재캘리브레이션 후 17-episode 데이터 (episode 0)"),
                ("ep9_new17_trajectory.png", "재캘리브레이션 후 17-episode 데이터 (episode 9)"),
            ]
        ):
            p = ds.get_asset_path(fname, dataset_id=dataset_id)
            if p.exists():
                with img_cols[i % 2]:
                    st.image(str(p), caption=caption)

        # 주의: st.expander는 다른 expander 안에 중첩할 수 없다(Streamlit 제약) --
        # 이 섹션이 원래 독립 expander였으나, 지금은 위 메타정보 expander 안에
        # 있으므로 일반 섹션(subheader + 상시 표시)으로 바꿨다.
        st.subheader("⚠️ 참고용: 예전 outlier_report.md (구식 — 시뮬레이션 2-episode 데이터 대상)")
        st.warning(
            "이 리포트는 학습에 쓴 `so101_teleop_real`이 아니라 훨씬 이전의 `so101_teleop`(시뮬레이션, "
            "2 episodes/109 frames) 데이터 대상입니다. 위 '설정' 카드의 실시간 수치가 현재 데이터 기준 최신입니다."
        )
        report_path = ds.get_asset_path("outlier_report.md", dataset_id=dataset_id)
        if report_path.exists():
            # 이 파일이 334KB나 돼서, 예전엔 접힌 상태에서도 st.markdown()이 매
            # rerun마다(위젯 클릭 한 번에도) 무조건 다시 파싱/렌더링돼 페이지가
            # 간헐적으로 멈추는 원인이었다 -- Streamlit은 접힌 영역도 "숨기기"만
            # 할 뿐 안의 코드는 그대로 실행한다. 버튼으로 명시적으로 요청할 때만
            # 읽고 렌더링하도록 바꿔서, 안 열어보면 이 비용 자체가 발생하지 않게 함.
            st.session_state.setdefault("qa_show_old_report", False)
            if st.button("📄 전체 내용 보기/숨기기", key="qa_toggle_old_report"):
                st.session_state["qa_show_old_report"] = not st.session_state["qa_show_old_report"]
            if st.session_state["qa_show_old_report"]:
                st.markdown(_load_old_report_text(report_path))
    else:
        st.caption(
            f"'{profile['label']}' 데이터셋은 아직 이상치 사례/시각화 산출물이 기록되지 않았습니다 — "
            "위 '설정' 카드의 실시간 수치가 이 데이터셋에 대한 유일한 QA 근거입니다."
        )

    st.subheader("저장된 스냅샷 이력 (전체)")
    if snapshots:
        st.dataframe(pd.DataFrame(snapshots), width='stretch', hide_index=True)
    else:
        st.caption("아직 저장된 스냅샷이 없습니다.")
