"""2단계: QA & 검증. team_robot_dataset.TeamRobotDataset의 읽기 전용
분석 메서드를 현재 so101_teleop_real 데이터셋에 대해 실제로 호출해서
최신 수치를 보여준다 (오래된 outlier_report.md는 시뮬레이션 2-episode
데이터 대상이라 참고용으로만 별도 표시). 아무 것도 필터링/삭제하지
않는다.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import qa_snapshot

st.set_page_config(page_title="② QA & 검증", page_icon="🔍", layout="wide")
st.title("② QA & 검증")
st.caption("`TeamRobotDataset` 커스텀 통계 · 아웃라이어 탐지 · 무결성 검증 (`team_robot_dataset.py`)")

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

st.divider()
st.subheader("라이브 QA 결과 (현재 so101_teleop_real 기준)")
st.caption("아래 수치는 페이지 로딩 시 실제로 `TeamRobotDataset`을 호출해서 계산한 것입니다 — 저장된 리포트가 아닙니다.")

zc1, zc2 = st.columns(2)
z_thresh_value = zc1.number_input("값 기준 z_thresh (detect_outliers)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
z_thresh_delta = zc2.number_input("delta 기준 z_thresh (detect_outliers_delta)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

with st.spinner("TeamRobotDataset 분석 실행 중..."):
    qa = ds.get_qa_results(z_thresh_value=z_thresh_value, z_thresh_delta=z_thresh_delta)

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

st.divider()
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
    img_path = ds.get_asset_path("wrist_roll_clip_diagnosis.png")
    if img_path.exists():
        st.image(str(img_path), caption="wrist_roll clip 진단 (수정 전)")

with st.container(border=True):
    st.markdown("#### episode 11 제외 (CLAUDE.md 24-3절)")
    st.write(
        "값/delta 기준 아웃라이어 0건, 최단 프레임(225), wrist_roll range 0.00°로 완전 무변화 — "
        "사용자 확인 결과 **S 키 오조작으로 인한 무효 데모**로 판명되어 학습 시 `episodes` 파라미터로 제외."
    )

st.divider()
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
    p = ds.get_asset_path(fname)
    if p.exists():
        with img_cols[i % 2]:
            st.image(str(p), caption=caption)

st.divider()
with st.expander("⚠️ 참고용: 예전 outlier_report.md (구식 — 시뮬레이션 2-episode 데이터 대상)"):
    st.warning(
        "이 리포트는 학습에 쓴 `so101_teleop_real`이 아니라 훨씬 이전의 `so101_teleop`(시뮬레이션, "
        "2 episodes/109 frames) 데이터 대상입니다. 위 '라이브 QA 결과'가 현재 데이터 기준 최신 수치입니다."
    )
    report_path = ds.get_asset_path("outlier_report.md")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))

st.divider()
st.subheader("학습용 스냅샷")
st.caption(
    "QA 결과를 확정된 시점의 train/val episode 구성으로 저장합니다 — 다음 세션에서 구현할 ③ 학습 페이지가 "
    "이 스냅샷을 자동으로 읽어 `--dataset.episodes`를 채우는 데 쓸 예정입니다 (CLAUDE.md 29-3절). "
    "비파괴적 — 데이터셋 자체는 전혀 건드리지 않고 인덱스 구성만 JSON으로 남깁니다."
)

if st.button("📌 이 결과를 학습용으로 확정 (스냅샷 저장)", type="primary"):
    saved_path = qa_snapshot.save_snapshot(qa, repo_id=ds.REPO_ID, root=str(ds.DATASET_ROOT))
    st.success(f"저장했습니다: {saved_path.name}")
    st.rerun()

snapshots = qa_snapshot.list_snapshots()
if snapshots:
    st.markdown("**저장된 스냅샷** (최신순)")
    st.dataframe(pd.DataFrame(snapshots), width='stretch', hide_index=True)
else:
    st.caption("아직 저장된 스냅샷이 없습니다.")
