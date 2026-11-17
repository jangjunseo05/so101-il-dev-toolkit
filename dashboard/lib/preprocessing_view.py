"""② QA 검증 페이지 하단 '(자세히 보기) 전처리 구조' 섹션.

예전 별도 페이지(pages/5_전처리_구조.py, AGENTS.md 42절)를 QA 페이지 안으로 옮긴 것.
버튼이 없는 정보 뷰라 사이드바 단계와 섞이지 않게 QA의 접힌 섹션에 둔다.
내용은 손으로 옮겨 적지 않고 TeamRobotDataset.describe_preprocessing()의 반환값을
그대로 렌더링한다(정규화 통계는 이 데이터셋의 meta/stats.json 실측값).
호출하는 쪽이 st.expander 안에서 부르므로, 여기서는 expander를 다시 열지 않는다.
"""

import pandas as pd
import streamlit as st

from lib import data_sources as ds
from lib.ui_components import render_card_header


def render() -> None:
    st.caption(
        "로봇 모방학습에서 흔히 필요한 전처리 항목들이 실제로 어디서(QA 계층 / ACT policy 자체 / "
        "lerobot-train CLI 옵션) 처리되는지 — `describe_preprocessing()`이 반환한 값을 그대로 표시합니다."
    )
    if not ds.has_episodes():
        st.info("아직 녹화된 데이터셋이 없습니다 — 정규화 통계는 데이터셋에서 읽으므로 ① 데이터 수집 후 표시됩니다.", icon="📭")
        return

    with st.spinner("전처리 구조 정보 로딩 중..."):
        desc = ds.get_preprocessing_description()

    with st.container(border=True):
        render_card_header("📐", "정규화(Normalization) — 결정 vs 적용", "settings")
        st.write(desc["normalization"]["summary"])
        st.caption(f"통계 출처: `{desc['normalization']['source_file']}` (이 데이터셋 실측값)")
        for sample in desc["normalization"]["samples"]:
            st.markdown(f"**`{sample['key']}`**")
            df = pd.DataFrame(sample["dims"])
            df.columns = ["관절", "mean", "std"]
            st.dataframe(df, width="stretch", hide_index=True)

    with st.container(border=True):
        render_card_header("📋", "표준 전처리 6항목 — 어디서 처리되는가", "settings")
        rows = [
            {
                "항목": item["name"],
                "QA 계층 (TeamRobotDataset)": item["qa_layer"],
                "ACT policy 자체": item["act_policy"],
                "lerobot-train CLI": item["lerobot_train_cli"],
                "상태": item["status_label"],
            }
            for item in desc["items"]
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    resize = next(item for item in desc["items"] if item["name"] == "이미지 리사이즈")
    with st.container(border=True):
        render_card_header("⚠️", f"{resize['name']} — {resize['status_label']}", "warning")
        st.write(resize["note"])

    st.markdown("**나머지 5항목 개별 설명**")
    for item in desc["items"]:
        if item["name"] == "이미지 리사이즈":
            continue
        st.markdown(f"**{item['name']}** — {item['status_label']}")
        st.caption(item["note"])
