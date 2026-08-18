"""② QA 검증 페이지의 메타정보 섹션에서 연결되는 상세 뷰. 행동(버튼)이
없는 순수 정보 페이지라 카드 위계(행동→설정→결과→메타)를 적용할 필요가
없음 -- 정보 카드만 나열한다(2026-08-17).

이 페이지는 CLAUDE.md 39/40절 내용을 손으로 옮겨 적지 않는다 -- 전부
TeamRobotDataset.describe_preprocessing()(pipeline/2_qa/team_robot_dataset.py)의
반환값을 그대로 렌더링한다. 정규화 통계는 실제 meta/stats.json에서 온
이 데이터셋의 실측값이고, 6항목 표는 파이프라인 코드 구조(ACT 정책 설정,
lerobot-train CLI 옵션 유무)에 대한 고정 설명이다.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib.ui_components import get_logo_icon, render_brand_header, render_card_header

st.set_page_config(page_title="전처리 구조", page_icon=get_logo_icon(), layout="wide")

render_brand_header()
st.title("🔍 이 파이프라인의 전처리 구조")
st.caption(
    "로봇 모방학습에서 흔히 필요한 전처리 항목들이 실제로 어디서(QA 계층 / ACT policy 자체 / "
    "lerobot-train CLI 옵션) 처리되는지 -- `describe_preprocessing()`이 반환한 값을 그대로 표시합니다."
)
st.page_link("pages/2_QA_검증.py", label="← ② QA 검증으로 돌아가기", icon="⬅️")

with st.spinner("전처리 구조 정보 로딩 중..."):
    desc = ds.get_preprocessing_description()

# 1. 정규화 -- 설정(중립) 카드
with st.container(border=True):
    render_card_header("📐", "정규화(Normalization) — 결정 vs 적용", "settings")
    st.write(desc["normalization"]["summary"])
    st.caption(f"통계 출처: `{desc['normalization']['source_file']}` (이 데이터셋 실측값)")

    for sample in desc["normalization"]["samples"]:
        st.markdown(f"**`{sample['key']}`**")
        df = pd.DataFrame(sample["dims"])
        df.columns = ["관절", "mean", "std"]
        st.dataframe(df, width="stretch", hide_index=True)

# 2. 6항목 표 -- 설정(중립) 카드
with st.container(border=True):
    render_card_header("📋", "표준 전처리 6항목 — 어디서 처리되는가", "settings")
    table_rows = [
        {
            "항목": item["name"],
            "QA 계층 (TeamRobotDataset)": item["qa_layer"],
            "ACT policy 자체": item["act_policy"],
            "lerobot-train CLI": item["lerobot_train_cli"],
            "상태": item["status_label"],
        }
        for item in desc["items"]
    ]
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

# 3. 이미지 리사이즈 -- 유일한 "진짜 빈 자리", warning role로 별도 강조
_resize_item = next(item for item in desc["items"] if item["name"] == "이미지 리사이즈")
with st.container(border=True):
    render_card_header("⚠️", f"{_resize_item['name']} — {_resize_item['status_label']}", "warning")
    st.write(_resize_item["note"])

with st.expander("(자세히 보기) 나머지 5항목 개별 설명"):
    for item in desc["items"]:
        if item["name"] == "이미지 리사이즈":
            continue
        st.markdown(f"**{item['name']}** — {item['status_label']}")
        st.caption(item["note"])
