"""홈 화면 동적 CTA + 단계별 진행 표시 판단 로직 (2026-08-17, 직관 UX 개선).

기존 state 파일(잠금/학습 run/QA 스냅샷/데이터셋)만 조회하며 새 state를
만들지 않는다. `get_next_action()`의 우선순위는 사용자가 명시적으로
확정한 규칙(첫 매치 채택):
  1. 살아있는 lock -> 진행 중인 세션 안내
  2. interrupted 학습 run 존재 -> 이어서 학습
  3. completed 학습 run 존재 -> 추론
  4. qa_snapshot 존재(완료된 run 없음) -> 학습 시작
  5. 데이터셋에 episode 존재(스냅샷 없음) -> QA 검증
  6. 아무 state도 없음 -> 데이터 수집부터

completed 학습 run과 그보다 새로운 qa_snapshot이 동시에 존재하는 경우
(예: 학습 후 episode를 더 모아 재QA한 경우)에도 규칙 3이 규칙 4보다
항상 우선한다 -- 순서 그대로, recency 비교 없음(사용자 확정 결정).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data_sources as ds
from lib import process_manager as pm
from lib import qa_snapshot
from lib import training_run as tr

# 잠금이 존재할 수 있는 stage만 -- QA는 인프로세스 계산이라 lock 개념이 없음.
_STAGE_INFO: dict[str, tuple[str, str]] = {
    "data_collection": ("① 데이터 수집", "pages/1_데이터_수집.py"),
    "training": ("③ ACT 학습", "pages/3_ACT_학습.py"),
    "inference": ("④ 추론", "pages/4_추론.py"),
}


def get_next_action() -> dict:
    """{"kind": "in_progress"|"cta", "icon": str, "message": str, "target_page": str}"""
    for stage, (name, page) in _STAGE_INFO.items():
        if pm.get_job_status(stage)["running"]:
            return {
                "kind": "in_progress",
                "icon": "🔵",
                "message": f"진행 중: {name} 세션 (보러가기)",
                "target_page": page,
            }

    runs = tr.list_runs()

    if any(r.get("status") == "interrupted" for r in runs):
        return {
            "kind": "cta",
            "icon": "▶️",
            "message": "이어서 학습할 수 있는 run이 있습니다 → ③ 학습",
            "target_page": "pages/3_ACT_학습.py",
        }

    if any(r.get("status") == "completed" for r in runs):
        return {
            "kind": "cta",
            "icon": "➡️",
            "message": "다음: ④ 추론으로 확인하기",
            "target_page": "pages/4_추론.py",
        }

    if qa_snapshot.list_snapshots():
        return {
            "kind": "cta",
            "icon": "➡️",
            "message": "다음: ③ 학습 시작하기",
            "target_page": "pages/3_ACT_학습.py",
        }

    if ds.get_dataset_info()["total_episodes"] > 0:
        return {
            "kind": "cta",
            "icon": "➡️",
            "message": "다음: ② QA 검증하기",
            "target_page": "pages/2_QA_검증.py",
        }

    return {
        "kind": "cta",
        "icon": "🚀",
        "message": "① 데이터 수집부터 시작하세요",
        "target_page": "pages/1_데이터_수집.py",
    }


def get_stage_progress() -> list[bool]:
    """[①,②,③,④] 각 단계의 "완료" 여부만 -- 진행 중 표시(🔵)는 이 값을
    쓰는 쪽에서 "처음으로 완료 안 된 단계"로 파생시킨다(watermark 방식).

    ④(추론)는 항상 False로 고정한다 -- 추론은 학습의 completed 같은 지속
    완료 기록이 없는 세션형 작업이라는 게 페이지 4 설계 단계에서 이미
    확인된 사실(트랙 A/B 둘 다 결과를 디스크에 남기지 않음). "완료"라는
    개념 자체가 성립하지 않으므로 거짓 신호를 만들지 않고 항상 미완료로
    둔다 -- ①②③이 전부 끝나면 자연히 ④가 "다음 할 일"로 표시된다.
    """
    done_1 = ds.get_dataset_info()["total_episodes"] > 0
    done_2 = bool(qa_snapshot.list_snapshots())
    done_3 = any(r.get("status") == "completed" for r in tr.list_runs())
    done_4 = False
    return [done_1, done_2, done_3, done_4]


def render_stage_progress() -> None:
    """모든 페이지(홈 + ①②③④) 상단에서 호출하는 공통 진행 표시.
    "처음으로 완료 안 된 단계"를 🔵(현재)로 표시하는 watermark 방식 --
    전부 완료면(①②③ 끝) ④가 항상 미완료라 자연히 ④가 🔵로 남는다.
    """
    done = get_stage_progress()
    current = next((i for i, d in enumerate(done) if not d), len(done) - 1)
    labels = ["①", "②", "③", "④"]
    parts = []
    for i, d in enumerate(done):
        if d:
            parts.append(f"{labels[i]}✅")
        elif i == current:
            parts.append(f"{labels[i]}🔵")
        else:
            parts.append(f"{labels[i]}⚪")
    st.caption(" ".join(parts))
