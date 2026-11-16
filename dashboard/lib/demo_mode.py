"""외부 공개용 체험 데모 모드 (2026-11).

환경 변수 `SO101_DEMO=1`로 켠다. 공개 서버에는 리더암(USB 시리얼)도 GPU도
없으므로, 장비가 필요한 행동(녹화·학습·MuJoCo 뷰어 추론)은 안내 문구로 바꾸고,
나머지는 실제 결과물(`demo_assets/`)을 읽어 그대로 보여 준다.
- QA 검증은 데모에서도 실제로 계산한다(이미지 열을 뺀 경량 데이터셋 사용).
- `demo_assets/`는 `tools/export_demo_assets.py`가 실제 녹화·학습·평가 결과에서 만든다.

HOLD THE FLOW 주문 웹의 공개 데모와 같은 원칙: 실제 장비와 연결되지 않는다는
사실을 모든 페이지 상단에 표시한다.
"""

import os
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENABLED = os.environ.get("SO101_DEMO", "").strip() == "1"
# SO101_DEMO_ASSETS로 다른 폴더를 가리킬 수 있다(시험용).
DEMO_ROOT = Path(os.environ.get("SO101_DEMO_ASSETS", str(PROJECT_ROOT / "demo_assets")))

REPO_URL = "https://github.com/jangjunseo05/so101-il-dev-toolkit"


def render_banner() -> None:
    if not ENABLED:
        return
    st.info(
        "**체험용 데모** — 이 화면은 실제 리더암·GPU와 연결되어 있지 않습니다. "
        "녹화·학습·시뮬레이션 실행 버튼은 막혀 있고, 실제로 수집·학습·평가한 결과를 그대로 보여 줍니다. "
        f"② QA 검증은 이 서버에서 실제로 계산됩니다. 코드: [GitHub]({REPO_URL})",
        icon="🧪",
    )
    if not has_results():
        st.warning(
            "실제 녹화·학습·평가 결과는 준비 중입니다. 결과가 올라오기 전까지는 각 단계의 화면 구성과 기능 설명만 볼 수 있습니다.",
            icon="⏳",
        )


def has_results() -> bool:
    """export_demo_assets.py로 만든 데이터셋이 들어 있는지."""
    return (DEMO_ROOT / "dataset" / "meta" / "info.json").is_file()


def render_locked(message: str) -> None:
    """데모에서 실행할 수 없는 행동 자리에 대신 그린다."""
    suffix = " 아래는 실제로 실행한 결과입니다." if has_results() else ""
    st.warning(f"{message}{suffix}", icon="🔒")
