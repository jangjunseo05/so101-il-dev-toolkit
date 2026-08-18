"""공통 UI 컴포넌트 -- 카드 헤더 색은 장식이 아니라 의미(role)에 고정.

page 코드는 색을 직접 고르거나 하드코딩하지 않고 항상 render_card_header()를
통해서만 카드 헤더를 그린다 -- 4개 단계 페이지 전체에서 같은 role은 항상
같은 색이 되도록 강제하기 위함("색상은 개체를 따르고 순위를 따르지 않는다",
lib/theme.py의 기존 원칙과 동일).

Streamlit 구현은 기존에 이미 쓰던 `st.container(border=True)` 패턴을 그대로
유지하고, 그 안 첫 줄에 이 함수로 그린 색상 헤더 한 줄만 추가하는 식으로
쓴다(호출부 예시):

    with st.container(border=True):
        render_card_header("🔴", "녹화 세션 제어", "action")
        ...

render_brand_header()(로고+"DAPIER")는 이 카드 role 색상 팔레트와 의도적으로
완전히 분리된 영역이다(2026-08-17) -- 로고 자체가 다색(주황/청록/보라)이라
action/settings/success/warning 색과 섞이면 그 색들의 "의미 고정" 원칙이
깨진다. 그래서 배경색을 아예 안 쓰고 로고+텍스트만, 얇은 회색 구분선으로
아래 콘텐츠와 분리한다 -- 카드 헤더처럼 배경색 블록으로 만들지 않음.
"""

import base64
from pathlib import Path

import streamlit as st
from PIL import Image

from lib.theme import AQUA, BLUE, GRID, TEXT_SECONDARY, YELLOW

_LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.png"

# role -> (배경색, 글자색). role은 이 4개로 고정 -- 새 역할이 필요해지면
# 색을 마음대로 고르지 말고 이 딕셔너리에 먼저 추가할 것.
_ROLE_STYLES = {
    "action": (BLUE, "#ffffff"),  # 버튼이 있는 행동 카드
    "settings": (TEXT_SECONDARY, "#ffffff"),  # 설정/구성 카드 (중립)
    "success": (AQUA, "#ffffff"),  # 결과/완료 카드
    "warning": (YELLOW, "#2b2200"),  # 경고/차단 카드 (밝은 배경이라 어두운 글자)
}


def render_card_header(icon: str, title: str, role: str) -> None:
    """`st.container(border=True)` 블록의 첫 줄에서 호출한다.

    role은 반드시 "action"/"settings"/"success"/"warning" 중 하나 -- 그 외
    값은 즉시 ValueError (조용히 기본색으로 넘어가지 않음, 잘못된 role을
    페이지 코드에 남겨두면 안 되므로).
    """
    if role not in _ROLE_STYLES:
        raise ValueError(f"unknown card role: {role!r} (expected one of {sorted(_ROLE_STYLES)})")
    bg, fg = _ROLE_STYLES[role]
    st.markdown(
        f'<div style="background-color:{bg};color:{fg};padding:0.4em 0.9em;'
        f'border-radius:0.4em;margin-bottom:0.7em;font-weight:600;font-size:1.05em;">'
        f"{icon} {title}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _logo_base64() -> str | None:
    if not _LOGO_PATH.is_file():
        return None
    return base64.b64encode(_LOGO_PATH.read_bytes()).decode()


def get_logo_icon() -> Image.Image | str:
    """`st.set_page_config(page_icon=...)`용. 로고 파일이 없으면(예: 이
    저장소를 아직 assets/logo.png 없이 clone한 경우) 조용히 기본 이모지로
    폴백 -- 브랜드 자산 누락이 앱 시작 자체를 막으면 안 되므로."""
    if _LOGO_PATH.is_file():
        return Image.open(_LOGO_PATH)
    return "🦾"


def render_brand_header() -> None:
    """페이지 최상단(제목보다 위)에서 호출 -- 로고 아이콘 + "DAPIER" 텍스트.

    카드 role 색상 팔레트와 완전히 분리된 영역(모듈 docstring 참고): 배경색
    없이 로고+텍스트만, 얇은 구분선으로 아래 콘텐츠와 분리한다. 로고 색을
    카드/버튼 색으로 끌어다 쓰지 않는다.
    """
    logo_b64 = _logo_base64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" width="36" height="36" '
        f'style="border-radius:6px;flex-shrink:0;">'
        if logo_b64
        else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6em;margin-bottom:0.6em;">'
        f"{logo_html}"
        f'<span style="font-weight:700;font-size:1.2em;letter-spacing:0.04em;">DAPIER</span>'
        f"</div>"
        f'<hr style="margin:0 0 1.2em 0;border:none;border-top:1px solid {GRID};">',
        unsafe_allow_html=True,
    )


# 로고 파일 코너 픽셀 실측(2026-08-17) -- 카드 role 색상 팔레트와는 무관,
# 로고 배경 자체의 남색과 통일하려고 이 색만 쓴다.
_SPLASH_NAVY = "#051644"


def render_splash_screen() -> None:
    """최초 접속 1회용 스플래시(app.py에서 welcome_seen=False일 때만 호출).

    카드 role 색상 팔레트(action/settings/success/warning)와 완전히 무관한
    독립 화면 -- 로고 자체의 남색(#051644, 로고 파일에서 실측)+흰색 톤만
    쓴다. `position:fixed` 전체 화면 덮개로 렌더링만 담당하고, 몇 초 뒤
    사라지게 하는 sleep+rerun은 호출부(app.py)의 책임.
    """
    logo_b64 = _logo_base64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" width="72" height="72" '
        f'style="border-radius:14px;">'
        if logo_b64
        else ""
    )
    st.markdown(
        f'<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;'
        f'background-color:{_SPLASH_NAVY};z-index:999999;display:flex;'
        f'flex-direction:column;align-items:center;justify-content:center;">'
        f"{logo_html}"
        f'<div style="margin-top:0.9em;color:#ffffff;font-size:19px;'
        f'font-weight:600;letter-spacing:0.06em;">DAPIER</div>'
        f'<div style="margin-top:1.4em;color:#ffffff;font-size:36px;'
        f'font-weight:700;text-align:center;line-height:1.4;padding:0 1em;">'
        f"누구나 모방학습을<br>쉽고 간단하게</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
