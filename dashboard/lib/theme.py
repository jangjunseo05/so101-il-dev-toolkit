"""공통 색상/스타일 상수. dataviz 스킬의 카테고리 팔레트(고정 순서)를 그대로
재사용해 이 세션에서 만든 다른 차트(checkpoint_sweep_loss.png)와 색이
일관되게 맞도록 한다.
"""

BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
YELLOW = "#eda100"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

# 색상 역할 고정: 항상 같은 지표에는 같은 색을 쓴다("색상은 개체를 따르고
# 순위를 따르지 않는다").
TRAIN_COLOR = BLUE
VAL_COLOR = AQUA
RATIO_COLOR = VIOLET

STAGE_NAMES = ["데이터 수집", "QA & 검증", "ACT 학습", "추론"]
STAGE_PAGES = [
    "pages/1_데이터_수집.py",
    "pages/2_QA_검증.py",
    "pages/3_ACT_학습.py",
    "pages/4_추론.py",
]
