"""
run_val_loss_check.py가 저장한 checkpoint_sweep_results.json을 읽어
step별 train/val loss와 val/train 비율을 2개의 단일축 subplot으로 그린다
(dual-axis는 쓰지 않음 -- loss와 비율은 스케일이 달라 각자 축을 가진
별도 subplot으로 분리).

색상은 dataviz 스킬의 카테고리 팔레트(고정 순서) 사용: train=slot1 blue,
val=slot2 aqua, ratio=slot5 violet(서로 다른 지표라 loss 두 색과 구분).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# so101_web/pipeline/3_training/plot_checkpoint_sweep.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_JSON = REPORTS_DIR / "checkpoint_sweep_results.json"
OUTPUT_PNG = REPORTS_DIR / "checkpoint_sweep_loss.png"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
TEXT_SECONDARY = "#52514e"

# val loss 개선폭이 급격히 줄어드는 지점(직전 분석에서 확인) -- early stopping
# 후보 구간 표시용.
KNEE_ZONE = (6000, 7500)


def main():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    steps = data["steps"]
    results = data["results"]
    train_loss = [results[str(s)]["train_loss"] for s in steps]
    val_loss = [results[str(s)]["val_loss"] for s in steps]
    ratio = [results[str(s)]["ratio"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("ACT checkpoint sweep: train vs val loss (episodes 0,1 vs 5,7,9)", fontsize=12)

    ax1.plot(steps, train_loss, color=BLUE, linewidth=2, marker="o", markersize=8, label="train loss")
    ax1.plot(steps, val_loss, color=AQUA, linewidth=2, marker="o", markersize=8, label="val loss")
    ax1.axvspan(*KNEE_ZONE, color=TEXT_SECONDARY, alpha=0.08)
    ax1.text(
        sum(KNEE_ZONE) / 2, max(train_loss + val_loss) * 0.92, "val loss\nplateau starts",
        ha="center", va="top", fontsize=9, color=TEXT_SECONDARY,
    )
    ax1.set_ylabel("loss")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="#e5e4e0", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    ax2.plot(steps, ratio, color=VIOLET, linewidth=2, marker="o", markersize=8)
    ax2.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax2.text(steps[0], 1.05, "ratio=1 (val ≈ train)", fontsize=8, color=TEXT_SECONDARY)
    ax2.axvspan(*KNEE_ZONE, color=TEXT_SECONDARY, alpha=0.08)
    ax2.set_ylabel("val / train loss ratio")
    ax2.set_xlabel("training step")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#e5e4e0", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    for s, t, v, r in zip(steps, train_loss, val_loss, ratio):
        ax1.annotate(f"{t:.2f}", (s, t), textcoords="offset points", xytext=(0, -14), fontsize=7, color=BLUE, ha="center")
        ax1.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(0, 6), fontsize=7, color=AQUA, ha="center")
        ax2.annotate(f"{r:.2f}x", (s, r), textcoords="offset points", xytext=(0, 6), fontsize=7, color=VIOLET, ha="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"저장: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
