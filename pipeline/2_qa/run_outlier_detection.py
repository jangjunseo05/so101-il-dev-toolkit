"""Runs TeamRobotDataset.detect_outliers() (value-based) and
detect_outliers_delta() (delta-based) against data/so101_teleop_real and
writes reports/outlier_report.md (spec v0.2 section 7/9, v0.4 section 14).

Detection only -- does not filter/remove any frames or episodes (out of scope
per section 8).
"""

from __future__ import annotations

from pathlib import Path

from team_robot_dataset import TeamRobotDataset

# so101_web/pipeline/2_qa/run_outlier_detection.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "so101_teleop_real"
REPO_ID = "so101_teleop_real"
Z_THRESH = 3.0
DELTA_Z_THRESH = 3.0
REPORT_PATH = PROJECT_ROOT / "reports" / "outlier_report.md"
# episode used for the combined red/blue overlay PNG at the end of the report
OVERLAY_EPISODE_IDX = 0
OVERLAY_PNG_PATH = PROJECT_ROOT / "reports" / f"episode_{OVERLAY_EPISODE_IDX}_trajectory_combined.png"


def _render_frame_sections(result: dict) -> list[str]:
    """Episode-level summary + frame-level detail table. Shared verbatim by
    both the value-based and delta-based sections of the report."""
    lines = []
    lines.append("## Episode-level summary (derived from frame-level results)")
    lines.append("")
    lines.append("| episode_index | outlier frame count |")
    lines.append("|---|---|")
    if result["episode_outlier_counts"]:
        for ep, count in result["episode_outlier_counts"].items():
            lines.append(f"| {ep} | {count} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")

    lines.append("## Frame-level detail")
    lines.append("")
    if not result["frame_outliers"]:
        lines.append(f"No frames exceeded |z| > {result['z_thresh']} for any checked key/dim.")
    else:
        lines.append("| frame_idx | episode_index | key | dim | joint_name | value | mean | std | z |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for o in result["frame_outliers"]:
            lines.append(
                f"| {o['frame_idx']} | {o['episode_index']} | {o['key']} | {o['dim']} | "
                f"{o['joint_name']} | {o['value']:.4f} | {o['mean']:.4f} | {o['std']:.4f} | {o['z']:+.3f} |"
            )
    lines.append("")
    return lines


def build_report(value_result: dict, delta_result: dict, dataset, overlay_png_path: Path) -> str:
    lines = []
    lines.append("# Outlier Report -- so101_teleop_real")
    lines.append("")
    lines.append(f"- keys checked: {', '.join(value_result['keys'])}")
    lines.append(f"- total frames: {value_result['num_frames']}")
    lines.append(f"- total episodes: {dataset.meta.total_episodes}")
    lines.append(
        "- mean/std source: `TeamRobotDataset.compute_custom_stats()` "
        "(reused as-is, not recomputed here)"
    )
    lines.append("")
    lines.append(
        "> Detection + reporting only. No frames/episodes are filtered or "
        "removed by this script (out of v0.2 scope, spec section 8)."
    )
    lines.append("")
    lines.append(
        "**요약 (CLAUDE.md 14절)**: 값 기준(`detect_outliers()`)과 delta 기준"
        "(`detect_outliers_delta()`)은 서로 다른 유형의 이상을 탐지하는 상호보완적"
        " 도구입니다 -- 값 기준은 평균에서 오래 벗어난 상태(지속적 극값)를, delta"
        " 기준은 직전 프레임 대비 급격한 변화(간헐적 전환)를 잡습니다. 하나로"
        " 통합하지 않고 둘 다 병행 유지합니다."
    )
    lines.append("")

    lines.append("## Value-based outliers (`detect_outliers()`)")
    lines.append("")
    lines.append(f"- z-score threshold: |z| > {value_result['z_thresh']}")
    lines.append("")
    lines.extend(_render_frame_sections(value_result))

    lines.append("## Delta-based outliers (`detect_outliers_delta()`)")
    lines.append("")
    lines.append(f"- z-score threshold: |z| > {delta_result['z_thresh']}")
    lines.append(
        "- delta = value[frame] - value[previous frame] within the same episode"
        " (never computed across an episode boundary; each episode's first"
        " frame has no delta and cannot be flagged)"
    )
    lines.append("")
    lines.extend(_render_frame_sections(delta_result))

    lines.append("## Combined overlay visualization")
    lines.append("")
    lines.append(
        f"`visualize_episode({OVERLAY_EPISODE_IDX}, outlier_results=<value-based>,"
        " delta_outlier_results=<delta-based>)` overlays both detections on the"
        " same trajectory plot: red = value-based outlier, blue = delta-based"
        " outlier."
    )
    lines.append("")
    lines.append(f"![combined overlay]({overlay_png_path.name})")
    lines.append("")
    lines.append(f"(file path: `{overlay_png_path}`)")
    lines.append("")

    return "\n".join(lines)


def main():
    dataset = TeamRobotDataset(REPO_ID, root=DATASET_ROOT)
    print(f"dataset: {DATASET_ROOT} (len={len(dataset)}, episodes={dataset.meta.total_episodes})")

    value_result = dataset.detect_outliers(z_thresh=Z_THRESH)
    delta_result = dataset.detect_outliers_delta(z_thresh=DELTA_Z_THRESH)

    print(f"[value] z_thresh={value_result['z_thresh']} keys={value_result['keys']}")
    print(f"[value] total outlier entries: {len(value_result['frame_outliers'])}")
    print(f"[value] episode_outlier_counts: {value_result['episode_outlier_counts']}")
    print()
    for o in value_result["frame_outliers"]:
        print(
            f"  [value] frame={o['frame_idx']:>3} ep={o['episode_index']} key={o['key']:<18} "
            f"dim={o['dim']} joint={o['joint_name']} value={o['value']:.4f} "
            f"mean={o['mean']:.4f} std={o['std']:.4f} z={o['z']:+.3f}"
        )
    print()

    print(f"[delta] z_thresh={delta_result['z_thresh']} keys={delta_result['keys']}")
    print(f"[delta] total outlier entries: {len(delta_result['frame_outliers'])}")
    print(f"[delta] episode_outlier_counts: {delta_result['episode_outlier_counts']}")
    print()
    for o in delta_result["frame_outliers"]:
        print(
            f"  [delta] frame={o['frame_idx']:>3} ep={o['episode_index']} key={o['key']:<18} "
            f"dim={o['dim']} joint={o['joint_name']} delta={o['value']:.4f} "
            f"mean={o['mean']:.4f} std={o['std']:.4f} z={o['z']:+.3f}"
        )
    print()

    dataset.visualize_episode(
        OVERLAY_EPISODE_IDX,
        outlier_results=value_result,
        delta_outlier_results=delta_result,
        save_path=OVERLAY_PNG_PATH,
    )
    print(f"wrote {OVERLAY_PNG_PATH}")

    report = build_report(value_result, delta_result, dataset, OVERLAY_PNG_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
