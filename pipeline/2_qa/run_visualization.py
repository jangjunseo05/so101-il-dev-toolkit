"""Runs TeamRobotDataset.visualize_episode() and plot_action_distribution()
against data/so101_teleop_real (spec v0.3, sections 10 & 11).

detect_outliers() is called once here (z_thresh=2.0, the threshold that
actually produced hits per the v0.2 sanity check) and its result is passed
into visualize_episode() -- visualize_episode itself never re-runs detection.
"""

from __future__ import annotations

from pathlib import Path

from team_robot_dataset import TeamRobotDataset

# so101_web/pipeline/2_qa/run_visualization.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "so101_teleop_real"
REPO_ID = "so101_teleop_real"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def main():
    ds = TeamRobotDataset(REPO_ID, root=DATASET_ROOT)
    print(f"dataset: {DATASET_ROOT} (len={len(ds)}, episodes={ds.meta.total_episodes})")

    outlier_results = ds.detect_outliers(z_thresh=2.0)
    print(
        f"detect_outliers(z_thresh=2.0): "
        f"{len(outlier_results['frame_outliers'])} entries, "
        f"episode_outlier_counts={outlier_results['episode_outlier_counts']}"
    )

    for ep_idx in (0, 1):
        path = ds.visualize_episode(
            ep_idx, outlier_results=outlier_results,
            save_path=OUTPUT_DIR / f"episode_{ep_idx}_trajectory.png",
        )
        print(f"wrote {path.resolve()} (exists={path.exists()}, size={path.stat().st_size} bytes)")

    dist_path = ds.plot_action_distribution(save_path=OUTPUT_DIR / "action_distribution.png")
    print(f"wrote {dist_path.resolve()} (exists={dist_path.exists()}, size={dist_path.stat().st_size} bytes)")

    # also demonstrate no-overlay path per spec ("outlier_results=None이면 오버레이 없이")
    path_no_overlay = ds.visualize_episode(
        0, outlier_results=None, save_path=OUTPUT_DIR / "episode_0_trajectory_no_overlay.png"
    )
    print(f"wrote {path_no_overlay.resolve()} (exists={path_no_overlay.exists()})")


if __name__ == "__main__":
    main()
