"""Usage example for TeamRobotDataset.get_clean_indices() (spec section 8/9).

Shows the intended pattern: run detect_outliers()/detect_outliers_delta(),
compute clean/excluded row indices, then wrap with torch.utils.data.Subset
(NOT LeRobotDataset(..., episodes=[...]) -- that filters by whole episode,
get_clean_indices() filters by individual frame) and feed that into a
DataLoader like any other dataset.

Detection + index computation only -- does not write, delete, or modify any
dataset file (spec section 8).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from team_robot_dataset import TeamRobotDataset

# so101_web/pipeline/2_qa/run_clean_subset_example.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "so101_teleop_real"
REPO_ID = "so101_teleop_real"


def main():
    dataset = TeamRobotDataset(REPO_ID, root=DATASET_ROOT)
    print(f"dataset: {DATASET_ROOT} (len={len(dataset)}, episodes={dataset.meta.total_episodes})")

    value_result = dataset.detect_outliers(z_thresh=2.0)
    delta_result = dataset.detect_outliers_delta(z_thresh=3.0)
    print(f"value-based frame_outliers: {len(value_result['frame_outliers'])}")
    print(f"delta-based frame_outliers: {len(delta_result['frame_outliers'])}")

    clean = dataset.get_clean_indices(value_result, delta_result)
    print(
        f"clean_indices={len(clean['clean_indices'])} "
        f"excluded_indices={clean['excluded_count']} "
        f"total={clean['total_count']}"
    )

    clean_subset = Subset(dataset, clean["clean_indices"])
    loader = DataLoader(clean_subset, batch_size=8, shuffle=False, num_workers=0)

    n_batches = 0
    n_frames = 0
    for batch in loader:
        n_batches += 1
        n_frames += batch["action"].shape[0]
    print(f"DataLoader over clean subset: {n_batches} batches, {n_frames} frames total")
    assert n_frames == len(clean["clean_indices"]), "frame count mismatch through DataLoader"
    print("OK -- Subset(dataset, clean_indices) + DataLoader works end to end")


if __name__ == "__main__":
    main()
