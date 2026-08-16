"""Spike acceptance test for TeamRobotDataset (spec section 5).

Runs the 3 acceptance criteria against the local dataset at
data/so101_teleop_real (v3.0 format: data/chunk-*/file-*.parquet,
meta/episodes/chunk-*/file-*.parquet, meta/stats.json).

1. DataLoader compatibility: DataLoader(TeamRobotDataset(...), batch_size=8, ...)
   batches without collate errors, same as base.
2. Meta delegation: .meta.stats / .meta.tasks / .meta.episodes reachable through
   __getattr__ and match the base dataset's values.
3. Performance: compute_custom_stats() frame-by-frame vs an equivalent loop over
   the base dataset directly -- flag if wrapper is >20% slower, and separate
   delegation overhead from I/O by also timing bare __getitem__ iteration
   (no stats computation) on both.

Read-only w.r.t. the dataset; only writes are to phase1_spike_result.md (by the
caller, from this script's stdout) -- this script itself does not write files.
"""

from __future__ import annotations

import itertools
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# so101_web/pipeline/2_qa/tests/test_team_robot_dataset.py -> so101_web/pipeline/2_qa
# team_robot_dataset.py lives one directory up from this tests/ folder, so it
# isn't importable via a plain `from team_robot_dataset import ...` unless that
# directory is on sys.path (only this file's own dir is added by default).
QA_DIR = Path(__file__).resolve().parents[1]
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))
PROJECT_ROOT = QA_DIR.parents[1]

from team_robot_dataset import TeamRobotDataset  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "so101_teleop_real"
REPO_ID = "so101_teleop_real"
# so101_teleop_real은 9022 frames -- 원래 so101_teleop(109 frames) 기준으로 잡힌
# N_PERF_REPEATS=20은 그대로 두면 비현실적으로 오래 걸려 3으로 줄임(타이밍 노이즈
# 감소라는 목적 자체는 유지, 반복 횟수만 데이터셋 규모에 맞게 조정).
N_PERF_REPEATS = 3


def make_datasets():
    wrapped = TeamRobotDataset(REPO_ID, root=DATASET_ROOT)
    base = LeRobotDataset(REPO_ID, root=DATASET_ROOT)
    return wrapped, base


def test_dataloader_compatibility(wrapped, base, max_batches: int = 20) -> dict:
    result = {"name": "1. DataLoader compatibility", "pass": False, "detail": ""}
    try:
        loader_wrapped = DataLoader(wrapped, batch_size=8, shuffle=False, num_workers=0)
        loader_base = DataLoader(base, batch_size=8, shuffle=False, num_workers=0)

        # so101_teleop_real(9022 frames, 이미지 포함)에서 list(loader)로 전체를
        # 한 번에 메모리에 올리면 수십 GB가 필요해 세그폴트로 죽는 것을 실측
        # 확인(원래 109-frame so101_teleop 기준으로 짜인 테스트라 거기선 문제
        #없었음). 배치 호환성(shape 일치) 확인이 목적이라 전체를 다 볼 필요는
        # 없으므로 앞 max_batches개만 비교.
        batches_wrapped = list(itertools.islice(loader_wrapped, max_batches))
        batches_base = list(itertools.islice(loader_base, max_batches))

        n_batches_w = len(batches_wrapped)
        n_batches_b = len(batches_base)
        shapes_match = True
        detail_lines = [
            f"wrapped: {n_batches_w} batches sampled, base: {n_batches_b} batches sampled "
            f"(dataset has {len(wrapped)} frames total, max_batches={max_batches})",
        ]
        for i, (bw, bb) in enumerate(zip(batches_wrapped, batches_base)):
            for key in ("action", "observation.state"):
                sw, sb = tuple(bw[key].shape), tuple(bb[key].shape)
                if sw != sb:
                    shapes_match = False
                detail_lines.append(f"  batch {i} [{key}] wrapped={sw} base={sb}")

        result["pass"] = shapes_match and n_batches_w == n_batches_b and n_batches_w > 0
        result["detail"] = "\n".join(detail_lines)
    except Exception:
        result["detail"] = "EXCEPTION:\n" + traceback.format_exc()
    return result


def test_meta_delegation(wrapped, base) -> dict:
    result = {"name": "2. Meta delegation (.meta.stats / .meta.tasks / .meta.episodes)", "pass": False, "detail": ""}
    try:
        checks = []

        stats_w, stats_b = wrapped.meta.stats, base.meta.stats
        checks.append(("meta.stats keys match", set(stats_w.keys()) == set(stats_b.keys())))

        tasks_w, tasks_b = wrapped.meta.tasks, base.meta.tasks
        checks.append(("meta.tasks equal", tasks_w.equals(tasks_b)))

        eps_w, eps_b = wrapped.meta.episodes, base.meta.episodes
        checks.append(("meta.episodes length match", len(eps_w) == len(eps_b)))

        # also confirm __getattr__ path specifically (not a class attribute) is exercised
        checks.append(("TeamRobotDataset has no own 'meta' attr (delegated)", "meta" not in vars(wrapped)))
        checks.append(("wrapped.features keys match base", set(wrapped.features.keys()) == set(base.features.keys())))

        result["pass"] = all(ok for _, ok in checks)
        result["detail"] = "\n".join(f"  {name}: {ok}" for name, ok in checks)
    except Exception:
        result["detail"] = "EXCEPTION:\n" + traceback.format_exc()
    return result


def _time_it(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - start


def test_performance(wrapped, base) -> dict:
    result = {"name": "3. Performance (compute_custom_stats vs base)", "pass": False, "detail": ""}
    try:
        keys = [k for k, ft in base.features.items() if ft["dtype"] in ("float32", "float64")]

        def base_manual_stats():
            # mirrors TeamRobotDataset.compute_custom_stats exactly, but calls the base
            # dataset directly -- isolates wrapper overhead from the stats math itself.
            import numpy as np

            values = {k: [] for k in keys}
            for i in range(len(base)):
                item = base[i]
                for k in keys:
                    v = item[k]
                    if isinstance(v, torch.Tensor):
                        v = v.numpy()
                    values[k].append(np.asarray(v, dtype=np.float64))
            stats = {}
            for k, vals in values.items():
                arr = np.stack(vals)
                stats[k] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0), "var": arr.var(axis=0)}
            return stats

        def wrapped_custom_stats():
            return wrapped.compute_custom_stats(keys=keys)

        # full stats computation timing
        t_base = _time_it(base_manual_stats, N_PERF_REPEATS)
        t_wrapped = _time_it(wrapped_custom_stats, N_PERF_REPEATS)
        overhead_pct = (t_wrapped - t_base) / t_base * 100

        # isolate delegation overhead: bare __getitem__ iteration only, no stats math
        def base_getitem_only():
            for i in range(len(base)):
                _ = base[i]

        def wrapped_getitem_only():
            for i in range(len(wrapped)):
                _ = wrapped[i]

        t_base_getitem = _time_it(base_getitem_only, N_PERF_REPEATS)
        t_wrapped_getitem = _time_it(wrapped_getitem_only, N_PERF_REPEATS)
        getitem_overhead_pct = (t_wrapped_getitem - t_base_getitem) / t_base_getitem * 100

        detail_lines = [
            f"repeats={N_PERF_REPEATS}, frames={len(base)}",
            f"full compute_custom_stats: base={t_base:.4f}s wrapped={t_wrapped:.4f}s overhead={overhead_pct:+.2f}%",
            f"bare __getitem__ loop:     base={t_base_getitem:.4f}s wrapped={t_wrapped_getitem:.4f}s overhead={getitem_overhead_pct:+.2f}%",
        ]
        if overhead_pct <= 20:
            detail_lines.append("=> within 20% threshold, PASS")
        else:
            if getitem_overhead_pct < overhead_pct / 2:
                detail_lines.append(
                    "=> exceeds 20% threshold; __getitem__ overhead is small, "
                    "so the extra cost is dominated by stats math / I/O, not composition delegation"
                )
            else:
                detail_lines.append(
                    "=> exceeds 20% threshold; bare __getitem__ overhead alone explains a large share, "
                    "suggesting composition/delegation overhead is the driver"
                )

        result["pass"] = overhead_pct <= 20
        result["detail"] = "\n".join(detail_lines)
    except Exception:
        result["detail"] = "EXCEPTION:\n" + traceback.format_exc()
    return result


def main():
    print(f"dataset root: {DATASET_ROOT}")
    print(f"dataset exists: {DATASET_ROOT.exists()}")
    wrapped, base = make_datasets()
    print(f"len(wrapped)={len(wrapped)} len(base)={len(base)}")
    print()

    results = [
        test_dataloader_compatibility(wrapped, base),
        test_meta_delegation(wrapped, base),
        test_performance(wrapped, base),
    ]

    print("=" * 70)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}")
        print(r["detail"])
        print("-" * 70)

    all_pass = all(r["pass"] for r in results)
    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
