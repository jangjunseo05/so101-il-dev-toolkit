"""TeamRobotDataset -- composition wrapper around lerobot.LeRobotDataset (v3.0).

Confirmed via direct source inspection (lerobot 0.4.4,
site-packages/lerobot/datasets/lerobot_dataset.py) -- not guessed:
- LeRobotDataset.__init__(repo_id, root=None, episodes=None, image_transforms=None,
  delta_timestamps=None, tolerance_s=1e-4, revision=None, force_cache_sync=False,
  download_videos=True, video_backend=None, batch_encoding_size=1, vcodec="libsvtav1",
  streaming_encoding=False, encoder_queue_maxsize=30, encoder_threads=None)
- __len__ -> self.num_frames ; __getitem__(idx) -> dict
- stats/tasks/episodes live on `.meta` (a LeRobotDatasetMetadata instance), NOT as
  top-level attributes on LeRobotDataset (i.e. dataset.meta.stats, not dataset.stats).
- `.features` is a property that returns `self.meta.features`
  (dict[key] -> {"dtype", "shape", "names"}), matching meta/info.json's "features" block.

Per team spec: base class (site-packages) source is never modified. All customization
(custom stats, future visualization) lives in this wrapper layer via composition.
"""

from __future__ import annotations

import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe: only ever used to save PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class TeamRobotDataset:
    def __init__(self, repo_id_or_path: str, **kwargs: Any) -> None:
        self._base = LeRobotDataset(repo_id_or_path, **kwargs)
        # Auto integrity check on load (spec section 8/18): a NaN/Inf in the
        # data would silently corrupt compute_custom_stats()'s mean/std (NaN
        # propagates through np.mean/np.std) and therefore poison every
        # downstream z-score, so this is deliberately run unconditionally at
        # construction time rather than left as an opt-in call. Warns, never
        # raises -- must not block training on a warning-worthy but
        # non-fatal condition. Stored so get_clean_indices() can fold it into
        # the same exclusion union as outlier_results/delta_outlier_results.
        self._integrity_result = self.check_integrity()
        self._warn_on_integrity_problems(self._integrity_result)
        # Auto stats-freshness check (spec section 19): normalize()/
        # unnormalize() read dataset.meta.stats directly, which lerobot
        # computes at write time and does NOT recompute automatically if the
        # dataset is later appended to -- so a stale meta.stats would
        # silently normalize against the wrong mean/std. Warns, never
        # raises, same lenient pattern as the integrity check above.
        self._warn_on_stale_meta_stats()

    def _warn_on_integrity_problems(self, result: dict[str, Any]) -> None:
        problems = result["frame_problems"]
        if not problems:
            return
        type_counts = dict(Counter(p["problem_type"] for p in problems))
        key_counts = dict(Counter(p["key"] for p in problems))
        warnings.warn(
            f"TeamRobotDataset: check_integrity() found {len(problems)} NaN/Inf "
            f"value(s) across {sum(result['episode_problem_counts'].values())} "
            f"frame(s). episode_problem_counts={result['episode_problem_counts']} "
            f"by_problem_type={type_counts} by_field={key_counts}. Training is "
            "NOT blocked by this warning -- call check_integrity() for the full "
            "detail, or get_clean_indices(integrity_results=...) to exclude the "
            "affected frames.",
            stacklevel=2,
        )

    def _warn_on_stale_meta_stats(self) -> None:
        # image/video feature stats are deliberately computed from a
        # subsample of frames per episode (lerobot's compute_stats.py:
        # sample_images()/estimate_num_samples()), so their "count" is
        # expected to be smaller than total_frames by design -- comparing
        # them here would be a false positive, not a staleness signal.
        total_frames = self._base.meta.total_frames
        mismatches: list[tuple[str, int, int]] = []
        for key, feature_stats in self._base.meta.stats.items():
            ft = self._base.features.get(key)
            if ft is not None and ft["dtype"] in ("image", "video"):
                continue
            count = feature_stats.get("count")
            if count is None:
                continue
            count_val = int(np.asarray(count).reshape(-1)[0])
            if count_val != total_frames:
                mismatches.append((key, count_val, total_frames))

        if mismatches:
            warnings.warn(
                f"TeamRobotDataset: dataset.meta.stats 'count' does not match "
                f"total_frames={total_frames} for {len(mismatches)} field(s): "
                f"{mismatches} (key, stats_count, total_frames). meta.stats may "
                "be stale (computed before the dataset was last appended to) -- "
                "normalize()/unnormalize() will still use these stats as-is.",
                stacklevel=2,
            )

    def __getitem__(self, idx):
        return self._base[idx]

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails on this instance,
        # i.e. delegates anything TeamRobotDataset doesn't define itself
        # (meta, features, stats access, etc.) to the wrapped base dataset.
        return getattr(self._base, name)

    def __dir__(self):
        # __getattr__ makes attribute *access* fully delegate to self._base
        # (confirmed, CLAUDE.md section 38-3: 37/37 public LeRobotDataset
        # attrs resolve via getattr()), but dir() doesn't consult
        # __getattr__ at all -- it only lists what's actually defined on
        # this class. Without this override, dir(team) showed 10 names
        # instead of the full delegated surface, which could mislead anyone
        # exploring the interface via dir()/tab-completion even though
        # nothing was actually broken functionally. Union, not replace, so
        # TeamRobotDataset's own methods (detect_outliers, normalize, etc.)
        # still show up alongside the delegated base ones.
        return sorted(set(super().__dir__()) | set(dir(self._base)))

    @staticmethod
    def _as_int(value) -> int:
        if isinstance(value, torch.Tensor):
            value = value.item()
        return int(value)

    def _light_items(self, keys: list[str]):
        """Column-projected view of the underlying hf_dataset for float-only
        scan passes (compute_custom_stats/detect_outliers/
        detect_outliers_delta/check_integrity) -- selects only `keys` plus
        episode_index/frame_index before iterating.

        Does NOT touch __getitem__ or any image-related logic; this is a
        separate, additional read path used only internally by those 4
        methods. Confirmed via direct inspection (not guessed): the image
        decode cost isn't in LeRobotDataset.__getitem__ itself -- it happens
        inside hf_dataset's own `set_transform(hf_transform_to_torch)`
        callback (lerobot/datasets/utils.py), which decodes *every column
        actually present in the requested row*, image or not. So
        `self._base.hf_dataset[i]` alone still decodes the image (verified:
        it already contains a decoded image tensor before
        LeRobotDataset.__getitem__ adds anything). What actually skips the
        decode is restricting the column set beforehand via
        `hf_dataset.select_columns(...)`, so the image column is never part
        of what the transform sees. Values returned for the selected columns
        are confirmed identical (spot-checked via torch.equal) to the
        corresponding fields in a full `self[i]`.
        """
        cols = list(dict.fromkeys([*keys, "episode_index", "frame_index"]))
        return self._base.hf_dataset.select_columns(cols)

    def compute_custom_stats(self, keys: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Frame-by-frame custom stats (mean/std/var) -- v0.1 spike minimal implementation.

        Scope per spec section 6: exact stats spec (per-joint outlier detection etc.)
        is deferred; this exists only to exercise acceptance criterion 3 (perf check).

        Reads frames via `_light_items()` (column-projected, skips image
        decode) rather than `self[i]` -- see that method's docstring. Values
        are confirmed identical either way; only the read path changed.
        """
        if keys is None:
            keys = [
                key
                for key, ft in self._base.features.items()
                if ft["dtype"] in ("float32", "float64")
            ]

        values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        items = self._light_items(keys)
        for i in range(len(self)):
            item = items[i]
            for key in keys:
                v = item[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                values[key].append(np.asarray(v, dtype=np.float64))

        stats: dict[str, dict[str, np.ndarray]] = {}
        for key, vals in values.items():
            arr = np.stack(vals)
            stats[key] = {
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "var": arr.var(axis=0),
            }
        return stats

    def detect_outliers(
        self,
        keys: list[str] | None = None,
        z_thresh: float = 3.0,
    ) -> dict[str, Any]:
        """Frame-level z-score outlier detection -- v0.2 spec section 7.

        Reuses the mean/std already computed by compute_custom_stats() (no
        duplicate stats pass). A value at (frame, key, dim) is an outlier if
        |z| > z_thresh, where z = (value - mean[dim]) / std[dim] -- i.e. this
        is per-joint (per-dim) since mean/std/std are computed per feature
        dimension (e.g. one mean/std per joint for "action"/"observation.state").

        `frame_idx` in the returned entries is the frame's within-episode
        `frame_index` (resets to 0 at each episode boundary), NOT a row index
        into `self`. `self[i]`'s row index is only meaningful relative to
        whatever subset this TeamRobotDataset instance happens to have loaded
        (e.g. a full load vs. a `episodes=[...]`-filtered load number the same
        physical frame differently), whereas `(episode_index, frame_idx)` is
        stable regardless of how the dataset was filtered -- this is what
        lets visualize_episode() overlay these results without any offset
        math, even when it's called on a differently-filtered instance than
        the one that produced them.

        Per spec: detection + reporting only. Does NOT filter/remove anything.

        Reads frames via `_light_items()` (column-projected, skips image
        decode) rather than `self[i]` -- see that method's docstring. Values
        are confirmed identical either way; only the read path changed.
        """
        stats = self.compute_custom_stats(keys=keys)
        keys = list(stats.keys())

        frame_outliers: list[dict[str, Any]] = []
        episode_outlier_frames: dict[int, set[int]] = {}

        items = self._light_items(keys)
        for i in range(len(self)):
            item = items[i]
            ep_idx = self._as_int(item["episode_index"])
            frame_idx = self._as_int(item["frame_index"])

            frame_has_outlier = False
            for key in keys:
                v = item[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                v = np.asarray(v, dtype=np.float64)

                mean = stats[key]["mean"]
                std = stats[key]["std"]
                names = self._base.features[key].get("names")

                z = np.divide(v - mean, std, out=np.zeros_like(v), where=std > 0)

                for dim_idx in np.ndindex(z.shape):
                    z_val = float(z[dim_idx])
                    if abs(z_val) > z_thresh:
                        frame_has_outlier = True
                        joint_name = None
                        if names and len(dim_idx) == 1 and dim_idx[0] < len(names):
                            joint_name = names[dim_idx[0]]
                        frame_outliers.append(
                            {
                                "frame_idx": frame_idx,
                                "episode_index": ep_idx,
                                "key": key,
                                "dim": dim_idx[0] if len(dim_idx) == 1 else dim_idx,
                                "joint_name": joint_name,
                                "value": float(v[dim_idx]),
                                "mean": float(mean[dim_idx]),
                                "std": float(std[dim_idx]),
                                "z": z_val,
                            }
                        )

            if frame_has_outlier:
                episode_outlier_frames.setdefault(ep_idx, set()).add(frame_idx)

        # episode-level counts are purely derived from the frame-level results above
        episode_outlier_counts = {
            ep: len(frames) for ep, frames in sorted(episode_outlier_frames.items())
        }

        return {
            "z_thresh": z_thresh,
            "keys": keys,
            "num_frames": len(self),
            "frame_outliers": frame_outliers,
            "episode_outlier_counts": episode_outlier_counts,
        }

    def detect_outliers_delta(
        self,
        keys: list[str] | None = None,
        z_thresh: float = 3.0,
    ) -> dict[str, Any]:
        """Frame-to-frame delta z-score outlier detection.

        Per CLAUDE.md section 8 (실물 데이터 진단 결론): detect_outliers()
        (raw-value z-score) only reliably catches the extremes of large
        deliberate movements and mostly misses intermittent step-like
        transitions (e.g. a person failing to hold perfectly still). This
        method instead scores delta = value[i] - value[i-1] within an
        episode -- deltas are never computed across an episode boundary, so
        each episode's first frame has no delta and cannot be flagged.

        Does NOT reuse compute_custom_stats()'s mean/std (those describe the
        raw value distribution, not the frame-to-frame delta distribution) --
        computes its own delta mean/std in a dedicated pass instead.

        `frame_idx` in the returned entries is the (later frame's)
        within-episode `frame_index`, not a row index into `self` -- same
        rationale as detect_outliers(): stable regardless of how this
        TeamRobotDataset instance was filtered when loaded.

        Per spec: detection + reporting only. Does NOT filter/remove anything.

        Reads frames via `_light_items()` (column-projected, skips image
        decode) rather than `self[i]` -- see that method's docstring. Values
        are confirmed identical either way; only the read path changed.
        """
        if keys is None:
            keys = [
                key
                for key, ft in self._base.features.items()
                if ft["dtype"] in ("float32", "float64")
            ]

        # first pass: raw values + episode_index/frame_index per frame, so
        # episode boundaries are known before any delta is computed
        values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        episode_indices: list[int] = []
        frame_indices: list[int] = []
        items = self._light_items(keys)
        for i in range(len(self)):
            item = items[i]
            episode_indices.append(self._as_int(item["episode_index"]))
            frame_indices.append(self._as_int(item["frame_index"]))
            for key in keys:
                v = item[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                values[key].append(np.asarray(v, dtype=np.float64))

        # per-key deltas, skipping any (i-1, i) pair that crosses an episode
        # boundary. delta_frame_idx/delta_episode_idx carry the *later*
        # frame's within-episode frame_index and episode_index, kept as
        # parallel lists (not derived by indexing back into episode_indices,
        # since frame_idx is no longer the same value as the loop index i).
        deltas: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        delta_frame_idx: list[int] = []
        delta_episode_idx: list[int] = []
        for i in range(1, len(self)):
            if episode_indices[i] != episode_indices[i - 1]:
                continue
            delta_frame_idx.append(frame_indices[i])
            delta_episode_idx.append(episode_indices[i])
            for key in keys:
                deltas[key].append(values[key][i] - values[key][i - 1])

        delta_stats: dict[str, dict[str, np.ndarray]] = {}
        for key in keys:
            arr = np.stack(deltas[key]) if deltas[key] else np.empty((0, *values[key][0].shape))
            delta_stats[key] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}

        frame_outliers: list[dict[str, Any]] = []
        episode_outlier_frames: dict[int, set[int]] = {}

        for local_i, (frame_idx, ep_idx) in enumerate(zip(delta_frame_idx, delta_episode_idx)):
            frame_has_outlier = False
            for key in keys:
                d = deltas[key][local_i]
                mean = delta_stats[key]["mean"]
                std = delta_stats[key]["std"]
                names = self._base.features[key].get("names")

                z = np.divide(d - mean, std, out=np.zeros_like(d), where=std > 0)

                for dim_idx in np.ndindex(z.shape):
                    z_val = float(z[dim_idx])
                    if abs(z_val) > z_thresh:
                        frame_has_outlier = True
                        joint_name = None
                        if names and len(dim_idx) == 1 and dim_idx[0] < len(names):
                            joint_name = names[dim_idx[0]]
                        frame_outliers.append(
                            {
                                "frame_idx": frame_idx,
                                "episode_index": ep_idx,
                                "key": key,
                                "dim": dim_idx[0] if len(dim_idx) == 1 else dim_idx,
                                "joint_name": joint_name,
                                "value": float(d[dim_idx]),
                                "mean": float(mean[dim_idx]),
                                "std": float(std[dim_idx]),
                                "z": z_val,
                            }
                        )

            if frame_has_outlier:
                episode_outlier_frames.setdefault(ep_idx, set()).add(frame_idx)

        # episode-level counts are purely derived from the frame-level results above
        episode_outlier_counts = {
            ep: len(frames) for ep, frames in sorted(episode_outlier_frames.items())
        }

        return {
            "z_thresh": z_thresh,
            "keys": keys,
            "num_frames": len(self),
            "frame_outliers": frame_outliers,
            "episode_outlier_counts": episode_outlier_counts,
        }

    def check_integrity(self, keys: list[str] | None = None) -> dict[str, Any]:
        """Scan every frame for NaN/Inf values -- pure detection, no
        filtering/removal (same non-destructive principle as
        detect_outliers()/get_clean_indices(), spec section 8).

        Independent of compute_custom_stats()/detect_outliers(): needs no
        mean/std at all (NaN/Inf detection doesn't depend on the value
        distribution), and doesn't touch their logic -- a NaN/Inf in the data
        would in fact corrupt compute_custom_stats()'s mean/std (np.nan
        propagates through np.mean/np.std) and therefore silently poison
        every z-score computed by detect_outliers()/detect_outliers_delta()
        downstream of it, which is the reason this check exists as a
        separate, prerequisite pass rather than piggybacking on those.

        keys=None defaults to the same float32/float64 feature filter used by
        compute_custom_stats().

        Reads frames via `_light_items()` (column-projected, skips image
        decode) rather than `self[i]` -- see that method's docstring. Values
        are confirmed identical either way; only the read path changed. This
        matters more here than elsewhere: check_integrity() runs
        automatically on every TeamRobotDataset construction, so its read
        path directly sets the fixed per-load cost for any dataset with
        image features.
        """
        if keys is None:
            keys = [
                key
                for key, ft in self._base.features.items()
                if ft["dtype"] in ("float32", "float64")
            ]

        frame_problems: list[dict[str, Any]] = []
        episode_problem_frames: dict[int, set[int]] = {}

        items = self._light_items(keys)
        for i in range(len(self)):
            item = items[i]
            ep_idx = self._as_int(item["episode_index"])
            frame_idx = self._as_int(item["frame_index"])

            frame_has_problem = False
            for key in keys:
                v = item[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                v = np.asarray(v, dtype=np.float64)
                names = self._base.features[key].get("names")

                nan_mask = np.isnan(v)
                inf_mask = np.isinf(v)
                if not (nan_mask.any() or inf_mask.any()):
                    continue

                for dim_idx in np.ndindex(v.shape):
                    is_nan = bool(nan_mask[dim_idx])
                    is_inf = bool(inf_mask[dim_idx])
                    if not (is_nan or is_inf):
                        continue
                    frame_has_problem = True
                    joint_name = None
                    if names and len(dim_idx) == 1 and dim_idx[0] < len(names):
                        joint_name = names[dim_idx[0]]
                    frame_problems.append(
                        {
                            "frame_idx": frame_idx,
                            "episode_index": ep_idx,
                            "key": key,
                            "dim": dim_idx[0] if len(dim_idx) == 1 else dim_idx,
                            "joint_name": joint_name,
                            "problem_type": "nan" if is_nan else "inf",
                        }
                    )

            if frame_has_problem:
                episode_problem_frames.setdefault(ep_idx, set()).add(frame_idx)

        # episode-level counts are purely derived from the frame-level results above
        episode_problem_counts = {
            ep: len(frames) for ep, frames in sorted(episode_problem_frames.items())
        }

        return {
            "keys": keys,
            "num_frames": len(self),
            "frame_problems": frame_problems,
            "episode_problem_counts": episode_problem_counts,
        }

    def get_clean_indices(
        self,
        outlier_results: dict[str, Any] | None = None,
        delta_outlier_results: dict[str, Any] | None = None,
        integrity_results: dict[str, Any] | None = None,
        exclude_keys: tuple[str, ...] = ("timestamp",),
    ) -> dict[str, Any]:
        """Compute which of `self`'s rows are "clean" (not flagged by any of
        the given detection results), without touching the dataset in any
        way -- pure index bookkeeping. Per spec section 8, filtering was
        deliberately deferred until there was enough evidence to judge what's
        safe to drop; this doesn't itself filter anything either -- it only
        computes index lists for the caller to act on (e.g. via
        torch.utils.data.Subset, see the example below).

        Takes the union of `outlier_results["frame_outliers"]`,
        `delta_outlier_results["frame_outliers"]`, and
        `integrity_results["frame_problems"]` (a check_integrity() return
        value -- any, all, or none of the three may be given; pass
        `self._integrity_result`, set automatically at construction time, to
        include the NaN/Inf check run on load). A frame only counts as an
        outlier if it was flagged on a key *outside* `exclude_keys` (default:
        `("timestamp",)`, since section 12/14 found "timestamp" over-triggers
        purely because its variance is tiny relative to the frame interval,
        not because of an actual anomaly) -- a frame flagged *only* on an
        excluded key is kept. A frame flagged on both an excluded key and a
        non-excluded key is still excluded (the non-excluded flag alone is
        enough). Note this means a NaN/Inf on an `exclude_keys` field alone
        would also be kept -- if that's ever a real scenario, pass
        `exclude_keys=()` to force every integrity problem to be excluded
        regardless of field.

        Frame entries identify frames by (episode_index, frame_idx), where
        frame_idx is the within-episode frame_index (see detect_outliers()'s
        docstring) -- so this works even if the results were computed
        against a differently-filtered TeamRobotDataset instance than `self`,
        as long as `self` contains the same frames. Any flagged frame not
        present in `self` is silently skipped (same lenient behavior as
        visualize_episode()'s overlay matching).

        Returns row indices into `self` (0..len(self)-1) -- i.e. directly
        usable as `torch.utils.data.Subset(self, result["clean_indices"])` --
        NOT (episode_index, frame_idx) pairs. This is the one place those get
        translated back into a concrete row position, since that's what
        Subset (and DataLoader) need.

        Example:
            outliers = dataset.detect_outliers(z_thresh=2.0)
            deltas = dataset.detect_outliers_delta(z_thresh=3.0)
            result = dataset.get_clean_indices(outliers, deltas, dataset._integrity_result)
            clean_subset = torch.utils.data.Subset(dataset, result["clean_indices"])
            loader = torch.utils.data.DataLoader(clean_subset, batch_size=8)

        Does not read or write any file, and does not mutate `self` or the
        underlying LeRobotDataset -- purely computes and returns index lists.
        """
        all_indices = list(range(len(self)))
        frame_key_to_local: dict[tuple[int, int], int] = {}
        for i in all_indices:
            item = self[i]
            ep_idx = self._as_int(item["episode_index"])
            frame_idx = self._as_int(item["frame_index"])
            frame_key_to_local[(ep_idx, frame_idx)] = i

        outlier_frame_keys: set[tuple[int, int]] = set()
        for result, list_field in (
            (outlier_results, "frame_outliers"),
            (delta_outlier_results, "frame_outliers"),
            (integrity_results, "frame_problems"),
        ):
            if result is None:
                continue
            for o in result[list_field]:
                if o["key"] in exclude_keys:
                    continue
                outlier_frame_keys.add((o["episode_index"], o["frame_idx"]))

        excluded_indices = sorted(
            frame_key_to_local[fk] for fk in outlier_frame_keys if fk in frame_key_to_local
        )
        excluded_set = set(excluded_indices)
        clean_indices = [i for i in all_indices if i not in excluded_set]

        return {
            "clean_indices": clean_indices,
            "excluded_indices": excluded_indices,
            "excluded_count": len(excluded_indices),
            "total_count": len(all_indices),
        }

    def split_episodes(
        self, val_ratio: float = 0.2, seed: int | None = None
    ) -> dict[str, Any]:
        """Compute an episode-level train/val split -- pure index bookkeeping,
        same non-destructive principle as get_clean_indices() (spec section
        17): does NOT construct any dataset itself. `LeRobotDataset` already
        natively supports loading a subset via its own `episodes=[...]`
        constructor kwarg (already relied on elsewhere in this project, e.g.
        section 3/17's filtered-load handling), so there is no reason to
        build a second, competing split/loading mechanism here -- this just
        decides *which* episode indices go where, and leaves constructing
        `TeamRobotDataset(..., episodes=train_episodes)` /
        `TeamRobotDataset(..., episodes=val_episodes)` to the caller.

        Episode indices come from `self._base.meta.episodes["episode_index"]`
        (a HF `datasets.Dataset` column, confirmed via direct inspection --
        plain Python ints, not the row-filtered dataset `self` itself, so
        this always reflects every episode in the underlying dataset
        regardless of whether `self` was loaded with an `episodes=[...]`
        filter).

        Shuffled with `random.Random(seed)` (stdlib), not `numpy`'s RNG --
        this is a plain permutation of a small list of episode indices, not
        a numeric array operation, so numpy (already a dependency here, e.g.
        compute_custom_stats()) would add nothing; stdlib `random.Random`
        gives the same seed-reproducibility with one less RNG state to
        reason about. `seed=None` (default) uses OS entropy, i.e.
        non-deterministic by design -- matches the default `random.Random()`
        behavior, so two no-seed calls are expected to differ.

        Split size: `n_val = round(n_episodes * val_ratio)`, then for
        `n_episodes >= 2` clamped to `[1, n_episodes - 1]` so both sides
        always get at least one episode regardless of `val_ratio` (e.g. 2
        episodes always split 1/1, never 2/0 or 0/2, even at val_ratio=0.2).
        For `n_episodes` 0 or 1, an actual two-sided split is impossible;
        the single episode (if any) goes to train and val is left empty.

        Warns (does not raise -- same lenient pattern as check_integrity(),
        spec section 18-1) whenever the computed split leaves either side
        empty, since an empty train or val set makes the split meaningless
        for its intended purpose (e.g. an empty val set can't be used to
        evaluate anything).

        Returns `{"train_episodes": [...], "val_episodes": [...]}`, each a
        sorted list of episode indices (sorted for readability/determinism
        of the returned lists themselves; the shuffle before splitting is
        what actually determines which episodes land in which list).
        """
        episode_indices = list(self._base.meta.episodes["episode_index"])
        n_episodes = len(episode_indices)

        rng = random.Random(seed)
        shuffled = episode_indices[:]
        rng.shuffle(shuffled)

        if n_episodes >= 2:
            n_val = round(n_episodes * val_ratio)
            n_val = max(1, min(n_episodes - 1, n_val))
        else:
            # 0 or 1 episodes: no two-sided split is possible either way.
            n_val = 0

        val_episodes = sorted(shuffled[:n_val])
        train_episodes = sorted(shuffled[n_val:])

        if n_val == 0 or len(train_episodes) == 0:
            warnings.warn(
                f"TeamRobotDataset.split_episodes(): with {n_episodes} total "
                f"episode(s) and val_ratio={val_ratio}, the computed split is "
                f"train={len(train_episodes)}/val={n_val} -- one side is "
                "empty, so this split is not usable as an actual train/val "
                "split. Returning it anyway (this is a warning, not a "
                "blocking error); add more episodes or adjust val_ratio if "
                "a real split is needed.",
                stacklevel=2,
            )

        return {"train_episodes": train_episodes, "val_episodes": val_episodes}

    # matches lerobot's _NormalizationMixin default eps exactly
    # (lerobot/processor/normalize_processor.py, _NormalizationMixin.eps: float = 1e-8)
    # -- not reused by import, since that class is tied to the policy
    # processor pipeline's EnvTransition/PolicyAction types (see normalize()
    # docstring); only the constant and the MEAN_STD formula are reused.
    _NORM_EPS = 1e-8

    def _default_normalize_keys(self) -> list[str]:
        keys = [k for k in ("action", "observation.state") if k in self._base.features]
        keys += [
            k
            for k, ft in self._base.features.items()
            if ft["dtype"] in ("image", "video")
        ]
        return keys

    def normalize(self, item: dict[str, Any], keys: list[str] | None = None) -> dict[str, Any]:
        """Normalize the given fields of `item` using `self._base.meta.stats`.

        norm_map (project decision log, spec section 19/20): MEAN_STD for
        action/observation.state, but **IDENTITY for image/video features**
        (a no-op copy-through, values stay in their original [0,1] scale).
        This deliberately deviates from ACT's out-of-the-box default (which
        uses MEAN_STD for VISUAL too) -- ACT's own norm_map is meant to be
        customized per-field, so this is a configuration choice within ACT's
        supported range, not a departure from it (SmolVLA/Pi0 also default
        VISUAL to IDENTITY, confirmed via source inspection in the prior
        investigation). Reason: this project's `observation.images.wrist_cam`
        is a MuJoCo render (flat shading, large solid-color regions), whose
        per-channel std (~1e-3 to 1e-5 depending on how much the arm moved in
        the recorded episodes) is 10-200x smaller than a natural image's --
        confirmed empirically: MEAN_STD normalization on this data produced
        values in the hundreds to thousands (e.g. -7611..+10917 on a
        near-static episode, -320..+385 even on a large-motion episode),
        far outside the roughly [-3, +3] a well-behaved MEAN_STD
        normalization should produce.

        For action/observation.state, reimplements (does not import) the
        MEAN_STD branch of lerobot's `_NormalizationMixin._apply_transform()`
        (lerobot/processor/normalize_processor.py): `(x - mean) / (std + eps)`,
        with `eps = 1e-8` matching that class's default exactly (confirmed via
        source inspection, not guessed). Not reusing that class directly
        because it operates on `EnvTransition`/`PolicyAction` (the policy
        processor pipeline's own types), which would pull the whole
        processor/pipeline framework into this project for no benefit here --
        this method instead works directly on the same plain dict shape
        `self[i]` / a DataLoader batch already returns.

        `item` may be a single frame (`self[i]`, unbatched tensors, e.g.
        action shape (6,)) or a batch (e.g. from DataLoader, shape (B, 6)) --
        both work unchanged because the stats' shapes ((6,) for
        action/observation.state) broadcast correctly against either a (6,)
        single item or a (B,6) batch (confirmed via broadcasting rules, not
        assumed). Image/video fields don't need this at all since they're
        passed through as-is.

        `keys=None` defaults to action + observation.state + every
        image/video feature present (e.g. `observation.images.wrist_cam` for
        so101_teleop_real) -- image/video keys are included in this default
        so callers don't have to special-case them, but they're copied
        through unchanged rather than transformed. Keys not present in `item`
        or without stats are silently skipped (same lenient behavior as
        visualize_episode()'s overlay matching).

        Does not mutate `item` -- returns a new dict; fields not in `keys`
        are carried over unchanged (by reference, not copied).
        """
        if keys is None:
            keys = self._default_normalize_keys()

        stats = self._base.meta.stats
        new_item = dict(item)
        for key in keys:
            if key not in item or key not in stats:
                continue
            v = item[key]
            ft = self._base.features.get(key)
            if ft is not None and ft["dtype"] in ("image", "video"):
                new_item[key] = v  # IDENTITY: no-op, see docstring
                continue
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            if isinstance(v, torch.Tensor):
                mean_t = torch.as_tensor(mean, dtype=v.dtype, device=v.device)
                std_t = torch.as_tensor(std, dtype=v.dtype, device=v.device)
                new_item[key] = (v - mean_t) / (std_t + self._NORM_EPS)
            else:
                v_arr = np.asarray(v, dtype=np.float64)
                new_item[key] = (v_arr - mean) / (std + self._NORM_EPS)
        return new_item

    def unnormalize(self, item: dict[str, Any], keys: list[str] | None = None) -> dict[str, Any]:
        """Inverse of normalize(): `x * std + mean` for action/observation.state,
        IDENTITY (no-op copy-through) for image/video fields -- symmetric with
        normalize()'s norm_map choice, see that method's docstring for why
        images are excluded from MEAN_STD.

        For the MEAN_STD fields, deliberately asymmetric with normalize()'s
        `(x - mean) / (std + eps)` -- the inverse does NOT add eps back
        (`x * std + mean`, not `x * (std + eps) + mean`), matching lerobot's
        own `_NormalizationMixin._apply_transform()` inverse branch exactly
        (confirmed via source inspection). eps only exists to keep the
        forward division stable; the inverse is a plain multiplication and
        doesn't need it.

        Same `keys=None` default, same lenient skip-if-missing behavior, same
        batch-or-single-item support, and does not mutate `item` -- see
        normalize()'s docstring for all of the above.
        """
        if keys is None:
            keys = self._default_normalize_keys()

        stats = self._base.meta.stats
        new_item = dict(item)
        for key in keys:
            if key not in item or key not in stats:
                continue
            v = item[key]
            ft = self._base.features.get(key)
            if ft is not None and ft["dtype"] in ("image", "video"):
                new_item[key] = v  # IDENTITY: no-op, see normalize()'s docstring
                continue
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            if isinstance(v, torch.Tensor):
                mean_t = torch.as_tensor(mean, dtype=v.dtype, device=v.device)
                std_t = torch.as_tensor(std, dtype=v.dtype, device=v.device)
                new_item[key] = v * std_t + mean_t
            else:
                v_arr = np.asarray(v, dtype=np.float64)
                new_item[key] = v_arr * std + mean
        return new_item

    def _warn_on_delta_timestamps_keys(self, keys: list[str], method_name: str) -> None:
        """Warn (not raise) if any of `keys` is chunk-expanded via
        self._base.delta_timestamps -- same lenient pattern as
        _warn_on_integrity_problems()/_warn_on_stale_meta_stats() (spec
        section 18-1): warn and keep going, never block.

        When the base LeRobotDataset is constructed with delta_timestamps=
        {"action": [...]} (or any other key), self[i] returns that key as a
        chunk-expanded sequence -- shape (chunk_len, dim) instead of a single
        frame's (dim,) value (confirmed by direct inspection of
        LeRobotDataset.__getitem__'s delta_timestamps handling). Both
        visualize_episode() and plot_action_distribution() read self[i] and
        assume one value per frame per key; for a chunk-expanded key, the
        chunk dimension silently gets treated as if it were an extra
        joint/value dimension. This produces no exception -- confirmed by
        direct measurement that the resulting plot renders successfully but
        with joint and chunk dimensions mixed together, i.e. a quiet
        misinterpretation of the data rather than a crash.
        """
        dt = self._base.delta_timestamps
        if not dt:
            return
        affected = [k for k in keys if k in dt]
        if not affected:
            return
        warnings.warn(
            f"TeamRobotDataset.{method_name}(): self._base.delta_timestamps is "
            f"set and includes key(s) {affected}. {method_name}() assumes every "
            "key is a single per-frame value, but self[i] returns delta_timestamps "
            "keys as a chunk-expanded sequence (shape (chunk_len, dim)) -- the "
            "chunk dimension will be silently treated as an extra joint/value "
            "dimension, so results for these key(s) may be inaccurate. Continuing "
            "anyway (this is a warning, not a blocking error).",
            stacklevel=2,
        )

    def visualize_episode(
        self,
        episode_idx: int,
        outlier_results: dict[str, Any] | None = None,
        delta_outlier_results: dict[str, Any] | None = None,
        save_path: str | Path | None = None,
    ) -> Path:
        """Line-plot action/observation.state per dim (subplot per dim) for one
        episode, x-axis = the dataset's own per-episode `frame_index` field
        (confirmed to reset to 0 at each episode boundary).

        Does NOT use `self.meta.episodes`' `dataset_from_index`/`dataset_to_index`
        to select rows -- those describe row positions in the *full, unfiltered*
        dataset (lerobot loads `.meta` independently of the `episodes=[...]`
        constructor filter, so those indices stay absolute even when `self`
        only has a filtered subset loaded; confirmed via direct source
        inspection of lerobot.datasets.lerobot_dataset.LeRobotDataset.__init__).
        `self[i]`'s own row index is likewise only meaningful relative to
        whatever subset `self` has loaded. Instead, this scans `self` and
        matches on each frame's own `episode_index`/`frame_index` fields,
        which are stable regardless of filtering.

        If `outlier_results` (a detect_outliers() return value) is given, overlays
        that episode's already-detected outlier frames as red points -- does NOT
        recompute or re-run outlier detection.

        If `delta_outlier_results` (a detect_outliers_delta() return value) is
        given, overlays that episode's delta-based outlier frames as blue points.
        Unlike `outlier_results`, whose "value" field is already the raw
        trajectory value, `detect_outliers_delta()`'s "value" field is the
        frame-to-frame delta -- not a point on the trajectory line -- so the
        blue point is plotted at that frame's actual series value instead, to
        mark *where on the curve* the flagged transition happened. Both
        overlays are independent and optional (either, both, or neither).

        Both `outlier_results` and `delta_outlier_results` match purely by
        (episode_index, frame_idx) -- since detect_outliers()/
        detect_outliers_delta() now record frame_idx as the within-episode
        frame_index (see their docstrings), this works with zero offset math
        even if they were computed against a *differently*-filtered
        TeamRobotDataset instance than the one `visualize_episode()` is called
        on, as long as both instances actually contain this episode's frames.
        """
        plot_keys = [k for k in ("action", "observation.state") if k in self._base.features]
        self._warn_on_delta_timestamps_keys(plot_keys, "visualize_episode")

        x_local: list[int] = []
        series: dict[str, list[np.ndarray]] = {k: [] for k in plot_keys}
        frame_index_to_pos: dict[int, int] = {}
        for i in range(len(self)):
            item = self[i]
            if self._as_int(item["episode_index"]) != episode_idx:
                continue
            fidx = self._as_int(item["frame_index"])
            frame_index_to_pos[fidx] = len(x_local)
            x_local.append(fidx)
            for k in plot_keys:
                v = item[k]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                series[k].append(np.asarray(v, dtype=np.float64))

        if not x_local:
            raise ValueError(
                f"episode_idx {episode_idx} not found in this dataset -- it may "
                "have been filtered out when this TeamRobotDataset was loaded "
                "(e.g. via episodes=[...])."
            )

        for k in plot_keys:
            series[k] = np.stack(series[k])  # (n_frames_in_episode, dim)

        # bucket this episode's already-computed outlier entries by (key, dim);
        # reused as-is from outlier_results, no recomputation. Matched by
        # frame_idx (== within-episode frame_index) directly, since that's
        # already the x-axis unit -- no offset needed.
        outlier_points: dict[tuple[str, int], list[tuple[int, float]]] = {}
        if outlier_results is not None:
            for o in outlier_results["frame_outliers"]:
                if o["episode_index"] != episode_idx or o["key"] not in plot_keys:
                    continue
                if o["frame_idx"] not in frame_index_to_pos:
                    continue  # that frame isn't loaded in this (possibly filtered) instance
                outlier_points.setdefault((o["key"], o["dim"]), []).append((o["frame_idx"], o["value"]))

        # same bucketing for delta-based outliers, but the y-coordinate is the
        # frame's actual series value (not o["value"], which is a delta) so the
        # point lands on the trajectory line itself.
        delta_outlier_points: dict[tuple[str, int], list[tuple[int, float]]] = {}
        if delta_outlier_results is not None:
            for o in delta_outlier_results["frame_outliers"]:
                if o["episode_index"] != episode_idx or o["key"] not in plot_keys:
                    continue
                pos = frame_index_to_pos.get(o["frame_idx"])
                if pos is None:
                    continue  # that frame isn't loaded in this (possibly filtered) instance
                y_val = float(series[o["key"]][pos, o["dim"]])
                delta_outlier_points.setdefault((o["key"], o["dim"]), []).append((o["frame_idx"], y_val))

        dims_per_key = {k: series[k].shape[1] for k in plot_keys}
        names_per_key = {k: self._base.features[k].get("names") for k in plot_keys}
        total_dims = sum(dims_per_key.values())

        fig, axes = plt.subplots(total_dims, 1, figsize=(10, 2.0 * total_dims), sharex=True)
        axes = np.atleast_1d(axes)

        ax_i = 0
        for k in plot_keys:
            names = names_per_key[k]
            for d in range(dims_per_key[k]):
                ax = axes[ax_i]
                ax.plot(x_local, series[k][:, d], linewidth=1)
                label = names[d] if names and d < len(names) else f"dim{d}"
                ax.set_ylabel(f"{k}\n{label}", fontsize=8)
                pts = outlier_points.get((k, d))
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, color="red", zorder=5, s=25, label="outlier (|z|>thresh)")

                delta_pts = delta_outlier_points.get((k, d))
                if delta_pts:
                    dxs, dys = zip(*delta_pts)
                    ax.scatter(dxs, dys, color="blue", zorder=6, s=25, label="delta outlier (|z|>thresh)")

                if pts or delta_pts:
                    ax.legend(fontsize=6, loc="upper right")
                ax_i += 1

        axes[-1].set_xlabel("frame_index (within episode)")
        title = f"episode {episode_idx} trajectory (frames {min(x_local)}-{max(x_local)})"
        if outlier_results is not None:
            title += f" -- outliers overlaid (z_thresh={outlier_results['z_thresh']})"
        if delta_outlier_results is not None:
            title += f" / delta outliers overlaid (z_thresh={delta_outlier_results['z_thresh']})"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()

        save_path = Path(save_path) if save_path else Path(f"episode_{episode_idx}_trajectory.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path

    def plot_action_distribution(
        self,
        keys: list[str] | None = None,
        save_path: str | Path | None = None,
    ) -> Path:
        """Histogram of the value distribution per (key, dim) across the whole
        dataset. Default keys match compute_custom_stats()'s own default filter
        (float32/float64 features)."""
        if keys is None:
            keys = [
                key
                for key, ft in self._base.features.items()
                if ft["dtype"] in ("float32", "float64")
            ]
        self._warn_on_delta_timestamps_keys(keys, "plot_action_distribution")

        values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for i in range(len(self)):
            item = self[i]
            for key in keys:
                v = item[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                values[key].append(np.asarray(v, dtype=np.float64))
        for key in keys:
            arr = np.stack(values[key])
            if arr.ndim == 1:
                # scalar-per-frame features (e.g. "timestamp" is a 0-d tensor at
                # runtime despite shape=[1] in meta/info.json) -> treat as 1 dim
                arr = arr.reshape(-1, 1)
            values[key] = arr  # (n_frames, dim)

        dims_per_key = {k: values[k].shape[1] for k in keys}
        names_per_key = {k: self._base.features[k].get("names") for k in keys}
        total_dims = sum(dims_per_key.values())

        ncols = 3
        nrows = -(-total_dims // ncols)  # ceil division
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)

        ax_i = 0
        for k in keys:
            names = names_per_key[k]
            for d in range(dims_per_key[k]):
                ax = axes[ax_i]
                ax.hist(values[k][:, d], bins=20, color="steelblue", edgecolor="black")
                label = names[d] if names and d < len(names) else f"dim{d}"
                ax.set_title(f"{k}[{label}]", fontsize=9)
                ax_i += 1
        for j in range(ax_i, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Value distribution -- so101_teleop (all episodes)", fontsize=11)
        fig.tight_layout()

        save_path = Path(save_path) if save_path else Path("action_distribution.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path

    def describe_preprocessing(self) -> dict[str, Any]:
        """Read-only summary of where each standard imitation-learning
        preprocessing step actually happens in this pipeline (CLAUDE.md
        sections 39/40) -- built for the QA dashboard's "전처리 구조 보기"
        detail view, so that page reads structured facts from here rather
        than a hand-copied paraphrase of the CLAUDE.md prose baked into the
        page itself (source-of-truth requirement from that request).

        Everything below is a description of *this specific dataset instance
        and the surrounding pipeline code*, not a live re-derivation --
        the normalization sample pulls this dataset's actual
        `self._base.meta.stats`; the 6-item preprocessing table is a fixed
        description of the pipeline's code structure (team_robot_dataset.py,
        the installed lerobot policies/act/*.py, lerobot-train's CLI
        surface), confirmed via direct source inspection in sections 39-41,
        not something that varies per-dataset. Does not mutate self or read
        any file beyond what `self._base.meta.stats`/`self._base.features`
        already loaded.
        """
        stats = self._base.meta.stats
        normalization_samples: list[dict[str, Any]] = []
        for key in ("action", "observation.state"):
            if key not in stats:
                continue
            feature_stats = stats[key]
            mean = np.asarray(feature_stats["mean"]).reshape(-1)
            std = np.asarray(feature_stats["std"]).reshape(-1)
            names = self._base.features.get(key, {}).get("names")
            dims = []
            for i in range(len(mean)):
                dims.append(
                    {
                        "name": names[i] if names and i < len(names) else f"dim{i}",
                        "mean": float(mean[i]),
                        "std": float(std[i]),
                    }
                )
            normalization_samples.append({"key": key, "dims": dims})

        normalization = {
            "summary": (
                "정규화 통계(mean/std)는 이 TeamRobotDataset이 계산하지 않습니다 -- "
                "lerobot 자신이 녹화 세션의 에피소드 저장(S/X 키 종료) 시점마다 "
                "직접 계산해 meta/stats.json에 기록합니다. normalize()/unnormalize()는 "
                "이 파일을 읽기만 할 뿐 새로 계산하거나 캐싱하지 않습니다. lerobot-train도 "
                "같은 파일을 자신의 LeRobotDataset 인스턴스로 독립적으로 다시 읽어 "
                "자체 NormalizerProcessorStep을 구성하므로, 이 TeamRobotDataset의 "
                "normalize()/unnormalize()는 실제 학습 경로와는 분리된 QA 진단 "
                "유틸리티입니다(호출자 0건, CLAUDE.md 39절)."
            ),
            "source_file": "meta/stats.json",
            "samples": normalization_samples,
        }

        preprocessing_items = [
            {
                "name": "이미지 리사이즈",
                "qa_layer": "안 함",
                "act_policy": "특정 해상도를 가정하지 않음 (2D 위치 임베딩이 실제 feature map 크기로 매 forward마다 동적 계산됨)",
                "lerobot_train_cli": "옵션 없음",
                "status": "gap",
                "status_label": "빈 자리",
                "note": (
                    "이 파이프라인 어디에도 리사이즈 로직이 없습니다. 지금 문제가 안 되는 "
                    "이유는 run_teleop_real.py/run_inference_mujoco.py 둘 다 캡처 해상도를 "
                    "480x640으로 고정해뒀기 때문(pipeline/camera_config.py, 41절)입니다. "
                    "카메라를 바꾸거나 해상도가 달라지면 실제로 필요해질 수 있는 자리입니다 -- "
                    "다만 41절에서 두 스크립트에 방어 코드(렌더 직후 shape assert)를 추가해, "
                    "두 스크립트의 캡처 해상도가 서로 어긋나면 조용히 넘어가지 않고 즉시 "
                    "에러로 실패하도록 만들어뒀습니다. 이 방어 코드는 불일치를 감지만 할 뿐, "
                    "리사이즈 자체를 자동으로 처리해주지는 않습니다."
                ),
            },
            {
                "name": "이미지 증강",
                "qa_layer": "안 함",
                "act_policy": "해당 없음 (데이터 로딩 레벨의 처리)",
                "lerobot_train_cli": "있음 -- dataset.image_transforms.* (brightness/contrast/saturation/hue/sharpness/affine), 기본값 enable=False",
                "status": "unused_option",
                "status_label": "미사용(옵션 존재)",
                "note": "config/train_main_run_config.yaml에 image_transforms 섹션 자체가 없어 기본값(꺼짐) 그대로 적용되는 중입니다. 코드가 없는 게 아니라 옵션을 안 켠 상태입니다.",
            },
            {
                "name": "액션 델타(상대 표현)",
                "qa_layer": "안 함",
                "act_policy": "절대 액션만 사용 (MEAN_STD 정규화만 적용, 델타 표현 없음)",
                "lerobot_train_cli": "lerobot에 빌딩블록(delta_action_processor.py)은 있으나 ACT 기본 경로엔 배선되지 않음",
                "status": "not_applicable",
                "status_label": "이 프로젝트엔 불필요",
                "note": "이 데이터셋의 action 필드는 처음부터 절대 관절 라디안 값입니다(leader->follower 비례 스케일링 변환 결과, 23절) -- 델타 표현 자체를 선택하는 지점이 설계에 없습니다.",
            },
            {
                "name": "액션 청킹",
                "qa_layer": "경고만 -- self._base.delta_timestamps가 설정된 채로 visualize_episode()/plot_action_distribution()을 호출하면 경고(_warn_on_delta_timestamps_keys)",
                "act_policy": "chunk_size=100 결정 (ACTConfig)",
                "lerobot_train_cli": "자동 구성 -- chunk_size로부터 delta_timestamps를 만들어 순정 LeRobotDataset에 전달",
                "status": "handled",
                "status_label": "정상 처리(lerobot 표준 경로)",
                "note": "실제 시퀀스 확장은 이 TeamRobotDataset이 아니라 lerobot-train이 직접 생성하는 순정 LeRobotDataset.__getitem__ 내부에서 일어납니다(20절).",
            },
            {
                "name": "프레임 스태킹(관측 이력)",
                "qa_layer": "관여 없음",
                "act_policy": "n_obs_steps=1 강제 -- 그 외 값이면 ACTConfig 생성 시점에 ValueError",
                "lerobot_train_cli": "관여 없음",
                "status": "policy_constraint",
                "status_label": "정책 자체 미지원(구조적 제약)",
                "note": "ACT 정책 구현 자체가 여러 관측 프레임을 스택하는 걸 지원하지 않습니다 -- 이 파이프라인이 놓친 게 아니라 정책 선택(ACT)에 따른 제약입니다.",
            },
            {
                "name": "센서 동기화",
                "qa_layer": "관여 없음",
                "act_policy": "관여 없음",
                "lerobot_train_cli": "관여 없음",
                "status": "not_applicable",
                "status_label": "필요 자체가 없음",
                "note": "run_teleop_real.py의 단일 제어 루프가 매 반복마다 action/observation.state/wrist_cam을 같은 순간에 한 번에 만들어내는 구조라, 비동기 센서 스트림을 사후에 정렬할 필요 자체가 없습니다.",
            },
        ]

        return {"normalization": normalization, "items": preprocessing_items}
