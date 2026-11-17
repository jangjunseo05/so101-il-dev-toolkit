"""녹화 세션이 finalize() 전에 강제 종료돼 마지막 data parquet에 footer가 없을 때(읽을 수 없음),
episode 목록(meta/episodes)에 기록된 episode까지로 데이터셋을 되돌린다.

배경 (2026-11): 대시보드 '세션 강제 종료'가 저장 중이던 세션을 5초 만에 강제 종료해,
녹화 중 열려 있던 data/chunk-000/file-005.parquet가 footer 없이 남았다. lerobot은 세션 중
episode 메타를 10개마다(또는 finalize 때) 쓰므로, 그 파일 속 episode들은 메타에도 없다.

하는 일 (지우지 않고 모두 백업으로 옮김):
  1. 데이터셋 폴더 전체와 <root>_scene/ 를 data/_backups/<이름>_before_repair_<시각>/ 로 복사
  2. 읽을 수 없는 data parquet, meta에 없는 episode의 임시 이미지·3인칭 영상을 백업 폴더로 이동
  3. meta/episodes에 있는 episode 수·프레임 수로 info.json을 맞추고,
     stats.json을 meta/episodes의 episode별 통계로 다시 집계(lerobot aggregate_stats)
  4. LeRobotDataset으로 다시 열어 행 수가 맞는지 확인
기본은 --dry-run(무엇을 할지만 출력). 실제로 하려면 --apply.
"""

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _deep_list(x):
    if isinstance(x, np.ndarray):
        return [_deep_list(i) for i in x] if x.dtype == object else x.tolist()
    if isinstance(x, (list, tuple)):
        return [_deep_list(i) for i in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo-id", default="so101_teleop_real")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    scene = root.parent / f"{root.name}_scene"

    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/chunk-*/file-*.parquet")))])
    meta = meta.sort_values("episode_index")
    kept_eps = meta["episode_index"].tolist()
    n_eps, n_frames = len(kept_eps), int(meta["length"].sum())
    assert kept_eps == list(range(n_eps)), "episode 번호가 0부터 연속이 아님 -- 수동 확인 필요"
    referenced = {(int(c), int(f)) for c, f in zip(meta["data/chunk_index"], meta["data/file_index"])}

    bad_files, rows = [], 0
    for f in sorted(root.glob("data/chunk-*/file-*.parquet")):
        key = (int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1]))
        try:
            t = pq.read_table(f, columns=["episode_index"])
            if key not in referenced:
                bad_files.append((f, "meta에 없는 파일"))
            else:
                rows += t.num_rows
        except Exception as e:
            bad_files.append((f, f"읽을 수 없음({type(e).__name__})"))
    if rows != n_frames:
        print(f"중단: 읽을 수 있는 meta 파일의 행 수({rows}) != meta 프레임 합({n_frames})")
        return 2
    stray_imgs = [d for d in (root / "images").glob("*/episode-*") if int(d.name.split("-")[1]) >= n_eps] if (root / "images").exists() else []
    stray_clips = [
        p for p in scene.glob("episode_*.mp4")
        if p.name.endswith(".tmp.mp4") or int(p.stem.split("_")[1]) >= n_eps
    ]
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))

    print(f"meta 기준 유지: episode 0~{n_eps - 1} ({n_eps}개, {n_frames} frames)")
    print(f"info.json 현재: episodes={info['total_episodes']} frames={info['total_frames']} -> {n_eps}/{n_frames}")
    for f, why in bad_files:
        print(f"백업으로 이동: {f.relative_to(root)} ({why})")
    for d in stray_imgs:
        print(f"백업으로 이동: {d.relative_to(root)} (저장 안 된 임시 이미지 {len(list(d.iterdir()))}장)")
    for p in stray_clips:
        print(f"백업으로 이동: {scene.name}/{p.name} (목록에 없는 episode의 3인칭 영상)")
    if not a.apply:
        print("\n(--dry-run) 아무것도 바꾸지 않았습니다. 실제로 하려면 --apply")
        return 0

    backup = root.parent / "_backups" / f"{root.name}_before_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"\n전체 백업 복사: {backup}")
    shutil.copytree(root, backup)
    if scene.exists():
        shutil.copytree(scene, backup.with_name(backup.name + "_scene"))
    for f, _ in bad_files:
        f.unlink()
    for d in stray_imgs:
        shutil.rmtree(d)
    for p in stray_clips:
        p.unlink()
    print("백업에 원본이 있으므로 데이터셋 쪽 사본만 정리함")

    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import unflatten_dict, write_info, write_stats

    image_keys = {k for k, f in info["features"].items() if f.get("dtype") in ("image", "video")}
    per_ep = []
    for _, row in meta.iterrows():
        flat = {}
        for k, v in row.items():
            if not k.startswith("stats/") or v is None:
                continue
            arr = np.array(_deep_list(v), dtype=np.float64)  # 이미지 통계는 중첩 object 배열로 저장돼 있음
            feat, stat = k[len("stats/"):].rsplit("/", 1)
            # meta parquet에는 이미지 통계가 (3,)으로 펴져 저장되지만 aggregate_stats는 (3,1,1)을 기대한다
            if feat in image_keys and stat != "count" and arr.shape == (3,):
                arr = arr.reshape(3, 1, 1)
            flat[k[len("stats/"):]] = arr
        per_ep.append(unflatten_dict(flat))
    write_stats(aggregate_stats(per_ep), root)
    info["total_episodes"], info["total_frames"] = n_eps, n_frames
    info["splits"] = {"train": f"0:{n_eps}"}
    write_info(info, root)
    print(f"info.json / stats.json 갱신: {n_eps} episodes, {n_frames} frames")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.repo_id, root=str(root))
    ok = ds.meta.total_episodes == n_eps and len(ds) == n_frames
    print(f"검증: LeRobotDataset 로드 {ds.meta.total_episodes} episodes, {len(ds)} frames -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
