"""녹화된 LeRobotDataset에서 episode를 지운다 (대시보드 ① 데이터 수집의 'episode 관리'가 호출).

lerobot 공식 `lerobot.datasets.dataset_tools.delete_episodes()`(= `lerobot-edit-dataset
--operation.type delete_episodes`)를 그대로 쓴다. 공식 CLI의 제자리 모드는 `<root>/<repo_id>`
경로 규칙을 가정해서 이 프로젝트의 `--root`(데이터셋 폴더 자체)와 맞지 않아, 같은 함수를
직접 호출한다.

순서:
  1. 원본 폴더를 같은 위치의 _backups/<이름>_<시각>/ (실제 데이터는 data/_backups/)로 **옮긴다**(이것이 백업. 복사보다 빠르고
     실패해도 원본이 그대로 남는다).
  2. 백업에서 읽어 지울 episode를 뺀 새 데이터셋을 원래 경로에 만든다.
     lerobot이 남은 episode 번호를 0부터 다시 매긴다(이전 QA 스냅샷 번호와 어긋날 수 있음).
  3. 실패하면 새로 만들던 폴더를 지우고 백업을 원래 자리로 되돌린다.

대시보드는 데이터셋 파일을 열어 둔 Streamlit 프로세스와 분리하려고 이 스크립트를
subprocess로 실행한다(Windows에서 열린 파일이 있으면 폴더 이동이 막힘).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path



def main() -> int:
    parser = argparse.ArgumentParser(description="delete episodes from a local LeRobotDataset (with backup)")
    parser.add_argument("--root", type=Path, required=True, help="데이터셋 폴더 (meta/info.json이 있는 곳)")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", required=True, help="지울 episode 번호, 쉼표 구분 (예: 3,7,12)")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    to_delete = sorted({int(x) for x in args.episodes.split(",") if x.strip()})
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    total = int(info["total_episodes"])
    bad = [e for e in to_delete if not 0 <= e < total]
    if bad:
        print(f"[delete_episodes] 없는 episode 번호: {bad} (총 {total}개)", flush=True)
        return 2
    if len(to_delete) >= total:
        print("[delete_episodes] 모든 episode를 지울 수는 없습니다(lerobot 제약). 폴더째 지우세요.", flush=True)
        return 2

    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    backup_dir = root.parent / "_backups"  # 실제 데이터는 data/_backups/ (data/는 git 제외)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{root.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    # 녹화 때 따로 저장한 3인칭 영상(<root>_scene/episode_XXXXXX.mp4)도 같이 백업하고 번호를 다시 매긴다
    scene_dir = root.parent / f"{root.name}_scene"
    scene_backup = backup_dir / f"{backup.name}_scene"
    print(f"[delete_episodes] 백업으로 이동: {root} -> {backup}", flush=True)
    shutil.move(str(root), str(backup))
    if scene_dir.is_dir():
        shutil.move(str(scene_dir), str(scene_backup))
    try:
        src = LeRobotDataset(args.repo_id, root=str(backup))
        print(f"[delete_episodes] 지울 episode {to_delete} / 총 {src.meta.total_episodes}개", flush=True)
        new = delete_episodes(src, to_delete, output_dir=root, repo_id=args.repo_id)
        if scene_backup.is_dir():
            kept = [e for e in range(total) if e not in to_delete]  # lerobot과 같은 재번호 규칙
            scene_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for new_idx, old_idx in enumerate(kept):
                src_clip = scene_backup / f"episode_{old_idx:06d}.mp4"
                if src_clip.is_file():
                    shutil.copy2(src_clip, scene_dir / f"episode_{new_idx:06d}.mp4")
                    n += 1
            print(f"[delete_episodes] 3인칭 영상 {n}개를 새 번호로 복사", flush=True)
        print(
            f"[delete_episodes] 완료: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames "
            f"(백업: {backup})",
            flush=True,
        )
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        print("[delete_episodes] 실패 -- 원본을 되돌립니다", flush=True)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        shutil.move(str(backup), str(root))
        if scene_backup.is_dir():
            if scene_dir.exists():
                shutil.rmtree(scene_dir, ignore_errors=True)
            shutil.move(str(scene_backup), str(scene_dir))
        print(f"[delete_episodes] 원본 복구 완료: {root}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
