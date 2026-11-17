"""① 데이터 수집 페이지의 episode 관리: 목록·길이 점검, 손목 카메라 미리보기 생성,
녹화 중 실시간 카운터, episode 삭제(pipeline/2_qa/delete_episodes.py를 job으로 실행).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib import dataset_registry
from lib import process_manager as pm

PROJECT_ROOT = dataset_registry.PROJECT_ROOT
DELETE_STAGE = "dataset_edit"
DELETE_SCRIPT = PROJECT_ROOT / "pipeline" / "2_qa" / "delete_episodes.py"
PREVIEW_CACHE = PROJECT_ROOT / "state" / "episode_previews"


def _root(dataset_id: str) -> Path:
    return dataset_registry.get_profile(dataset_id)["dataset_root"]


def dataset_exists(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> bool:
    from lib import data_sources as ds  # 순환 import 방지용 지연 import

    return ds.has_episodes(dataset_id)


def _info(dataset_id: str) -> dict:
    return json.loads((_root(dataset_id) / "meta" / "info.json").read_text(encoding="utf-8"))


def episode_table(dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> pd.DataFrame:
    """episode별 번호·frame 수·길이(초)와 길이 점검 표시. 모든 episodes 메타 파일을 읽는다."""
    root = _root(dataset_id)
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"]) for f in files], ignore_index=True)
    df = df.sort_values("episode_index").reset_index(drop=True)
    fps = float(_info(dataset_id).get("fps", 30))
    df["seconds"] = (df["length"] / fps).round(1)
    med = float(df["length"].median()) if len(df) else 0.0
    def flag(n: int) -> str:
        if len(df) < 5:
            return ""  # 너무 적으면 기준이 불안정해 표시하지 않음
        if n < 0.5 * med:
            return "짧음 — 오조작인지 확인"
        if n > 1.8 * med:
            return "김 — 머뭇거림이 있었는지 확인"
        return ""
    df["check"] = df["length"].map(flag)
    return df


def scene_clip_path(episode_index: int, dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> Path | None:
    """녹화 때 함께 저장한 3인칭 MuJoCo 화면 mp4(run_teleop_real.py SceneClipWriter). 이 기능을 넣기
    전에 녹화한 episode에는 없다."""
    root = _root(dataset_id)
    path = root.parent / f"{root.name}_scene" / f"episode_{episode_index:06d}.mp4"
    return path if path.is_file() else None


def preview_path(episode_index: int, dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> Path | None:
    """손목 카메라 미리보기 mp4. export 스크립트가 만든 것이 있으면 그걸 쓰고, 없으면 데이터셋
    이미지로 만들어 state/episode_previews/에 캐시한다. 데이터셋이 바뀌면(episode 추가·삭제)
    캐시 폴더 이름이 달라져 새로 만든다. 이미지 필드가 없으면 None."""
    exported = dataset_registry.get_profile(dataset_id)["reports_dir"] / "episode_previews" / f"episode_{episode_index:03d}.mp4"
    if exported.is_file():
        return exported
    info = _info(dataset_id)
    key = next((k for k, f in info["features"].items() if f.get("dtype") == "image"), None)
    if key is None:
        return None
    # 개수만으로는 다른 데이터셋과 겹칠 수 있어 info.json 수정 시각도 넣는다(녹화·삭제 때마다 바뀜)
    mtime = (_root(dataset_id) / "meta" / "info.json").stat().st_mtime_ns
    tag = f"{dataset_id}_{info['total_episodes']}ep_{info['total_frames']}f_{mtime}"
    out = PREVIEW_CACHE / tag / f"episode_{episode_index:03d}.mp4"
    if out.is_file():
        return out
    _write_preview(_root(dataset_id), key, episode_index, int(info.get("fps", 30)), out)
    return out if out.is_file() else None


def _write_preview(root: Path, key: str, episode_index: int, fps: int, out: Path) -> None:
    import av
    import pyarrow.dataset as pads
    from PIL import Image
    import io

    table = pads.dataset(str(root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", key], filter=pads.field("episode_index") == episode_index
    )
    if table.num_rows == 0:
        return
    order = np.argsort(np.asarray(table.column("frame_index")))
    images = table.column(key).to_pylist()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    container = av.open(str(tmp), mode="w")
    stream = None
    for i in order:
        cell = images[int(i)]
        img = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB")) if cell.get("bytes") else None
        if img is None:
            continue
        if stream is None:
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "26", "movflags": "+faststart"}
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[: stream.height, : stream.width]), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    if stream is not None:
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    if stream is None:
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(out)


_SAVED = re.compile(r"\[REC\] 저장 완료: (\d+) frames \(총 (\d+) episodes\)")


def recording_status(log_path: str | None) -> dict:
    """녹화 세션 로그에서 현재 상태와 저장 개수를 읽는다(run_teleop_real.py의 [REC] 출력).
    {"state": "recording"|"saving"|"idle", "saved_this_session": int, "total": int|None, "current_ep": int|None}"""
    text = pm.tail_log(log_path, n_lines=4000) if log_path else ""
    state, current, total, saved = "idle", None, None, 0
    slow, short_saved = None, None  # 이번 녹화 중 루프 느림 경고 / 마지막 저장 episode가 짧게 기록됐다는 경고
    for line in text.splitlines():
        if "[경고] 제어 루프 느림" in line:
            slow = line.strip()
        elif "[REC] ⚠" in line:
            short_saved = line.strip()
        if "[REC] 녹화 시작" in line:
            state, slow, short_saved = "recording", None, None
            m = re.search(r"episode (\d+)", line)
            current = int(m.group(1)) if m else None
        elif "[REC] 저장 중" in line:
            state = "saving"
        elif (m := _SAVED.search(line)):
            state, total, saved = "idle", int(m.group(2)), saved + 1
        elif "[REC] episode 버림" in line or "[REC] 빈 episode" in line:
            state = "idle"
    return {"state": state, "saved_this_session": saved, "total": total, "current_ep": current,
            "slow_warning": slow, "short_saved_warning": short_saved}


def start_delete(episodes: list[int], dataset_id: str = dataset_registry.DEFAULT_DATASET_ID) -> dict:
    profile = dataset_registry.get_profile(dataset_id)
    cmd = [
        sys.executable, str(DELETE_SCRIPT),
        "--root", str(profile["dataset_root"]),
        "--repo-id", profile["repo_id"],
        "--episodes", ",".join(str(e) for e in sorted(episodes)),
    ]
    return pm.start_job(DELETE_STAGE, cmd, extra_meta={"episodes": sorted(episodes)})
