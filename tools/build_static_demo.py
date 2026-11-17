"""Netlify용 정적 체험 데모(web_demo/)를 빌드한다.

`tools/export_demo_assets.py`가 만든 demo_assets/(경량 데이터셋·영상·학습 로그·평가 결과)를
브라우저가 바로 읽을 수 있는 파일로 바꿔 build/static_demo/에 모은다.

- data/frames.bin   : 모든 frame의 [observation.state(6), action(6), timestamp(1)] float32 (원본 값 그대로)
                      -> 브라우저가 QA(z-score 아웃라이어·delta 아웃라이어·NaN/Inf)를 직접 계산한다.
- data/manifest.json: episode 범위, 관절 이름, train/val 분할(Python split_episodes와 같은 값),
                      Python QA 기준값(z=3), 학습 로그·스윕·평가 결과, 영상 목록
- media/            : 손목 카메라 미리보기, 평가 영상, QA 사례 그림

demo_assets/가 없으면 manifest 없이 빌드되고, 화면은 '결과 준비 중' 상태로 뜬다.

사용:
  python tools/build_static_demo.py                 # build/static_demo/
  npx -y netlify-cli deploy --dir build/static_demo --prod --site <사이트 이름>
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = ROOT / "web_demo"
LAYOUT = [("observation.state", 6), ("action", 6), ("timestamp", 1)]


def _load_trd(dataset_root: Path):
    sys.path.insert(0, str(ROOT / "pipeline" / "2_qa"))
    from team_robot_dataset import TeamRobotDataset

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    return TeamRobotDataset("so101_teleop_real", root=str(dataset_root)), info


def export_frames(trd, out_bin: Path) -> list[dict]:
    cols = ["episode_index", "frame_index"] + [k for k, _ in LAYOUT]
    df = trd._base.hf_dataset.select_columns(cols).with_format(None).to_pandas()
    rows = []
    for state, action, ts in zip(df["observation.state"], df["action"], df["timestamp"]):
        ts_val = ts[0] if hasattr(ts, "__len__") else ts
        rows.append(np.concatenate([np.asarray(state, np.float32), np.asarray(action, np.float32),
                                    np.asarray([ts_val], np.float32)]))
    arr = np.stack(rows).astype("<f4")
    out_bin.write_bytes(arr.tobytes())

    episodes = []
    ep_col = df["episode_index"].to_numpy()
    fr_col = df["frame_index"].to_numpy()
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or ep_col[i] != ep_col[start]:
            length = i - start
            # 브라우저는 frame_index를 episode 안의 위치로 계산한다 -- 실제 값과 같은지 확인
            if not np.array_equal(fr_col[start:i], np.arange(length)):
                raise SystemExit(f"episode {ep_col[start]}의 frame_index가 0부터 연속이 아닙니다")
            episodes.append({"index": int(ep_col[start]), "from": start, "length": length})
            start = i
    return episodes


def python_qa_reference(trd, z: float = 3.0) -> dict:
    v = trd.detect_outliers(z_thresh=z)
    d = trd.detect_outliers_delta(z_thresh=z)
    integ = trd.check_integrity()
    clean = trd.get_clean_indices(outlier_results=v, delta_outlier_results=d, integrity_results=integ)
    return {
        "z_thresh_value": z,
        "z_thresh_delta": z,
        "value_episode_outlier_counts": {str(k): n for k, n in v["episode_outlier_counts"].items()},
        "delta_episode_outlier_counts": {str(k): n for k, n in d["episode_outlier_counts"].items()},
        "integrity_problem_count": sum(integ["episode_problem_counts"].values()),
        "excluded_count": clean["excluded_count"],
        "total_count": clean["total_count"],
    }


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build(assets: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB_SRC, out)
    shutil.copy2(ROOT / "dashboard" / "assets" / "logo.png", out / "logo.png")

    dataset_root = assets / "dataset"
    if not (dataset_root / "meta" / "info.json").is_file():
        print(f"[static] {assets}에 데이터셋이 없어 '결과 준비 중' 상태로 빌드합니다.")
        return

    reports = assets / "reports"
    (out / "data").mkdir()
    media = out / "media"
    (media / "episodes").mkdir(parents=True)

    trd, info = _load_trd(dataset_root)
    episodes = export_frames(trd, out / "data" / "frames.bin")
    split = trd.split_episodes(val_ratio=0.2, seed=42)
    names = info["features"]["observation.state"].get("names") or [f"joint{i}" for i in range(6)]

    previews = []
    for mp4 in sorted((reports / "episode_previews").glob("episode_*.mp4")):
        shutil.copy2(mp4, media / "episodes" / mp4.name)
        previews.append(int(mp4.stem.split("_")[1]))

    for name in ("wrist_roll_clip_diagnosis.png",):
        if (reports / name).is_file():
            shutil.copy2(reports / name, media / name)

    train_log = None
    if (reports / "train_log.csv").is_file():
        with open(reports / "train_log.csv", encoding="utf-8") as f:
            train_log = [[int(r["step"]), float(r["loss"])] for r in csv.DictReader(f)]

    evals = sorted(p for p in reports.glob("eval_*.json") if p.name != "eval_batch.json")
    eval_report = _read_json(evals[-1]) if evals else None
    has_video = (reports / "inference_demo.mp4").is_file()
    if has_video:
        shutil.copy2(reports / "inference_demo.mp4", media / "inference_demo.mp4")

    import yaml

    cfg_path = assets / "train_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    policy = cfg.get("policy", {})

    manifest = {
        "fps": info["fps"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "image_shape": next((f["shape"] for f in info["features"].values() if f["dtype"] in ("image", "video")), None),
        "joint_names": list(names),
        "layout": [{"key": k, "dims": n} for k, n in LAYOUT],
        "episodes": episodes,
        "split": split,
        "qa_reference": python_qa_reference(trd),
        "summary": _read_json(reports / "pipeline_summary.json") or {},
        "train_config": {k: cfg.get(k) for k in ("steps", "batch_size", "num_workers", "save_freq", "log_freq")}
        | {"chunk_size": policy.get("chunk_size"), "n_action_steps": policy.get("n_action_steps")},
        "train_log": train_log,
        "sweep": _read_json(reports / "checkpoint_sweep_results.json"),
        "eval": eval_report,
        "eval_batch": _read_json(reports / "eval_batch.json"),
        "inference_video": "media/inference_demo.mp4" if has_video else None,
        "previews": previews,
        "has_wrist_roll_case": (media / "wrist_roll_clip_diagnosis.png").is_file(),
    }
    (out / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[static] {out} ({size:.1f} MB, {len(episodes)} episodes, {info['total_frames']} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=ROOT / "demo_assets")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "static_demo")
    args = ap.parse_args()
    build(args.assets, args.out)


if __name__ == "__main__":
    main()
