"""공개 웹 데모(SO101_DEMO=1)에 쓸 `demo_assets/`를 실제 결과물에서 만든다.

공개 서버에는 수 GB짜리 이미지 데이터셋과 GPU가 없으므로, 아래만 추려서 담는다.
- dataset/            : 이미지 열을 뺀 경량 LeRobotDataset(관절 상태·액션 전체 episode)
                        -> ② QA 검증을 데모 서버에서도 실제로 계산할 수 있다.
- reports/episode_previews/episode_XXX.mp4 : 몇 개 episode의 손목 카메라 영상(원본 프레임 그대로)
- reports/train_log.csv : lerobot-train 로그에서 뽑은 step별 train loss
- reports/eval_*.json, inference_demo.mp4 : eval_success_mujoco.py 결과(그대로 복사)
- reports/checkpoint_sweep_results.json : (있으면) run_val_loss_check.py 결과
- reports/pipeline_summary.json : GPU·학습 시간 등 화면에 표시할 요약
- train_config.yaml : 학습에 쓴 설정

예:
  python tools/export_demo_assets.py --dataset-root data/so101_teleop_real \
      --train-log logs/jobs/training_XXXX.log --eval-report reports/eval_XXXX.json \
      --eval-video reports/inference_demo.mp4 --gpu "GTX 1650 Ti (4GB)" --train-time "약 3시간 30분"
"""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# lerobot-train 로그 한 줄 예: "... step:2K smpl:32K ep:54 epch:0.88 loss:0.214 grdn:12.3 lr:1.0e-05 ..."
# step 값은 "2K"처럼 반올림돼 찍혀 같은 값이 반복되므로, 몇 번째 로그 줄인지 x log_freq로 step을 계산한다.
_LOSS_RE = re.compile(r"\bstep:\S+.*?\bloss:([0-9.eE+-]+)")


def export_lite_dataset(src_root: Path, dst_root: Path, repo_id: str) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(repo_id, root=str(src_root))
    feats = src.meta.features
    keep = {k: feats[k] for k in ("observation.state", "action")}
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst = LeRobotDataset.create(repo_id=repo_id, fps=src.meta.fps, features=keep, root=dst_root,
                                robot_type=src.meta.robot_type, use_videos=False)

    tasks = src.meta.tasks  # v3: index=task 문자열, 열 task_index
    index_to_task = {int(row.task_index): task for task, row in tasks.iterrows()}

    df = (src.hf_dataset.select_columns(["episode_index", "observation.state", "action", "task_index"])
          .with_format(None).to_pandas())
    for ep, g in df.groupby("episode_index", sort=True):
        for state, action, task_index in zip(g["observation.state"], g["action"], g["task_index"]):
            dst.add_frame({
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "task": index_to_task[int(task_index)],
            })
        dst.save_episode()
    dst.finalize()
    return {"episodes": int(df["episode_index"].nunique()), "frames": int(len(df))}


def export_previews(src_root: Path, repo_id: str, out_dir: Path, episodes: list[int]) -> list[str]:
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(repo_id, root=str(src_root))
    image_keys = [k for k, f in src.meta.features.items() if f["dtype"] == "image"]
    if not image_keys:
        return []
    key = image_keys[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    hf = src.hf_dataset.select_columns(["episode_index", key]).with_format(None)
    ep_col = np.asarray(hf["episode_index"])
    written = []
    for ep in episodes:
        idx = np.nonzero(ep_col == ep)[0]
        if len(idx) == 0:
            continue
        path = out_dir / f"episode_{ep:03d}.mp4"
        container = av.open(str(path), mode="w")
        stream = None
        for i in idx:
            img = np.asarray(hf[int(i)][key].convert("RGB"))
            if stream is None:
                stream = container.add_stream("libx264", rate=int(src.meta.fps))
                stream.height, stream.width = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "26", "movflags": "+faststart"}
            frame = av.VideoFrame.from_ndarray(img[: stream.height, : stream.width], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        written.append(path.name)
    return written


def export_train_log(log_path: Path, out_csv: Path, log_freq: int) -> int:
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LOSS_RE.search(line)
        if m:
            rows.append({"step": (len(rows) + 1) * log_freq, "loss": float(m.group(1))})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="공개 데모용 demo_assets/ 생성")
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--repo-id", default="so101_teleop_real")
    ap.add_argument("--out", default=PROJECT_ROOT / "demo_assets", type=Path)
    ap.add_argument("--preview-episodes", type=int, nargs="*", default=None,
                    help="손목 카메라 미리보기를 만들 episode 번호(기본: 앞에서 6개)")
    ap.add_argument("--train-config", default=PROJECT_ROOT / "config" / "train_main_run_config.yaml", type=Path)
    ap.add_argument("--train-log", type=Path, default=None, help="lerobot-train 로그 파일(logs/jobs/...)")
    ap.add_argument("--sweep-json", type=Path, default=None, help="run_val_loss_check.py 결과 JSON")
    ap.add_argument("--eval-report", type=Path, default=None)
    ap.add_argument("--eval-video", type=Path, default=None)
    ap.add_argument("--eval-batch", type=Path, default=None,
                    help="eval_batch.py 결과 폴더(reports/eval_runs/<시각>) -- 체크포인트별 성공률 비교표로 쓴다")
    ap.add_argument("--gpu", default=None, help='예: "GTX 1650 Ti (4GB)"')
    ap.add_argument("--train-time", default=None, help='예: "약 3시간 30분"')
    args = ap.parse_args()

    import yaml

    out = args.out
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    print("[export] 경량 데이터셋 생성 중...", flush=True)
    stats = export_lite_dataset(args.dataset_root, out / "dataset", args.repo_id)
    print(f"[export] dataset: {stats}")

    eps = args.preview_episodes if args.preview_episodes is not None else list(range(min(6, stats["episodes"])))
    previews = export_previews(args.dataset_root, args.repo_id, reports / "episode_previews", eps)
    print(f"[export] previews: {previews}")

    shutil.copy2(args.train_config, out / "train_config.yaml")
    cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))

    # QA 페이지의 '실제로 있었던 이상치 필터링 사례' 그림 등 기존 reports/ 산출물(예전 데이터 기준 기록)
    for name in ("wrist_roll_clip_diagnosis.png", "action_distribution.png", "outlier_report.md"):
        src = PROJECT_ROOT / "reports" / name
        if src.is_file():
            shutil.copy2(src, reports / name)

    if args.train_log:
        n = export_train_log(args.train_log, reports / "train_log.csv", int(cfg.get("log_freq", 200)))
        print(f"[export] train_log.csv: {n} rows")
    if args.sweep_json:
        shutil.copy2(args.sweep_json, reports / "checkpoint_sweep_results.json")
    if args.eval_report:
        report = json.loads(args.eval_report.read_text(encoding="utf-8"))
        report["checkpoint"] = Path(report.get("checkpoint", "")).name or report.get("checkpoint")
        report["video"] = "inference_demo.mp4" if args.eval_video else None
        # build_static_demo.py가 reports/eval_*.json을 찾으므로 원래 이름이 eval_로 시작하지 않으면 붙인다.
        name = args.eval_report.name if args.eval_report.name.startswith("eval_") else f"eval_{args.eval_report.name}"
        (reports / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.eval_video:
        shutil.copy2(args.eval_video, reports / "inference_demo.mp4")
    if args.eval_batch:
        # 로컬 경로는 빼고 비교에 필요한 값만 남긴다(공개 데모에 PC 경로가 드러나지 않게).
        batch = json.loads((args.eval_batch / "batch.json").read_text(encoding="utf-8"))
        slim = {k: batch.get(k) for k in ("episodes", "obj_jitter_cm", "seed", "max_seconds")}
        slim["items"] = [{k: it.get(k) for k in ("label", "status", "successes", "success_rate", "failure_counts")}
                         | {"step": int(Path(it["checkpoint"]).parent.name)}
                         for it in batch["items"]]
        (reports / "eval_batch.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "exported_at": time.strftime("%Y-%m-%d %H:%M"),
        "gpu": args.gpu,
        "steps_label": f"{cfg.get('steps')} steps · batch {cfg.get('batch_size')}",
        "train_time_label": args.train_time,
        "dataset": stats,
    }
    (reports / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[export] 완료: {out} ({total:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
