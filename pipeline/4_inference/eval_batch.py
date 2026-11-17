"""여러 체크포인트를 같은 조건(episode 수·물체 위치 변화·seed)으로 차례로 평가한다.

대시보드 ④의 "자동 성공률 평가"가 job 하나로 띄우는 스크립트. 체크포인트마다
eval_success_mujoco.py를 별도 프로세스로 실행해(GPU 메모리를 체크포인트마다 비움)
결과를 한 폴더에 모은다.

  reports/eval_runs/<시각>/
    batch.json             -- 설정 + 체크포인트별 요약(진행 중에도 갱신)
    <label>.json           -- eval_success_mujoco.py 결과(episode별)
    <label>.mp4            -- --video일 때 3인칭 영상

사용 예:
  python pipeline/4_inference/eval_batch.py --episodes 20 --obj-jitter-cm 1 --video \
      --ckpt "main_run step 6000=train_runs/.../checkpoints/006000/pretrained_model" --ckpt ...
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
EVAL_SCRIPT = THIS_DIR / "eval_success_mujoco.py"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, help='"표시이름=pretrained_model 경로" (여러 번)')
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--obj-jitter-cm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "reports" / "eval_runs" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        items.append({"label": label, "checkpoint": path, "status": "pending"})
    batch = {"started_at": time.time(), "ended_at": None, "episodes": args.episodes,
             "obj_jitter_cm": args.obj_jitter_cm, "seed": args.seed, "max_seconds": args.max_seconds,
             "video": args.video, "items": items}

    def save():
        (out / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    print(f"[batch] {len(items)}개 체크포인트 × {args.episodes} episode -- 결과 폴더: {out}", flush=True)
    for i, it in enumerate(items, 1):
        name = _safe(it["label"])
        report = out / f"{name}.json"
        cmd = [sys.executable, str(EVAL_SCRIPT), "--checkpoint", it["checkpoint"],
               "--episodes", str(args.episodes), "--obj-jitter-cm", str(args.obj_jitter_cm),
               "--seed", str(args.seed), "--max-seconds", str(args.max_seconds),
               "--label", it["label"], "--report", str(report)]
        if args.video:
            cmd += ["--video", str(out / f"{name}.mp4")]
        print(f"[batch] ({i}/{len(items)}) {it['label']} 평가 시작", flush=True)
        it["status"] = "running"
        save()
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        it["wall_seconds"] = round(time.time() - t0, 1)
        if rc == 0 and report.is_file():
            r = json.loads(report.read_text(encoding="utf-8"))
            it.update(status="done", report=report.name, successes=r["successes"],
                      success_rate=r["success_rate"], failure_counts=r.get("failure_counts", {}),
                      video=(f"{name}.mp4" if args.video else None))
            print(f"[batch] ({i}/{len(items)}) {it['label']}: 성공 {r['successes']}/{r['episodes']}", flush=True)
        else:
            it.update(status="failed", returncode=rc)
            print(f"[batch] ({i}/{len(items)}) {it['label']}: 평가 실패 (exit {rc})", flush=True)
        save()
    batch["ended_at"] = time.time()
    save()
    print("[batch] 완료", flush=True)


if __name__ == "__main__":
    main()
