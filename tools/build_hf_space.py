"""공개 데모용 Hugging Face Space 폴더를 만들고(기본), 요청 시 업로드한다.

1) 먼저 tools/export_demo_assets.py로 demo_assets/를 만든다.
2) python tools/build_hf_space.py               -> build/hf_space/ 에 Space 내용만 모음(업로드 안 함)
3) python tools/build_hf_space.py --upload --space-id jun-seo/so101-il-dashboard
                                               -> Space 생성(없으면) + 업로드. 공개 주소가 생기므로 확인 후 실행.

Space에는 대시보드 실행에 필요한 코드와 demo_assets/만 담는다. 리더암 캘리브레이션,
로컬 설정, 원본 데이터셋, 체크포인트는 담지 않는다.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_SRC = ROOT / "deploy" / "hf_space"

INCLUDE = [
    "dashboard",
    "pipeline/camera_config.py",
    "pipeline/2_qa/team_robot_dataset.py",
    "config/train_main_run_config.yaml",
    "demo_assets",
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "local_settings.json", "*calibration*.json")


def build(out: Path, allow_empty: bool = False) -> None:
    if not (ROOT / "demo_assets" / "dataset" / "meta" / "info.json").is_file() and not allow_empty:
        raise SystemExit("demo_assets/가 없습니다 -- 먼저 tools/export_demo_assets.py를 실행하세요 "
                         "(결과 없이 '준비 중' 상태로 올리려면 --allow-empty).")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in INCLUDE:
        src, dst = ROOT / rel, out / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
    shutil.copy2(SPACE_SRC / "Dockerfile", out / "Dockerfile")
    shutil.copy2(SPACE_SRC / "requirements.txt", out / "requirements.txt")
    shutil.copy2(SPACE_SRC / "README_SPACE.md", out / "README.md")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[build] {out} ({size:.1f} MB)")


def upload(out: Path, space_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # delete_patterns: 예전 업로드에 있던 demo_assets 파일이 남지 않도록 폴더 내용을 그대로 맞춘다.
    api.upload_folder(folder_path=str(out), repo_id=space_id, repo_type="space",
                      commit_message="Update demo dashboard", delete_patterns=["*"])
    print(f"[upload] https://huggingface.co/spaces/{space_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf_space")
    ap.add_argument("--upload", action="store_true", help="Space에 업로드(공개 주소 생성)")
    ap.add_argument("--space-id", default="jun-seo/so101-il-dashboard")
    ap.add_argument("--allow-empty", action="store_true", help="demo_assets 없이 '결과 준비 중' 상태로 빌드")
    args = ap.parse_args()
    build(args.out, allow_empty=args.allow_empty)
    if args.upload:
        upload(args.out, args.space_id)


if __name__ == "__main__":
    main()
