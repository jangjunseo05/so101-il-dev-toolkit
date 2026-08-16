"""
학습된 ACT 체크포인트들(기본: train_main_run_output/checkpoints/ 아래 저장된
1500~10000 스텝 7개)에 대해 val episode(기본 5,7,9)와 train episode 일부
(기본 0,1) loss를 스텝별로 측정해서, val loss가 어디서부터 정체/악화되는지
(=early stopping 적정 지점)와 val/train 격차가 스텝에 따라 어떻게 벌어지는지
보는 읽기 전용 분석 스크립트.

lerobot-train은 held-out 데이터셋에 대한 val loss를 기본 제공하지 않는다(조사로
확인: eval_freq/eval.*는 gym 환경(cfg.env) rollout 평가 전용이라 cfg.env가 None이면
아예 실행되지 않음, lerobot/scripts/lerobot_train.py:230). 그래서 학습 루프
(lerobot/scripts/lerobot_train.py의 update_policy())가 실제로 하는 것과 동일한
과정을 여기서 직접 재현한다:
  batch = next(dl_iter)
  batch = preprocessor(batch)   # 체크포인트에 저장된 정규화 설정 자동 적용
  loss, loss_dict = policy.forward(batch)

주의(중요): ACTPolicy.forward()는 use_vae=True일 때 self.training이 True인
경우에만 VAE 인코더가 실제 action을 조건으로 latent를 계산한다
(lerobot/policies/act/modeling_act.py:397,405). eval() 모드로 두면 VAE 인코더
경로 자체가 스킵되고 mu_hat/log_sigma_x2_hat이 None이 되어 KLD 계산에서 그대로
크래시한다 -- 즉 이 체크포인트들(use_vae=True)에서 policy.forward()로 loss를
재려면 policy.train() 모드가 사실상 강제된다. 이 모드는 dropout과 VAE latent
재매개변수화 샘플링이 매 호출마다 stochastic하게 들어가지만, 단일 체크포인트를
3회 반복 측정해 패스 간 표준편차가 평균의 0.02~0.04% 수준으로 무시할 만함을
이미 확인했다(run_val_loss_check.py 이전 버전 실행 결과) -- 그래서 이번
스윕에서는 체크포인트당 1회 측정(--passes 1, 기본값)으로 시간을 절약한다.

성능 최적화(중요): 체크포인트마다 val/train 데이터셋을 매번 새로 순회하면
(이전 버전 방식) 체크포인트 수만큼 이미지 디코딩 비용이 반복돼 매우 느리다
(실측: 체크포인트 1개, 3-pass 기준 val+train 계산에 약 1시간). 이번 버전은
val/train 데이터셋을 각각 "한 번만" 순회하면서, 그 안에서 미리 로딩해둔
7개 체크포인트의 policy.forward()를 배치마다 전부 돌리는 구조로 바꿨다 --
데이터 로딩(이미지 디코딩)이 병목이라는 게 실측으로 이미 확인됐으므로,
이 병목 비용을 체크포인트 수(7)가 아니라 1회만 지불하면 되게 한 것.
preprocessor(batch)를 같은 원본 batch에 대해 여러 policy가 반복 호출해도
안전한지는 소스로 확인함: normalize_processor.py의 `_apply_transform()`이
모든 정규화 모드(MEAN_STD/MIN_MAX/IDENTITY)에서 새 텐서를 반환하는 순수
함수형 연산(예: `(tensor - mean) / denom`)이라 원본 batch 텐서를 in-place로
훼손하지 않는다 -- 즉 여러 policy가 같은 원본 batch를 재사용해도 서로 간섭하지
않는다.

사용법:
  python run_val_loss_check.py                        # so101_web에 있는 체크포인트(010000)만
  python run_val_loss_check.py --checkpoints-dir <원본 train_main_run_output/checkpoints 경로> --steps 1500 3000 4500 6000 7500 9000 10000
  python run_val_loss_check.py --checkpoint <경로>     # 단일 체크포인트만(이전 방식과 동일 용도)
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# so101_web/pipeline/3_training/run_val_loss_check.py -> so101_web
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_DIR = str(PROJECT_ROOT / "checkpoints")
# so101_web으로 이전 시 체크포인트는 010000(최종 스텝) 하나만 옮겨왔다 -- 나머지
# (1500~9000)는 checkpoint_sweep_results.json에 이미 정제된 결과로 보존돼 있어
# 원본 체크포인트 파일 자체는 용량 절약을 위해 옮기지 않음(so101_web 마이그레이션
# 논의 참고). 여러 체크포인트를 다시 스윕하려면 --checkpoints-dir/--steps로
# 원래 프로젝트의 train_main_run_output/checkpoints를 직접 가리키면 된다.
DEFAULT_STEPS = [10000]
DEFAULT_REPO_ID = "so101_teleop_real"
DEFAULT_ROOT = str(PROJECT_ROOT / "data" / "so101_teleop_real")
DEFAULT_RESULTS_JSON = str(PROJECT_ROOT / "reports" / "checkpoint_sweep_results.json")


def checkpoint_path(checkpoints_dir: str, step: int) -> str:
    return f"{checkpoints_dir}/{step:06d}/pretrained_model"


def load_policy_and_preprocessor(path: str, device: str):
    policy = ACTPolicy.from_pretrained(path, device=device)
    # use_vae=True인 ACT는 policy.train() 모드에서만 VAE 인코더가 동작한다
    # (모듈 상단 docstring 참고) -- eval()로는 forward()가 크래시한다.
    policy.train()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor


def build_dataset(repo_id: str, root: str, episodes: list, delta_timestamps: dict) -> LeRobotDataset:
    return LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps)


@torch.no_grad()
def sweep_one_dataset_pass(
    dataloader: DataLoader, policies: dict, preprocessors: dict
) -> dict:
    """dataloader를 한 번만 순회하면서, 로딩된 모든 체크포인트(step)의
    loss/l1_loss/kld_loss를 동시에 프레임 수 가중 평균으로 계산한다."""
    sums = {step: {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0} for step in policies}
    total_frames = 0
    for batch in dataloader:
        n = batch["action"].shape[0]
        total_frames += n
        for step, policy in policies.items():
            processed = preprocessors[step](batch)
            loss, loss_dict = policy.forward(processed)
            sums[step]["loss"] += loss.item() * n
            sums[step]["l1_loss"] += loss_dict["l1_loss"] * n
            sums[step]["kld_loss"] += loss_dict.get("kld_loss", 0.0) * n
    return {
        step: {**{k: v / total_frames for k, v in s.items()}, "n_frames": total_frames}
        for step, s in sums.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="설정 시 --checkpoints-dir/--steps 스윕 대신 이 경로 하나만 평가(레거시 단일 체크포인트 모드)",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--val-episodes", type=int, nargs="+", default=[5, 7, 9])
    parser.add_argument("--train-episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON)
    args = parser.parse_args()

    if args.checkpoint is not None:
        steps = [0]
        paths = {0: args.checkpoint}
    else:
        steps = args.steps
        paths = {step: checkpoint_path(args.checkpoints_dir, step) for step in steps}

    policies, preprocessors = {}, {}
    for step in steps:
        print(f"체크포인트 로딩: step={step} path={paths[step]}")
        policy, preprocessor = load_policy_and_preprocessor(paths[step], args.device)
        policies[step] = policy
        preprocessors[step] = preprocessor

    first_policy = policies[steps[0]]
    print(
        f"policy 설정 확인: use_vae={first_policy.config.use_vae} "
        f"chunk_size={first_policy.config.chunk_size} dropout={first_policy.config.dropout}"
    )

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = ds_meta.fps
    delta_timestamps = {"action": [i / fps for i in range(first_policy.config.chunk_size)]}
    print(f"dataset fps={fps}, action delta_timestamps 길이={len(delta_timestamps['action'])}")

    val_dataset = build_dataset(args.repo_id, args.root, args.val_episodes, delta_timestamps)
    train_dataset = build_dataset(args.repo_id, args.root, args.train_episodes, delta_timestamps)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"VAL   episodes={args.val_episodes}  frames={len(val_dataset)}")
    print(f"TRAIN episodes={args.train_episodes}  frames={len(train_dataset)}")

    val_passes, train_passes = [], []
    for p in range(args.passes):
        print(f"\n--- pass {p + 1}/{args.passes}: VAL 순회 ---")
        val_passes.append(sweep_one_dataset_pass(val_loader, policies, preprocessors))
        print(f"--- pass {p + 1}/{args.passes}: TRAIN 순회 ---")
        train_passes.append(sweep_one_dataset_pass(train_loader, policies, preprocessors))

    results = {}
    for step in steps:
        val_losses = torch.tensor([p[step]["loss"] for p in val_passes])
        train_losses = torch.tensor([p[step]["loss"] for p in train_passes])
        results[step] = {
            "val_loss": val_losses.mean().item(),
            "val_loss_std": val_losses.std().item() if args.passes > 1 else None,
            "train_loss": train_losses.mean().item(),
            "train_loss_std": train_losses.std().item() if args.passes > 1 else None,
            "val_l1_loss": val_passes[-1][step]["l1_loss"],
            "train_l1_loss": train_passes[-1][step]["l1_loss"],
            "ratio": val_losses.mean().item() / train_losses.mean().item(),
        }

    print("\n" + "=" * 70)
    print(f"{'step':>7} {'train_loss':>11} {'val_loss':>10} {'ratio':>7}")
    print("=" * 70)
    for step in steps:
        r = results[step]
        print(f"{step:>7} {r['train_loss']:>11.4f} {r['val_loss']:>10.4f} {r['ratio']:>6.2f}x")

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": steps,
                "results": {str(k): v for k, v in results.items()},
                "val_episodes": args.val_episodes,
                "train_episodes": args.train_episodes,
                "passes": args.passes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n결과 저장: {args.results_json}")


if __name__ == "__main__":
    main()
