from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mario_rl.config import EnvConfig, TrainingConfig
from mario_rl.train_loop import train


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(prog="mario-train")
    p.add_argument("--env-id", type=str, default=EnvConfig.env_id)
    p.add_argument("--seed", type=int, default=TrainingConfig.seed)
    p.add_argument("--total-steps", type=int, default=TrainingConfig.total_steps)
    p.add_argument("--learning-starts", type=int, default=TrainingConfig.learning_starts)
    p.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    p.add_argument("--replay-capacity", type=int, default=TrainingConfig.replay_capacity)
    p.add_argument("--lr", type=float, default=TrainingConfig.lr)
    p.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    p.add_argument("--n-quantiles", type=int, default=TrainingConfig.n_quantiles)
    p.add_argument("--checkpoint-dir", type=Path, default=TrainingConfig.checkpoint_dir)
    p.add_argument("--checkpoint-every", type=int, default=TrainingConfig.checkpoint_every)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = TrainingConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        lr=args.lr,
        gamma=args.gamma,
        n_quantiles=args.n_quantiles,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )
    env_cfg = EnvConfig(env_id=args.env_id)

    device = _device_from_args(args)
    train(cfg=cfg, env_cfg=env_cfg, device=device)


if __name__ == "__main__":
    main()
