from __future__ import annotations

import argparse

import torch

from mario_rl.config import EnvConfig
from mario_rl.train_loop import train_mario_new


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(prog="mario_main.py")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-id", type=str, default=EnvConfig.env_id)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    env_cfg = EnvConfig(env_id=args.env_id)
    device = _device_from_args(args)
    train_mario_new(
        env_cfg=env_cfg,
        device=device,
        episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
