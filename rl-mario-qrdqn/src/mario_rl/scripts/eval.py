from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mario_rl.config import EnvConfig
from mario_rl.eval_loop import evaluate


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(prog="mario-eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default=EnvConfig.env_id)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-dir", type=Path, default=Path("videos"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    env_cfg = EnvConfig(env_id=args.env_id)
    device = _device_from_args(args)
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        env_cfg=env_cfg,
        device=device,
        episodes=args.episodes,
        record_video=args.record_video,
        video_dir=args.video_dir,
    )
    print(metrics)


if __name__ == "__main__":
    main()
