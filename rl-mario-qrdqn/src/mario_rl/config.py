from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 1_000_000


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42

    total_steps: int = 5_000_000
    learning_starts: int = 50_000
    train_every: int = 4
    target_update_every: int = 10_000

    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 500_000
    warmup_fill: int = 20_000

    lr: float = 2.5e-4
    grad_clip_norm: float = 10.0

    n_quantiles: int = 200
    huber_kappa: float = 1.0

    eps: EpsilonSchedule = field(default_factory=EpsilonSchedule)

    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 50_000

    log_every: int = 10_000


@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    frame_skip: int = 4
    frame_stack: int = 4
    resize_hw: tuple[int, int] = (84, 84)
    grayscale: bool = True
