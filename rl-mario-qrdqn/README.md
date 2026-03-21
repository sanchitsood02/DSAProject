# Mario QR-DQN Agent (Super Mario Bros)

Train a distributional RL agent for NES Super Mario Bros using Quantile Regression DQN (QR-DQN), a CNN encoder over stacked frames, epsilon-greedy exploration with decay, and an experience replay buffer.

## Highlights

- QR-DQN with quantile Huber loss
- Double DQN target selection + target network syncing
- Frame skip, max-pooling over skipped frames, grayscale + resize, frame stacking
- Checkpointing and a small evaluation runner (optional video recording)

## Project Layout

- src/mario_rl/agent.py: QR-DQN agent + training step
- src/mario_rl/networks.py: CNN quantile network
- src/mario_rl/losses.py: quantile Huber loss
- src/mario_rl/replay.py: replay buffer
- src/mario_rl/env.py: Mario environment wrappers (optional deps)
- src/mario_rl/train_loop.py: training loop + checkpointing
- src/mario_rl/eval_loop.py: evaluation loop + optional video recording
- src/mario_rl/scripts/: CLI entrypoints
- tests/: unit tests (loss, replay, network)

## Install

Create a virtual environment, then:

```bash
pip install -e ".[dev]"
```

To train on Super Mario Bros, install the optional environment dependencies:

```bash
pip install -e ".[dev,mario]"
```

## Quickstart

Run a short smoke training run (CPU):

```bash
mario-train --cpu --total-steps 10000 --learning-starts 1000
```

Evaluate a checkpoint:

```bash
mario-eval --cpu --checkpoint checkpoints/qrdqn_final.pt --episodes 3
```

Record videos (requires the `mario` extra):

```bash
mario-eval --checkpoint checkpoints/qrdqn_final.pt --episodes 1 --record-video --video-dir videos
```

## Dev Commands

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
```
