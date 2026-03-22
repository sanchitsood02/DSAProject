# Mario RL (DQN + QR-DQN) for Super Mario Bros

This repo contains a small reinforcement learning codebase for training agents on NES Super Mario Bros.

There are two training paths:
- QR-DQN (distributional DQN with quantile regression) via the CLI entrypoints
- A MARIO_NEW-style DQN runner script (`mario_main.py`) that trains from pixels and checkpoints to `models/mario_dqn.pt`

## What it does
- Wraps the Mario environment with frame skipping, max-pooling, grayscale/resize, and frame stacking
- Trains a convolutional network with experience replay and a target network
- Optionally applies reward shaping + an anti-freeze rule (truncate the episode if Mario stops progressing)
- Saves checkpoints so training can be resumed

## Install

Create a virtual environment, then:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

To run against Super Mario Bros, install the optional environment dependencies:

```bash
pip install -e ".[dev,mario]"
```

## Quickstart

MARIO_NEW-style DQN training (renders by default):

```bash
python mario_main.py
```

Headless (faster):

```bash
python mario_main.py --no-render
```

QR-DQN smoke training run (CPU):

```bash
mario-train --cpu --total-steps 10000 --learning-starts 1000
```

Evaluate a checkpoint (CPU):

```bash
mario-eval --cpu --checkpoint checkpoints/qrdqn_final.pt --episodes 3
```

## Code map
- [agent.py](file:///c:/Users/sanch/Documents/trae_projects/Git/rl-mario-qrdqn/src/mario_rl/agent.py): DQN + QR-DQN agents
- [env.py](file:///c:/Users/sanch/Documents/trae_projects/Git/rl-mario-qrdqn/src/mario_rl/env.py): wrappers and reward/freeze logic
- [networks.py](file:///c:/Users/sanch/Documents/trae_projects/Git/rl-mario-qrdqn/src/mario_rl/networks.py): CNN models
- [train_loop.py](file:///c:/Users/sanch/Documents/trae_projects/Git/rl-mario-qrdqn/src/mario_rl/train_loop.py): training loops + checkpointing
