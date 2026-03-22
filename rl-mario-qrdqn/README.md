# 🍄 Mario RL: Deep Q-Network Agent

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/0/03/Super_Mario_Bros._box.png" alt="Super Mario Bros" width="300"/>
</div>

<p align="center">
  <strong>A high-performance Reinforcement Learning agent that learns to play Super Mario Bros using Deep Q-Networks (DQN).</strong>
</p>

## 🚀 Overview

This project implements a completely autonomous AI agent capable of playing the classic NES game **Super Mario Bros**. Using a Convolutional Neural Network (CNN) and Q-Learning, the agent learns directly from raw pixels to navigate the level, avoid enemies, and maximize its score.

### ✨ Key Features
- **Deep Q-Learning (DQN):** Learns optimal policies via experience replay and target networks.
- **Computer Vision (CNN):** Processes stacked, grayscale game frames (84x84) to understand motion and the environment.
- **Custom Reward Shaping:** Incentivizes speed, coin collection, and distance traveled while penalizing stagnation.
- **Anti-Freeze Mechanism:** Automatically resets if the agent attempts to "hide" in safe corners.
- **Hardware Accelerated:** Full CUDA 12.4 PyTorch support for rapid GPU training.

## 🧠 Model Architecture

The agent's "brain" is a 3-layer Convolutional Neural Network followed by 2 fully connected layers:
1. `Conv2d(in=4, out=32, kernel=8, stride=4)`
2. `Conv2d(in=32, out=64, kernel=4, stride=2)`
3. `Conv2d(in=64, out=64, kernel=3, stride=1)`
4. `Linear(enc_dim, 512)` -> `ReLU` -> `Linear(512, n_actions)`

## 🛠️ Installation

**Requirements:**
- Python 3.10+
- NVIDIA GPU (Recommended, requires CUDA 12.4)

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mario-rl-agent.git
cd mario-rl-agent
```

2. Install dependencies (including CUDA bindings):
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 🎮 Quickstart

Train the agent and watch it play in real-time!

```bash
python mario_main.py --episodes 500
```
*Note: The agent will naturally perform poorly at first (epsilon-greedy exploration). Watch it improve over hundreds of episodes!*

**Headless Training (No GUI, much faster):**
```bash
python mario_main.py --episodes 1000 --no-render
```

## 📂 Repository Structure

- `mario_main.py` - The main training entrypoint.
- `src/mario_rl/agent.py` - DQN algorithm and optimization logic.
- `src/mario_rl/env.py` - Gym environment wrappers, frame stacking, and reward shaping.
- `src/mario_rl/networks.py` - PyTorch CNN architectures.
- `src/mario_rl/train_loop.py` - Episode tracking, rendering, and checkpointing.

## 🤝 License
This project is open-source and available under the MIT License.
