# Card-Sum Market Making RL

A reproducible research codebase for studying market-taking strategies in a stylized card-sum market with realistic microstructure (Tier-2 liquidity, impact, events).

## Overview
The goal is to trade (long/short) on the sum of 3 hidden cards (2-10, J, Q, K, A).
- **Information**: Hints (revealed cards), Market Quotes (Mid/Spread), Displayed Depth.
- **Dynamics**: Tier-2 liquidity (hidden vs displayed), market impact, slippage.
- **Events**: Probabilistic constraints (e.g., "Even cards only", "Sum >= 10").

See [docs/architecture.md](docs/architecture.md) for detailed system design.

## Installation

```bash
pip install -e .
```

Requirements: Python 3.10+, `torch`, `gymnasium`, `numpy`, `pandas`, `pyyaml`.

## Quickstart

### 1. Human Play (CLI)
Play against the market or a random opponent.

```bash
# Single Player
python -m mmrl.human.cli_play --mode single --events on --impact on

# Two Player vs Random
python -m mmrl.human.cli_play --mode two --opponent random
```

### 2. Run Baselines
Evaluate the Random-Valid policy.

```bash
python -m mmrl.eval.evaluate --policy random --episodes 100
```

### 3. Train Agents
(Example commands, assuming training scripts are hooked up to Hydra/Config)

```bash
# Train DQN (Single Player)
python -m mmrl.agents.dqn.train_dqn

# Train IPPO (Two Player)
python -m mmrl.agents.ippo.train_ippo
```

### 4. Evaluation & OOD Tests
Run out-of-distribution tests on spread/impact shifts.

```bash
python -m mmrl.eval.ood_tests
```

## Configuration
Configs are located in `src/mmrl/config/`.
- `env.yaml`: Market parameters (sigma, spread, liquidity, events).
- `dqn.yaml`, `ippo.yaml`: Algorithm hyperparameters.

## Repository Structure
- `src/mmrl/env`: Gym environments (Single/Two Player) & Core logic (Cards, Quotes, Liquidity).
- `src/mmrl/agents`: RL implementations (DQN, IPPO, MAPPO).
- `src/mmrl/baselines`: Simple policies (Random, EV Oracle).
- `src/mmrl/human`: CLI game.
- `src/mmrl/eval`: Metrics and plotting.

## License
MIT
