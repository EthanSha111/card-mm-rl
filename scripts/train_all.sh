#!/bin/bash
# Master training script to train all RL agents

set -e

echo "========================================"
echo "Training All RL Agents"
echo "========================================"

# Train DQN (Single Player)
echo ""
echo "1/3: Training DQN..."
python -m mmrl.agents.dqn.train_dqn --episodes 1000 --seed 42

# Train IPPO (Two Player)
echo ""
echo "2/3: Training IPPO..."
python -m mmrl.agents.ippo.train_ippo --steps 50000 --seed 42

# Train MAPPO (Two Player)
echo ""
echo "3/3: Training MAPPO..."
python -m mmrl.agents.mappo.train_mappo --steps 50000 --seed 42

echo ""
echo "========================================"
echo "All Training Completed!"
echo "Logs: data/logs/{dqn,ippo,mappo}"
echo "Models: data/models/{dqn,ippo,mappo}"
echo "========================================"

