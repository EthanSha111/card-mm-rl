#!/usr/bin/env python
"""
Evaluate all trained agents and baselines, compare performance.
"""
import argparse
import os
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.baselines.random_valid import RandomValidAgent
from mmrl.baselines.ev_oracle import EVOracleAgent
from mmrl.baselines.level1_crowding import Level1Policy
from mmrl.agents.dqn.dqn_agent import DQNAgent
from mmrl.agents.ippo.ippo_agent import IPPOAgent
from mmrl.agents.mappo.mappo_agent import MAPPOAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE
from mmrl.eval.evaluate import evaluate_policy

def load_dqn(checkpoint_path: str) -> DQNAgent:
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    cfg = {
        "device": "cpu",
        "gamma": 0.99,
        "batch_size": 32,
        "lr": 1e-3,
        "hidden_dims": [64, 64]
    }
    
    agent = DQNAgent(obs_dim, act_dim, cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.q_net.load_state_dict(checkpoint['q_net'])
    agent.q_net.eval()
    return agent

def load_ippo(checkpoint_path: str) -> IPPOAgent:
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    cfg = {
        "device": "cpu",
        "lr": 3e-4,
        "rollout_steps": 2048,
        "train_iters": 4
    }
    
    agent = IPPOAgent(obs_dim, act_dim, cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.ac.load_state_dict(checkpoint['ac'])
    agent.ac.eval()
    return agent

def load_mappo(checkpoint_path: str) -> MAPPOAgent:
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    cfg = {
        "device": "cpu",
        "lr": 3e-4,
        "rollout_steps": 2048,
        "train_iters": 4
    }
    
    agent = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.ac.load_state_dict(checkpoint['ac'])
    agent.ac.eval()
    return agent

def evaluate_single_agent(agent, agent_name: str, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate single-player agent."""
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": True}
    }
    
    def make_env(c):
        return SingleCardEnv(c)
    
    metrics = evaluate_policy(make_env, agent, cfg, n_episodes=n_episodes, seed=123)
    metrics["agent"] = agent_name
    metrics["mode"] = "single"
    return metrics

def evaluate_two_agent(agent_a, agent_b, agent_name: str, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate two-player agent."""
    # For two-player, we need a wrapper or custom eval logic
    # For simplicity, evaluate agent_a vs random
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": True}
    }
    
    env = TwoPlayerCardEnv(cfg)
    opponent = RandomValidAgent()
    
    returns_a = []
    returns_b = []
    
    for i in range(n_episodes):
        (obs_a, obs_b), info = env.reset(seed=123 + i)
        done = False
        ep_ret_a = 0.0
        ep_ret_b = 0.0
        
        while not done:
            mask_a = info["mask_a"]
            mask_b = info["mask_b"]
            
            # Agent A acts
            if hasattr(agent_a, 'act'):
                act_a = agent_a.act(obs_a, mask_a, eval_mode=True)
            else:
                # MAPPO/IPPO might have different signature
                act_a, _, _ = agent_a.step(obs_a, mask_a) if not hasattr(agent_a, 'ac') else agent_a.ac.act(torch.tensor(obs_a), torch.tensor(mask_a))
                
            # Opponent acts
            act_b = opponent.act(obs_b, mask_b, eval_mode=True)
            
            (obs_a, obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
            done = term or trunc
            
            ep_ret_a += r_a
            ep_ret_b += r_b
            
        returns_a.append(ep_ret_a)
        returns_b.append(ep_ret_b)
    
    return {
        "agent": agent_name,
        "mode": "two",
        "return_mean": float(np.mean(returns_a)),
        "return_std": float(np.std(returns_a)),
        "sharpe": float(np.mean(returns_a) / (np.std(returns_a) + 1e-8))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/results/eval_all.csv")
    args = parser.parse_args()
    
    results = []
    
    print("Evaluating Baselines...")
    
    # Random-Valid (Single)
    random_agent = RandomValidAgent()
    results.append(evaluate_single_agent(random_agent, "Random", args.n_episodes))
    
    # EV Oracle (Single)
    # Note: EV Oracle needs `info` with `mu`. Our evaluate_policy doesn't pass info to agent.act.
    # We need to modify evaluate_policy or create custom eval loop.
    # For now, skip or use a wrapper.
    print("  [Warning] EV Oracle and Level-1 require custom eval loop (need info). Skipping for now.")
    
    # DQN
    print("Evaluating DQN...")
    dqn_path = "data/models/dqn/dqn_final.pt"
    if not os.path.exists(dqn_path):
        dqn_path = "data/models/dqn/dqn_ep1000.pt"
    
    if os.path.exists(dqn_path):
        dqn = load_dqn(dqn_path)
        results.append(evaluate_single_agent(dqn, "DQN", args.n_episodes))
    else:
        print(f"  [Warning] DQN checkpoint not found at {dqn_path}")
    
    # IPPO
    print("Evaluating IPPO...")
    ippo_path = "data/models/ippo/ippo_final.pt"
    if os.path.exists(ippo_path):
        ippo_a = load_ippo(ippo_path)
        ippo_b = load_ippo(ippo_path)
        results.append(evaluate_two_agent(ippo_a, ippo_b, "IPPO", args.n_episodes))
    else:
        print(f"  [Warning] IPPO checkpoint not found at {ippo_path}")
    
    # MAPPO
    print("Evaluating MAPPO...")
    mappo_path = "data/models/mappo/mappo_final.pt"
    if os.path.exists(mappo_path):
        mappo_a = load_mappo(mappo_path)
        mappo_b = load_mappo(mappo_path)
        results.append(evaluate_two_agent(mappo_a, mappo_b, "MAPPO", args.n_episodes))
    else:
        print(f"  [Warning] MAPPO checkpoint not found at {mappo_path}")
    
    # Save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*50)
    print("Evaluation Summary:")
    print("="*50)
    print(df.to_string(index=False))
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()

