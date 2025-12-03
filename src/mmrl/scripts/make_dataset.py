import argparse
import numpy as np
import pandas as pd
import os
import yaml
from typing import Dict, Any
import gymnasium as gym

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.baselines.random_valid import RandomValidAgent
from mmrl.baselines.ev_oracle import EVOracleAgent

def make_dataset(
    output_path: str,
    steps: int,
    env_kind: str,
    policy_kind: str,
    cfg_path: str = None
):
    """
    Generate offline dataset from policy rollouts.
    """
    # Load config
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "W0": 500.0, 
            "episode_length": 10,
            "flags": {"enable_events": True, "enable_impact": True}
        }

    # Setup Env
    if env_kind == "single":
        env = SingleCardEnv(cfg)
        is_multi = False
    elif env_kind == "two":
        env = TwoPlayerCardEnv(cfg)
        is_multi = True
    else:
        raise ValueError(f"Unknown env kind: {env_kind}")
        
    # Setup Agent
    if policy_kind == "random":
        agent = RandomValidAgent()
    elif policy_kind == "oracle":
        agent = EVOracleAgent()
    else:
        # Could load trained agent here if needed
        raise ValueError(f"Unknown policy: {policy_kind}")
        
    obs, info = env.reset()
    
    records = []
    
    for step_i in range(steps):
        # Get Mask
        # Assuming single agent interface or wrapper for now for simplicity
        # Or handling two player explicitly
        
        if is_multi:
            # Just random for both for now or simple
            # Logic complicates for two player data generation without proper agent abstraction
            # For now, let's assume single player for dataset generation in this stub
            # or implement basic two player loop
            
            mask_a = info.get("mask_a", np.ones(21, dtype=bool))
            mask_b = info.get("mask_b", np.ones(21, dtype=bool))
            
            act_a = agent.act(obs[0], mask_a)
            act_b = agent.act(obs[1], mask_b)
            action = (act_a, act_b)
            
            next_obs, rewards, term, trunc, next_info = env.step(action)
            
            # Record A
            records.append({
                "step": step_i,
                "agent": "A",
                "obs_flat": obs[0].flatten().tolist(), # Parquet handles lists? better as binary or separate cols
                "action": act_a,
                "reward": rewards[0],
                "done": term or trunc,
                "info": str(info) # Serialize info dict
            })
            # Record B
            records.append({
                "step": step_i,
                "agent": "B",
                "obs_flat": obs[1].flatten().tolist(),
                "action": act_b,
                "reward": rewards[1],
                "done": term or trunc,
                "info": str(info)
            })
            
            obs, info = next_obs, next_info
            
        else:
            mask = info.get("mask", np.ones(21, dtype=bool))
            action = agent.act(obs, mask)
            
            next_obs, reward, term, trunc, next_info = env.step(action)
            
            # Record
            # Flatten obs for storage
            obs_flat = obs.flatten()
            
            record = {
                "step": step_i,
                "obs": obs_flat.tolist(), # List of floats
                "action": int(action),
                "reward": float(reward),
                "done": bool(term or trunc),
                "mu": float(info.get("mu", 0.0)),
                "sigma": float(info.get("sigma", 0.0)),
                "exec_price": float(info.get("exec_price", 0.0)),
                "true_sum": float(info.get("true_sum", 0.0))
            }
            records.append(record)
            
            obs, info = next_obs, next_info
            
        if term or trunc:
            obs, info = env.reset()
            
    df = pd.DataFrame(records)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
    except Exception as e:
        print(f"Failed to save to {output_path}: {e}")
        # Fallback
        fallback = output_path.replace(".parquet", ".csv")
        df.to_csv(fallback, index=False)
        print(f"Saved to {fallback} instead")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/rollouts/data.parquet")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--env", type=str, default="single")
    parser.add_argument("--policy", type=str, default="random")
    
    args = parser.parse_args()
    
    make_dataset(args.output, args.steps, args.env, args.policy)

