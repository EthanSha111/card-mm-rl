import argparse
import numpy as np
import torch
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.agents.mappo.mappo_agent import MAPPOAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": True}
    }
    
    env = TwoPlayerCardEnv(cfg)
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    agent_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 3e-4,
        "rollout_steps": 2048,
        "train_iters": 4
    }
    
    # Two agents with shared parameters
    agent_a = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b.ac = agent_a.ac
    agent_b.optimizer = agent_a.optimizer
    
    (obs_a, obs_b), info = env.reset(seed=args.seed)
    
    total_steps = 0
    while total_steps < args.steps:
        mask_a = info["mask_a"]
        mask_b = info["mask_b"]
        
        # Construct joint obs
        joint_obs = np.concatenate([obs_a, obs_b])
        
        act_a, logp_a, val_a = agent_a.step(obs_a, joint_obs, mask_a)
        act_b, logp_b, val_b = agent_b.step(obs_b, joint_obs, mask_b)
        
        (next_obs_a, next_obs_b), (r_a, r_b), term, trunc, next_info = env.step((act_a, act_b))
        done = term or trunc
        
        agent_a.store(obs_a, joint_obs, act_a, r_a, logp_a, val_a, done, mask_a)
        agent_b.store(obs_b, joint_obs, act_b, r_b, logp_b, val_b, done, mask_b)
        
        obs_a, obs_b = next_obs_a, next_obs_b
        info = next_info
        total_steps += 1
        
        if done:
            next_joint_obs = np.concatenate([next_obs_a, next_obs_b])
            _, _, last_val_a = agent_a.step(obs_a, next_joint_obs, next_info["mask_a"])
            _, _, last_val_b = agent_b.step(obs_b, next_joint_obs, next_info["mask_b"])
            
            if agent_a.buffer.ptr >= 64:
                agent_a.update(last_val_a)
                agent_b.update(last_val_b)
                
            (obs_a, obs_b), info = env.reset()
            
    print("MAPPO Training finished.")

if __name__ == "__main__":
    main()

