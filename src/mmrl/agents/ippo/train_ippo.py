import argparse
import numpy as np
import torch
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.agents.ippo.ippo_agent import IPPOAgent
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
    
    agent_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 3e-4,
        "gamma": 0.99,
        "rollout_steps": 2048, # large enough
        "train_iters": 4
    }
    
    # Instantiate two agents but share the network (Parameter Sharing)
    agent_a = IPPOAgent(obs_dim, act_dim, agent_cfg)
    agent_b = IPPOAgent(obs_dim, act_dim, agent_cfg)
    
    # Share weights and optimizer
    agent_b.ac = agent_a.ac
    agent_b.optimizer = agent_a.optimizer
    
    (obs_a, obs_b), info = env.reset(seed=args.seed)
    
    total_steps = 0
    while total_steps < args.steps:
        # Collect rollouts until buffer full (or episode ends, but PPO usually runs fixed steps)
        # My buffer is simple sequential. I'll run one episode at a time.
        
        mask_a = info["mask_a"]
        mask_b = info["mask_b"]
        
        act_a, logp_a, val_a = agent_a.step(obs_a, mask_a)
        act_b, logp_b, val_b = agent_b.step(obs_b, mask_b)
        
        (next_obs_a, next_obs_b), (r_a, r_b), term, trunc, next_info = env.step((act_a, act_b))
        done = term or trunc
        
        # Store
        agent_a.store(obs_a, act_a, r_a, logp_a, val_a, done, mask_a)
        agent_b.store(obs_b, act_b, r_b, logp_b, val_b, done, mask_b)
        
        obs_a, obs_b = next_obs_a, next_obs_b
        info = next_info
        total_steps += 1
        
        if done:
            # Finish path
            # Need last val for bootstrap
            # We can use next_obs (which is valid next state or terminal dummy)
            # But mask for next_obs? If done, mask irrelevant.
            # We compute value of next_obs
            _, _, last_val_a = agent_a.step(obs_a, next_info["mask_a"])
            _, _, last_val_b = agent_b.step(obs_b, next_info["mask_b"])
            
            # If done, last val should be 0 usually?
            # GAE handles done flag (next_non_terminal = 0).
            # So last_val only matters if NOT done (truncated).
            # My buffer finish_path uses dones array.
            
            if agent_a.buffer.ptr >= 64: # Update freq
                agent_a.update(last_val_a)
                agent_b.update(last_val_b)
                
            (obs_a, obs_b), info = env.reset()
            
    print("Training finished.")

if __name__ == "__main__":
    main()

