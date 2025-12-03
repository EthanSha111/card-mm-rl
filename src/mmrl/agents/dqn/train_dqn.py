import argparse
import numpy as np
import torch
from mmrl.env.single_env import SingleCardEnv
from mmrl.agents.dqn.dqn_agent import DQNAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Simple config
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": True}
    }
    
    env = SingleCardEnv(cfg)
    
    agent_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gamma": 0.99,
        "batch_size": 32,
        "lr": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "buffer_size": 1000,
        "hidden_dims": [64, 64]
    }
    
    agent = DQNAgent(get_obs_shape()[0], ACTION_SPACE_SIZE, agent_cfg)
    
    returns = []
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask)
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store next_mask for replay buffer?
            # My replay buffer implementation stores `mask` (current).
            # Update expects `next_masks` for target calc.
            # So I should store `next_mask` as well?
            # Currently my `dqn_agent.step` takes `mask` argument.
            # Is it current mask or next mask?
            # `agent.act` used current mask.
            # `agent.step` adds to buffer.
            # The buffer `add` stores `mask`.
            # In `update`, `next_masks` are sampled.
            # Wait, if I store current mask, I only have mask for `obs`.
            # For `next_obs`, I need mask for `next_obs`.
            # My buffer `add` takes `mask` (singular).
            # If I only store current mask, how do I get next_mask?
            # The buffer usually stores `(s, a, r, s', d)`.
            # If I need masking for Q(s', a'), I need mask(s').
            # So I should store `next_mask` in the buffer?
            # Or just store `mask` and shift it? But `s'` might be terminal.
            # Let's assume I passed `next_mask` to `agent.step`?
            # Let's check `dqn_agent.step`.
            
            next_mask = next_info.get("mask")
            if next_mask is None:
                # If done, maybe mask is all False or irrelevant
                next_mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            
            # I'll modify agent.step to accept next_mask and store it as mask?
            # Standard DQN doesn't mask next actions, but here we must.
            # Let's verify `dqn_agent.py`.
            
            agent.step(obs, action, reward, next_obs, done, next_mask)
            
            obs = next_obs
            info = next_info
            ep_ret += reward
            
        returns.append(ep_ret)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{args.episodes} | Return: {ep_ret:.2f} | Avg10: {np.mean(returns[-10:]):.2f} | Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    main()

