import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Any, List, Optional, Union, Callable
import torch
import os
import yaml

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv

def evaluate_policy(
    env_factory: Callable[[Dict[str, Any]], gym.Env],
    agent: Any,
    cfg: Dict[str, Any],
    n_episodes: int = 100,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a policy (agent) on an environment.
    
    Args:
        env_factory: Function to create env given config
        agent: Agent with act(obs, mask, eval_mode=True) method
        cfg: Environment config
        n_episodes: Number of episodes to run
        seed: Base seed
        output_dir: If provided, save detailed results to CSV here
    
    Returns:
        Dict of aggregate metrics
    """
    
    # Set up recording
    episode_returns = []
    episode_lens = []
    episode_final_wealth = []
    
    # Action stats
    valid_actions = 0
    total_actions = 0
    pass_actions = 0
    
    # Execution stats
    slippage_list = []
    fill_ratios = []
    impact_costs = []
    
    rng = np.random.RandomState(seed)
    
    # Create env
    env = env_factory(cfg)
    
    detailed_records = []
    
    for i in range(n_episodes):
        episode_seed = int(rng.randint(0, 1e9))
        obs, info = env.reset(seed=episode_seed)
        
        ep_ret = 0.0
        ep_len = 0
        done = False
        
        # For MDD calculation within episode (if needed, or use wealth curve)
        wealth_curve = [info.get("state", {}).get("W", 500.0)] # Try to get initial wealth from info or assume W0
        # Actually SingleEnv info doesn't have 'state' key in the reset return usually?
        # Let's check single_env.py: reset returns obs, info. info has mu, sigma, event, hidden_cards, true_sum, true_depths, mask.
        # Obs has state info (W, W0, t, T).
        # We can extract W from obs if we had the encoder. Or simpler, just track cumulative reward + W0.
        
        W0 = getattr(env, "W0", 500.0)
        current_wealth = W0
        
        while not done:
            # Extract mask from info if available (SingleEnv puts it in info)
            # TwoPlayerEnv puts it in info per agent?
            # Assume Single Agent for now based on signature
            
            mask = info.get("mask")
            if mask is None:
                # Fallback if not in info (should be there)
                mask = np.ones(env.action_space.n, dtype=bool)
                
            # Agent act
            if hasattr(agent, "act"):
                action = agent.act(obs, mask, eval_mode=True)
            else:
                # Baseline might be function or class
                action = agent(obs, mask)
            
            # Count validity (agent should only pick valid)
            if mask[action]:
                valid_actions += 1
            else:
                # If agent picked invalid, env will likely force pass or error. 
                # We count it as invalid attempt.
                pass 
            
            if action == 0:
                pass_actions += 1
                
            total_actions += 1
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_ret += reward
            ep_len += 1
            
            current_wealth += reward
            wealth_curve.append(current_wealth)
            
            # Track execution metrics from info
            # SingleEnv info: exec_price, slippage, true_sum, action, budget
            if "slippage" in info:
                slippage_list.append(info["slippage"])
            
            # Two player might differ, handle later or assume single for now
            
            # Detailed logging
            if output_dir:
                detailed_records.append({
                    "episode": i,
                    "step": ep_len,
                    "action": action,
                    "reward": reward,
                    "wealth": current_wealth,
                    "slippage": info.get("slippage", 0.0),
                    "exec_price": info.get("exec_price", 0.0),
                    "true_sum": info.get("true_sum", 0.0)
                })

        episode_returns.append(ep_ret)
        episode_lens.append(ep_len)
        episode_final_wealth.append(current_wealth)
        
    # Compute Aggregates
    returns = np.array(episode_returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / (std_ret + 1e-8) # Simplified Sharpe per episode
    
    # Max Drawdown (per episode, then mean? or over equity curve of sequence?)
    # Usually MDD is over a trajectory. Here we have independent episodes.
    # We can report mean MDD per episode.
    
    # Ruin probability: Fraction of episodes ending with W < threshold or truncated by ruin
    # env has stop_out. If terminated and W < W0*stop_out
    stop_out = getattr(env, "_get_cfg", lambda k, d: d)("stop_out", 0.2)
    ruin_count = sum(1 for w in episode_final_wealth if w < W0 * stop_out * 1.01) # tolerance
    prob_ruin = ruin_count / n_episodes
    
    metrics = {
        "return_mean": float(mean_ret),
        "return_std": float(std_ret),
        "sharpe": float(sharpe),
        "ruin_prob": float(prob_ruin),
        "valid_rate": float(valid_actions / total_actions) if total_actions > 0 else 0.0,
        "pass_rate": float(pass_actions / total_actions) if total_actions > 0 else 0.0,
        "mean_slippage": float(np.mean(slippage_list)) if slippage_list else 0.0,
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(detailed_records)
        df.to_csv(os.path.join(output_dir, "eval_details.csv"), index=False)
        
        # Save summary
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)
            
    return metrics

if __name__ == "__main__":
    # Simple test harness
    from mmrl.baselines.random_valid import RandomValidAgent
    from mmrl.env.single_env import SingleCardEnv
    
    cfg = {
        "W0": 500.0,
        "episode_length": 10,
        "flags": {"enable_events": False, "enable_impact": True}
    }
    
    agent = RandomValidAgent()
    
    def make_env(c): return SingleCardEnv(c)
    
    print("Running eval...")
    m = evaluate_policy(make_env, agent, cfg, n_episodes=10)
    print(m)
