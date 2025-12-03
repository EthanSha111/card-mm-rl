import copy
import pandas as pd
import os
from typing import Dict, Any, Callable, List

from mmrl.eval.evaluate import evaluate_policy

def run_ood_tests(
    env_factory: Callable,
    agent: Any,
    base_cfg: Dict[str, Any],
    variations: Dict[str, List[Any]],
    output_dir: str
):
    """
    Run evaluation across multiple configuration variations (OOD tests).
    
    Args:
        env_factory: Function to make env
        agent: Policy to test
        base_cfg: Base configuration
        variations: Dict mapping config key (dot-separated) to list of values to test
        output_dir: Where to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    
    # Baseline run
    print("Running Baseline...")
    metrics = evaluate_policy(env_factory, agent, base_cfg, n_episodes=100)
    metrics["variation"] = "baseline"
    metrics["param"] = "none"
    metrics["value"] = "none"
    summary.append(metrics)
    
    for param_path, values in variations.items():
        for val in values:
            print(f"Running OOD: {param_path} = {val}")
            
            # Clone config
            test_cfg = copy.deepcopy(base_cfg)
            
            # Set value by path
            keys = param_path.split(".")
            target = test_cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = val
            
            # Eval
            res = evaluate_policy(env_factory, agent, test_cfg, n_episodes=100)
            res["variation"] = param_path
            res["param"] = param_path
            res["value"] = val
            summary.append(res)
            
    # Save summary
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_dir, "ood_summary.csv"), index=False)
    print(f"OOD results saved to {output_dir}")
    return df

if __name__ == "__main__":
    from mmrl.baselines.random_valid import RandomValidAgent
    from mmrl.env.single_env import SingleCardEnv
    
    cfg = {
        "W0": 500.0,
        "episode_length": 10,
        "flags": {"enable_events": False, "enable_impact": True},
        "alpha": 0.3,
        "spread": {"beta": 0.25}
    }
    
    agent = RandomValidAgent()
    
    def make_env(c): return SingleCardEnv(c)
    
    variations = {
        "alpha": [0.0, 0.5, 1.0], # Impact scaling
        "spread.beta": [0.1, 0.5, 1.0], # Spread width
    }
    
    run_ood_tests(make_env, agent, cfg, variations, "ood_results")
