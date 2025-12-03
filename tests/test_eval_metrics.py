import pytest
import numpy as np
from mmrl.eval.evaluate import evaluate_policy

class MockAgent:
    def act(self, obs, mask, eval_mode=True):
        return 0 # Pass

class MockEnv:
    def __init__(self, cfg):
        self.action_space = type('obj', (object,), {'n': 21})
        self.cfg = cfg
        self.t = 0
    
    def reset(self, seed=None):
        self.t = 0
        return np.zeros(10), {"mask": np.ones(21, dtype=bool), "state": {"W": 500}}
        
    def step(self, action):
        self.t += 1
        done = self.t >= 2
        reward = 1.0
        return np.zeros(10), reward, done, False, {"slippage": 0.1, "mask": np.ones(21, dtype=bool)}

def test_metrics_calculation():
    cfg = {"stop_out": 0.2}
    metrics = evaluate_policy(
        lambda c: MockEnv(c),
        MockAgent(),
        cfg,
        n_episodes=5,
        seed=42
    )
    
    # 5 episodes. Each 2 steps. Reward 1.0 per step. Total 2.0 per episode.
    assert metrics["return_mean"] == 2.0
    assert metrics["return_std"] == 0.0 # All identical
    assert metrics["sharpe"] > 0 # 2 / small_std
    assert metrics["ruin_prob"] == 0.0
    assert metrics["mean_slippage"] == 0.1
