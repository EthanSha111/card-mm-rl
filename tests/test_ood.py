import pytest
import os
import shutil
from mmrl.eval.ood_tests import run_ood_tests

class MockAgent:
    def act(self, obs, mask, eval_mode=True):
        return 0

class MockEnv:
    def __init__(self, cfg):
        self.action_space = type('obj', (object,), {'n': 21})
        self.cfg = cfg
        self.t = 0
    
    def reset(self, seed=None):
        self.t = 0
        return np.zeros(10), {"mask": np.ones(21, dtype=bool)}
        
    def step(self, action):
        self.t += 1
        done = self.t >= 1
        return np.zeros(10), 0.0, done, False, {"slippage": 0.0, "mask": np.ones(21, dtype=bool)}

import numpy as np

def test_ood_execution():
    out_dir = "test_ood_output"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    variations = {
        "param_a": [1, 2]
    }
    
    cfg = {"param_a": 0}
    
    df = run_ood_tests(
        lambda c: MockEnv(c),
        MockAgent(),
        cfg,
        variations,
        out_dir
    )
    
    # Baseline (1) + Variations (2) = 3 rows
    assert len(df) == 3
    assert os.path.exists(os.path.join(out_dir, "ood_summary.csv"))
    
    # Cleanup
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
