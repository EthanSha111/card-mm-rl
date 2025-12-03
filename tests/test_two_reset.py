import numpy as np
import pytest
from mmrl.env.two_player_env import TwoPlayerCardEnv

def test_two_reset_structure():
    cfg = {}
    env = TwoPlayerCardEnv(cfg)
    (obs_a, obs_b), info = env.reset(seed=42)
    
    assert obs_a.shape == (35,)
    assert obs_b.shape == (35,)
    
    # Check quote parts are same (first 4 elements)
    assert np.allclose(obs_a[:4], obs_b[:4])
    
    # Check opponent init (last 2 elements)
    assert np.allclose(obs_a[-2:], [0.0, 0.0])
    assert np.allclose(obs_b[-2:], [0.0, 0.0])
    
    assert "mask_a" in info
    assert "mask_b" in info

def test_hints_shared():
    cfg = {"hints": {"count": 3}}
    env = TwoPlayerCardEnv(cfg)
    (obs_a, obs_b), info = env.reset(seed=42)
    
    # Hints slice 10:23
    hints_a = obs_a[10:23]
    hints_b = obs_b[10:23]
    assert np.allclose(hints_a, hints_b)

