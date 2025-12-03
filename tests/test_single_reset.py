import numpy as np
import pytest
from mmrl.env.single_env import SingleCardEnv
from mmrl.env.events import EVENT_NONE

def test_reset_structure():
    cfg = {}
    env = SingleCardEnv(cfg)
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (35,)
    assert isinstance(info, dict)
    assert "mu" in info
    assert "sigma" in info
    assert "mask" in info
    assert len(info["mask"]) == 21

def test_reset_determinism():
    cfg = {}
    env = SingleCardEnv(cfg)
    
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    
    assert np.allclose(obs1, obs2)
    assert info1["mu"] == info2["mu"]
    assert info1["true_sum"] == info2["true_sum"]
    
    obs3, info3 = env.reset(seed=999)
    assert not np.allclose(obs1, obs3)

def test_events_toggle():
    # Disable events
    cfg = {"flags": {"enable_events": False}}
    env = SingleCardEnv(cfg)
    
    # Run multiple resets to ensure we don't get random events
    for i in range(10):
        obs, info = env.reset(seed=i)
        assert info["event"]["type"] == EVENT_NONE

    # Enable events with ge10 only
    cfg = {
        "flags": {"enable_events": True},
        "events": {"none": 0.0, "ge10_only": 1.0}
    }
    env = SingleCardEnv(cfg)
    obs, info = env.reset(seed=42)
    assert info["event"]["type"] == "ge10_only"

