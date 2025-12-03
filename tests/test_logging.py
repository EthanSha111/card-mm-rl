import pytest
from mmrl.env.single_env import SingleCardEnv

def test_logging_hook():
    cfg = {}
    env = SingleCardEnv(cfg)
    env.reset(seed=42)
    
    # Step
    obs, reward, terminated, truncated, info = env.step(0) # Pass
    
    assert "log" in info
    log = info["log"]
    
    expected_keys = ["t", "W", "action", "reward", "exec_price", "slippage", "true_sum", "mu", "sigma", "mask", "event_type", "L_bid", "L_ask"]
    for k in expected_keys:
        assert k in log
        
    # Check values
    assert log["action"] == 0
    assert log["reward"] == 0.0
    assert log["t"] == 0

