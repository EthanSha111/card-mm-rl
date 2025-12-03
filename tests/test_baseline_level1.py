import numpy as np
import pytest
from mmrl.baselines.level1_crowding import Level1Policy

def test_level1_no_history():
    # Behaves like Oracle
    # mu=30, Y=20. Edge=10. D=100.
    obs = np.zeros(35, dtype=np.float32)
    obs[0] = 10.0 # X
    obs[1] = 20.0 # Y
    obs[5] = 100.0 # D_ask
    
    mask = np.ones(21, dtype=bool)
    info = {"mu": 30.0}
    
    policy = Level1Policy()
    action = policy.act(obs, mask, info)
    assert action == 10 # Max buy

def test_level1_crowding_reduction():
    # Y=20, mu=22. Edge=2.
    # D_ask = 5.
    # Alpha = 0.3.
    # Opponent history: always Buy 5.
    
    obs = np.zeros(35, dtype=np.float32)
    obs[1] = 20.0 # Y
    obs[5] = 5.0 # D_ask
    
    mask = np.ones(21, dtype=bool)
    info = {"mu": 22.0}
    
    policy = Level1Policy(alpha=0.3)
    # Seed history
    for _ in range(10):
        policy.update(1, 5.0) # Buy 5
        
    # Prediction: q_opp = 5.0.
    # If I buy 10: Total=15. Overflow=10. Impact=3. Price=23. Edge = 22-23 = -1. BAD.
    # If I buy 1: Total=6. Overflow=1. Impact=0.3. Price=20.3. Edge = 1.7. GOOD.
    # Find max profitable size.
    # Limit: Price < 22.
    # 20 + 0.3 * (q + 5 - 5) < 22
    # 20 + 0.3 * q < 22
    # 0.3 * q < 2
    # q < 6.66.
    # So max valid size is 6.
    
    action = policy.act(obs, mask, info)
    assert action == 6

