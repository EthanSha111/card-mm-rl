import numpy as np
import pytest
from mmrl.baselines.ev_oracle import act_ev_oracle

def test_ev_buy():
    # mu=30, Y=20. Edge=10.
    # X=10.
    obs = np.zeros(35, dtype=np.float32)
    obs[0] = 10.0 # X
    obs[1] = 20.0 # Y
    
    mask = np.ones(21, dtype=bool)
    info = {"mu": 30.0}
    
    action = act_ev_oracle(obs, mask, info)
    # Expect max buy size (10) -> action 10
    assert action == 10
    
    # With mask restriction
    mask[10] = False
    action = act_ev_oracle(obs, mask, info)
    assert action == 9

def test_ev_sell():
    # mu=10, X=20. Edge=10.
    obs = np.zeros(35, dtype=np.float32)
    obs[0] = 20.0 # X
    obs[1] = 30.0 # Y
    
    mask = np.ones(21, dtype=bool)
    info = {"mu": 10.0}
    
    action = act_ev_oracle(obs, mask, info)
    # Expect max sell size (10) -> action 20
    assert action == 20

def test_ev_pass():
    # mu=15, X=10, Y=20. No edge.
    obs = np.zeros(35, dtype=np.float32)
    obs[0] = 10.0
    obs[1] = 20.0
    
    mask = np.ones(21, dtype=bool)
    info = {"mu": 15.0}
    
    action = act_ev_oracle(obs, mask, info)
    assert action == 0

