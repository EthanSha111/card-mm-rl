import numpy as np
import pytest
from mmrl.env.liquidity import draw_true_depth, displayed_depth

def test_display_le_true():
    cfg = {"liquidity": {"display_cap": 5.0}}
    l_true = 10.0
    d = displayed_depth(l_true, cfg)
    assert d == 5.0
    
    l_true_small = 3.0
    d2 = displayed_depth(l_true_small, cfg)
    assert d2 == 3.0

def test_monotonicity():
    rng = np.random.RandomState(42)
    cfg = {"liquidity": {"k": 100.0, "tau": 0.1, "min": 0.0, "max": 1000.0}}
    
    # Case A: Low uncertainty
    sigma_low = 1.0
    spread = 1.0
    # L_bar = 100 / (1*1) = 100
    draws_low = [draw_true_depth(sigma_low, spread, cfg, rng) for _ in range(100)]
    avg_low = np.mean(draws_low)
    
    # Case B: High uncertainty
    sigma_high = 10.0
    # L_bar = 100 / (10*1) = 10
    draws_high = [draw_true_depth(sigma_high, spread, cfg, rng) for _ in range(100)]
    avg_high = np.mean(draws_high)
    
    assert avg_low > avg_high
    assert avg_low > 80.0 # close to 100
    assert avg_high < 20.0 # close to 10

def test_clipping():
    rng = np.random.RandomState(42)
    cfg = {"liquidity": {"k": 10.0, "min": 2.0, "max": 5.0}}
    
    # 1. Force high
    # sigma small, spread small -> L_bar huge
    sigma = 0.0001
    spread = 0.1
    l = draw_true_depth(sigma, spread, cfg, rng)
    assert l == 5.0
    
    # 2. Force low
    # sigma large
    sigma = 10000.0
    l_low = draw_true_depth(sigma, spread, cfg, rng)
    assert l_low == 2.0

