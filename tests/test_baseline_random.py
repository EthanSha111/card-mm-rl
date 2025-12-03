import numpy as np
import pytest
from mmrl.baselines.random_valid import act_random_valid

def test_random_valid_always_valid():
    rng = np.random.RandomState(42)
    
    # Mask: only 0, 5, 10 allowed
    mask = np.zeros(21, dtype=bool)
    mask[[0, 5, 10]] = True
    
    for _ in range(100):
        action = act_random_valid(None, mask, rng)
        assert action in [0, 5, 10]

def test_random_valid_distribution():
    rng = np.random.RandomState(42)
    mask = np.zeros(21, dtype=bool)
    mask[[0, 1]] = True
    
    actions = [act_random_valid(None, mask, rng) for _ in range(1000)]
    
    # Should be roughly 50/50
    counts = np.bincount(actions, minlength=2)
    assert 400 < counts[0] < 600
    assert 400 < counts[1] < 600

