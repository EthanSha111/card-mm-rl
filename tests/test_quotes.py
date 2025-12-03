import numpy as np
import pytest
from mmrl.env.quotes import make_quote

def test_quote_structure():
    mu, sigma = 20.0, 2.0
    cfg = {
        "sigma_q": 0.0, # No noise
        "spread": {"s0": 1.0, "beta": 0.5, "min": 0.5, "max": 5.0}
    }
    rng = np.random.RandomState(42)
    
    q = make_quote(mu, sigma, cfg, rng)
    
    assert np.isclose(q.mid, 20.0)
    # spread = 1.0 + 0.5 * 2.0 = 2.0
    assert np.isclose(q.spread, 2.0)
    assert np.isclose(q.bid, 19.0)
    assert np.isclose(q.ask, 21.0)
    assert np.isclose(q.ask - q.bid, q.spread)

def test_quote_noise():
    mu, sigma = 20.0, 1.0
    cfg = {
        "sigma_q": 1.0,
        "spread": {"s0": 0.0, "beta": 0.0, "min": 0.0, "max": 10.0}
    }
    rng = np.random.RandomState(42)
    
    mids = []
    for _ in range(1000):
        q = make_quote(mu, sigma, cfg, rng)
        mids.append(q.mid)
    
    # Mean should be close to mu
    assert np.abs(np.mean(mids) - mu) < 0.2 # Standard error is 1/sqrt(1000) approx 0.03

def test_spread_clipping():
    mu, sigma = 20.0, 100.0 # Huge sigma
    cfg = {
        "sigma_q": 0.0,
        "spread": {"s0": 1.0, "beta": 1.0, "min": 1.0, "max": 5.0}
    }
    rng = np.random.RandomState(42)
    
    # Spread should be clipped to max 5.0
    q = make_quote(mu, sigma, cfg, rng)
    assert np.isclose(q.spread, 5.0)
    
    # Small sigma -> clipped to min
    sigma_small = 0.0
    # raw = 1.0 + 0 = 1.0. Min is 2.0 say.
    cfg["spread"]["min"] = 2.0
    q2 = make_quote(mu, sigma_small, cfg, rng)
    assert np.isclose(q2.spread, 2.0)

