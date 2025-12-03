import numpy as np
import pytest
from mmrl.env.liquidity import exec_price_buy, exec_price_sell

def test_no_impact():
    Y = 20.0
    q = 10.0
    L = 5.0 # overflow
    alpha = 0.5
    
    p = exec_price_buy(Y, q, L, alpha, enable_impact=False)
    assert p == Y

def test_buy_under_capacity():
    Y = 20.0
    q = 4.0
    L = 5.0
    alpha = 0.5
    
    p = exec_price_buy(Y, q, L, alpha, enable_impact=True)
    assert p == Y

def test_buy_overflow():
    Y = 20.0
    q = 10.0
    L = 5.0
    alpha = 0.5
    
    # overflow = 5.0
    # price = 20 + 0.5 * 5 = 22.5
    p = exec_price_buy(Y, q, L, alpha, enable_impact=True)
    assert np.isclose(p, 22.5)

def test_sell_overflow():
    X = 20.0
    q = 10.0
    L = 5.0
    alpha = 0.5
    
    # overflow = 5.0
    # price = 20 - 0.5 * 5 = 17.5
    p = exec_price_sell(X, q, L, alpha, enable_impact=True)
    assert np.isclose(p, 17.5)

