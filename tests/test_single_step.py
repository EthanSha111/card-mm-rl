import numpy as np
import pytest
from mmrl.env.single_env import SingleCardEnv
from mmrl.env.quotes import Quote
from mmrl.env.events import Event, EVENT_NONE

class MockSingleCardEnv(SingleCardEnv):
    def reset_mock(self, S, X, Y, L_bid, L_ask, alpha=0.0, enable_impact=False, W=500.0):
        self.t = 0
        self.W = W
        self.W0 = W
        self.T = 10
        self.true_sum = S
        self.quote = Quote(mid=(X+Y)/2, spread=Y-X, bid=X, ask=Y)
        self.true_depths = (L_bid, L_ask)
        self.disp_depths = (L_bid, L_ask)
        self.current_event = Event(type=EVENT_NONE)
        self.metrics = {}
        # Override config for impact
        self.cfg = {
            "flags": {"enable_impact": enable_impact},
            "alpha": alpha,
            "stop_out": 0.2
        }

def test_step_buy_profit():
    env = MockSingleCardEnv({})
    # S=30, X=24, Y=26. Spread=2.
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=10)
    
    # Action: Buy 1 (idx 1)
    obs, reward, terminated, truncated, info = env.step(1)
    
    assert reward == 4.0 # 1 * (30 - 26)
    assert env.W == 504.0
    assert not terminated

def test_step_sell_loss():
    env = MockSingleCardEnv({})
    # S=30, X=24, Y=26
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=10)
    
    # Action: Sell 1 (idx 11)
    obs, reward, terminated, truncated, info = env.step(11)
    
    # Reward = 1 * (24 - 30) = -6
    assert reward == -6.0
    assert env.W == 494.0

def test_step_impact():
    env = MockSingleCardEnv({})
    # S=30, Y=26, L_ask=2. Alpha=1.
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=2, alpha=1.0, enable_impact=True)
    
    # Action: Buy 4 (idx 4)
    # Overflow = 4 - 2 = 2.
    # Price = 26 + 1 * 2 = 28.
    # Reward = 4 * (30 - 28) = 4 * 2 = 8.
    
    obs, reward, terminated, truncated, info = env.step(4)
    
    assert reward == 8.0
    assert info["exec_price"] == 28.0
    assert info["slippage"] == 2.0 # 28 - 26

def test_termination_time():
    env = MockSingleCardEnv({})
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=10)
    env.T = 1
    
    obs, reward, terminated, truncated, info = env.step(0) # Pass
    assert terminated

def test_termination_ruin():
    env = MockSingleCardEnv({})
    env.reset_mock(S=100, X=20, Y=22, L_bid=10, L_ask=10)
    # Sell 10. Reward = 10 * (20 - 100) = -800.
    # W = 100 - 800 = -700.
    # Stop out is 20.
    
    obs, reward, terminated, truncated, info = env.step(20) # Sell 10
    assert terminated
    assert env.W == -300.0

