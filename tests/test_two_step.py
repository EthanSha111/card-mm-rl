import numpy as np
import pytest
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.env.quotes import Quote
from mmrl.env.events import Event, EVENT_NONE

class MockTwoPlayerEnv(TwoPlayerCardEnv):
    def reset_mock(self, S, X, Y, L_bid, L_ask, alpha=0.0, enable_impact=False, W=500.0):
        self.t = 0
        self.W_a = W
        self.W_b = W
        self.W0 = W
        self.T = 10
        self.true_sum = S
        self.quote = Quote(mid=(X+Y)/2, spread=Y-X, bid=X, ask=Y)
        self.true_depths = (L_bid, L_ask)
        self.disp_depths = (L_bid, L_ask)
        self.current_event = Event(type=EVENT_NONE)
        self.hints_a = []
        self.hints_b = []
        self.metrics_a = {}
        self.metrics_b = {}
        self.last_act_a = {"side": 0.0, "size": 0.0}
        self.last_act_b = {"side": 0.0, "size": 0.0}
        self.mu = 0.0
        self.sigma = 0.0
        
        self.cfg = {
            "flags": {"enable_impact": enable_impact},
            "alpha": alpha,
            "stop_out": 0.2,
            "hints": {"count": 0}
        }

def test_crowding_impact():
    env = MockTwoPlayerEnv({})
    # S=30, Y=26. L_ask=5. Alpha=1.0.
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=5, alpha=1.0, enable_impact=True)
    
    # A Buys 5 (idx 5), B Buys 5 (idx 5)
    # Total Buy = 10. Overflow = 10 - 5 = 5.
    # Price = 26 + 1 * 5 = 31.
    # Reward = 5 * (30 - 31) = -5 each.
    
    (obs, _), (r_a, r_b), term, trunc, info = env.step((5, 5))
    
    assert r_a == -5.0
    assert r_b == -5.0
    assert info["exec_price_buy"] == 31.0

def test_offsetting_flow_no_impact():
    env = MockTwoPlayerEnv({})
    # L=5.
    env.reset_mock(S=30, X=24, Y=26, L_bid=5, L_ask=5, alpha=1.0, enable_impact=True)
    
    # A Buys 5 (idx 5), B Sells 5 (idx 15)
    # q_buy=5 (<= L), q_sell=5 (<= L).
    # Price Buy = 26. Price Sell = 24.
    # Reward A = 5 * (30 - 26) = 20.
    # Reward B = 5 * (24 - 30) = -30.
    
    (obs, _), (r_a, r_b), term, trunc, info = env.step((5, 15))
    
    assert r_a == 20.0
    assert r_b == -30.0
    assert info["exec_price_buy"] == 26.0
    assert info["exec_price_sell"] == 24.0

def test_opponent_observation():
    env = MockTwoPlayerEnv({})
    env.reset_mock(S=30, X=24, Y=26, L_bid=10, L_ask=10)
    
    # A Buys 2 (side 1, size 2). B Sells 3 (side -1, size 3).
    # Next obs_a should see B: side -1, size 3.
    # Next obs_b should see A: side 1, size 2.
    
    (obs_a, obs_b), _, _, _, _ = env.step((2, 13)) # 13 is Sell 3
    
    # Opponent slice is last 2 dims
    opp_a = obs_a[-2:] # Should be B
    opp_b = obs_b[-2:] # Should be A
    
    assert np.allclose(opp_a, [-1.0, 3.0])
    assert np.allclose(opp_b, [1.0, 2.0])

