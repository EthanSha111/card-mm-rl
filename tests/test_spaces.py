import numpy as np
import pytest
from mmrl.env.spaces import get_obs_shape, build_obs, EVENT_TYPES
from mmrl.env.events import Event, EVENT_GE10
from mmrl.env.quotes import Quote

def test_obs_shape():
    assert get_obs_shape() == (35,)

def test_build_obs():
    # Mock inputs
    quote = Quote(mid=20.0, spread=2.0, bid=19.0, ask=21.0)
    depths = (5.0, 5.0)
    metrics = {"slippage_bid": 0.1, "slippage_ask": 0.0, "fill_ratio_bid": 1.0, "fill_ratio_ask": 0.0}
    hints = [2, 2, 14] # Two 2s, one Ace
    event = Event(type=EVENT_GE10)
    state = {"W": 500.0, "W0": 500.0, "t": 1, "T": 10}
    flags = {"enable_events": True, "enable_impact": False}
    
    obs = build_obs(quote, depths, metrics, hints, event, state, flags)
    
    assert obs.shape == (35,)
    assert obs.dtype == np.float32
    
    # Check values
    # Quote: 19, 21, 20, 2
    assert np.allclose(obs[0:4], [19.0, 21.0, 20.0, 2.0])
    
    # Depths
    assert np.allclose(obs[4:6], [5.0, 5.0])
    
    # Metrics
    assert np.allclose(obs[6:10], [0.1, 0.0, 1.0, 0.0])
    
    # Hints: 2 is at index 0 (2-2=0). 14 is at index 12 (14-2=12).
    # hints[0] should be 2.0, hints[12] should be 1.0.
    hints_slice = obs[10:23]
    assert hints_slice[0] == 2.0
    assert hints_slice[12] == 1.0
    assert np.sum(hints_slice) == 3.0
    
    # Event: GE10 is index 1 in EVENT_TYPES
    # EVENT_TYPES = [EVENT_NONE, EVENT_GE10, ...]
    ev_slice = obs[23:29]
    assert ev_slice[1] == 1.0
    assert np.sum(ev_slice) == 1.0
    
    # State
    # W/W0 = 1.0, t/T = 0.1
    assert np.allclose(obs[29:31], [1.0, 0.1])
    
    # Flags
    # events=1, impact=0
    assert np.allclose(obs[31:33], [1.0, 0.0])
    
    # Opponent (default 0)
    assert np.allclose(obs[33:35], [0.0, 0.0])

def test_opponent_obs():
    quote = Quote(mid=0, spread=0, bid=0, ask=0)
    depths = (0,0)
    metrics = {}
    hints = []
    event = Event(type="none")
    state = {"W": 100, "W0": 100, "t":0, "T":10}
    flags = {}
    
    opp = {"side": -1.0, "size": 5.0}
    obs = build_obs(quote, depths, metrics, hints, event, state, flags, opponent_last=opp)
    
    assert np.allclose(obs[33:35], [-1.0, 5.0])

