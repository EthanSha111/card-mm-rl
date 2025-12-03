import numpy as np
import pytest
from mmrl.env.execution import get_max_sum, get_action_mask
from mmrl.env.events import Event, EVENT_NONE, EVENT_LE7, EVENT_REMAP_VALUE

def test_s_max():
    # None: 14+14+14 = 42
    ev = Event(type=EVENT_NONE)
    assert get_max_sum(ev) == 42
    
    # LE7: Max rank is 7. 7+7+7 = 21
    ev = Event(type=EVENT_LE7)
    assert get_max_sum(ev) == 21
    
    # Remap 2 to 50.
    # Max is 50+50+50 = 150 (since we have 4 twos)
    ev = Event(type=EVENT_REMAP_VALUE, params={"rank_from": 2, "value_to": 50})
    assert get_max_sum(ev) == 150
    
    # Remap 14 to 0.
    # Next max is 13. 13+13+13 = 39.
    ev = Event(type=EVENT_REMAP_VALUE, params={"rank_from": 14, "value_to": 0})
    assert get_max_sum(ev) == 39

def test_mask_buy():
    ev = Event(type=EVENT_NONE) # S_max=42
    W = 10.0
    Y = 2.0 # Price 2.0. Can buy 5.
    X = 10.0
    
    mask = get_action_mask(W, X, Y, ev)
    assert mask[0] # Pass
    # Buys 1..5 valid
    assert mask[1] and mask[5]
    # Buys 6..10 invalid
    assert not mask[6] and not mask[10]

def test_mask_sell():
    ev = Event(type=EVENT_NONE) # S_max=42
    W = 100.0
    Y = 20.0
    X = 32.0 # Risk = 42 - 32 = 10.
    # Can sell W / 10 = 10 units.
    
    mask = get_action_mask(W, X, Y, ev)
    assert mask[20] # Sell 10 is valid (index 20)
    
    # Lower W
    W = 10.0
    # Can sell 1 unit.
    mask = get_action_mask(W, X, Y, ev)
    assert mask[11] # Sell 1 valid
    assert not mask[12] # Sell 2 invalid

def test_mask_sell_risk_free():
    ev = Event(type=EVENT_NONE) # S_max=42
    X = 45.0 # Above S_max
    W = 1.0
    Y = 50.0
    
    mask = get_action_mask(W, X, Y, ev)
    # All sells valid
    assert all(mask[11:21])
    # Buys invalid (Y=50 > W=1)
    assert not any(mask[1:11])

