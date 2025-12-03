import numpy as np
import pytest
from mmrl.env.events import (
    apply_event, sample_event, Event,
    EVENT_NONE, EVENT_GE10, EVENT_LE7, EVENT_EVEN, EVENT_ODD, EVENT_REMAP_VALUE
)
from mmrl.env.cards import make_deck, ALL_RANKS

def test_enable_events_false():
    rng = np.random.RandomState(42)
    cfg = {"flags": {"enable_events": False}}
    ev = sample_event(rng, cfg)
    assert ev.type == EVENT_NONE

def test_apply_none():
    deck = make_deck()
    ev = Event(type=EVENT_NONE)
    filtered, vmap = apply_event(deck, ev)
    assert filtered == deck
    for r in ALL_RANKS:
        assert vmap[r] == r

def test_apply_ge10():
    deck = make_deck()
    ev = Event(type=EVENT_GE10)
    filtered, vmap = apply_event(deck, ev)
    
    # Check logic
    assert len(filtered) > 0
    for rank in filtered:
        assert vmap[rank] >= 10
    
    # Check counts (should be 4 of each 10, J, Q, K, A -> 5 ranks * 4 = 20)
    assert len(filtered) == 20
    
def test_apply_even():
    deck = make_deck()
    ev = Event(type=EVENT_EVEN)
    filtered, vmap = apply_event(deck, ev)
    
    for rank in filtered:
        assert vmap[rank] % 2 == 0
    
    # Evens: 2, 4, 6, 8, 10, 12(Q), 14(A) -> 7 ranks * 4 = 28
    assert len(filtered) == 28

def test_apply_remap():
    deck = make_deck()
    # Remap 2 to 20
    ev = Event(type=EVENT_REMAP_VALUE, params={"rank_from": 2, "value_to": 20})
    filtered, vmap = apply_event(deck, ev)
    
    assert filtered == deck # Should not filter
    assert vmap[2] == 20
    assert vmap[3] == 3 # Unchanged

def test_sample_frequencies():
    rng = np.random.RandomState(123)
    cfg = {
        "flags": {"enable_events": True},
        "events": {
            "none": 0.0,
            "ge10_only": 1.0
        }
    }
    ev = sample_event(rng, cfg)
    assert ev.type == EVENT_GE10

def test_sample_remap_params():
    rng = np.random.RandomState(123)
    cfg = {
        "flags": {"enable_events": True},
        "events": {
            "remap_value": 1.0
        }
    }
    ev = sample_event(rng, cfg)
    assert ev.type == EVENT_REMAP_VALUE
    assert "rank_from" in ev.params
    assert "value_to" in ev.params
    assert ev.params["rank_from"] != ev.params["value_to"]

def test_persistence():
    rng = np.random.RandomState(123)
    # Setup config where persistence is 1.0
    cfg = {
        "flags": {"enable_events": True},
        "event_persist": 1.0,
        "events": {"none": 0.5, "ge10_only": 0.5}
    }
    
    last_ev = Event(type=EVENT_ODD) # Something not in distribution
    ev = sample_event(rng, cfg, last_event=last_ev)
    assert ev == last_ev
    
    # Persistence 0.0
    cfg["event_persist"] = 0.0
    ev2 = sample_event(rng, cfg, last_event=last_ev)
    assert ev2.type in ["none", "ge10_only"]

