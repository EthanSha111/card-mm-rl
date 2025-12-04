import pytest
import numpy as np
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.env.events import EVENT_NONE

def test_event_config_forcing():
    # Config mirroring cli_play.py structure
    cfg = {
        "W0": 500.0,
        "episode_length": 10,
        "event_persist": 0.0,
        "events": {
            "freq": {
                "none": 0.0,       # Force 0.0
                "ge10_only": 0.25,
                "le7_only": 0.25,
                "even_only": 0.25,
                "remap_value": 0.25 
            }
        },
        "flags": {
            "enable_events": True,
            "enable_impact": True
        }
    }
    
    env = TwoPlayerCardEnv(cfg)
    
    # Test multiple resets to ensure we don't get 'none'
    for i in range(20):
        env.reset(seed=i)
        event_type = env.current_event.type
        print(f"Reset {i}: {event_type}")
        
        # Assert that we NEVER get 'none' because prob is 0.0
        assert event_type != EVENT_NONE, \
            f"Got 'none' event despite 0.0 probability! Config passed: {cfg['events']}"

def test_event_config_flat():
    # Test checking if flat config works (old style)
    cfg = {
        "W0": 500.0,
        "episode_length": 10,
        "event_persist": 0.0,
        "events": {
            "none": 0.0,
            "ge10_only": 1.0
        },
        "flags": {"enable_events": True}
    }
    env = TwoPlayerCardEnv(cfg)
    env.reset()
    assert env.current_event.type == "ge10_only"

