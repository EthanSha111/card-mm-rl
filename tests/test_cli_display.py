import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from mmrl.human.cli_play import print_state

def test_print_state_remap_event():
    # Mock inputs
    info = {
        "event": {
            "type": "remap_value",
            "params": {"rank_from": 10, "value_to": 2}
        }
    }
    obs_info = {
        "hints": [],
        "quote_bid": 10.0,
        "quote_ask": 12.0,
        "depth_bid": 5.0,
        "depth_ask": 5.0,
        "W": 500.0
    }
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        print_state(info, obs_info, 0, 10)
        output = fake_out.getvalue()
        
        assert "EVENT: remap_value" in output
        assert "(Remap 10 -> 2)" in output

def test_print_state_none_event():
    info = {
        "event": {"type": "none"}
    }
    obs_info = {"hints": [], "quote_bid": 0, "quote_ask": 0, "depth_bid": 0, "depth_ask": 0, "W": 0}
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        print_state(info, obs_info, 0, 10)
        output = fake_out.getvalue()
        assert "EVENT: none" in output
        assert "Remap" not in output

