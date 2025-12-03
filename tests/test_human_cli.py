import pytest
import sys
from io import StringIO
from unittest.mock import patch
from mmrl.human.cli_play import run_game

def test_cli_single_player_quit():
    # Test that we can start and quit/finish a game
    # We mock input to provide "0" (Pass) for all steps
    
    # We need 10 inputs for 10 rounds
    inputs = ["0"] * 12 # Extra just in case
    
    with patch('builtins.input', side_effect=inputs):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            run_game(mode="single", events=False, impact=False)
            
            output = fake_out.getvalue()
            assert "ROUND 1/10" in output
            assert "GAME OVER" in output
            assert "Final Wealth" in output

def test_cli_two_player_random():
    inputs = ["0"] * 12
    with patch('builtins.input', side_effect=inputs):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            run_game(mode="two", events=False, impact=True, opponent_type="random")
            output = fake_out.getvalue()
            assert "Opponent Action" in output
            assert "GAME OVER" in output

