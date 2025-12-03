import numpy as np
import pytest
from mmrl.env.cards import make_deck, draw_cards, calculate_sum, FULL_DECK

def test_deck_composition():
    deck = make_deck()
    assert len(deck) == 52
    # Check counts: 4 of each rank 2..14
    for r in range(2, 15):
        assert deck.count(r) == 4

def test_draw_cards_structure():
    rng = np.random.RandomState(42)
    deck = make_deck()
    drawn, remaining = draw_cards(deck, 3, rng)
    
    assert len(drawn) == 3
    assert len(remaining) == 49
    
    # Check that drawn + remaining contains same elements as original deck
    combined = drawn + remaining
    combined.sort()
    original = list(deck)
    original.sort()
    assert combined == original

def test_draw_values():
    rng = np.random.RandomState(123)
    deck = make_deck()
    for _ in range(100):
        drawn, _ = draw_cards(deck, 3, rng)
        s = calculate_sum(drawn)
        # Min sum: 2+2+2=6 (actually 2,2,2 is possible since we have 4 twos)
        # Max sum: 14+14+14=42
        assert 6 <= s <= 42
        for card in drawn:
            assert 2 <= card <= 14

def test_draw_without_replacement():
    # Create a small deck to force collision if replacement was used
    # e.g. deck = [10, 11, 12]
    deck = [10, 11, 12]
    rng = np.random.RandomState(1)
    drawn, remaining = draw_cards(deck, 3, rng)
    assert len(remaining) == 0
    assert sorted(drawn) == [10, 11, 12]
    
    # Try to draw 4 from 3
    with pytest.raises(ValueError):
        draw_cards(deck, 4, rng)

