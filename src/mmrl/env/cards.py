import numpy as np
from typing import List, Tuple, Optional

# Constants for Ranks
TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN = 2, 3, 4, 5, 6, 7, 8, 9, 10
JACK, QUEEN, KING, ACE = 11, 12, 13, 14

ALL_RANKS = [
    TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN,
    JACK, QUEEN, KING, ACE
]

# Standard deck: 4 suits * 13 ranks = 52 cards
# We represent cards just by their rank integer since suits don't matter for value.
FULL_DECK = sorted(ALL_RANKS * 4)

def get_rank_str(rank: int) -> str:
    """Return string representation of a rank."""
    if 2 <= rank <= 9:
        return str(rank)
    mapping = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    return mapping.get(rank, '?')

def make_deck(ranks: Optional[List[int]] = None) -> List[int]:
    """
    Create a deck from a list of ranks. 
    If ranks is None, returns a standard 52-card deck (4 of each rank 2..14).
    """
    if ranks is None:
        return list(FULL_DECK)
    # If we want a specific set of ranks (e.g. 4 of each in the input list)
    # The requirement says 'eligible deck (filtered by event)'.
    # So we might just pass the full deck and filter it.
    # This function might just be a helper to get a fresh full deck.
    return list(FULL_DECK)

def draw_cards(deck: List[int], n: int, rng: np.random.RandomState) -> Tuple[List[int], List[int]]:
    """
    Draw n cards from the deck without replacement.
    Returns: (drawn_cards, remaining_deck)
    """
    if n > len(deck):
        raise ValueError(f"Cannot draw {n} cards from deck of size {len(deck)}")
    
    # We operate on indices to handle duplicate rank values correctly (standard deck has duplicates)
    indices = rng.choice(len(deck), size=n, replace=False)
    drawn = [deck[i] for i in indices]
    
    # To get remaining deck, we need to remove the specific instances we drew
    # Since we have indices, we can do this efficiently.
    # Note: rng.choice with replace=False returns unique indices.
    
    # Faster way: mask
    mask = np.ones(len(deck), dtype=bool)
    mask[indices] = False
    
    # remaining = [deck[i] for i in range(len(deck)) if mask[i]]
    # But we want to return lists
    drawn = sorted(drawn) # Sort drawn cards for canonical representation usually? 
    # Actually, "hidden multiset" -> order doesn't matter for sum.
    # Let's just return them.
    
    remaining = []
    # Make a mutable copy to remove
    temp_deck = list(deck)
    # Removing by value is risky if we don't match the specific instances drawn
    # But since cards are identical by rank, removing *a* card of that rank is equivalent to removing *the* card.
    # However, using indices is cleaner if we want to simulate physical draw.
    
    # Let's just use the mask approach on the list
    remaining = [deck[i] for i in range(len(deck)) if mask[i]]
    
    return drawn, remaining

def calculate_sum(cards: List[int]) -> int:
    return sum(cards)

