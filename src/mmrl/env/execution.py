import numpy as np
from typing import List, Tuple, Any
from mmrl.env.cards import ALL_RANKS
from mmrl.env.events import Event, EVENT_NONE, EVENT_REMAP_VALUE, EVENT_LE7, EVENT_GE10, EVENT_ODD, EVENT_EVEN, apply_event

# Action constants
ACTION_PASS = 0
# Buy 1..10 mapped to 1..10
# Sell 1..10 mapped to 11..20

def get_max_sum(event: Event) -> int:
    """
    Calculate S_max possible under the given event.
    Assumes standard deck counts (4 of each).
    """
    # We can use apply_event to get the value map and eligible ranks.
    # Then pick top 3.
    
    # Full deck ranks
    # We don't need the full deck list, just the distinct ranks and their counts.
    # Since apply_event returns a filtered list of *all* cards in deck, we can just sort it.
    
    # Construct a full deck to filter
    # 4 of each rank 2..14
    full_deck = sorted(ALL_RANKS * 4)
    
    filtered_deck, value_map = apply_event(full_deck, event)
    
    if len(filtered_deck) < 3:
        # Should not happen in valid game, but return sum of all
        vals = [value_map[r] for r in filtered_deck]
        return sum(vals)
    
    # Map to values
    vals = [value_map[r] for r in filtered_deck]
    # Sort descending
    vals.sort(reverse=True)
    # Take top 3
    return sum(vals[:3])

def get_action_mask(
    W: float, 
    X: float, 
    Y: float, 
    event: Event
) -> np.ndarray:
    """
    Returns boolean mask of size 21.
    mask[0] = True (Pass always allowed)
    mask[1..10] = True if Buy i feasible
    mask[11..20] = True if Sell i feasible
    """
    mask = np.zeros(21, dtype=bool)
    mask[0] = True # Pass
    
    S_max = get_max_sum(event)
    
    # Buy 1..10
    # Constraint: i * Y <= W
    # i <= W / Y
    # If Y <= 0 (should not happen normally), unbounded.
    # Assuming Y > 0.
    if Y > 1e-9:
        max_buy = int(W / Y)
    else:
        max_buy = 10 # unlimited
    
    for i in range(1, 11):
        if i <= max_buy:
            mask[i] = True
            
    # Sell 1..10 (Indices 11..20 correspond to sizes 1..10)
    # Constraint: i * (S_max - X) <= W
    # Risk per unit: R = S_max - X.
    # If R <= 0 (X >= S_max), risk is zero (profit guaranteed or break even), unlimited size.
    R = S_max - X
    if R > 1e-9:
        max_sell = int(W / R)
    else:
        max_sell = 10
        
    for i in range(1, 11):
        idx = 10 + i
        if i <= max_sell:
            mask[idx] = True
            
    return mask

