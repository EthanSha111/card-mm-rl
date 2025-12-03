import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from mmrl.env.cards import calculate_sum

def posterior_mean_var(
    hints: List[int], 
    filtered_deck: List[int], 
    value_map: Dict[int, int]
) -> Tuple[float, float]:
    """
    Compute posterior mean (mu) and standard deviation (sigma) of the sum of 3 cards,
    given the revealed hints and the eligible deck (after event filtering).
    
    Args:
        hints: list of ranks revealed (subset of the 3 hidden cards).
        filtered_deck: list of ranks available to be drawn from.
        value_map: mapping from rank to value (for the current event).
        
    Returns:
        mu: Expected sum.
        sigma: Standard deviation of sum.
    """
    # 1. Map deck and hints to values
    # Note: hints are specific cards drawn. We must remove them from the deck to get the population
    # for the remaining draws.
    
    # Identify indices of hints in filtered_deck to remove them.
    # Since deck might have duplicates, we just need to remove one instance per hint.
    
    remaining_deck_values = []
    
    # Create a frequency map or copy list to remove
    temp_deck = list(filtered_deck)
    
    hints_sum = 0
    for h in hints:
        val = value_map[h]
        hints_sum += val
        
        # Remove h from temp_deck
        try:
            temp_deck.remove(h)
        except ValueError:
            # This should not happen in a valid game state
            # But if hints contains a card not in deck, we have an issue.
            # For now assume valid.
            pass
            
    remaining_deck_values = [value_map[r] for r in temp_deck]
    
    num_total_cards = 3
    num_hints = len(hints)
    m = num_total_cards - num_hints
    
    if m == 0:
        return float(hints_sum), 0.0
    
    if len(remaining_deck_values) < m:
        # Not enough cards to draw. Should not happen.
        # Return best guess
        return float(hints_sum), 0.0

    # Population statistics
    pop = np.array(remaining_deck_values, dtype=np.float64)
    N = len(pop)
    
    pop_mean = np.mean(pop)
    pop_var = np.var(pop) # by default numpy var is population variance (ddof=0)
    
    # Expected sum of m cards
    expected_rem_sum = m * pop_mean
    mu = hints_sum + expected_rem_sum
    
    # Variance of sum of m cards without replacement
    # Var(Sum) = m * Var(Population) * (N - m) / (N - 1)
    if N > 1:
        fpc = (N - m) / (N - 1)
        var_rem_sum = m * pop_var * fpc
    else:
        var_rem_sum = 0.0
        
    sigma = np.sqrt(var_rem_sum)
    
    return mu, sigma

@dataclass
class Quote:
    mid: float
    spread: float
    bid: float
    ask: float

def make_quote(
    mu: float, 
    sigma: float, 
    cfg: Any, 
    rng: np.random.RandomState
) -> Quote:
    """
    Generate a quote based on posterior moments and config.
    
    Args:
        mu: posterior mean of sum S.
        sigma: posterior std dev of sum S.
        cfg: config object with:
            - sigma_q: float (noise std dev)
            - spread: dict/object with s0, beta, min, max.
        rng: random number generator.
        
    Returns:
        Quote object with mid, spread, bid, ask.
    """
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # 1. Mid price
    sigma_q = get_val(cfg, "sigma_q", 0.5)
    noise = rng.normal(0, sigma_q)
    mid = mu + noise
    
    # 2. Spread
    # spread_t = clip(s0 + beta * sigma_{S,t}, s_min, s_max)
    spread_cfg = get_val(cfg, "spread", {})
    s0 = get_val(spread_cfg, "s0", 0.8)
    beta = get_val(spread_cfg, "beta", 0.25)
    s_min = get_val(spread_cfg, "min", 1.0)
    s_max = get_val(spread_cfg, "max", 3.0)
    
    raw_spread = s0 + beta * sigma
    spread = np.clip(raw_spread, s_min, s_max)
    
    # 3. Bid/Ask
    # X = mid - spread/2, Y = mid + spread/2
    bid = mid - spread / 2.0
    ask = mid + spread / 2.0
    
    return Quote(mid=mid, spread=spread, bid=bid, ask=ask)
