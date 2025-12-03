import numpy as np
import pytest
from itertools import combinations
from mmrl.env.quotes import posterior_mean_var
from mmrl.env.events import get_default_value_map

def brute_force_stats(deck_values, k_draw):
    # Enumerate all combinations of k_draw from deck_values
    sums = []
    for combo in combinations(deck_values, k_draw):
        sums.append(sum(combo))
    return np.mean(sums), np.std(sums)

def test_posterior_3_hints():
    hints = [2, 5, 10]
    deck = [2, 5, 10, 8, 9] # Must contain hints
    vmap = get_default_value_map()
    
    mu, sigma = posterior_mean_var(hints, deck, vmap)
    assert mu == 17.0
    assert sigma == 0.0

def test_posterior_small_deck_0_hints():
    # Deck: [2, 3, 4, 5]
    deck = [2, 3, 4, 5]
    hints = []
    vmap = get_default_value_map()
    
    mu, sigma = posterior_mean_var(hints, deck, vmap)
    
    # Brute force
    # Combos of 3: (2,3,4)=9, (2,3,5)=10, (2,4,5)=11, (3,4,5)=12
    # Mean = 10.5
    # Var = ((9-10.5)^2 + ...)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 5/4 = 1.25
    # Std = sqrt(1.25) approx 1.118
    
    assert np.isclose(mu, 10.5)
    assert np.isclose(sigma, np.sqrt(1.25))

def test_posterior_small_deck_1_hint():
    # Deck: [2, 3, 4, 5]
    # Hint: [2]
    # Remaining: [3, 4, 5]. Need to draw 2.
    # Combos of 2 from [3,4,5]: (3,4)=7, (3,5)=8, (4,5)=9.
    # Add hint 2: Sums = 9, 10, 11.
    # Mean = 10.
    # Var = ((9-10)^2 + 0 + 1)/3 = 2/3 approx 0.666
    # Std = sqrt(0.666)
    
    deck = [2, 3, 4, 5]
    hints = [2]
    vmap = get_default_value_map()
    
    mu, sigma = posterior_mean_var(hints, deck, vmap)
    
    assert np.isclose(mu, 10.0)
    assert np.isclose(sigma, np.sqrt(2/3))

def test_value_mapping():
    # Deck: [10, 10, 10] mapped to [0, 0, 0]
    deck = [10, 10, 10]
    hints = []
    vmap = {10: 0}
    
    mu, sigma = posterior_mean_var(hints, deck, vmap)
    assert mu == 0.0
    assert sigma == 0.0

