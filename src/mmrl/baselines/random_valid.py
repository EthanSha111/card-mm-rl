import numpy as np
from typing import Any

def act_random_valid(
    obs: np.ndarray, 
    mask: np.ndarray, 
    rng: np.random.RandomState
) -> int:
    """
    Sample a random valid action.
    Args:
        obs: Observation vector (unused)
        mask: Boolean validity mask (21,)
        rng: Random state
    Returns:
        action index
    """
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return 0 # Pass if nothing valid (should not happen as Pass is always valid)
        
    return rng.choice(valid_indices)

class RandomValidAgent:
    def __init__(self, rng: np.random.RandomState = None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        
    def act(self, obs: np.ndarray, mask: np.ndarray, eval_mode: bool = True) -> int:
        return act_random_valid(obs, mask, self.rng)
