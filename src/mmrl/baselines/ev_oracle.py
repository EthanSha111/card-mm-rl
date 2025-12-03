import numpy as np
from typing import Optional, Dict, Any

def act_ev_oracle(
    obs: np.ndarray, 
    mask: np.ndarray, 
    info: Optional[Dict[str, Any]] = None
) -> int:
    """
    Level-0 EV Oracle.
    Uses mu from info (privileged) or computes it (not implemented here).
    """
    if info is None or "mu" not in info:
        # Fallback: acts randomly or passes?
        # For now, return 0 (Pass) if no info.
        return 0
        
    mu = info["mu"]
    
    # Extract Quote from obs
    # Obs 0: Bid(X), 1: Ask(Y)
    X = obs[0]
    Y = obs[1]
    
    # Check edge
    edge_buy = mu - Y
    edge_sell = X - mu
    
    best_action = 0
    max_edge = 0.0
    
    if edge_buy > 0 and edge_buy > edge_sell:
        # Buy side
        # Find max size
        # Buy actions 1..10
        for i in range(10, 0, -1):
            if mask[i]:
                return i
                
    elif edge_sell > 0 and edge_sell >= edge_buy:
        # Sell side
        # Sell actions 11..20 (size 1..10)
        for i in range(10, 0, -1):
            idx = 10 + i
            if mask[idx]:
                return idx
                
    return 0 # Pass

class EVOracleAgent:
    def act(self, obs: np.ndarray, mask: np.ndarray, info: Optional[Dict[str, Any]] = None, eval_mode: bool = True) -> int:
        return act_ev_oracle(obs, mask, info)
