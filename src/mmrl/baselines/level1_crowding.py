import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque
from mmrl.baselines.ev_oracle import act_ev_oracle

class Level1Policy:
    def __init__(self, history_len: int = 10, alpha: float = 0.3):
        self.history_len = history_len
        self.alpha = alpha
        # History of opponent actions: list of (side, size)
        self.opp_history = deque(maxlen=history_len)
        
    def update(self, opp_side: int, opp_size: float):
        """
        Update belief about opponent based on last round.
        opp_side: -1, 0, 1
        """
        self.opp_history.append((opp_side, opp_size))
        
    def _estimate_opp_demand(self, my_side: int) -> float:
        if len(self.opp_history) == 0:
            return 0.0
            
        # Simple frequency of opponent being on the same side
        count_same = sum(1 for s, z in self.opp_history if s == my_side)
        prob = count_same / len(self.opp_history)
        
        # Avg size when on that side
        sizes = [z for s, z in self.opp_history if s == my_side]
        avg_size = np.mean(sizes) if sizes else 5.0 # Default guess
        
        return prob * avg_size

    def act(self, obs: np.ndarray, mask: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        # 1. Get Level-0 proposal (max size)
        # We use the same logic as EV Oracle to find direction and max feasible size
        # But we don't want to just call act_ev_oracle because we need to iterate sizes.
        
        if info is None or "mu" not in info:
            return 0
            
        mu = info["mu"]
        X = obs[0]
        Y = obs[1]
        
        # Displayed depths
        D_bid = obs[4]
        D_ask = obs[5]
        
        # Determine direction
        edge_buy = mu - Y
        edge_sell = X - mu
        
        side = 0
        if edge_buy > 0 and edge_buy > edge_sell:
            side = 1
        elif edge_sell > 0 and edge_sell >= edge_buy:
            side = -1
        else:
            return 0
            
        # Estimate opponent crowding
        q_opp = self._estimate_opp_demand(side)
        
        # Iterate sizes downwards from max feasible
        # Buy: 1..10. Sell: 1..10 (actions 11..20)
        
        best_action = 0
        
        if side == 1:
            # Buy
            # Check sizes 10 down to 1
            for i in range(10, 0, -1):
                if mask[i]:
                    q_me = float(i)
                    q_total = q_me + q_opp
                    
                    # Est impact against Displayed Depth
                    # Note: True depth might be higher, but we risk averse on displayed.
                    overflow = max(0.0, q_total - D_ask)
                    p_est = Y + self.alpha * overflow
                    
                    # Check edge
                    if mu - p_est > 0:
                        return i
                        
        else:
            # Sell
            for i in range(10, 0, -1):
                idx = 10 + i
                if mask[idx]:
                    q_me = float(i)
                    q_total = q_me + q_opp
                    
                    overflow = max(0.0, q_total - D_bid)
                    p_est = X - self.alpha * overflow
                    
                    if p_est - mu > 0:
                        return idx
                        
        return 0

