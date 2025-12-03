import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, List
from mmrl.env.events import Event, EVENT_NONE, EVENT_GE10, EVENT_LE7, EVENT_EVEN, EVENT_ODD, EVENT_REMAP_VALUE

# Action Space
# 0: Pass
# 1..10: Buy 1..10
# 11..20: Sell 1..10
ACTION_SPACE_SIZE = 21
ACTION_SPACE = spaces.Discrete(ACTION_SPACE_SIZE)

# Event types for one-hot
EVENT_TYPES = [EVENT_NONE, EVENT_GE10, EVENT_LE7, EVENT_EVEN, EVENT_ODD, EVENT_REMAP_VALUE]

def get_obs_shape() -> Tuple[int,]:
    """
    Returns shape of observation vector.
    Components:
    - Quote (X, Y, mid, spread): 4
    - Displayed Depth (bid, ask): 2
    - Last Step Metrics (slip_bid, slip_ask, fill_bid, fill_ask): 4
    - Hints (counts of 2..14): 13
    - Event (one-hot): 6
    - State (W/W0, t/T): 2
    - Flags (events, impact): 2
    - Opponent (side, size): 2
    Total: 35
    """
    return (35,)

def build_obs(
    quote: Any, # Quote object
    depths: Tuple[float, float], # (bid_depth, ask_depth)
    metrics: Dict[str, float], # slippage, fill ratios
    hints: List[int],
    event: Event,
    state: Dict[str, float], # W, W0, t, T
    flags: Dict[str, bool],
    opponent_last: Dict[str, float] = None
) -> np.ndarray:
    
    obs = []
    
    # 1. Quote: X, Y, mid, spread
    obs.extend([quote.bid, quote.ask, quote.mid, quote.spread])
    
    # 2. Depths: bid, ask
    obs.extend([depths[0], depths[1]])
    
    # 3. Metrics (default 0 if missing)
    obs.append(metrics.get("slippage_bid", 0.0))
    obs.append(metrics.get("slippage_ask", 0.0))
    obs.append(metrics.get("fill_ratio_bid", 0.0))
    obs.append(metrics.get("fill_ratio_ask", 0.0))
    
    # 4. Hints: 13-dim count vector (ranks 2..14)
    # Map 2->0, ..., 14->12
    hint_counts = np.zeros(13, dtype=np.float32)
    for h in hints:
        if 2 <= h <= 14:
            hint_counts[h-2] += 1
    obs.extend(hint_counts)
    
    # 5. Event: one-hot
    ev_vec = np.zeros(len(EVENT_TYPES), dtype=np.float32)
    try:
        idx = EVENT_TYPES.index(event.type)
        ev_vec[idx] = 1.0
    except ValueError:
        # Unknown event, maybe map to none or all zeros?
        pass
    obs.extend(ev_vec)
    
    # 6. State: W/W0, t/T
    w_frac = state["W"] / max(1e-3, state["W0"])
    t_frac = state["t"] / max(1.0, state["T"])
    obs.extend([w_frac, t_frac])
    
    # 7. Flags
    obs.append(1.0 if flags.get("enable_events", False) else 0.0)
    obs.append(1.0 if flags.get("enable_impact", False) else 0.0)
    
    # 8. Opponent: side (-1, 0, 1), size
    if opponent_last:
        obs.append(opponent_last.get("side", 0.0))
        obs.append(opponent_last.get("size", 0.0))
    else:
        obs.extend([0.0, 0.0])
        
    return np.array(obs, dtype=np.float32)

