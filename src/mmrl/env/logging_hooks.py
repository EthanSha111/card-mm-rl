from typing import Dict, Any, Optional
import numpy as np

def make_log_entry(
    step_info: Dict[str, Any],
    obs: np.ndarray,
    action: int,
    mask: np.ndarray,
    state_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construct a flat dictionary for logging/dataset creation.
    """
    # Flatten relevant fields
    log_entry = {
        "t": state_info.get("t"),
        "W": state_info.get("W"),
        "action": action,
        "reward": step_info.get("reward"),
        "exec_price": step_info.get("exec_price"),
        "slippage": step_info.get("slippage"),
        "true_sum": step_info.get("true_sum"),
        "mu": state_info.get("mu"),
        "sigma": state_info.get("sigma"),
        # Store mask as list or string? Parquet handles lists usually, or fixed size array.
        # We might just want validity of taken action?
        # Or full mask.
        "mask": mask.tolist() if isinstance(mask, np.ndarray) else mask,
        # Snapshot of obs?
        # "obs": obs.tolist() # Might be large.
    }
    
    # Add event info
    event = state_info.get("event", {})
    if event:
        log_entry["event_type"] = event.get("type")
    
    # Add depths
    depths = state_info.get("true_depths", (0,0))
    log_entry["L_bid"] = depths[0]
    log_entry["L_ask"] = depths[1]
    
    return log_entry

