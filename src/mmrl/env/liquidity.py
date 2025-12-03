import numpy as np
from typing import Tuple, Any

def draw_true_depth(
    sigma_S: float,
    spread: float,
    cfg: Any,
    rng: np.random.RandomState
) -> float:
    """
    Draw true liquidity depth L_true based on uncertainty and spread.
    L_bar = k / (sigma_S * (spread + epsilon))
    L_true ~ LogNormal(mean=ln(L_bar) - 0.5 * tau^2, sigma=tau)
    Clipped to [L_min, L_max].
    """
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    
    liq_cfg = get_val(cfg, "liquidity", {})
    k = get_val(liq_cfg, "k", 10.0)
    tau = get_val(liq_cfg, "tau", 0.6)
    l_min = get_val(liq_cfg, "min", 2.0)
    l_max = get_val(liq_cfg, "max", 20.0)
    epsilon = 1e-8
    
    # Nominal depth scale
    # Avoid division by zero
    denom = sigma_S * (spread + epsilon)
    if denom <= 1e-9:
        L_bar = k / 1e-9
    else:
        L_bar = k / denom
        
    mu_ln = np.log(L_bar) - 0.5 * (tau ** 2)
    l_sample = rng.lognormal(mean=mu_ln, sigma=tau)
    return float(np.clip(l_sample, l_min, l_max))

def displayed_depth(L_true: float, cfg: Any) -> float:
    """
    D_disp = min(L_true, L_cap)
    """
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    liq_cfg = get_val(cfg, "liquidity", {})
    l_cap = get_val(liq_cfg, "display_cap", 10.0)
    
    return float(min(L_true, l_cap))

def exec_price_buy(
    Y: float, 
    q_total: float, 
    L_true: float, 
    alpha: float, 
    enable_impact: bool = True
) -> float:
    """
    Compute average execution price for buy order of size q_total.
    If impact enabled and q > L: price increases linearly.
    """
    if not enable_impact:
        return Y
    
    overflow = max(0.0, q_total - L_true)
    if overflow <= 0:
        return Y
    
    return Y + alpha * overflow

def exec_price_sell(
    X: float, 
    q_total: float, 
    L_true: float, 
    alpha: float, 
    enable_impact: bool = True
) -> float:
    """
    Compute average execution price for sell order of size q_total.
    If impact enabled and q > L: price decreases linearly.
    """
    if not enable_impact:
        return X
        
    overflow = max(0.0, q_total - L_true)
    if overflow <= 0:
        return X
        
    return X - alpha * overflow
