import yaml
import os
from typing import Dict, Any

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config(name: str) -> Dict[str, Any]:
    """
    Load a YAML config by name (e.g., 'env', 'dqn') from the config directory.
    Recursively merges with defaults if needed (not implemented here, just raw load).
    """
    if not name.endswith(".yaml"):
        name += ".yaml"
        
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
        
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_merged_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load defaults and env config, merge with overrides.
    """
    base = load_config("defaults")
    env_cfg = load_config("env")
    
    # Simple merge: defaults < env < overrides
    cfg = base.copy()
    
    # Helper to deep merge? For now shallow merge of top keys is fine 
    # but env has nested keys.
    # Let's do a naive update.
    cfg.update(env_cfg)
    
    if overrides:
        cfg.update(overrides)
        
    return cfg

