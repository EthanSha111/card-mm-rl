from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from mmrl.env.cards import ALL_RANKS, FULL_DECK

# Event Constants
EVENT_NONE = "none"
EVENT_REMAP_VALUE = "remap_value"
EVENT_GE10 = "ge10_only"
EVENT_LE7 = "le7_only"
EVENT_EVEN = "even_only"
EVENT_ODD = "odd_only"

@dataclass
class Event:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "params": self.params}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Event':
        return Event(type=d["type"], params=d.get("params", {}))
    
    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.type == other.type and self.params == other.params

def get_default_value_map() -> Dict[int, int]:
    """Returns identity map for all ranks."""
    return {r: r for r in ALL_RANKS}

def apply_event(full_deck: List[int], event: Event) -> Tuple[List[int], Dict[int, int]]:
    """
    Applies the event to the deck.
    Returns:
        filtered_deck: List of ranks available to be drawn.
        value_map: Dict mapping rank -> value for this round.
    """
    # Default: all cards eligible, identity value map
    value_map = get_default_value_map()
    
    if event.type == EVENT_NONE:
        return list(full_deck), value_map
    
    elif event.type == EVENT_REMAP_VALUE:
        # remap_value X to Y
        x = event.params["rank_from"]
        y = event.params["value_to"]
        value_map[x] = y
        # Deck remains full
        return list(full_deck), value_map
    
    elif event.type == EVENT_GE10:
        # Only cards worth 10 or more
        # Note: if we had remapped, we would check the NEW value.
        # But events are mutually exclusive.
        # Standard values: 2..14.
        filtered = [r for r in full_deck if value_map[r] >= 10]
        return filtered, value_map
        
    elif event.type == EVENT_LE7:
        filtered = [r for r in full_deck if value_map[r] <= 7]
        return filtered, value_map
        
    elif event.type == EVENT_EVEN:
        filtered = [r for r in full_deck if value_map[r] % 2 == 0]
        return filtered, value_map
        
    elif event.type == EVENT_ODD:
        filtered = [r for r in full_deck if value_map[r] % 2 != 0]
        return filtered, value_map
        
    else:
        # Unknown event, treat as none? Or raise?
        # Let's treat as none to be safe, or warn.
        return list(full_deck), value_map

def sample_event(rng: np.random.RandomState, cfg: Any, last_event: Optional[Event] = None) -> Event:
    """
    Sample an event based on config.
    cfg should be a DictConfig or object with:
        - flags.enable_events (bool)
        - event_persist (float, optional)
        - events (dict of frequencies)
    """
    # Access config in a way compatible with dict or object
    def get_cfg(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    
    # Handle nested flags.enable_events
    enable_events = False
    flags = get_cfg("flags")
    if flags:
        if isinstance(flags, dict):
            enable_events = flags.get("enable_events", False)
        else:
            enable_events = getattr(flags, "enable_events", False)
    
    if not enable_events:
        return Event(type=EVENT_NONE)
    
    # Persistence
    persist_prob = get_cfg("event_persist", 0.0)
    if last_event is not None and rng.rand() < persist_prob:
        return last_event
    
    # Sample new event
    # Frequencies from cfg.events
    # Expected structure: cfg.events = { "none": 0.6, "ge10_only": 0.1, ... }
    # Or if nested: cfg.events.freq = { ... }
    
    events_cfg = get_cfg("events", {})
    if isinstance(events_cfg, dict) and "freq" in events_cfg:
        event_probs = events_cfg["freq"]
    else:
        event_probs = events_cfg
        
    # If empty, default to none
    if not event_probs:
        # print("DEBUG: event_probs is empty, defaulting to none")
        return Event(type=EVENT_NONE)
    
    keys = sorted(list(event_probs.keys()))
    probs = [event_probs[k] for k in keys]
    total = sum(probs)
    if total <= 0:
        return Event(type=EVENT_NONE)
    
    # Normalize
    probs = np.array(probs) / total
    
    chosen_type = rng.choice(keys, p=probs)
    
    if chosen_type == EVENT_REMAP_VALUE:
        # We need to sample parameters for remap_value.
        # "remap_value X to Y".
        # Maybe random rank X (2..14) to random value Y (2..14)?
        # Or configured? Prompt doesn't specify. "remap_value X to Y this round."
        # Let's pick X from ALL_RANKS, and Y from ALL_RANKS.
        x = rng.choice(ALL_RANKS)
        y = rng.choice(ALL_RANKS)
        # Avoid trivial remap
        while x == y:
             y = rng.choice(ALL_RANKS)
        return Event(type=EVENT_REMAP_VALUE, params={"rank_from": int(x), "value_to": int(y)})
        
    return Event(type=chosen_type)

