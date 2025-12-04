import argparse
import numpy as np
import sys
from typing import Dict, Any, Optional

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.baselines.random_valid import RandomValidAgent

def format_money(amount: float) -> str:
    return f"${amount:+.2f}"

def print_state(info: Dict[str, Any], obs_info: Dict[str, Any], t: int, T: int):
    """Print the current market state to the console."""
    print("\n" + "="*50)
    print(f"ROUND {t+1}/{T}")
    print("="*50)
    
    # Event
    event = info.get("event", {})
    ev_type = event.get("type", "none")
    print(f"EVENT: {ev_type}")
    if ev_type == "remap_value":
        # event is a dict here: {'type': '...', 'params': {'rank_from': ..., 'value_to': ...}}
        # OR it might be flattened if to_dict() behaved differently?
        # Let's check Event.to_dict() in events.py: returns {"type": ..., "params": ...}
        # So event.get("params", {}) is where the data is.
        params = event.get("params", {})
        print(f"  (Remap {params.get('rank_from')} -> {params.get('value_to')})")
    
    # Hints
    hints = obs_info.get("hints", [])
    print(f"HINTS: {hints}")
    
    # Quote
    q_bid = obs_info.get("quote_bid", 0.0)
    q_ask = obs_info.get("quote_ask", 0.0)
    print(f"QUOTE: {q_bid:.2f} @ {q_ask:.2f}")
    
    # Depth
    d_bid = obs_info.get("depth_bid", 0.0)
    d_ask = obs_info.get("depth_ask", 0.0)
    print(f"DEPTH: {d_bid:.1f} | {d_ask:.1f} (Displayed)")
    
    # Wallet
    w = obs_info.get("W", 0.0)
    print(f"WALLET: {w:.2f}")
    print("-" * 50)

def get_valid_input(mask: np.ndarray) -> int:
    """Get valid integer action from user."""
    valid_indices = np.where(mask)[0]
    while True:
        print(f"Available Actions: [0=PASS]")
        # Group buys/sells for display
        buys = [int(i) for i in valid_indices if 1 <= i <= 10]
        sells = [int(i) for i in valid_indices if 11 <= i <= 20]
        
        if buys:
            print(f"  Buy sizes: {buys} (Input 1-10)")
        if sells:
            # Map back to size 1..10
            sell_sizes = [i-10 for i in sells]
            print(f"  Sell sizes: {sell_sizes} (Input 11-20)")
            
        try:
            user_input = input("Enter Action ID > ")
            action = int(user_input)
            if action in valid_indices:
                return action
            else:
                print(f"Invalid action {action}. Must be one of {valid_indices}")
        except ValueError:
            print("Please enter an integer.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

def run_game(mode: str, events: bool, impact: bool, opponent_type: str = "random"):
    # Config
    cfg = {
        "W0": 500.0,
        "episode_length": 10,
        "event_persist": 0.0, # Force 0.0 for debug
        "events": {
            "freq": {
                "none": 0.3,       # Force 0.0
                "ge10_only": 0.2,
                "le7_only": 0.2,
                "even_only": 0.1,
                "remap_value": 0.2 
            }
        },
        "flags": {
            "enable_events": events,
            "enable_impact": impact
        },
        # "hints": {"count": 2} # Remove to allow random 0..3
        
        # Volatile Liquidity Config
        "liquidity": {
            "k": 20.0,      # Higher base depth
            "tau": 1.0,     # High variance (violent swings)
            "min": 1.0,
            "max": 50.0,    # Allow deep markets
            "cap": 10.0     # Display cap (hidden iceberg)
        }
    }
    
    # Setup Env
    if mode == "single":
        env = SingleCardEnv(cfg)
        opponent = None
    elif mode == "two":
        env = TwoPlayerCardEnv(cfg)
        if opponent_type == "random":
            opponent = RandomValidAgent()
        else:
            raise ValueError(f"Unknown opponent: {opponent_type}")
    else:
        raise ValueError("Mode must be single or two")
        
    obs, info = env.reset(seed=np.random.randint(0, 10000))
    done = False
    
    total_reward = 0.0
    
    while not done:
        # Extract info for display
        # Single env: info has keys directly.
        # Two player: info might be split or shared.
        
        # We need specific display values. 
        # In single_env step, info has 'mu', 'sigma', 'event', etc.
        # But we also need 'quote' and 'depth' which are in 'obs' usually, 
        # or we can cheat and pull from env object for CLI since we have access.
        # Pulling from env is easier for CLI formatting than decoding Obs.
        
        if mode == "single":
            # info has 'event', 'hidden_cards' (cheat?), 'true_sum' (cheat?)
            # We shouldn't show cheat info.
            # We reconstruct displayable info from env attributes
            display_info = {
                "hints": env.hints,
                "quote_bid": env.quote.bid,
                "quote_ask": env.quote.ask,
                "depth_bid": env.disp_depths[0],
                "depth_ask": env.disp_depths[1],
                "W": env.W,
                "event": env.current_event.to_dict() if env.current_event else {}
            }
            mask = info["mask"]
        else:
            # Two player
            # User is Agent A (index 0)
            display_info = {
                "hints": env.hints_a, # User hints
                "quote_bid": env.quote.bid,
                "quote_ask": env.quote.ask,
                "depth_bid": env.disp_depths[0],
                "depth_ask": env.disp_depths[1],
                "W": env.W_a, # Access W_a directly, env has no agents list
                "event": env.current_event.to_dict() if env.current_event else {}
            }
            # mask for agent A
            # Use mask from info
            if "mask_a" in info:
                mask = info["mask_a"]
            else:
                # Fallback if info structure is different (e.g. terminal step)
                # But we check done loop.
                mask = np.ones(21, dtype=bool)
            
        print_state(info, display_info, env.t, env.T)
        
        # User Action
        action = get_valid_input(mask)
        
        # Step
        if mode == "single":
            obs, reward, term, trunc, next_info = env.step(action)
            print(f"Executed! Price: {next_info['exec_price']:.2f}, True Sum: {next_info['true_sum']}")
            print(f"PnL: {format_money(reward)}")
            if next_info.get("slippage", 0) > 0:
                print(f"Slippage: {next_info['slippage']:.2f}")
            
            total_reward += reward
            info = next_info
            done = term or trunc
            
        else:
            # Two player
            # Get Opponent Action
            mask_b = env._get_mask(1)
            action_b = opponent.act(obs[1], mask_b)
            
            actions = (action, action_b)
            obs, rewards, term, trunc, next_info = env.step(actions)
            
            r_user = rewards[0]
            r_opp = rewards[1]
            
            # Info likely contains details for both
            # Let's assume next_info has 'agent_0' dict etc or flat keys
            # Based on typical multi-agent envs. 
            # I'll use the return values directly where possible
            
            print(f"You PnL: {format_money(r_user)} | Opponent PnL: {format_money(r_opp)}")
            print(f"Opponent Action: {action_b}")
            
            # Reveal Hidden State
            print(f"True Sum: {next_info.get('true_sum', '???')}")
            print(f"True Liquidity (Bid/Ask): {env.true_depths[0]:.1f} / {env.true_depths[1]:.1f}")
            
            if next_info.get("exec_price_buy"):
                p_buy = next_info['exec_price_buy']
                # Use quote from START of round (display_info), not current env.quote (next round)
                impact_buy = p_buy - display_info['quote_ask']
                print(f"Exec Buy Price: {p_buy:.2f} (Impact: {impact_buy:+.2f})")
                
            if next_info.get("exec_price_sell"):
                p_sell = next_info['exec_price_sell']
                impact_sell = display_info['quote_bid'] - p_sell
                print(f"Exec Sell Price: {p_sell:.2f} (Impact: {impact_sell:+.2f})")
            
            total_reward += r_user
            info = next_info # Need to ensure this structure works for next loop
            done = term or trunc
            
    print("="*50)
    print(f"GAME OVER. Total PnL: {format_money(total_reward)}")
    if mode == "single":
        print(f"Final Wealth: {env.W:.2f}")
    else:
        print(f"Final Wealth: {env.W_a:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "two"])
    parser.add_argument("--events", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--impact", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--opponent", type=str, default="random")
    
    args = parser.parse_args()
    
    events_bool = (args.events == "on")
    impact_bool = (args.impact == "on")
    
    run_game(args.mode, events_bool, impact_bool, args.opponent)
