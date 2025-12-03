import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Optional

def plot_action_histogram(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot histogram of actions taken.
    Action map: 0=Pass, 1..10=Buy, 11..20=Sell
    """
    plt.figure(figsize=(10, 6))
    
    # Bin actions
    # We can group by type: Pass, Buy, Sell
    # Or raw histogram
    
    counts, bins, patches = plt.hist(df['action'], bins=range(0, 22), rwidth=0.8, align='left')
    plt.title("Action Distribution")
    plt.xlabel("Action ID (0=Pass, 1-10=Buy Size, 11-20=Sell Size)")
    plt.ylabel("Count")
    plt.xticks(range(0, 21))
    plt.grid(axis='y', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_pnl_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot distribution of rewards (PnL).
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['reward'], bins=50, alpha=0.7, color='green')
    plt.title("Reward Distribution (Per Step)")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_wealth_curves(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot wealth trajectories for a subset of episodes.
    """
    plt.figure(figsize=(10, 6))
    
    episodes = df['episode'].unique()
    # Plot first 10 episodes
    for ep in episodes[:10]:
        subset = df[df['episode'] == ep]
        plt.plot(subset['step'], subset['wealth'], label=f'Ep {ep}')
        
    plt.title("Wealth Trajectories (First 10 Episodes)")
    plt.xlabel("Step")
    plt.ylabel("Wealth")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_diagnostics(csv_path: str, output_dir: str):
    """
    Load results CSV and generate all plots.
    """
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plot_action_histogram(df, os.path.join(output_dir, "action_hist.png"))
    plot_pnl_distribution(df, os.path.join(output_dir, "reward_dist.png"))
    plot_wealth_curves(df, os.path.join(output_dir, "wealth_curves.png"))
    
    print(f"Diagnostics saved to {output_dir}")

if __name__ == "__main__":
    # Test run
    # Create dummy data
    data = {
        "episode": [0, 0, 1, 1],
        "step": [1, 2, 1, 2],
        "action": [1, 0, 15, 5],
        "reward": [5, 0, -2, 3],
        "wealth": [505, 505, 498, 501]
    }
    df = pd.DataFrame(data)
    df.to_csv("test_diag.csv", index=False)
    run_diagnostics("test_diag.csv", "test_plots")
