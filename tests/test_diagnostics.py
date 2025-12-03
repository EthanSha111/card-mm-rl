import pytest
import pandas as pd
import os
import shutil
from mmrl.eval.diagnostics import run_diagnostics

def test_diagnostics_generation():
    # Create dummy CSV
    data = {
        "episode": [0, 0, 1, 1],
        "step": [1, 2, 1, 2],
        "action": [1, 0, 15, 5],
        "reward": [5, 0, -2, 3],
        "wealth": [505, 505, 498, 501]
    }
    df = pd.DataFrame(data)
    df.to_csv("test_diag_input.csv", index=False)
    
    out_dir = "test_diag_output"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    run_diagnostics("test_diag_input.csv", out_dir)
    
    # Check files exist
    assert os.path.exists(os.path.join(out_dir, "action_hist.png"))
    assert os.path.exists(os.path.join(out_dir, "reward_dist.png"))
    assert os.path.exists(os.path.join(out_dir, "wealth_curves.png"))
    
    # Cleanup
    if os.path.exists("test_diag_input.csv"):
        os.remove("test_diag_input.csv")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
