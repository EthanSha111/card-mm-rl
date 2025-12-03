import pytest
import os
import pandas as pd
import shutil
from mmrl.scripts.make_dataset import make_dataset

def test_make_dataset():
    out_file = "test_data/rollout.parquet"
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        
    # Run with very few steps
    make_dataset(out_file, steps=10, env_kind="single", policy_kind="random")
    
    # Check existence
    # Might fallback to csv if parquet missing
    if os.path.exists(out_file):
        df = pd.read_parquet(out_file)
    elif os.path.exists(out_file.replace(".parquet", ".csv")):
        df = pd.read_csv(out_file.replace(".parquet", ".csv"))
    else:
        pytest.fail("Output file not created")
        
    assert len(df) >= 10
    required_cols = ["step", "obs", "action", "reward", "done"]
    for c in required_cols:
        assert c in df.columns
        
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")

