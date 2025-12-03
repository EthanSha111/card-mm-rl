import pytest
import os
from mmrl.config.loader import load_config, load_merged_config, CONFIG_DIR

def test_load_defaults():
    cfg = load_config("defaults")
    assert "seed" in cfg
    assert cfg["seed"] == 42

def test_load_env():
    cfg = load_config("env")
    assert "kind" in cfg
    assert "spread" in cfg
    assert cfg["spread"]["s0"] == 0.8

def test_merged_config():
    # Should have both seed (defaults) and kind (env)
    cfg = load_merged_config()
    assert "seed" in cfg
    assert "kind" in cfg
    assert cfg["W0"] == 500.0

def test_all_configs_exist():
    expected = ["defaults.yaml", "env.yaml", "dqn.yaml", "ippo.yaml", "mappo.yaml", "eval.yaml"]
    for f in expected:
        assert os.path.exists(os.path.join(CONFIG_DIR, f))

