import numpy as np
import pytest
from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.agents.dqn.dqn_agent import DQNAgent
from mmrl.agents.ippo.ippo_agent import IPPOAgent
from mmrl.agents.mappo.mappo_agent import MAPPOAgent
from mmrl.baselines.random_valid import act_random_valid
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

def test_dqn_smoke():
    # Config
    cfg = {
        "episode_length": 5,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": False}
    }
    
    env = SingleCardEnv(cfg)
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    agent_cfg = {
        "device": "cpu",
        "gamma": 0.9,
        "batch_size": 16,
        "lr": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9,
        "buffer_size": 500,
        "hidden_dims": [32]
    }
    
    agent = DQNAgent(obs_dim, act_dim, agent_cfg)
    
    # Train for 50 episodes
    returns = []
    for ep in range(50):
        obs, info = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_mask = info.get("mask")
            if next_mask is None:
                next_mask = np.zeros(act_dim, dtype=bool)
                
            agent.step(obs, action, reward, next_obs, terminated or truncated, next_mask)
            obs = next_obs
            ep_ret += reward
            done = terminated or truncated
            
        returns.append(ep_ret)
        
    assert len(returns) == 50

def test_ippo_smoke():
    cfg = {
        "episode_length": 5,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": False}
    }
    
    env = TwoPlayerCardEnv(cfg)
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    agent_cfg = {
        "device": "cpu",
        "rollout_steps": 100,
        "train_iters": 1,
        "hidden_dims": [32]
    }
    
    agent_a = IPPOAgent(obs_dim, act_dim, agent_cfg)
    agent_b = IPPOAgent(obs_dim, act_dim, agent_cfg)
    agent_b.ac = agent_a.ac
    agent_b.optimizer = agent_a.optimizer
    
    (obs_a, obs_b), info = env.reset(seed=42)
    
    for _ in range(50):
        mask_a = info["mask_a"]
        mask_b = info["mask_b"]
        
        act_a, logp_a, val_a = agent_a.step(obs_a, mask_a)
        act_b, logp_b, val_b = agent_b.step(obs_b, mask_b)
        
        (next_obs_a, next_obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
        
        agent_a.store(obs_a, act_a, r_a, logp_a, val_a, term, mask_a)
        agent_b.store(obs_b, act_b, r_b, logp_b, val_b, term, mask_b)
        
        obs_a, obs_b = next_obs_a, next_obs_b
        
        if term or trunc:
            agent_a.update()
            agent_b.update()
            (obs_a, obs_b), info = env.reset()
            
    assert True

def test_mappo_smoke():
    cfg = {
        "episode_length": 5,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": False}
    }
    
    env = TwoPlayerCardEnv(cfg)
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    agent_cfg = {
        "device": "cpu",
        "rollout_steps": 100,
        "train_iters": 1,
        "hidden_dims": [32]
    }
    
    agent_a = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b.ac = agent_a.ac
    agent_b.optimizer = agent_a.optimizer
    
    (obs_a, obs_b), info = env.reset(seed=42)
    
    for _ in range(50):
        mask_a = info["mask_a"]
        mask_b = info["mask_b"]
        joint_obs = np.concatenate([obs_a, obs_b])
        
        act_a, logp_a, val_a = agent_a.step(obs_a, joint_obs, mask_a)
        act_b, logp_b, val_b = agent_b.step(obs_b, joint_obs, mask_b)
        
        (next_obs_a, next_obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
        
        agent_a.store(obs_a, joint_obs, act_a, r_a, logp_a, val_a, term, mask_a)
        agent_b.store(obs_b, joint_obs, act_b, r_b, logp_b, val_b, term, mask_b)
        
        obs_a, obs_b = next_obs_a, next_obs_b
        
        if term or trunc:
            next_joint_obs = np.concatenate([obs_a, obs_b])
            agent_a.update()
            agent_b.update()
            (obs_a, obs_b), info = env.reset()
            
    assert True
