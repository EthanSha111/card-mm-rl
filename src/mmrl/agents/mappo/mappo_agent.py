import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any
from mmrl.agents.common.nets import MLP
from mmrl.agents.common.rollout import RolloutBuffer

class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, joint_obs_dim, action_dim, hidden_dims):
        super().__init__()
        self.actor = MLP(obs_dim, hidden_dims, action_dim)
        self.critic = MLP(joint_obs_dim, hidden_dims, 1)
        
    def forward(self, obs, joint_obs, mask=None):
        logits = self.actor(obs)
        if mask is not None:
            logits = torch.where(mask, logits, torch.tensor(-1e8, device=obs.device))
        
        val = self.critic(joint_obs)
        return logits, val

class MAPPORolloutBuffer(RolloutBuffer):
    def __init__(self, steps, obs_dim, joint_obs_dim, device, gamma, gae_lambda):
        super().__init__(steps, obs_dim, device, gamma, gae_lambda)
        self.joint_obs = np.zeros((steps, joint_obs_dim), dtype=np.float32)
        
    def add(self, obs, joint_obs, action, reward, log_prob, value, done, mask):
        self.joint_obs[self.ptr] = joint_obs
        super().add(obs, action, reward, log_prob, value, done, mask)
        
    def get(self):
        # We need joint_obs for value update if we recompute values?
        # Usually PPO recomputes values.
        tensors = super().get()
        # Tensors: obs, act, logp, ret, adv, val, mask
        return tensors + (torch.as_tensor(self.joint_obs[:self.ptr], device=self.device),)

class MAPPOAgent:
    def __init__(self, obs_dim, joint_obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self.obs_dim = obs_dim
        self.joint_obs_dim = joint_obs_dim
        
        hidden_dims = cfg.get("hidden_dims", [64, 64])
        self.ac = MAPPOActorCritic(obs_dim, joint_obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=cfg.get("lr", 3e-4))
        
        self.rollout_steps = cfg.get("rollout_steps", 2048)
        self.buffer = MAPPORolloutBuffer(self.rollout_steps, obs_dim, joint_obs_dim, self.device, cfg.get("gamma", 0.99), cfg.get("gae_lambda", 0.95))
        
        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.train_iters = cfg.get("train_iters", 10)
        
    def step(self, obs, joint_obs, mask):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            
            logits, val = self.ac(obs_t, joint_obs_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), val.item()
        
    def store(self, obs, joint_obs, action, reward, log_prob, val, done, mask):
        self.buffer.add(obs, joint_obs, action, reward, log_prob, val, done, mask)
        
    def update(self, last_val=0.0):
        self.buffer.finish_path(last_val)
        obs, act, log_prob_old, ret, adv, val_old, masks, joint_obs = self.buffer.get()
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for i in range(self.train_iters):
            logits, val = self.ac(obs, joint_obs, masks)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(act)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(log_prob - log_prob_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(val.squeeze(), ret)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.buffer.reset()
        return loss.item()

