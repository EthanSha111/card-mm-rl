import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any
from mmrl.agents.common.nets import MLP
from mmrl.agents.common.rollout import RolloutBuffer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        # Shared or separate? Separate is safer for stability often.
        self.actor = MLP(obs_dim, hidden_dims, action_dim)
        self.critic = MLP(obs_dim, hidden_dims, 1)
        
    def forward(self, x, mask=None):
        # Actor
        logits = self.actor(x)
        if mask is not None:
            # Apply mask
            logits = torch.where(mask, logits, torch.tensor(-1e8, device=x.device))
            
        return logits, self.critic(x)

class IPPOAgent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        hidden_dims = cfg.get("hidden_dims", [64, 64])
        self.ac = ActorCritic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=cfg.get("lr", 3e-4))
        
        # Rollout buffer
        self.rollout_steps = cfg.get("rollout_steps", 2048)
        self.buffer = RolloutBuffer(self.rollout_steps, obs_dim, self.device, gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95))
        
        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.target_kl = cfg.get("target_kl", 0.01)
        self.train_iters = cfg.get("train_iters", 10)
        
    def step(self, obs, mask):
        # Select action
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            
            logits, val = self.ac(obs_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), val.item()
        
    def store(self, obs, action, reward, log_prob, val, done, mask):
        self.buffer.add(obs, action, reward, log_prob, val, done, mask)
        
    def update(self, last_val=0.0):
        self.buffer.finish_path(last_val)
        data = self.buffer.get()
        obs, act, log_prob_old, ret, adv, val_old, masks = data
        
        # Normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for i in range(self.train_iters):
            # Recalculate
            logits, val = self.ac(obs, masks)
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
            
            # Optional: KL check for early stopping
            with torch.no_grad():
                approx_kl = (log_prob_old - log_prob).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break
                
        self.buffer.reset()
        return {"loss": loss.item(), "kl": approx_kl, "entropy": entropy.item()}

