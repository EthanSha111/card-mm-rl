import numpy as np
import torch
from typing import List, Tuple

class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, device: str = "cpu", gamma: float = 0.99, gae_lambda: float = 0.95):
        self.steps = steps
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.obs = np.zeros((steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps,), dtype=np.int64)
        self.rewards = np.zeros((steps,), dtype=np.float32)
        self.log_probs = np.zeros((steps,), dtype=np.float32)
        self.values = np.zeros((steps,), dtype=np.float32)
        self.masks = np.zeros((steps, 21), dtype=np.bool_)
        self.dones = np.zeros((steps,), dtype=np.float32)
        
        self.ptr = 0
        
    def add(self, obs, action, reward, log_prob, value, done, mask):
        if self.ptr >= self.steps:
            raise IndexError("Rollout buffer overflow")
            
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.masks[self.ptr] = mask
        
        self.ptr += 1
        
    def finish_path(self, last_val: float = 0.0):
        # Compute GAE
        # We need to handle multiple episodes if they are concatenated.
        # But simplest is to run for N steps then bootstrap.
        # If buffer contains multiple episodes, `dones` handles it.
        
        advs = np.zeros_like(self.rewards)
        last_gae = 0.0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_val
                next_non_terminal = 1.0 - self.dones[t] # If done, 0.
            else:
                next_val = self.values[t+1]
                next_non_terminal = 1.0 - self.dones[t]
                
            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            advs[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        returns = advs[:self.ptr] + self.values[:self.ptr]
        
        self.returns = returns
        self.advantages = advs[:self.ptr]
        
    def get(self):
        if self.ptr != self.steps:
            # Can return partial if needed, but usually we wait for full rollout
            pass
            
        return (
            torch.as_tensor(self.obs[:self.ptr], device=self.device),
            torch.as_tensor(self.actions[:self.ptr], device=self.device),
            torch.as_tensor(self.log_probs[:self.ptr], device=self.device),
            torch.as_tensor(self.returns, device=self.device),
            torch.as_tensor(self.advantages, device=self.device),
            torch.as_tensor(self.values[:self.ptr], device=self.device),
            torch.as_tensor(self.masks[:self.ptr], device=self.device)
        )
        
    def reset(self):
        self.ptr = 0

