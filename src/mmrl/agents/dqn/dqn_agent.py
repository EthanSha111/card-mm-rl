import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Any, Tuple
from mmrl.agents.common.nets import MLP
from mmrl.agents.common.replay import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Dict[str, Any]
    ):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self.gamma = cfg.get("gamma", 0.99)
        self.batch_size = cfg.get("batch_size", 64)
        self.lr = cfg.get("lr", 3e-4)
        self.target_update_freq = cfg.get("target_update_freq", 100)
        self.epsilon = cfg.get("epsilon_start", 1.0)
        self.epsilon_min = cfg.get("epsilon_min", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        
        hidden_dims = cfg.get("hidden_dims", [64, 64])
        
        self.q_net = MLP(obs_dim, hidden_dims, action_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(cfg.get("buffer_size", 10000), obs_dim, self.device)
        
        self.steps = 0
        self.action_dim = action_dim
        
    def act(self, obs: np.ndarray, mask: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.random() < self.epsilon:
            # Random valid
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0: return 0
            return np.random.choice(valid_indices)
        
        # Greedy
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t)
            
            # Mask invalid actions (set to -inf)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            q_values[~mask_t] = -float('inf')
            
            action = torch.argmax(q_values, dim=1).item()
            return action
            
    def step(self, obs, action, reward, next_obs, done, mask):
        self.buffer.add(obs, action, reward, next_obs, done, mask)
        self.steps += 1
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.buffer.size >= self.batch_size:
            return self.update()
        return None
        
    def update(self):
        obs, actions, rewards, next_obs, dones, next_masks = self.buffer.sample(self.batch_size)
        
        # Compute Target
        with torch.no_grad():
            # Double DQN or Standard? Standard for now.
            # Masking in target?
            # If next state has valid actions, take max over valid.
            # If terminal, target is reward.
            
            next_q = self.target_net(next_obs)
            # Mask next_q
            # next_masks is boolean tensor
            next_q[~next_masks.bool()] = -float('inf')
            
            max_next_q, _ = next_q.max(dim=1)
            # If all masked (should not happen if not done), max is -inf. Handle?
            # Generally done implies no next state value.
            
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        # Current Q
        q_values = self.q_net(obs)
        curr_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = F.huber_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()

