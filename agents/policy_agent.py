import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.optimizer = optim.Adam(self.policy_network.parameters())
        self.gamma = 0.99
        
    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy_network(state_tensor)
        action_probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs.squeeze(0)[action])
        return action, log_prob
    
    def update_policy(self, log_probs, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()