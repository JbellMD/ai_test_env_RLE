import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class MultiAgentSystem:
    def __init__(self, state_size, action_size, num_agents, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        self.agents = [
            DDPGAgent(state_size, action_size, hidden_size)
            for _ in range(num_agents)
        ]
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.01
        
    def act(self, state, noise_scale=1.0):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(state[i], noise_scale)
            actions.append(action)
        return np.array(actions)
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size, gamma, tau):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        for i, agent in enumerate(self.agents):
            agent.update(
                states[:, i],
                actions[:, i],
                rewards[:, i],
                next_states[:, i],
                dones[:, i],
                gamma,
                tau
            )

class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.target_actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        
        self.target_critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
    
    def act(self, state, noise_scale=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()
        noise = noise_scale * np.random.normal(size=self.action_size)
        return np.clip(action + noise, -1, 1)
    
    def update(self, states, actions, rewards, next_states, dones, gamma, tau):
        # Update Critic
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(torch.cat([next_states, next_actions], dim=1))
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks(tau)
    
    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)