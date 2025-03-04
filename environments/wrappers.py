import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._modify_observation_space(env.observation_space)
        self.action_space = self._modify_action_space(env.action_space)
        
    def _modify_observation_space(self, obs_space):
        """Override to modify observation space"""
        return obs_space
    
    def _modify_action_space(self, action_space):
        """Override to modify action space"""
        return action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info
    
    def step(self, action):
        action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_observation(obs)
        reward = self._process_reward(reward)
        return obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        """Override to process observation"""
        return obs
    
    def _process_action(self, action):
        """Override to process action"""
        return action
    
    def _process_reward(self, reward):
        """Override to process reward"""
        return reward