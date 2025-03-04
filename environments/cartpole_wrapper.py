from environments.wrappers import BaseWrapper
import numpy as np

class CartPoleWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def _modify_observation_space(self, obs_space):
        """Add additional features to observation space"""
        return spaces.Box(
            low=np.append(obs_space.low, [-np.inf]),
            high=np.append(obs_space.high, [np.inf]),
            dtype=np.float32
        )
    
    def _process_observation(self, obs):
        """Add angle velocity as an additional feature"""
        angle_velocity = (obs[3] - self.last_angle) if hasattr(self, 'last_angle') else 0
        self.last_angle = obs[3]
        return np.append(obs, [angle_velocity])
    
    def _process_reward(self, reward):
        """Modify reward to encourage stability"""
        return reward + (1 - abs(self.env.state[2]) / (np.pi / 2))