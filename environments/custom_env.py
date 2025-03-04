import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomGridEnv(gym.Env):
    def __init__(self, grid_size=5):
        super().__init__()
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # Initialize state
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size-1, grid_size-1])
        self.current_pos = self.start_pos.copy()
        
    def reset(self, **kwargs):
        self.current_pos = self.start_pos.copy()
        return self.current_pos, {}
    
    def step(self, action):
        # Calculate new position
        new_pos = self.current_pos.copy()
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        self.current_pos = new_pos
        
        # Calculate reward and done
        done = np.array_equal(self.current_pos, self.goal_pos)
        reward = 1.0 if done else -0.01
        
        return self.current_pos, reward, done, False, {}
    
    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[self.start_pos[0], self.start_pos[1]] = 'S'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.current_pos[0], self.current_pos[1]] = 'A'
        
        print('+' + '---+' * self.grid_size)
        for row in grid:
            print('| ' + ' | '.join(row) + ' |')
            print('+' + '---+' * self.grid_size)
    
    def close(self):
        pass