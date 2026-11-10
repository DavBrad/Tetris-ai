from __future__ import annotations 

import gymnasium as gym
import numpy as np

class HeightRewardWrapper(gym.Wrapper):
    #Reward based on the height of the agent in the environment
    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        self.height = base.height
        self.width = base.width
        self.padding = base.padding

        self.previous_height = 0
        self._action_mask = None

    def get_total_height(self, observation):
            board = observation
            # Board without padding
            board = board[:self.height, self.padding:self.padding + self.width]
            occupied = board != 0
    
            column_heights = np.where(occupied.any(axis = 0), self.height - occupied.argmax(axis=0), 0)
            return int(column_heights.sum())

    def get_max_height(self, observation):
        board = observation
        # Board without padding
        board = board[:self.height, self.padding:self.padding + self.width]
        occupied = board != 0

        column_heights = np.where(occupied.any(axis = 0), self.height - occupied.argmax(axis=0), 0)
        return int(column_heights.max())

    @property
    def action_mask(self):
        return self._action_mask

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_height = self.get_total_height(observation)
        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = (self.env.step(action))
        current_height = self.get_max_height(observation)


        # Reward function
        delta = current_height - self.previous_height
        if delta <= 0:
            reward = 0.1 + (abs(delta) * 0.2)
        else:
            reward = -(0.1 * delta + 0.05 * delta**2)
        if terminated:
            reward -= 1.0 # Terminal penalty

        self.previous_height = current_height
        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        return (observation, float(reward), terminated, truncated, info)