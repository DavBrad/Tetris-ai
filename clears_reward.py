from __future__ import annotations 

import gymnasium as gym
import numpy as np

class ClearRewardWrapper(gym.Wrapper):
    max_pieces = 0
    max_reward = 0.0
    #Reward based on the height of the agent in the environment
    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped

        self.base = base
        self.height = base.height
        self.width = base.width
        self.padding = base.padding

        self.pieces_placed = 0
        self.previous_height = 0
        self.rows_clear = 0
        self._action_mask = None

    @property
    def action_mask(self):
        return self._action_mask

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.rows_clear = 0
        self.pieces_placed = 0
        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        return observation, info

    def get_max_height(self):
            board = self.base.board
            board = board[:self.height, self.padding:self.padding + self.width]
            occupied = board != 0
            column_heights = np.where(occupied.any(axis = 0), self.height - occupied.argmax(axis=0), 0)
            return int(column_heights.max())

    def step(self, action):
        observation, _, terminated, truncated, info = (self.env.step(action))
        self.pieces_placed += 1
        current_height = self.get_max_height()

        # M1 - Solely reward lines cleared
        lines_cleared = int(info["lines_cleared"])
        reward = 10.0 * lines_cleared

        # M2 - Reward based on survival
        reward += 0.1

        # M3 - Reward based on not increasing height
        #delta = current_height - self.previous_height
        #reward -= delta

        #if terminated:
            #reward -= 5.0 # Terminal penalty

        # if self.pieces_placed >= self.max_pieces:
        #     self.max_pieces = self.pieces_placed
        #     print(f"New max pieces: {self.max_pieces}")
        #     print(f"Reward: {reward}")
        # if reward >= self.max_reward:
        #     self.max_reward = reward
        #     #print(f"New max reward: {self.max_reward}")
        # if reward >= 10:
        #     print(f"Line cleared")
        
        self.previous_height = current_height
        self.rows_clear = lines_cleared
        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        return (observation, float(reward), terminated, truncated, info)