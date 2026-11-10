from __future__ import annotations 

import gymnasium as gym
import numpy as np

class PiecesRewardWrapper(gym.Wrapper):
    max_pieces = 0
    #Reward based on the number of pieces placed in the environment
    def __init__(self, env):
        super().__init__(env)
        self.pieces_placed = 0
        self._action_mask = None

    @property
    def action_mask(self):
        return self._action_mask

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.pieces_placed = 0
        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = (self.env.step(action))
        self.pieces_placed += 1

        # Reward function
        reward = 1.0  # Reward for placing a piece

        if self.pieces_placed > self.max_pieces:
            self.max_pieces = self.pieces_placed
            print(f"New max pieces: {self.max_pieces}")

        self._action_mask = np.asarray(info["action_mask"], dtype=bool)
        info = dict(info)
        info["pieces_placed"] = self.pieces_placed
        return (observation, float(reward), terminated, truncated, info)