import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN

import torch

from environment import make_placement_env
from clears_reward import ClearRewardWrapper

SEED = 42

def create_env():
    env = make_placement_env(render_mode="rgb_array")
    #env = gym.wrappers.HumanRendering(env)
    env = ClearRewardWrapper(env)
    return env

def main(model_name):
    np.random.seed(SEED)
    env = create_env()
    model = DQN("MlpPolicy",
                 env,
                 learning_rate=1e-4,
                 buffer_size=100_000,
                 learning_starts=10_000,
                 batch_size=64,
                 gamma=1,
                 tau=1,
                 train_freq=4,
                 target_update_interval=10_000,
                 gradient_steps=1,
                 exploration_fraction=0.3,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05,
                 verbose=1,
                 tensorboard_log="./tensorboard",
                 device="auto",
                 seed=SEED,
                 policy_kwargs = dict(
                     activation_fn = torch.nn.ReLU,
                     net_arch=[128, 64]))
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    #model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(model_name)
    env.close()

def continue_training(load_name, save_name, timesteps=100_000): # Total timestep goal
    env = create_env()
    np.random.seed(SEED)
    model = DQN.load(load_name, env=env)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)
    model.save(save_name)
    env.close()

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    main("tetris_dqnclears_m4_1000") #Format: m1 (Model 1) 100(100k timesteps)
    #for i in range(2, 24):
    #            continue_training(f"tetris_dqnclears_m1_{i * 1000}", f"tetris_dqnclears_m1_{(i+1)*1000}", 1_000_000)

    