import gymnasium as gym
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from environment import make_placement_env
from pieces_reward import PiecesRewardWrapper

SEED = 42

def create_env():
    env = make_placement_env(render_mode="rgb_array")
    env = PiecesRewardWrapper(env)
    env = ActionMasker(env, lambda e: e.action_mask)
    env.reset()
    return env

def main(model_name):
    np.random.seed(SEED)
    env = create_env()
    model = MaskablePPO("MlpPolicy", 
                        env,
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        gamma=1,
                        gae_lambda=0.95,
                        ent_coef=0.01,
                        verbose=1,
                        tensorboard_log="./tensorboard",
                        device="auto",
                        seed=SEED)
    #model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.learn(total_timesteps=10_000_000, progress_bar=True)
    model.save(model_name)
    env.close()

def continue_training(load_name, save_name, timesteps=100_000): # Total timestep goal
    env = create_env()
    np.random.seed(SEED)
    model = MaskablePPO.load(load_name, env=env)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)
    model.save(save_name)
    env.close()

if __name__ == "__main__":
    main("tetris_pieces_m3_10000") #Format: m1 (Model 1) 100(100k timesteps)
    #continue_training("tetris_pieces_m1_1000", "tetris_pieces_m1_10000", 9_000_000)

    