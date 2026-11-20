import numpy as np
from stable_baselines3 import DQN
from dqn_clears_train import create_env

def evaluate(path, episodes=1000):
    env = create_env()
    model = DQN.load(path)
    final_heights = []
    pieces_placed = []

    for episode in range(episodes): 
        observation, info = env.reset(seed=42 + episode)
        terminated = truncated = False
        count = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = (env.step(action))
            count += 1
        pieces_placed.append(count)
        
    print(f"Mean pieces placed: {np.mean(pieces_placed):.2f}")
    base = env.unwrapped

    env.close()

    return final_heights, pieces_placed

if __name__ == "__main__":
    evaluate("tetris_dqnclears_m1_100")
    evaluate("tetris_dqnclears_m1-1_100")
    evaluate("tetris_dqnclears_m1-2_100")
    evaluate("tetris_dqnclears_m1-3_100")
