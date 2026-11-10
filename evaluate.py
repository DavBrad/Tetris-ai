import numpy as np
from sb3_contrib import MaskablePPO
from height_train import create_env

def evaluate(path, episodes=100):
    env = create_env()
    model = MaskablePPO.load(path)
    final_heights = []
    pieces_placed = []

    for episode in range(episodes):
        observation, info = env.reset(seed=42 + episode)
        terminated = truncated = False
        count = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
            observation, reward, terminated, truncated, info = (env.step(action))
            count += 1
        final_height = env.env.get_total_height(observation)
        final_heights.append(final_height)
        pieces_placed.append(count)

    print(f"Mean final height: {np.mean(final_heights):.2f}")
    print(f"Std final height: {np.std(final_heights):.2f}")
    print(f"Mean pieces placed: {np.mean(pieces_placed):.2f}")
    base = env.unwrapped

    env.close()

    return final_heights, pieces_placed

if __name__ == "__main__":
    evaluate("tetris_pieces_m5_100")
