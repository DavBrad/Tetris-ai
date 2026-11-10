from __future__ import annotations

import time
import argparse
import cv2
import numpy as np
from sb3_contrib import MaskablePPO
from height_train import create_env

WINDOW_NAME = "Tetris AI - Trained Model"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tetris_pieces_m4_100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-ms", type=int, default=500)
    return parser.parse_args()

def main():
    args = parse_args()
    env = create_env()
    model = MaskablePPO.load(args.model)
    observation, info = env.reset()
    terminated = truncated = False
    pieces = 0
    frame = env.render()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
    while not (terminated or truncated):
        frame = env.render()
        cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(args.delay_ms)
        if key in (27, ord("q")):
            break
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks()
        )

        observation, reward, terminated, truncated, info = (env.step(action))

        pieces += 1

        print(f"Episode Finished")
        print(f"Pieces placed: {pieces}")
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()