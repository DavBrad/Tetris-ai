"""Render a random agent choosing complete, legal piece placements."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from environment import make_placement_env


WINDOW_NAME = "Tetris AI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_placement_env(render_mode="rgb_array")

    try:
        _, info = env.reset(seed=args.seed)
        rng = np.random.default_rng(args.seed)
        terminated = truncated = False
        frame = env.render()

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])

        while not (terminated or truncated):
            frame = env.render()
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(args.delay_ms)
            if key in (27, ord("q")):
                break

            legal_actions = np.flatnonzero(info["action_mask"])
            if legal_actions.size == 0:
                raise RuntimeError("The environment reported no legal placements.")

            action = int(rng.choice(legal_actions))
            _, _, terminated, truncated, info = env.step(action)
    finally:
        env.close()
        cv2.destroyAllWindows()

    print("Game over!")

if __name__ == "__main__":
    main()
