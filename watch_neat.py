from __future__ import annotations

import argparse
import pickle

import cv2
import neat
import numpy as np

from neat_clears_train import create_env, CONFIG_PATH, choose_action

WINDOW_NAME = "Tetris AI - Trained Model"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, default="best_genome.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-ms", type=int, default=500)
    return parser.parse_args()

def main():
    args = parse_args()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                         CONFIG_PATH)
    with open(args.genome, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print("Loaded genome:", genome.key)
    print("Fitness:", genome.fitness)

    env = create_env()

    try:
        observation,info = env.reset(seed=args.seed)
        terminated = False
        truncated = False
        pieces = 0
        total_reward = 0.0
        frame = env.render()
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            frame.shape[1],
            frame.shape[0]
        )

        while not (terminated or truncated):
            frame = env.render()

            cv2.imshow(
                WINDOW_NAME,
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )

            key = cv2.waitKey(args.delay_ms)

            if key in (27, ord("q")):
                print("Playback stopped.")
                break

            action = choose_action(
                net,
                observation,
                info["action_mask"]
            )

            # Execute placement
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            pieces += 1
            total_reward += reward
        print("\nEpisode finished!")
        print("Pieces placed:", pieces)
        print("Total reward:", total_reward)
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()