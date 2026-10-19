

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

import gymnasium as gym
import numpy as np

from environment import make_placement_env


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    total_return: float
    lines_cleared: int
    pieces_placed: int
    terminated: bool
    truncated: bool
    capped: bool


def run_episode(
    env: gym.Env,
    episode: int,
    seed: int,
    max_pieces: int,
) -> EpisodeResult:
    """Run one random-policy episode using only legal placements."""
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_return = 0.0
    lines_cleared = 0
    pieces_placed = 0
    terminated = truncated = False

    while not (terminated or truncated) and pieces_placed < max_pieces:
        legal_actions = np.flatnonzero(info["action_mask"])
        if legal_actions.size == 0:
            raise RuntimeError("The environment reported no legal placement actions.")

        action = int(rng.choice(legal_actions))
        _, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        lines_cleared += int(info["lines_cleared"])
        pieces_placed += 1

    return EpisodeResult(
        episode=episode,
        seed=seed,
        total_return=total_return,
        lines_cleared=lines_cleared,
        pieces_placed=pieces_placed,
        terminated=terminated,
        truncated=truncated,
        capped=not (terminated or truncated),
    )


def evaluate_random_agent(
    episodes: int,
    base_seed: int,
    max_pieces: int,
) -> list[EpisodeResult]:
    """Evaluate a random legal-placement policy over consecutive seeds."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_pieces < 1:
        raise ValueError("max_pieces must be at least 1")

    env = make_placement_env()
    try:
        return [
            run_episode(env, episode, base_seed + episode, max_pieces)
            for episode in range(episodes)
        ]
    finally:
        env.close()


def write_results(results: list[EpisodeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def format_summary(results: list[EpisodeResult]) -> str:
    returns = [result.total_return for result in results]
    lines = [result.lines_cleared for result in results]
    pieces = [result.pieces_placed for result in results]

    return "\n".join(
        [
            f"Episodes: {len(results)}",
            f"Return: mean={fmean(returns):.2f}, std={pstdev(returns):.2f}",
            f"Lines cleared: mean={fmean(lines):.2f}, std={pstdev(lines):.2f}",
            f"Pieces placed: mean={fmean(pieces):.2f}, std={pstdev(pieces):.2f}",
            f"Safety-capped episodes: {sum(result.capped for result in results)}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pieces", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_random_agent(args.episodes, args.seed, args.max_pieces)

    if args.output is not None:
        write_results(results, args.output)
        print(f"Wrote results to {args.output}")

    print(format_summary(results))


if __name__ == "__main__":
    main()
