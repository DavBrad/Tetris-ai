"""Tests for the seeded random-policy baseline."""

from __future__ import annotations

from evaluate_random import evaluate_random_agent


def test_random_baseline_is_reproducible() -> None:
    first_run = evaluate_random_agent(episodes=3, base_seed=42, max_pieces=200)
    second_run = evaluate_random_agent(episodes=3, base_seed=42, max_pieces=200)

    assert first_run == second_run
    assert all(result.terminated for result in first_run)
    assert all(not result.capped for result in first_run)
    assert all(result.pieces_placed > 0 for result in first_run)
