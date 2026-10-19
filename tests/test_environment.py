"""Contract tests for the shared Tetris environment constructors."""

from __future__ import annotations

import numpy as np

from environment import BASE_ACTIONS, make_placement_env, make_step_env


def test_step_environment_accepts_every_declared_action() -> None:
    env = make_step_env()
    try:
        assert env.action_space.n == len(BASE_ACTIONS) == 8

        for action in BASE_ACTIONS:
            observation, _ = env.reset(seed=42)
            assert env.observation_space.contains(observation)

            observation, _, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(observation)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "lines_cleared" in info
    finally:
        env.close()


def test_reset_is_reproducible_for_a_fixed_seed() -> None:
    first_env = make_step_env()
    second_env = make_step_env()
    try:
        first_observation, _ = first_env.reset(seed=123)
        second_observation, _ = second_env.reset(seed=123)

        assert first_observation.keys() == second_observation.keys()
        for key in first_observation:
            np.testing.assert_array_equal(
                first_observation[key], second_observation[key]
            )
    finally:
        first_env.close()
        second_env.close()


def test_rgb_rendering_with_current_numpy() -> None:
    env = make_step_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        frame = env.render()

        assert frame.shape == (480, 360, 3)
        assert frame.dtype == np.uint8
        assert not np.any(np.all(frame == [128, 128, 128], axis=-1))
    finally:
        env.close()


def test_completed_horizontal_line_is_cleared_and_rewarded() -> None:
    env = make_step_env()
    try:
        env.reset(seed=42)
        base_env = env.unwrapped

        # Leave a four-cell gap in the bottom row for a horizontal I piece.
        playable_columns = slice(
            base_env.padding,
            base_env.padding + base_env.width,
        )
        bottom_row = base_env.height - 1
        base_env.board[bottom_row, playable_columns] = 2
        gap_start = base_env.padding + 3
        base_env.board[bottom_row, gap_start : gap_start + 4] = 0

        base_env.active_tetromino = base_env.tetrominoes[0]
        base_env.x = gap_start
        base_env.y = 0

        _, reward, terminated, truncated, info = env.step(
            base_env.actions.hard_drop
        )

        assert info["lines_cleared"] == 1
        assert reward == 11
        assert not terminated
        assert not truncated
        assert not base_env.board[
            : base_env.height,
            playable_columns,
        ].any()
    finally:
        env.close()


def test_placement_environment_matches_its_contract() -> None:
    env = make_placement_env()
    try:
        observation, info = env.reset(seed=42)

        assert env.action_space.n == 40
        assert observation.shape == (40, 24, 18)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)

        action_mask = info["action_mask"]
        assert action_mask.shape == (40,)
        assert set(np.unique(action_mask)).issubset({0, 1})
        assert action_mask.any()

        action = int(np.flatnonzero(action_mask)[0])
        next_observation, _, _, _, next_info = env.step(action)

        assert env.observation_space.contains(next_observation)
        assert next_info["action_mask"].shape == (40,)
        assert "lines_cleared" in next_info
    finally:
        env.close()


def test_placement_matches_selected_candidate_board() -> None:
    env = make_placement_env()
    try:
        candidate_boards, info = env.reset(seed=42)
        action = int(np.flatnonzero(info["action_mask"])[5])
        expected_board = candidate_boards[action].astype(np.uint8)

        env.step(action)

        np.testing.assert_array_equal(env.unwrapped.board, expected_board)
    finally:
        env.close()


def test_next_piece_promotion_and_preview_stay_synchronized() -> None:
    env = make_placement_env()
    try:
        _, info = env.reset(seed=42)
        base_env = env.unwrapped
        queue_before = base_env.queue.get_queue()
        expected_next_piece = base_env.tetrominoes[queue_before[0]]

        action = int(np.flatnonzero(info["action_mask"])[0])
        env.step(action)

        queue_after = base_env.queue.get_queue()
        assert base_env.active_tetromino is expected_next_piece
        assert queue_after[:-1] == queue_before[1:]

        observation = base_env._get_obs()
        preview_chunks = np.hsplit(
            observation["queue"], base_env.queue.size
        )
        for preview, piece_index in zip(preview_chunks, queue_after):
            piece = base_env.tetrominoes[piece_index]
            expected_preview = np.pad(
                piece.matrix,
                (
                    (0, base_env.padding - piece.matrix.shape[0]),
                    (0, base_env.padding - piece.matrix.shape[1]),
                ),
            )
            np.testing.assert_array_equal(preview, expected_preview)
    finally:
        env.close()
