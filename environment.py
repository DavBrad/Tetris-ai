

"""Shared constructors for the project's Tetris environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations


BASE_ACTIONS = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "hard_drop",
    6: "swap",
    7: "no_op",
}


class TetrisEnvironment(Tetris):
    """Tetris environment with a compact, NumPy-compatible RGB renderer."""

    def get_rgb(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        board = observation["board"][
            : self.height,
            self.padding : self.padding + self.width,
        ]
        queue = observation["queue"].copy()
        queue[queue == 1] = 0

        queue_piece_count = queue.shape[1] // self.padding
        queue_vertical = np.vstack(np.hsplit(queue, queue_piece_count))

        panel_width = self.padding
        panel_start = self.width + 1
        render_grid = np.zeros(
            (self.height, panel_start + panel_width), dtype=np.intp
        )
        render_grid[:, : self.width] = board

        queue_height = min(queue_vertical.shape[0], self.height - self.padding)
        render_grid[:queue_height, panel_start:] = queue_vertical[:queue_height]

        colors = np.array(
            [pixel.color_rgb for pixel in self.pixels], dtype=np.uint8
        )
        return colors[render_grid]


def make_step_env(
    render_mode: str | None = None,
    render_upscale: int = 24,
) -> gym.Env:
    """Create the raw, step-level Tetris environment."""
    return TetrisEnvironment(
        render_mode=render_mode,
        render_upscale=render_upscale,
    )


def make_placement_env(
    render_mode: str | None = None,
    render_upscale: int = 24,
) -> gym.Env:
    """Create the placement-level environment used by learning agents.

    The upstream grouped-action wrapper declares float32 observations but emits
    uint8 arrays. TransformObservation keeps the runtime values consistent with
    the declared observation space so Gymnasium agents and checkers can rely on
    the environment contract.
    """
    grouped_env = GroupedActionsObservations(
        TetrisEnvironment(
            render_mode=render_mode,
            render_upscale=render_upscale,
        )
    )
    return gym.wrappers.TransformObservation(
        grouped_env,
        lambda observation: observation.astype(np.float32),
        observation_space=grouped_env.observation_space,
    )
