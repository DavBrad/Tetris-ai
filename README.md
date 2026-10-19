# Tetris AI

This repository contains a university artificial intelligence project focused on training reinforcement-learning agents to play Tetris. The core goal is to build and evaluate an agent that learns from structured game-state information rather than screenshots.

Potential stretch goals include comparing differently sized agents through knowledge distillation or teacher-guided learning and exploring competitive Tetris. These are secondary to delivering a reproducible single-agent experiment.

## Current milestone

The current milestone establishes:

- a reproducible Tetris environment;
- a placement-level action interface suitable for reinforcement learning;
- a seeded random-policy baseline;
- automated environment-contract tests; and
- shared project scope and evaluation metrics.

No learning model has been implemented yet.

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

To run the visual random-placement demo:

```powershell
python quickstart.py
```

To run the headless random placement baseline:

```powershell
python evaluate_random.py --episodes 50 --seed 42 --output results/random_baseline.csv
```

To run the tests:

```powershell
python -m pytest -q -p no:cacheprovider
```

## Environment contract

The project currently uses `tetris-gymnasium==0.3.1`.

The shared environment constructor includes a compact renderer that hides the
library's internal collision padding, shows only the upcoming pieces in the
right-side preview, and fixes RGB rendering with current NumPy releases. Use
the constructors in `environment.py` instead of instantiating the third-party
environment directly.

The raw environment exposes a dictionary observation:

| Field | Shape | Description |
| --- | ---: | --- |
| `board` | `(24, 18)` | A 20×10 board surrounded by collision padding |
| `active_tetromino_mask` | `(24, 18)` | Location of the currently controlled piece |
| `holder` | `(4, 4)` | Held piece |
| `queue` | `(4, 16)` | Upcoming pieces |

The raw environment has eight step-level actions: move left, move right, move down, rotate clockwise, rotate counterclockwise, hard drop, swap, and no-op.

For initial learning experiments, `environment.make_placement_env()` applies the library's grouped-action wrapper. Its 40 discrete actions represent every combination of 10 columns and 4 rotations:

```text
action = column * 4 + rotations
```

Each action places one complete piece. The observation contains the 40 candidate post-placement boards with shape `(40, 24, 18)`. The `info["action_mask"]` array identifies legal placements. The local factory normalizes these observations to `float32` to match the declared Gymnasium observation space.

The default reward is one point for placing and surviving a piece, plus `10 × lines_cleared²`. A terminal placement receives zero reward. Evaluation therefore records return, lines cleared, and pieces placed separately.

## Evaluation protocol

Until a trained agent exists, the random placement policy is the baseline. Experiments should:

- use the placement-level environment;
- choose only actions allowed by `info["action_mask"]`;
- report results over multiple consecutive seeds;
- record total return, lines cleared, pieces placed, and termination state; and
- compare models on the same seeds and environment configuration.

The baseline evaluator uses seed `42` and 50 episodes by default. An episode is capped at 1,000 placed pieces as a safety limit.

### Initial random baseline

The first 50-episode run on October 12, 2026 produced:

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Return | 17.34 | 3.63 |
| Lines cleared | 0.00 | 0.00 |
| Pieces placed | 18.34 | 3.63 |

All episodes terminated normally, and none reached the safety cap. These intentionally weak results establish a concrete threshold that a learning agent must exceed.

## Minimum viable project

The core project is complete when it has:

1. A tested and documented Tetris environment.
2. A random-policy baseline.
3. At least one trained reinforcement-learning agent.
4. Reproducible evaluation across fixed seeds.
5. Training curves and a final comparison against the baseline.
6. A visual gameplay demonstration.

Knowledge distillation is the first stretch goal. Competitive Tetris is attempted only after the core project is complete.

## Working schedule

| Target week | Outcome |
| --- | --- |
| Oct. 12–18 | Environment contract, tests, and random baseline |
| Oct. 19–25 | Final observations, actions, rewards, and experiment metrics |
| Oct. 26–Nov. 1 | First reinforcement-learning training pipeline |
| Nov. 2–8 | Meaningful training runs and baseline comparison |
| Nov. 9–15 | Reward, representation, and hyperparameter improvements |
| Nov. 16–22 | Freeze and evaluate a reliable core model |
| Nov. 23–29 | Smaller model and distillation experiment, if core work is complete |
| Nov. 30–Dec. 6 | Final comparisons, plots, and gameplay demonstrations |
| Dec. 7–12 | Report, presentation, repository cleanup, and reproduction pass |
| Dec. 13–15 | Submission buffer |

An internal deadline of November 22 keeps the stretch goals from putting the core submission at risk.

## Suggested ownership

- **Environment track:** wrappers, environment tests, reward experiments, and visualization.
- **Experiment track:** baseline evaluation, training pipeline, metrics, plots, and result tracking.

Both partners should review changes to the environment contract because those decisions affect every experiment.
