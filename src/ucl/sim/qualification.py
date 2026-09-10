"""Monte Carlo over the remaining league phase.

The league-phase format makes this both harder and more interesting than a group
stage. All 36 clubs sit in one table, each plays a different set of eight
opponents, and the cut lines are:

    1-8    direct to the round of 16
    9-24   knockout-phase playoff
    25-36  eliminated

Because opponents differ, a club's fate depends on the entire fixture graph, not
just its own remaining matches. That is only answerable by simulation, and it is
the output that stays useful all season - after matchday 3 a single result can
swing several clubs across the top-8 line.

Scorelines are sampled from each arm's joint score matrix, so the simulation
inherits whatever the arm believes about goal distributions rather than assuming
Poisson a second time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AUTO_R16_CUTOFF, PLAYOFF_CUTOFF


@dataclass
class SimulationResult:
    table: pd.DataFrame          # per-club qualification probabilities
    simulations: int
    points: pd.DataFrame         # per-club points distribution summary


def _sample_scores(matrix: np.ndarray, size: int, rng: np.random.Generator):
    """Draw `size` scorelines from a joint score matrix."""
    flat = matrix.ravel()
    flat = flat / flat.sum()
    picks = rng.choice(len(flat), size=size, p=flat)
    return np.unravel_index(picks, matrix.shape)


def simulate(
    fixtures: pd.DataFrame,
    predictions: dict[tuple[str, str], np.ndarray],
    *,
    simulations: int = 20_000,
    seed: int = 20262027,
) -> SimulationResult:
    """Simulate the league phase.

    `fixtures` must contain every league-phase match, played and unplayed, with
    `home`, `away`, `home_goals`, `away_goals` and `played`. Played matches are
    held fixed; unplayed ones are sampled from `predictions`, keyed by
    (home, away). A fixture with no prediction is skipped and reported, never
    silently treated as a draw.
    """
    rng = np.random.default_rng(seed)
    clubs = sorted(set(fixtures["home"]) | set(fixtures["away"]))
    index = {c: i for i, c in enumerate(clubs)}
    n = len(clubs)

    points = np.zeros((simulations, n), dtype=np.int16)
    goal_diff = np.zeros((simulations, n), dtype=np.int16)
    goals_for = np.zeros((simulations, n), dtype=np.int16)

    missing: list[tuple[str, str]] = []

    for row in fixtures.itertuples(index=False):
        h, a = index[row.home], index[row.away]

        if bool(row.played):
            hg = np.full(simulations, int(row.home_goals), dtype=np.int16)
            ag = np.full(simulations, int(row.away_goals), dtype=np.int16)
        else:
            matrix = predictions.get((row.home, row.away))
            if matrix is None:
                missing.append((row.home, row.away))
                continue
            hg, ag = _sample_scores(matrix, simulations, rng)
            hg = hg.astype(np.int16)
            ag = ag.astype(np.int16)

        home_win = hg > ag
        away_win = ag > hg
        draw = hg == ag

        points[:, h] += np.where(home_win, 3, np.where(draw, 1, 0)).astype(np.int16)
        points[:, a] += np.where(away_win, 3, np.where(draw, 1, 0)).astype(np.int16)
        goal_diff[:, h] += (hg - ag).astype(np.int16)
        goal_diff[:, a] += (ag - hg).astype(np.int16)
        goals_for[:, h] += hg
        goals_for[:, a] += ag

    # Rank within each simulation: points, then goal difference, then goals for.
    # Lexsort takes the LAST key as primary, so order is reversed here.
    order = np.lexsort((goals_for, goal_diff, points), axis=1)[:, ::-1]
    positions = np.empty_like(order)
    rows = np.arange(simulations)[:, None]
    positions[rows, order] = np.arange(1, n + 1)[None, :]

    table = pd.DataFrame({
        "club": clubs,
        "top8": (positions <= AUTO_R16_CUTOFF).mean(axis=0),
        "playoff": ((positions > AUTO_R16_CUTOFF) & (positions <= PLAYOFF_CUTOFF)).mean(axis=0),
        "eliminated": (positions > PLAYOFF_CUTOFF).mean(axis=0),
        "mean_points": points.mean(axis=0),
        "mean_position": positions.mean(axis=0),
        "p10_points": np.percentile(points, 10, axis=0),
        "p90_points": np.percentile(points, 90, axis=0),
    })
    table["advance"] = table["top8"] + table["playoff"]
    table = table.sort_values(["top8", "advance", "mean_points"], ascending=False)
    table = table.reset_index(drop=True)
    table.attrs["missing_predictions"] = missing

    points_summary = pd.DataFrame({
        "club": clubs,
        "mean": points.mean(axis=0),
        "sd": points.std(axis=0),
        "min": points.min(axis=0),
        "max": points.max(axis=0),
    }).sort_values("mean", ascending=False).reset_index(drop=True)

    return SimulationResult(table=table, simulations=simulations, points=points_summary)
