"""Everything derivable from a joint score matrix.

A score matrix `M[i, j]` is the probability of exactly i home goals and j away
goals. Every market this project quotes is a sum over cells of that one matrix.

This is a deliberate structural defence. The EPL lab shipped a bug where a fair
value computed for Over 2.5 was compared against a price for Over 5.5, because
totals were matched on selection ("over") without checking the line, and the two
numbers came from different places. Here they cannot: a line is an argument to a
function over the matrix, so a probability and its line are produced together and
can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_GOALS = 12  # matrix is (MAX_GOALS+1) x (MAX_GOALS+1); P(>12 goals) is ~0


@dataclass(frozen=True)
class Outcome1X2:
    home: float
    draw: float
    away: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.home, self.draw, self.away


def normalise(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total <= 0:
        raise ValueError("Score matrix has non-positive mass")
    return matrix / total


def poisson_matrix(home_lambda: float, away_lambda: float,
                   max_goals: int = MAX_GOALS) -> np.ndarray:
    """Independent Poisson score matrix. The starting point, not the finish."""
    from scipy.stats import poisson
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, max(home_lambda, 1e-6))
    away = poisson.pmf(goals, max(away_lambda, 1e-6))
    return normalise(np.outer(home, away))


def apply_dixon_coles(matrix: np.ndarray, home_lambda: float, away_lambda: float,
                      rho: float) -> np.ndarray:
    """Dixon-Coles low-score dependence correction.

    Independent Poisson systematically misprices 0-0, 1-0, 0-1 and 1-1, which is
    precisely where draws live - the class every model in the CSCI 635 study
    failed on. This correction is the cheapest real improvement available.
    """
    adjusted = matrix.copy()
    lh, la = max(home_lambda, 1e-6), max(away_lambda, 1e-6)
    adjusted[0, 0] *= 1.0 - lh * la * rho
    adjusted[0, 1] *= 1.0 + lh * rho
    adjusted[1, 0] *= 1.0 + la * rho
    adjusted[1, 1] *= 1.0 - rho
    adjusted = np.clip(adjusted, 1e-15, None)
    return normalise(adjusted)


# --- market derivations ------------------------------------------------------

def result_1x2(matrix: np.ndarray) -> Outcome1X2:
    home = float(np.tril(matrix, -1).sum())   # home goals > away goals
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return Outcome1X2(home, draw, away)


def total_goals_distribution(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0] + matrix.shape[1] - 1
    totals = np.zeros(size)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            totals[i + j] += matrix[i, j]
    return totals


def over_under(matrix: np.ndarray, line: float) -> tuple[float, float]:
    """P(total > line), P(total < line) for any line, whole or half.

    Whole lines (2.0, 3.0) push on an exact total, so over + under < 1. The
    caller is handed both and must not assume they sum to one.
    """
    totals = total_goals_distribution(matrix)
    goals = np.arange(len(totals))
    over = float(totals[goals > line].sum())
    under = float(totals[goals < line].sum())
    return over, under


def both_teams_score(matrix: np.ndarray) -> float:
    return float(matrix[1:, 1:].sum())


def asian_handicap(matrix: np.ndarray, line: float) -> tuple[float, float]:
    """P(home covers), P(away covers) for handicap `line` applied to home."""
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    margin = home_goals - away_goals + line
    return float(matrix[margin > 0].sum()), float(matrix[margin < 0].sum())


def correct_score_top(matrix: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    flat = [(i, j, float(matrix[i, j]))
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    flat.sort(key=lambda row: row[2], reverse=True)
    return flat[:n]


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    home_goals = np.arange(matrix.shape[0])
    away_goals = np.arange(matrix.shape[1])
    return (float((matrix.sum(axis=1) * home_goals).sum()),
            float((matrix.sum(axis=0) * away_goals).sum()))


def clean_sheet(matrix: np.ndarray) -> tuple[float, float]:
    """P(home keeps clean sheet), P(away keeps clean sheet)."""
    return float(matrix[:, 0].sum()), float(matrix[0, :].sum())


def summarise_markets(matrix: np.ndarray, lines: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)) -> dict:
    """One dict carrying every market, each tagged with the line it belongs to."""
    result = result_1x2(matrix)
    home_xg, away_xg = expected_goals(matrix)
    totals = {}
    for line in lines:
        over, under = over_under(matrix, line)
        totals[f"over_{line}"] = over
        totals[f"under_{line}"] = under
    return {
        "home": result.home,
        "draw": result.draw,
        "away": result.away,
        "btts": both_teams_score(matrix),
        "home_xg": home_xg,
        "away_xg": away_xg,
        "totals": totals,
        "top_scores": correct_score_top(matrix, 5),
    }
