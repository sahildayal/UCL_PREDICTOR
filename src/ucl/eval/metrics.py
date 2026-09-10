"""Scoring rules and calibration diagnostics.

Accuracy is close to useless here and the CSCI 635 project shows why: every model
in it could reach ~60% accuracy by predicting home and away well and never
predicting a draw, while remaining badly calibrated on the class that decides
whether a forecast is worth anything.

So the primary metric is log-loss, which is a proper scoring rule and punishes
confident errors, measured against two references:

  * outcomes, the ground truth;
  * the de-vigged closing line, the hardest honest benchmark available.

The second is the one that matters. Beating a coin flip is easy; beating the
market's closing opinion is the question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-15
OUTCOMES = ("home", "draw", "away")


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, EPSILON, 1.0)


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass log-loss. `outcomes` holds column indices 0=home,1=draw,2=away."""
    probabilities = _clip(np.asarray(probabilities, dtype=float))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    chosen = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(chosen)))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score (lower is better)."""
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.zeros_like(probabilities)
    target[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def ranked_probability_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS - the standard football forecasting metric.

    Unlike log-loss it respects the natural ordering home > draw > away, so
    predicting a draw when the away side wins is penalised less than predicting a
    home win. Widely used in the football forecasting literature, which makes
    results here comparable to published work.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.zeros_like(probabilities)
    target[np.arange(len(outcomes)), outcomes] = 1.0
    cumulative_p = np.cumsum(probabilities, axis=1)
    cumulative_t = np.cumsum(target, axis=1)
    return float(np.mean(np.sum((cumulative_p - cumulative_t) ** 2, axis=1)
                         / (probabilities.shape[1] - 1)))


def accuracy(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == outcomes))


def draw_recall(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Recall on draws - the class every model in the CSCI 635 study failed on."""
    predicted = np.argmax(probabilities, axis=1)
    actual_draws = outcomes == 1
    if actual_draws.sum() == 0:
        return float("nan")
    return float(np.mean(predicted[actual_draws] == 1))


def calibration_table(probabilities: np.ndarray, outcomes: np.ndarray,
                      bins: int = 10) -> pd.DataFrame:
    """Reliability table pooled over all three outcome classes."""
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.zeros_like(probabilities)
    target[np.arange(len(outcomes)), outcomes] = 1.0
    flat_p = probabilities.ravel()
    flat_t = target.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append({
            "bin_low": edges[b],
            "bin_high": edges[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(flat_p[mask].mean()),
            "observed_rate": float(flat_t[mask].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray,
                               bins: int = 10) -> float:
    table = calibration_table(probabilities, outcomes, bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def outcome_index(home_goals, away_goals) -> np.ndarray:
    """Map scorelines to 0=home win, 1=draw, 2=away win."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


def summarise(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "n": int(len(outcomes)),
        "log_loss": log_loss(probabilities, outcomes),
        "brier": brier_score(probabilities, outcomes),
        "rps": ranked_probability_score(probabilities, outcomes),
        "accuracy": accuracy(probabilities, outcomes),
        "draw_recall": draw_recall(probabilities, outcomes),
        "ece": expected_calibration_error(probabilities, outcomes),
    }


def compare_to_market(model_probabilities: np.ndarray,
                      market_probabilities: np.ndarray,
                      outcomes: np.ndarray) -> dict:
    """The headline comparison: did the arm beat the de-vigged line?

    A positive `edge_vs_market` means the model's log-loss is LOWER than the
    market's, i.e. the model was better. The EPL lab found this quantity was
    never reliably positive over sixteen seasons. Finding the same here would be
    a replication, not a disappointment.
    """
    model_ll = log_loss(model_probabilities, outcomes)
    market_ll = log_loss(market_probabilities, outcomes)
    return {
        "model_log_loss": model_ll,
        "market_log_loss": market_ll,
        "edge_vs_market": market_ll - model_ll,
        "model_rps": ranked_probability_score(model_probabilities, outcomes),
        "market_rps": ranked_probability_score(market_probabilities, outcomes),
        "n": int(len(outcomes)),
    }
