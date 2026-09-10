"""Cross-league Elo. The control arm.

Deliberately the least clever model in the stack. Elo has one state variable per
club, no league term, no attack/defence split, and no notion of uncertainty. It
learns cross-league strength only implicitly, by clubs carrying their rating into
European matches.

It is here to answer a question the other arms cannot ask of themselves: was any
of the complexity worth it? The CSCI 635 study's most durable finding was that
simple, well-calibrated models kept pace with elaborate ones on football data. If
Elo tracks the Bayesian backbone all season, that finding replicates and the
honest conclusion is to prefer the simple model. Reporting that would be a result,
not a failure.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..scoring.matrix import apply_dixon_coles, poisson_matrix
from .base import ModelArm, Prediction


class CrossLeagueElo(ModelArm):
    name = "elo_xleague"
    hypothesis = "A single rating per club, updated match by match, is enough."

    def __init__(
        self,
        *,
        k_factor: float = 20.0,
        home_advantage: float = 60.0,
        goal_difference_scaling: bool = True,
        initial_rating: float = 1500.0,
        regress_to_mean: float = 0.0,
        max_goals: int = 12,
        rho: float = -0.04,
    ) -> None:
        super().__init__()
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.goal_difference_scaling = goal_difference_scaling
        self.initial_rating = initial_rating
        self.regress_to_mean = regress_to_mean
        self.max_goals = max_goals
        self.rho = rho

        self.ratings: dict[str, float] = {}
        self.match_counts: dict[str, int] = {}
        # Mapping from rating difference to goal expectations, fitted on history.
        self._intercept: float = 0.0
        self._slope: float = 0.0
        self._home_bump: float = 0.0

    # -- fitting --------------------------------------------------------------
    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "CrossLeagueElo":
        data = self._training_slice(corpus, as_of).sort_values("date")
        if data.empty:
            raise ValueError("No training data for CrossLeagueElo")

        ratings: dict[str, float] = {}
        counts: dict[str, int] = {}
        records: list[tuple[float, float, float]] = []

        for row in data.itertuples(index=False):
            home_rating = ratings.get(row.home, self.initial_rating)
            away_rating = ratings.get(row.away, self.initial_rating)

            diff = home_rating + self.home_advantage - away_rating
            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            home_goals, away_goals = float(row.home_goals), float(row.away_goals)
            if home_goals > away_goals:
                actual = 1.0
            elif home_goals < away_goals:
                actual = 0.0
            else:
                actual = 0.5

            # Record the pre-match state so the goal mapping is fitted only on
            # information available before kickoff.
            records.append((diff, home_goals, away_goals))

            k = self.k_factor
            if self.goal_difference_scaling:
                # A three-goal win should move ratings more than a one-goal win,
                # damped so blowouts do not dominate.
                margin = abs(home_goals - away_goals)
                k *= float(np.sqrt(max(margin, 1.0)))

            change = k * (actual - expected)
            ratings[row.home] = home_rating + change
            ratings[row.away] = away_rating - change
            counts[row.home] = counts.get(row.home, 0) + 1
            counts[row.away] = counts.get(row.away, 0) + 1

        self.ratings = ratings
        self.match_counts = counts
        self._fit_goal_mapping(records)
        self.fitted = True
        return self

    def _fit_goal_mapping(self, records: list[tuple[float, float, float]]) -> None:
        """Map a rating difference onto Poisson rates.

        Elo natively predicts a win probability, not a scoreline. To honour the
        shared score-matrix interface - so this arm quotes totals and BTTS like
        every other - the rating gap is mapped to expected goals by fitting
            log(home rate) = a + b * diff + c
            log(away rate) = a - b * diff
        by weighted Poisson likelihood over history.
        """
        if not records:
            return
        diffs = np.array([r[0] for r in records]) / 400.0
        home_goals = np.array([r[1] for r in records])
        away_goals = np.array([r[2] for r in records])

        def negll(theta: np.ndarray) -> float:
            intercept, slope, home_bump = theta
            log_lh = intercept + home_bump + slope * diffs
            log_la = intercept - slope * diffs
            lam_h = np.exp(np.clip(log_lh, -8, 3))
            lam_a = np.exp(np.clip(log_la, -8, 3))
            return float(np.sum(lam_h - home_goals * log_lh + lam_a - away_goals * log_la))

        result = minimize(negll, np.array([np.log(1.3), 0.35, 0.2]), method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        self._intercept, self._slope, self._home_bump = (float(v) for v in result.x)

    # -- prediction -----------------------------------------------------------
    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        if not self.fitted:
            raise RuntimeError("CrossLeagueElo.predict called before fit")
        if home not in self.ratings or away not in self.ratings:
            return None
        diff = (self.ratings[home] + self.home_advantage - self.ratings[away]) / 400.0
        lam_h = float(np.exp(np.clip(self._intercept + self._home_bump + self._slope * diff, -8, 3)))
        lam_a = float(np.exp(np.clip(self._intercept - self._slope * diff, -8, 3)))
        matrix = poisson_matrix(lam_h, lam_a, self.max_goals)
        if self.rho:
            matrix = apply_dixon_coles(matrix, lam_h, lam_a, self.rho)
        return Prediction(
            arm=self.name, home=home, away=away, matrix=matrix,
            home_lambda=lam_h, away_lambda=lam_a,
            meta={"home_rating": self.ratings[home], "away_rating": self.ratings[away],
                  "home_matches": self.match_counts.get(home, 0),
                  "away_matches": self.match_counts.get(away, 0)},
        )

    def rating_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"club": c, "rating": r, "matches": self.match_counts.get(c, 0)}
             for c, r in self.ratings.items()]
        ).sort_values("rating", ascending=False).reset_index(drop=True)
