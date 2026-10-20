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


class LeagueOffsetElo(CrossLeagueElo):
    """Version 2 of the control arm: Elo with a rating per league as well.

    Version 1 pools every club into one rating scale, but domestic leagues are
    closed systems: a club that wins most of its league games climbs whether the
    league is the Bundesliga or the Azerbaijan Premier League, because almost
    every rating point it gains comes from domestic opponents. Only the handful
    of European matches carry information across leagues, and they are drowned
    out. After two matchdays v1 scored worse than the market by 0.12 log-loss
    per match.

    Here a club's strength is `club rating + league offset`. Domestic matches
    move club ratings only (the offset cancels between two clubs of the same
    league). Cross-border matches move both, and they are the only thing that
    moves the offsets. Ratings also regress part of the way to the mean at
    each new season, which v1 never did.
    """

    name = "elo_xleague_v2"
    hypothesis = ("A single club rating is enough once each league carries its own "
                  "offset, learned only from cross-border matches.")

    def __init__(
        self,
        *,
        k_factor: float = 20.0,
        european_k_factor: float = 30.0,
        league_k_factor: float = 6.0,
        season_regression: float = 0.2,
        burn_in_matches: int = 20,
        off_season_days: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(k_factor=k_factor, **kwargs)
        self.european_k_factor = european_k_factor
        self.league_k_factor = league_k_factor
        self.season_regression = season_regression
        self.burn_in_matches = burn_in_matches
        self.off_season_days = off_season_days
        self.league_offsets: dict[str, float] = {}
        self.club_league: dict[str, str] = {}

    @staticmethod
    def _club_leagues(data: pd.DataFrame) -> dict[str, str]:
        """A club's league is the country it appears with most often."""
        pairs = pd.concat([
            data[["home", "home_country"]].set_axis(["club", "country"], axis=1),
            data[["away", "away_country"]].set_axis(["club", "country"], axis=1),
        ])
        pairs = pairs[pairs["country"].astype(bool)]
        return {club: group["country"].mode().iloc[0]
                for club, group in pairs.groupby("club")}

    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "LeagueOffsetElo":
        data = self._training_slice(corpus, as_of).sort_values("date")
        if data.empty:
            raise ValueError("No training data for LeagueOffsetElo")

        club_league = self._club_leagues(data)
        ratings: dict[str, float] = {}
        offsets: dict[str, float] = {}
        counts: dict[str, int] = {}
        last_played: dict[str, pd.Timestamp] = {}
        records: list[tuple[float, float, float]] = []

        for row in data.itertuples(index=False):
            home_league = club_league.get(row.home, "")
            away_league = club_league.get(row.away, "")
            # A long gap is an off-season. The corpus's season labels cannot be
            # used: a Norwegian club alternates "2025" domestic and "2025-26"
            # European rows, which would regress it every few weeks.
            for club in (row.home, row.away):
                last = last_played.get(club)
                if last is not None and (row.date - last).days > self.off_season_days:
                    rating = ratings[club]
                    ratings[club] = rating - self.season_regression * (rating - self.initial_rating)
                last_played[club] = row.date

            home_club = ratings.get(row.home, self.initial_rating)
            away_club = ratings.get(row.away, self.initial_rating)
            home_offset = offsets.get(home_league, 0.0)
            away_offset = offsets.get(away_league, 0.0)
            cross = home_league != away_league

            diff = (home_club + home_offset) + self.home_advantage - (away_club + away_offset)
            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            home_goals, away_goals = float(row.home_goals), float(row.away_goals)
            actual = 1.0 if home_goals > away_goals else (0.0 if home_goals < away_goals else 0.5)

            settled_in = (counts.get(row.home, 0) >= self.burn_in_matches
                          and counts.get(row.away, 0) >= self.burn_in_matches)
            if cross and settled_in:
                records.append((diff, home_goals, away_goals))

            margin = float(np.sqrt(max(abs(home_goals - away_goals), 1.0)))
            surprise = actual - expected
            k = (self.european_k_factor if cross else self.k_factor) * margin
            ratings[row.home] = home_club + k * surprise
            ratings[row.away] = away_club - k * surprise
            if cross:
                step = self.league_k_factor * margin * surprise
                offsets[home_league] = home_offset + step
                offsets[away_league] = away_offset - step
            counts[row.home] = counts.get(row.home, 0) + 1
            counts[row.away] = counts.get(row.away, 0) + 1

        # predict() reads self.ratings, so fold each league's offset in.
        self.ratings = {club: rating + offsets.get(club_league.get(club, ""), 0.0)
                        for club, rating in ratings.items()}
        self.league_offsets = offsets
        self.club_league = club_league
        self.match_counts = counts
        self._fit_goal_mapping(records)
        self.fitted = True
        return self
