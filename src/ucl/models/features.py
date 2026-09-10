"""Pre-match feature construction.

Every feature here is computed from matches strictly *before* the one being
described. That discipline is the whole ballgame: the CSCI 635 project's data
section spends most of its length on exactly this point, because shots-on-target
and possession are wonderful predictors of a match that has already finished and
worthless for one that has not.

Features are built by a single chronological pass, updating per-club state as it
goes, so a row can only ever see its own past.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

FORM_WINDOW = 8

FEATURE_NAMES = [
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_points_avg", "away_points_avg",
    "home_rest_days", "away_rest_days",
    "home_matches_seen", "away_matches_seen",
    "is_cross_border", "is_ucl",
    "home_euro_gf_avg", "home_euro_ga_avg",
    "away_euro_gf_avg", "away_euro_ga_avg",
    "backbone_log_lh", "backbone_log_la", "backbone_log_ratio",
]


class _ClubState:
    __slots__ = ("goals_for", "goals_against", "points", "last_date",
                 "euro_gf", "euro_ga", "seen")

    def __init__(self) -> None:
        self.goals_for: deque[float] = deque(maxlen=FORM_WINDOW)
        self.goals_against: deque[float] = deque(maxlen=FORM_WINDOW)
        self.points: deque[float] = deque(maxlen=FORM_WINDOW)
        self.euro_gf: deque[float] = deque(maxlen=FORM_WINDOW)
        self.euro_ga: deque[float] = deque(maxlen=FORM_WINDOW)
        self.last_date = None
        self.seen = 0

    @staticmethod
    def _mean(values: deque[float], default: float) -> float:
        return float(np.mean(values)) if values else default


def build_features(
    corpus: pd.DataFrame,
    backbone=None,
    *,
    rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a feature frame aligned to `rows` (defaults to the whole corpus).

    `backbone` is an optional fitted arm whose lambdas become features, which is
    what makes the tree models *residual* learners: they start from the
    Bayesian/Dixon-Coles view of a fixture and learn where it goes wrong, rather
    than rediscovering that Bayern are good.
    """
    target = corpus if rows is None else rows
    target_keys = set(zip(target["date"], target["home"], target["away"]))

    state: dict[str, _ClubState] = defaultdict(_ClubState)
    records: dict[tuple, dict] = {}

    for row in corpus.sort_values("date").itertuples(index=False):
        home_state = state[row.home]
        away_state = state[row.away]
        key = (row.date, row.home, row.away)

        if key in target_keys:
            home_rest = ((row.date - home_state.last_date).days
                         if home_state.last_date is not None else 7)
            away_rest = ((row.date - away_state.last_date).days
                         if away_state.last_date is not None else 7)

            feature_row = {
                "home_gf_avg": home_state._mean(home_state.goals_for, 1.3),
                "home_ga_avg": home_state._mean(home_state.goals_against, 1.3),
                "away_gf_avg": away_state._mean(away_state.goals_for, 1.3),
                "away_ga_avg": away_state._mean(away_state.goals_against, 1.3),
                "home_points_avg": home_state._mean(home_state.points, 1.3),
                "away_points_avg": away_state._mean(away_state.points, 1.3),
                "home_rest_days": float(min(home_rest, 30)),
                "away_rest_days": float(min(away_rest, 30)),
                "home_matches_seen": float(home_state.seen),
                "away_matches_seen": float(away_state.seen),
                "is_cross_border": float(bool(row.cross_border)),
                "is_ucl": float(bool(row.is_ucl)),
                "home_euro_gf_avg": home_state._mean(home_state.euro_gf, 1.2),
                "home_euro_ga_avg": home_state._mean(home_state.euro_ga, 1.2),
                "away_euro_gf_avg": away_state._mean(away_state.euro_gf, 1.2),
                "away_euro_ga_avg": away_state._mean(away_state.euro_ga, 1.2),
            }

            if backbone is not None:
                pair = None
                if hasattr(backbone, "lambdas"):
                    pair = backbone.lambdas(row.home, row.away,
                                            euro=bool(row.cross_border))
                if pair is None:
                    feature_row["backbone_log_lh"] = np.nan
                    feature_row["backbone_log_la"] = np.nan
                    feature_row["backbone_log_ratio"] = np.nan
                else:
                    lam_h, lam_a = pair
                    feature_row["backbone_log_lh"] = float(np.log(max(lam_h, 1e-6)))
                    feature_row["backbone_log_la"] = float(np.log(max(lam_a, 1e-6)))
                    feature_row["backbone_log_ratio"] = float(
                        np.log(max(lam_h, 1e-6)) - np.log(max(lam_a, 1e-6)))
            else:
                feature_row["backbone_log_lh"] = np.nan
                feature_row["backbone_log_la"] = np.nan
                feature_row["backbone_log_ratio"] = np.nan

            records[key] = feature_row

        # --- update state AFTER the row has been described -------------------
        if not row.played:
            continue
        home_goals, away_goals = float(row.home_goals), float(row.away_goals)
        home_points = 3.0 if home_goals > away_goals else (1.0 if home_goals == away_goals else 0.0)
        away_points = 3.0 if away_goals > home_goals else (1.0 if home_goals == away_goals else 0.0)

        home_state.goals_for.append(home_goals)
        home_state.goals_against.append(away_goals)
        home_state.points.append(home_points)
        home_state.last_date = row.date
        home_state.seen += 1

        away_state.goals_for.append(away_goals)
        away_state.goals_against.append(home_goals)
        away_state.points.append(away_points)
        away_state.last_date = row.date
        away_state.seen += 1

        if row.cross_border:
            home_state.euro_gf.append(home_goals)
            home_state.euro_ga.append(away_goals)
            away_state.euro_gf.append(away_goals)
            away_state.euro_ga.append(home_goals)

    frame = pd.DataFrame(
        [records.get((d, h, a), {name: np.nan for name in FEATURE_NAMES})
         for d, h, a in zip(target["date"], target["home"], target["away"])],
        index=target.index,
    )
    return frame.reindex(columns=FEATURE_NAMES)
