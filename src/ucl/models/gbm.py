"""Gradient-boosted residual arm.

The direct descendant of the CSCI 635 work, with one structural change. That
project asked tree ensembles to predict match outcomes from engineered features
and found they plateaued at roughly the same place as logistic regression - the
signal-to-noise ceiling, not a modelling failure.

Here the trees are not asked to rediscover that Bayern are good. The Dixon-Coles
backbone supplies log-rate features, and LightGBM predicts goals on top of them
with a Poisson objective. What is left for the trees is the part the backbone
structurally cannot see: fatigue, rest asymmetry, recent form diverging from
long-run strength, and whether a club travels better or worse than its rating.

If the trees add nothing over the backbone, this arm converges to it, and that is
a clean, interpretable result rather than a wasted arm.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..scoring.matrix import apply_dixon_coles, poisson_matrix
from .base import ModelArm, Prediction
from .features import FEATURE_NAMES, build_features


class GradientBoostedGoals(ModelArm):
    name = "gbm"
    hypothesis = (
        "Tree ensembles on form, rest and fatigue features improve on a "
        "ratings-only backbone."
    )

    def __init__(
        self,
        *,
        backbone=None,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 60,
        recent_years: float = 8.0,
        max_goals: int = 12,
        rho: float = -0.04,
        seed: int = 20262027,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.recent_years = recent_years
        self.max_goals = max_goals
        self.rho = rho
        self.seed = seed
        self.model_home = None
        self.model_away = None
        self._corpus: pd.DataFrame | None = None

    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "GradientBoostedGoals":
        import lightgbm as lgb

        self._corpus = corpus
        data = self._training_slice(corpus, as_of)
        data = data[data["days_ago"] <= self.recent_years * 365.25]
        if data.empty:
            raise ValueError("No training data for GradientBoostedGoals")

        features = build_features(corpus[corpus["date"] <= data["date"].max()],
                                  self.backbone, rows=data)
        weights = data["weight"].to_numpy(dtype=float)

        common = dict(
            objective="poisson",              # goals are counts, not a Gaussian
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=self.seed,
            verbose=-1,
        )
        self.model_home = lgb.LGBMRegressor(**common).fit(
            features, data["home_goals"].to_numpy(dtype=float), sample_weight=weights)
        self.model_away = lgb.LGBMRegressor(**common).fit(
            features, data["away_goals"].to_numpy(dtype=float), sample_weight=weights)

        self.fitted = True
        return self

    def predict_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict (lambda_home, lambda_away) for a frame of fixtures."""
        features = build_features(self._corpus, self.backbone, rows=rows)
        lam_h = np.clip(self.model_home.predict(features), 0.05, 8.0)
        lam_a = np.clip(self.model_away.predict(features), 0.05, 8.0)
        return np.column_stack([lam_h, lam_a])

    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        if not self.fitted or self._corpus is None:
            raise RuntimeError("GradientBoostedGoals.predict called before fit")
        corpus = self._corpus
        mask = (corpus["home"] == home) & (corpus["away"] == away)
        if when is not None:
            mask &= corpus["date"] == pd.Timestamp(when)
        rows = corpus[mask]
        if rows.empty:
            return None
        rates = self.predict_rows(rows.head(1))
        lam_h, lam_a = float(rates[0, 0]), float(rates[0, 1])
        matrix = poisson_matrix(lam_h, lam_a, self.max_goals)
        if self.rho:
            matrix = apply_dixon_coles(matrix, lam_h, lam_a, self.rho)
        return Prediction(arm=self.name, home=home, away=away, matrix=matrix,
                          home_lambda=lam_h, away_lambda=lam_a)

    def importances(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": FEATURE_NAMES,
            "home": self.model_home.feature_importances_,
            "away": self.model_away.feature_importances_,
        }).sort_values("home", ascending=False).reset_index(drop=True)
