"""The one interface every model arm implements.

Arms are deliberately uniform: each returns a full joint score matrix, never a
1X2 triple. Markets are then derived from that matrix in `scoring.matrix`, so
every arm quotes every market consistently and no arm can invent a probability
for one line while holding a fair value for another.

An arm that cannot price a fixture returns None. It does not guess, and it does
not fall back to a league average - a fabricated price is worse than an absent
one, because the ledger will happily bet on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """One arm's view of one fixture."""
    arm: str
    home: str
    away: str
    matrix: np.ndarray
    home_lambda: float | None = None
    away_lambda: float | None = None
    meta: dict = field(default_factory=dict)

    def markets(self) -> dict:
        from ..scoring.matrix import summarise_markets
        return summarise_markets(self.matrix)


class ModelArm(ABC):
    """A named, independently-scored predictor with its own bankroll."""

    name: str = "unnamed"
    #: Human-readable statement of what this arm is testing.
    hypothesis: str = ""

    def __init__(self) -> None:
        self.fitted: bool = False

    @abstractmethod
    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "ModelArm":
        """Fit on matches strictly before `as_of`.

        Implementations must filter on date themselves. Leakage here is invisible
        and fatal: a model that has seen the match it is predicting will look
        excellent all season and be worthless.
        """

    @abstractmethod
    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        """Return a prediction, or None if this arm cannot price the fixture."""

    # -- shared helpers -------------------------------------------------------
    @staticmethod
    def _training_slice(corpus: pd.DataFrame, as_of: date | None) -> pd.DataFrame:
        played = corpus[corpus["played"]]
        if as_of is not None:
            played = played[played["date"] < pd.Timestamp(as_of)]
        return played

    def __repr__(self) -> str:
        state = "fitted" if self.fitted else "unfitted"
        return f"<{type(self).__name__} name={self.name!r} {state}>"


class ArmRegistry:
    """Holds the arms for a run so scripts and CI agree on the roster."""

    def __init__(self) -> None:
        self._arms: dict[str, ModelArm] = {}

    def register(self, arm: ModelArm) -> ModelArm:
        if arm.name in self._arms:
            raise ValueError(f"Duplicate arm name: {arm.name}")
        self._arms[arm.name] = arm
        return arm

    def get(self, name: str) -> ModelArm | None:
        return self._arms.get(name)

    def names(self) -> list[str]:
        return list(self._arms)

    def all(self) -> list[ModelArm]:
        return list(self._arms.values())

    def __len__(self) -> int:
        return len(self._arms)

    def __iter__(self):
        return iter(self._arms.values())
