"""Recurrent sequence arm over each club's recent match history.

This is the "temporal convolutional network" line from the CSCI 635 report's
Future Improvements section, finally attempted. A GRU is used rather than a TCN:
the sequences here are short (a handful of recent matches) and a gated recurrent
unit handles irregular gaps between matches more naturally, which matters when a
club alternates between weekly domestic fixtures and a European match after four
days.

What this arm can see that a ratings model cannot: trajectory. A club whose last
six results are improving and one whose are decaying can hold identical ratings.
Whether that trajectory is real signal or noise the model will happily overfit is
exactly the question the season-long record is there to answer.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date

import numpy as np
import pandas as pd

from ..scoring.matrix import apply_dixon_coles, poisson_matrix
from .base import ModelArm, Prediction

SEQUENCE_LENGTH = 10
STEP_FEATURES = 6          # gf, ga, home flag, rest days, cross-border flag, points


class SequenceForm(ModelArm):
    name = "sequence"
    hypothesis = "Recent-form trajectory carries signal beyond a static rating."

    def __init__(
        self,
        *,
        hidden_dim: int = 24,
        epochs: int = 40,
        batch_size: int = 512,
        learning_rate: float = 0.005,
        recent_years: float = 8.0,
        max_goals: int = 12,
        rho: float = -0.04,
        seed: int = 20262027,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recent_years = recent_years
        self.max_goals = max_goals
        self.rho = rho
        self.seed = seed
        self._net = None
        self._sequences: dict[str, np.ndarray] = {}
        self._corpus: pd.DataFrame | None = None

    # -- sequence construction ------------------------------------------------
    @staticmethod
    def _build_sequences(corpus: pd.DataFrame, upto: pd.Timestamp | None = None):
        """Chronological pass yielding each match's pre-match sequences.

        Returns (records, final_state) where records maps (date, home, away) to a
        pair of arrays and final_state holds each club's latest sequence, used to
        predict fixtures that have not happened yet.
        """
        history: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
        last_date: dict[str, pd.Timestamp] = {}
        records: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        def snapshot(club: str) -> np.ndarray:
            arr = np.zeros((SEQUENCE_LENGTH, STEP_FEATURES), dtype=np.float32)
            entries = list(history[club])
            if entries:
                arr[-len(entries):] = np.array(entries, dtype=np.float32)
            return arr

        for row in corpus.sort_values("date").itertuples(index=False):
            if upto is not None and row.date > upto:
                break
            records[(row.date, row.home, row.away)] = (snapshot(row.home), snapshot(row.away))

            if not row.played:
                continue
            hg, ag = float(row.home_goals), float(row.away_goals)
            cross = float(bool(row.cross_border))
            for club, gf, ga, is_home in ((row.home, hg, ag, 1.0), (row.away, ag, hg, 0.0)):
                rest = 7.0
                if club in last_date:
                    rest = float(min((row.date - last_date[club]).days, 30))
                points = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
                history[club].append([gf, ga, is_home, rest / 30.0, cross, points / 3.0])
                last_date[club] = row.date

        final_state = {club: snapshot(club) for club in history}
        return records, final_state

    # -- fitting --------------------------------------------------------------
    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "SequenceForm":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        self._corpus = corpus
        data = self._training_slice(corpus, as_of)
        data = data[data["days_ago"] <= self.recent_years * 365.25]
        if data.empty:
            raise ValueError("No training data for SequenceForm")

        cutoff = pd.Timestamp(as_of) if as_of is not None else None
        records, final_state = self._build_sequences(corpus, upto=cutoff)
        self._sequences = final_state

        keys = list(zip(data["date"], data["home"], data["away"]))
        usable = [k for k in keys if k in records]
        if not usable:
            raise ValueError("No sequence records aligned to training rows")
        index = {k: i for i, k in enumerate(usable)}
        rows = data[[k in index for k in keys]]

        home_seq = np.stack([records[k][0] for k in usable])
        away_seq = np.stack([records[k][1] for k in usable])
        home_goals = rows["home_goals"].to_numpy(dtype=np.float32)
        away_goals = rows["away_goals"].to_numpy(dtype=np.float32)
        weights = rows["weight"].to_numpy(dtype=np.float32)
        weights = weights / weights.mean()

        hidden = self.hidden_dim

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(STEP_FEATURES, hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
                self.mu = nn.Parameter(torch.tensor(0.15))
                self.home_adv = nn.Parameter(torch.tensor(0.2))

            def forward(self, home_x, away_x):
                _, h_home = self.gru(home_x)
                _, h_away = self.gru(away_x)
                joined = torch.cat([h_home[-1], h_away[-1]], dim=-1)
                out = self.head(joined)
                log_lh = self.mu + self.home_adv + out[:, 0]
                log_la = self.mu + out[:, 1]
                return log_lh.clamp(-8, 3), log_la.clamp(-8, 3)

        net = Net()
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        home_t = torch.tensor(home_seq)
        away_t = torch.tensor(away_seq)
        hg_t = torch.tensor(home_goals)
        ag_t = torch.tensor(away_goals)
        w_t = torch.tensor(weights)
        n = len(usable)

        for _ in range(self.epochs):
            permutation = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch = permutation[start:start + self.batch_size]
                optimiser.zero_grad()
                log_lh, log_la = net(home_t[batch], away_t[batch])
                lam_h, lam_a = torch.exp(log_lh), torch.exp(log_la)
                loss = torch.mean(w_t[batch] * (lam_h - hg_t[batch] * log_lh
                                                + lam_a - ag_t[batch] * log_la))
                loss.backward()
                optimiser.step()

        net.eval()
        self._net = net
        self.fitted = True
        return self

    # -- prediction -----------------------------------------------------------
    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        import torch

        if not self.fitted:
            raise RuntimeError("SequenceForm.predict called before fit")
        if home not in self._sequences or away not in self._sequences:
            return None
        with torch.no_grad():
            log_lh, log_la = self._net(
                torch.tensor(self._sequences[home][None, ...]),
                torch.tensor(self._sequences[away][None, ...]),
            )
        lam_h = float(torch.exp(log_lh)[0])
        lam_a = float(torch.exp(log_la)[0])
        matrix = poisson_matrix(lam_h, lam_a, self.max_goals)
        if self.rho:
            matrix = apply_dixon_coles(matrix, lam_h, lam_a, self.rho)
        return Prediction(arm=self.name, home=home, away=away, matrix=matrix,
                          home_lambda=lam_h, away_lambda=lam_a)
