"""Graph neural network over the club-versus-club network.

The motivating observation: every club in Europe is connected to every other by a
short chain of matches, even when they have never met. Sabah played Azerbaijani
opposition and a handful of European qualifiers; those opponents played others;
within three or four hops you reach Manchester United. A rating model only uses
that chain implicitly, through a shared scale. Message passing uses it directly -
a club's representation is built from its opponents' representations, so evidence
propagates along the fixture graph and across league boundaries.

Honest expectation: with a corpus this size and a target this noisy, this arm is
unlikely to beat the Bayesian backbone early. It is included because the season is
an experiment, its score is recorded like every other arm's, and losing on the
record is a result. It is not blended into anything until it earns that.

Implemented in plain PyTorch. torch-geometric would add a fragile dependency for
a two-line sparse matmul.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..scoring.matrix import apply_dixon_coles, poisson_matrix
from .base import ModelArm, Prediction


class ClubGraphNet(ModelArm):
    name = "gnn"
    hypothesis = (
        "Propagating club representations along the fixture graph captures "
        "cross-league strength better than a scalar rating."
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
        layers: int = 3,
        epochs: int = 300,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        recent_years: float = 8.0,
        max_goals: int = 12,
        rho: float = -0.04,
        seed: int = 20262027,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.recent_years = recent_years
        self.max_goals = max_goals
        self.rho = rho
        self.seed = seed

        self.clubs: list[str] = []
        self._club_index: dict[str, int] = {}
        self._attack: np.ndarray = np.array([])
        self._defence: np.ndarray = np.array([])
        self._mu = 0.0
        self._home_adv = 0.0
        self.history: list[float] = []

    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "ClubGraphNet":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        data = self._training_slice(corpus, as_of)
        data = data[data["days_ago"] <= self.recent_years * 365.25]
        if data.empty:
            raise ValueError("No training data for ClubGraphNet")

        clubs = sorted(set(data["home"]) | set(data["away"]))
        self.clubs = clubs
        self._club_index = {c: i for i, c in enumerate(clubs)}
        n = len(clubs)

        home_idx = torch.tensor(data["home"].map(self._club_index).to_numpy(), dtype=torch.long)
        away_idx = torch.tensor(data["away"].map(self._club_index).to_numpy(), dtype=torch.long)
        home_goals = torch.tensor(data["home_goals"].to_numpy(), dtype=torch.float32)
        away_goals = torch.tensor(data["away_goals"].to_numpy(), dtype=torch.float32)
        weights = torch.tensor(data["weight"].to_numpy(), dtype=torch.float32)
        weights = weights / weights.mean()

        # Symmetric, recency-weighted adjacency with symmetric normalisation.
        # Normalising by degree stops clubs in high-fixture leagues from
        # dominating the message passing purely by playing more matches.
        indices = torch.stack([
            torch.cat([home_idx, away_idx]),
            torch.cat([away_idx, home_idx]),
        ])
        values = torch.cat([weights, weights])
        adjacency = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1e-6)
        inv_sqrt = degree.pow(-0.5)
        norm_values = inv_sqrt[adjacency.indices()[0]] * adjacency.values() * inv_sqrt[adjacency.indices()[1]]
        adjacency = torch.sparse_coo_tensor(adjacency.indices(), norm_values, (n, n)).coalesce()

        dim = self.embedding_dim

        class Net(nn.Module):
            def __init__(self, n_clubs: int, layers: int):
                super().__init__()
                self.embedding = nn.Embedding(n_clubs, dim)
                nn.init.normal_(self.embedding.weight, std=0.1)
                self.self_weights = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))
                self.neighbour_weights = nn.ModuleList(nn.Linear(dim, dim, bias=False)
                                                       for _ in range(layers))
                self.attack_head = nn.Linear(dim, 1)
                self.defence_head = nn.Linear(dim, 1)
                self.mu = nn.Parameter(torch.tensor(0.1))
                self.home_adv = nn.Parameter(torch.tensor(0.2))

            def forward(self, adj):
                h = self.embedding.weight
                for self_w, neigh_w in zip(self.self_weights, self.neighbour_weights):
                    messages = torch.sparse.mm(adj, h)
                    h = torch.relu(self_w(h) + neigh_w(messages))
                return self.attack_head(h).squeeze(-1), self.defence_head(h).squeeze(-1)

        net = Net(n, self.layers)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            optimiser.zero_grad()
            attack, defence = net(adjacency)
            log_lh = net.mu + net.home_adv + attack[home_idx] - defence[away_idx]
            log_la = net.mu + attack[away_idx] - defence[home_idx]
            lam_h = torch.exp(log_lh.clamp(-8, 3))
            lam_a = torch.exp(log_la.clamp(-8, 3))
            loss = torch.mean(weights * (lam_h - home_goals * log_lh
                                         + lam_a - away_goals * log_la))
            loss.backward()
            optimiser.step()
            self.history.append(float(loss.item()))

        with torch.no_grad():
            attack, defence = net(adjacency)
            self._attack = attack.numpy()
            self._defence = defence.numpy()
            self._mu = float(net.mu.item())
            self._home_adv = float(net.home_adv.item())

        self.fitted = True
        return self

    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        if not self.fitted:
            raise RuntimeError("ClubGraphNet.predict called before fit")
        if home not in self._club_index or away not in self._club_index:
            return None
        h, a = self._club_index[home], self._club_index[away]
        lam_h = float(np.exp(np.clip(self._mu + self._home_adv
                                     + self._attack[h] - self._defence[a], -8, 3)))
        lam_a = float(np.exp(np.clip(self._mu + self._attack[a] - self._defence[h], -8, 3)))
        matrix = poisson_matrix(lam_h, lam_a, self.max_goals)
        if self.rho:
            matrix = apply_dixon_coles(matrix, lam_h, lam_a, self.rho)
        return Prediction(arm=self.name, home=home, away=away, matrix=matrix,
                          home_lambda=lam_h, away_lambda=lam_a,
                          meta={"final_loss": self.history[-1] if self.history else None})
