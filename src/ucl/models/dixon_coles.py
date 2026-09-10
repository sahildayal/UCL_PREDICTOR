"""Cross-league Dixon-Coles with partial pooling toward league strength.

The standard Dixon-Coles model fits attack and defence ratings inside a single
league, where they are identified only up to a per-league constant. That is fine
domestically and useless here: nothing in a Norwegian league table says what
Bodo/Glimt is worth against Bayern.

Two changes make it work across leagues:

1. **One global fit.** Domestic and European matches are fitted together with a
   single normalisation, so ratings live on one scale. The club graph is
   connected through European competition, which is what identifies the
   league-to-league offsets at all. Qualifying rounds carry most of that weight
   for small nations - they are often the only matches where an Azerbaijani or
   Kazakh club meets the rest of Europe.

2. **Partial pooling.** Each club's rating is `league effect + club deviation`,
   with the deviation penalised harder than the league effect. A club with eight
   European matches is therefore shrunk toward its league's level instead of
   being fitted to noise. This is the MAP version of what `hier_bayes` does with
   full posterior sampling, and it is what stops Sabah from getting a rating
   built out of three matches.

Fitted by L-BFGS with analytic gradients. Numerical gradients over ~1,300
parameters would be hopeless.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..scoring.matrix import apply_dixon_coles, poisson_matrix
from .base import ModelArm, Prediction


class DixonColesXLeague(ModelArm):
    name = "dixon_coles"
    hypothesis = (
        "Club strength is league strength plus a shrunk club deviation; goals are "
        "Poisson with a low-score correction."
    )

    def __init__(
        self,
        *,
        club_penalty: float = 1.0,
        league_penalty: float = 0.05,
        max_goals: int = 12,
        fit_rho: bool = True,
    ) -> None:
        super().__init__()
        # Club deviations are penalised ~20x harder than league effects, which is
        # what produces the shrinkage: with few matches, a club keeps its league's
        # level rather than inventing its own.
        self.club_penalty = club_penalty
        self.league_penalty = league_penalty
        self.max_goals = max_goals
        self.fit_rho = fit_rho

        self.clubs: list[str] = []
        self.leagues: list[str] = []
        self._club_index: dict[str, int] = {}
        self._league_index: dict[str, int] = {}
        self._club_league: np.ndarray = np.array([], dtype=int)

        self.mu: float = 0.0
        self.home_adv: float = 0.0
        self.euro_home_adv: float = 0.0
        self.rho: float = 0.0
        self.attack: np.ndarray = np.array([])
        self.defence: np.ndarray = np.array([])
        self.league_attack: np.ndarray = np.array([])
        self.league_defence: np.ndarray = np.array([])
        self.match_counts: dict[str, int] = {}

    # -- fitting --------------------------------------------------------------
    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "DixonColesXLeague":
        data = self._training_slice(corpus, as_of)
        data = data[data["weight"] > 1e-6]
        if data.empty:
            raise ValueError("No training data for DixonColesXLeague")

        clubs = sorted(set(data["home"]) | set(data["away"]))
        self.clubs = clubs
        self._club_index = {c: i for i, c in enumerate(clubs)}

        # A club's league is the country it appears with most often, so a club
        # that changed country (or a mislabelled row) cannot silently create a
        # one-club league.
        country_of: dict[str, str] = {}
        pairs = pd.concat([
            data[["home", "home_country"]].rename(columns={"home": "club", "home_country": "country"}),
            data[["away", "away_country"]].rename(columns={"away": "club", "away_country": "country"}),
        ])
        pairs = pairs[pairs["country"].astype(bool)]
        for club, group in pairs.groupby("club"):
            country_of[club] = group["country"].mode().iloc[0]

        leagues = sorted(set(country_of.values()))
        self.leagues = leagues
        self._league_index = {l: i for i, l in enumerate(leagues)}
        self._club_league = np.array(
            [self._league_index.get(country_of.get(c, ""), 0) for c in clubs], dtype=int
        )

        home_idx = data["home"].map(self._club_index).to_numpy()
        away_idx = data["away"].map(self._club_index).to_numpy()
        home_goals = data["home_goals"].to_numpy(dtype=float)
        away_goals = data["away_goals"].to_numpy(dtype=float)
        weights = data["weight"].to_numpy(dtype=float)
        is_euro = data["cross_border"].to_numpy(dtype=float)

        counts = np.bincount(home_idx, minlength=len(clubs)) + \
            np.bincount(away_idx, minlength=len(clubs))
        self.match_counts = {c: int(counts[i]) for c, i in self._club_index.items()}

        n_clubs, n_leagues = len(clubs), len(leagues)
        club_league = self._club_league

        def unpack(theta: np.ndarray):
            offset = 0
            mu = theta[offset]; offset += 1
            home_adv = theta[offset]; offset += 1
            euro_home_adv = theta[offset]; offset += 1
            attack = theta[offset:offset + n_clubs]; offset += n_clubs
            defence = theta[offset:offset + n_clubs]; offset += n_clubs
            league_attack = theta[offset:offset + n_leagues]; offset += n_leagues
            league_defence = theta[offset:offset + n_leagues]
            return mu, home_adv, euro_home_adv, attack, defence, league_attack, league_defence

        def objective(theta: np.ndarray):
            mu, home_adv, euro_home_adv, attack, defence, l_att, l_def = unpack(theta)

            # Effective rating = league effect + club deviation.
            eff_att = attack + l_att[club_league]
            eff_def = defence + l_def[club_league]

            log_lh = mu + home_adv + euro_home_adv * is_euro + eff_att[home_idx] - eff_def[away_idx]
            log_la = mu + eff_att[away_idx] - eff_def[home_idx]
            lam_h = np.exp(np.clip(log_lh, -8, 3))
            lam_a = np.exp(np.clip(log_la, -8, 3))

            nll = float(np.sum(weights * (lam_h - home_goals * log_lh
                                          + lam_a - away_goals * log_la)))
            nll += self.club_penalty * float(attack @ attack + defence @ defence)
            nll += self.league_penalty * float(l_att @ l_att + l_def @ l_def)

            rh = weights * (lam_h - home_goals)
            ra = weights * (lam_a - away_goals)

            g_mu = float(rh.sum() + ra.sum())
            g_home = float(rh.sum())
            g_euro = float((rh * is_euro).sum())

            # attack of home team enters lam_h; attack of away team enters lam_a
            g_att = (np.bincount(home_idx, weights=rh, minlength=n_clubs)
                     + np.bincount(away_idx, weights=ra, minlength=n_clubs))
            # defence enters with a negative sign, opponent's lambda
            g_def = -(np.bincount(away_idx, weights=rh, minlength=n_clubs)
                      + np.bincount(home_idx, weights=ra, minlength=n_clubs))

            g_l_att = np.bincount(club_league, weights=g_att, minlength=n_leagues)
            g_l_def = np.bincount(club_league, weights=g_def, minlength=n_leagues)

            g_att = g_att + 2.0 * self.club_penalty * attack
            g_def = g_def + 2.0 * self.club_penalty * defence
            g_l_att = g_l_att + 2.0 * self.league_penalty * l_att
            g_l_def = g_l_def + 2.0 * self.league_penalty * l_def

            grad = np.concatenate((
                np.array([g_mu, g_home, g_euro]), g_att, g_def, g_l_att, g_l_def
            ))
            return nll, grad

        theta0 = np.zeros(3 + 2 * n_clubs + 2 * n_leagues)
        theta0[0] = np.log(max(np.average(home_goals, weights=weights), 0.2))
        theta0[1] = 0.25

        result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": 600, "maxfun": 900})

        (self.mu, self.home_adv, self.euro_home_adv, self.attack, self.defence,
         self.league_attack, self.league_defence) = unpack(result.x)
        self._opt_result = result

        if self.fit_rho:
            self.rho = self._fit_rho(home_idx, away_idx, home_goals, away_goals,
                                     weights, is_euro)
        self.fitted = True
        return self

    def _fit_rho(self, home_idx, away_idx, home_goals, away_goals, weights, is_euro) -> float:
        """Fit the low-score correlation on the low-score cells only.

        rho is a one-dimensional nuisance parameter that only touches 0-0, 1-0,
        0-1 and 1-1, so it is fitted after the ratings rather than jointly. This
        is standard practice and keeps the main optimisation clean.
        """
        lam_h, lam_a = self._lambdas_indexed(home_idx, away_idx, is_euro)
        low = (home_goals <= 1) & (away_goals <= 1)
        if low.sum() == 0:
            return 0.0
        lh, la = lam_h[low], lam_a[low]
        hg, ag = home_goals[low], away_goals[low]
        w = weights[low]

        def negll(rho_arr: np.ndarray) -> float:
            rho = float(rho_arr[0])
            tau = np.ones_like(lh)
            tau = np.where((hg == 0) & (ag == 0), 1.0 - lh * la * rho, tau)
            tau = np.where((hg == 0) & (ag == 1), 1.0 + lh * rho, tau)
            tau = np.where((hg == 1) & (ag == 0), 1.0 + la * rho, tau)
            tau = np.where((hg == 1) & (ag == 1), 1.0 - rho, tau)
            tau = np.clip(tau, 1e-9, None)
            return -float(np.sum(w * np.log(tau)))

        out = minimize(negll, np.array([0.0]), method="L-BFGS-B",
                       bounds=[(-0.25, 0.25)])
        return float(out.x[0])

    def _lambdas_indexed(self, home_idx, away_idx, is_euro):
        eff_att = self.attack + self.league_attack[self._club_league]
        eff_def = self.defence + self.league_defence[self._club_league]
        log_lh = (self.mu + self.home_adv + self.euro_home_adv * is_euro
                  + eff_att[home_idx] - eff_def[away_idx])
        log_la = self.mu + eff_att[away_idx] - eff_def[home_idx]
        return np.exp(np.clip(log_lh, -8, 3)), np.exp(np.clip(log_la, -8, 3))

    # -- prediction -----------------------------------------------------------
    def lambdas(self, home: str, away: str, *, euro: bool = True) -> tuple[float, float] | None:
        if home not in self._club_index or away not in self._club_index:
            return None
        h, a = self._club_index[home], self._club_index[away]
        lam_h, lam_a = self._lambdas_indexed(np.array([h]), np.array([a]),
                                             np.array([1.0 if euro else 0.0]))
        return float(lam_h[0]), float(lam_a[0])

    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        if not self.fitted:
            raise RuntimeError("DixonColesXLeague.predict called before fit")
        pair = self.lambdas(home, away, euro=True)
        if pair is None:
            return None                       # unknown club: refuse, never guess
        lam_h, lam_a = pair
        matrix = poisson_matrix(lam_h, lam_a, self.max_goals)
        if self.rho:
            matrix = apply_dixon_coles(matrix, lam_h, lam_a, self.rho)
        return Prediction(
            arm=self.name, home=home, away=away, matrix=matrix,
            home_lambda=lam_h, away_lambda=lam_a,
            meta={"rho": self.rho,
                  "home_matches": self.match_counts.get(home, 0),
                  "away_matches": self.match_counts.get(away, 0)},
        )

    # -- inspection -----------------------------------------------------------
    def league_table(self) -> pd.DataFrame:
        """League strength, the quantity this whole project turns on."""
        return pd.DataFrame({
            "league": self.leagues,
            "attack": self.league_attack,
            "defence": self.league_defence,
            "strength": self.league_attack + self.league_defence,
        }).sort_values("strength", ascending=False).reset_index(drop=True)

    def club_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "club": self.clubs,
            "league": [self.leagues[i] for i in self._club_league],
            "attack": self.attack + self.league_attack[self._club_league],
            "defence": self.defence + self.league_defence[self._club_league],
            "matches": [self.match_counts.get(c, 0) for c in self.clubs],
        }).assign(strength=lambda df: df["attack"] + df["defence"]) \
          .sort_values("strength", ascending=False).reset_index(drop=True)
