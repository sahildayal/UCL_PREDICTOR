"""Hierarchical Bayesian Poisson backbone.

This is the arm that exists because of the sample-starvation problem. Dixon-Coles
gives a point estimate of every club's rating; with eight European matches a
season, that point estimate is confidently wrong for exactly the clubs the
Champions League keeps introducing - Sabah, Bodo/Glimt, Pafos, Qarabag.

Two properties matter here and nowhere else in the stack:

**Partial pooling.** A club's rating is its league's level plus a deviation drawn
from a fitted distribution. The width of that distribution is itself estimated, so
the amount of shrinkage is learned rather than assumed. A club with 200 matches
overrides its league prior; a club with 8 mostly does not.

**Uncertainty propagation.** Predictions integrate over the rating posterior
instead of conditioning on its mode. This is what stops the model quoting 96% on
Manchester United against a club it has barely observed: some posterior draws have
Sabah much better than the mode, and those draws pull the quoted probability in.
Point-estimate models cannot express that and are systematically overconfident on
precisely the mismatches this competition is full of.

Fitted with ADVI rather than NUTS. NUTS over ~1,600 parameters and tens of
thousands of matches is hours; ADVI is minutes and the posterior width - which is
the entire reason this arm exists - survives the approximation well enough. NUTS
is available via `method="nuts"` for offline validation runs.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..scoring.matrix import apply_dixon_coles, normalise
from .base import ModelArm, Prediction


class HierarchicalBayes(ModelArm):
    name = "hier_bayes"
    hypothesis = (
        "Club ratings are drawn from league-level distributions; integrating over "
        "rating uncertainty beats conditioning on its mode, especially for clubs "
        "with few observed matches."
    )

    def __init__(
        self,
        *,
        recent_years: float = 6.0,
        draws: int = 400,
        advi_iterations: int = 25_000,
        method: str = "advi",
        max_goals: int = 12,
        rho: float = -0.04,
        seed: int = 20262027,
    ) -> None:
        super().__init__()
        self.recent_years = recent_years
        self.draws = draws
        self.advi_iterations = advi_iterations
        self.method = method
        self.max_goals = max_goals
        self.rho = rho
        self.seed = seed

        self.clubs: list[str] = []
        self._club_index: dict[str, int] = {}
        self.leagues: list[str] = []
        self._club_league: np.ndarray = np.array([], dtype=int)
        self.match_counts: dict[str, int] = {}
        # Posterior draws, shape (draws, n_clubs) and (draws,)
        self.att_draws: np.ndarray = np.array([])
        self.def_draws: np.ndarray = np.array([])
        self.mu_draws: np.ndarray = np.array([])
        self.home_draws: np.ndarray = np.array([])
        self.euro_draws: np.ndarray = np.array([])
        self._poisson_cache: dict[int, np.ndarray] = {}

    # -- fitting --------------------------------------------------------------
    def fit(self, corpus: pd.DataFrame, *, as_of: date | None = None) -> "HierarchicalBayes":
        import pymc as pm

        data = self._training_slice(corpus, as_of)
        cutoff_days = self.recent_years * 365.25
        data = data[data["days_ago"] <= cutoff_days]
        data = data[data["weight"] > 1e-6]
        if data.empty:
            raise ValueError("No training data for HierarchicalBayes")

        clubs = sorted(set(data["home"]) | set(data["away"]))
        self.clubs = clubs
        self._club_index = {c: i for i, c in enumerate(clubs)}

        pairs = pd.concat([
            data[["home", "home_country"]].rename(columns={"home": "club", "home_country": "country"}),
            data[["away", "away_country"]].rename(columns={"away": "club", "away_country": "country"}),
        ])
        pairs = pairs[pairs["country"].astype(bool)]
        country_of = {club: group["country"].mode().iloc[0]
                      for club, group in pairs.groupby("club")}
        leagues = sorted(set(country_of.values()))
        self.leagues = leagues
        league_index = {l: i for i, l in enumerate(leagues)}
        club_league = np.array([league_index.get(country_of.get(c, ""), 0) for c in clubs],
                               dtype=int)
        self._club_league = club_league

        home_idx = data["home"].map(self._club_index).to_numpy()
        away_idx = data["away"].map(self._club_index).to_numpy()
        home_goals = data["home_goals"].to_numpy(dtype=float)
        away_goals = data["away_goals"].to_numpy(dtype=float)
        weights = data["weight"].to_numpy(dtype=float)
        # Rescale to mean 1. Raw exponential-decay weights average ~0.35 over a
        # six-year window, so the weighted likelihood would carry barely a third
        # of the sample's information and the posterior would be far too wide -
        # the arm quoted 77% on Manchester United against Sabah, where the honest
        # answer is nearer 90%. Rescaling preserves the recency ordering while
        # restoring the effective sample size to n.
        weights = weights / weights.mean()
        is_euro = data["cross_border"].to_numpy(dtype=float)

        counts = (np.bincount(home_idx, minlength=len(clubs))
                  + np.bincount(away_idx, minlength=len(clubs)))
        self.match_counts = {c: int(counts[i]) for c, i in self._club_index.items()}

        n_clubs, n_leagues = len(clubs), len(leagues)

        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0)
            home_adv = pm.Normal("home_adv", 0.2, 0.2)
            euro_adv = pm.Normal("euro_adv", 0.0, 0.2)

            league_att = pm.Normal("league_att", 0.0, 0.5, shape=n_leagues)
            league_def = pm.Normal("league_def", 0.0, 0.5, shape=n_leagues)

            # The learned shrinkage: how far clubs typically deviate from their
            # league's level. Estimated, not assumed.
            sigma_att = pm.HalfNormal("sigma_att", 0.75)
            sigma_def = pm.HalfNormal("sigma_def", 0.75)

            att_raw = pm.Normal("att_raw", 0.0, 1.0, shape=n_clubs)
            def_raw = pm.Normal("def_raw", 0.0, 1.0, shape=n_clubs)

            attack = pm.Deterministic("attack", league_att[club_league] + att_raw * sigma_att)
            defence = pm.Deterministic("defence", league_def[club_league] + def_raw * sigma_def)

            log_lh = mu + home_adv + euro_adv * is_euro + attack[home_idx] - defence[away_idx]
            log_la = mu + attack[away_idx] - defence[home_idx]

            # Recency weighting via a weighted log-likelihood. Older matches still
            # inform small-league clubs, just less.
            logp = (weights * pm.logp(pm.Poisson.dist(mu=pm.math.exp(log_lh)), home_goals)
                    + weights * pm.logp(pm.Poisson.dist(mu=pm.math.exp(log_la)), away_goals))
            pm.Potential("weighted_likelihood", logp.sum())

            if self.method == "nuts":
                idata = pm.sample(draws=self.draws, tune=1000, chains=2, cores=1,
                                  target_accept=0.9, progressbar=False,
                                  random_seed=self.seed)
                posterior = idata.posterior
                stack = {k: posterior[k].stack(sample=("chain", "draw")).values
                         for k in ("attack", "defence", "mu", "home_adv", "euro_adv")}
                self.att_draws = stack["attack"].T
                self.def_draws = stack["defence"].T
                self.mu_draws = stack["mu"]
                self.home_draws = stack["home_adv"]
                self.euro_draws = stack["euro_adv"]
            else:
                approx = pm.fit(n=self.advi_iterations, method="advi",
                                progressbar=False, random_seed=self.seed)
                trace = approx.sample(self.draws)
                post = trace.posterior
                self.att_draws = post["attack"].values[0]
                self.def_draws = post["defence"].values[0]
                self.mu_draws = post["mu"].values[0]
                self.home_draws = post["home_adv"].values[0]
                self.euro_draws = post["euro_adv"].values[0]
                self._approx = approx

        self._model = model
        self.fitted = True
        return self

    # -- prediction -----------------------------------------------------------
    def _poisson_pmf(self, lam: np.ndarray) -> np.ndarray:
        """Poisson pmf for a vector of rates, shape (n, max_goals+1)."""
        goals = np.arange(self.max_goals + 1)
        log_pmf = (goals[None, :] * np.log(lam[:, None])
                   - lam[:, None]
                   - np.array([float(np.sum(np.log(np.arange(1, g + 1)))) for g in goals])[None, :])
        return np.exp(log_pmf)

    def predict(self, home: str, away: str, *, when: date | None = None) -> Prediction | None:
        if not self.fitted:
            raise RuntimeError("HierarchicalBayes.predict called before fit")
        if home not in self._club_index or away not in self._club_index:
            return None
        h, a = self._club_index[home], self._club_index[away]

        lam_h = np.exp(np.clip(
            self.mu_draws + self.home_draws + self.euro_draws
            + self.att_draws[:, h] - self.def_draws[:, a], -8, 3))
        lam_a = np.exp(np.clip(
            self.mu_draws + self.att_draws[:, a] - self.def_draws[:, h], -8, 3))

        # Posterior predictive: average the per-draw score matrices rather than
        # building one matrix from average rates. Averaging the rates first would
        # throw away exactly the uncertainty this arm exists to capture.
        pmf_h = self._poisson_pmf(lam_h)
        pmf_a = self._poisson_pmf(lam_a)
        matrix = np.einsum("si,sj->ij", pmf_h, pmf_a) / len(lam_h)
        matrix = normalise(matrix)
        if self.rho:
            matrix = apply_dixon_coles(matrix, float(lam_h.mean()), float(lam_a.mean()), self.rho)

        return Prediction(
            arm=self.name, home=home, away=away, matrix=matrix,
            home_lambda=float(lam_h.mean()), away_lambda=float(lam_a.mean()),
            meta={
                "home_lambda_sd": float(lam_h.std()),
                "away_lambda_sd": float(lam_a.std()),
                "home_matches": self.match_counts.get(home, 0),
                "away_matches": self.match_counts.get(away, 0),
                "draws": int(len(lam_h)),
            },
        )

    # -- inspection -----------------------------------------------------------
    def club_table(self) -> pd.DataFrame:
        """Posterior mean and sd per club. The sd column is the point of the arm."""
        return pd.DataFrame({
            "club": self.clubs,
            "league": [self.leagues[i] for i in self._club_league],
            "attack": self.att_draws.mean(axis=0),
            "attack_sd": self.att_draws.std(axis=0),
            "defence": self.def_draws.mean(axis=0),
            "defence_sd": self.def_draws.std(axis=0),
            "matches": [self.match_counts.get(c, 0) for c in self.clubs],
        }).assign(strength=lambda df: df["attack"] + df["defence"]) \
          .sort_values("strength", ascending=False).reset_index(drop=True)
