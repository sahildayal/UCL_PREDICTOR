# UCL Predictor — Design

A single-season machine-learning laboratory for the 2026/27 UEFA Champions League
league phase. Not a betting bot. It produces calibrated probabilistic forecasts,
places **fake** bets to keep an honest scoreboard, and leaves every real decision
to a human.

## 1. Why this is not the EPL/La Liga lab

`EPL_LALIGA_PREDICTOR` is a market-divergence engine: it treats the sharp line as
truth and hunts for mispricing. That project already established, over 16 seasons
of walk-forward CV, that no model beat the de-vigged sharp line and that the fitted
market/model blend weight converged to 0.00.

This project asks a different question, so it is allowed a different answer:

> Given how little data a 36-team league phase produces, what modelling approach
> best predicts matches between clubs from *different* leagues?

The market is the **scoreboard**, never an input. See §6.

## 2. The three problems that shape every decision

**P1 — Sample starvation.** 36 teams x 8 matches = 189 league-phase matches per
season, and each team faces 8 *different* opponents. There are no home-and-away
pairs and no 380-match domestic rhythm. Any model trained on UCL matches alone is
undertrained by construction.

**P2 — Cross-league strength is the actual task.** "Bayern vs Bodo/Glimt" is a
question about how many goals an Eliteserien attack is worth against a Bundesliga
defence. This is a latent-variable inference problem, and it is where the real
modelling lives.

**P3 — Violent non-stationarity.** Rotation once qualification is secure, manager
changes, UCL-specific tactical identities, midweek fatigue, and travel (Glasgow to
Almaty is a real feature). Small effects domestically, large ones here.

## 3. Architecture: backbone + residual arms

The core inversion versus the CSCI 635 project, where six model families competed
head-to-head on raw features and all hit the same noise ceiling (SVM best at ~0.62
accuracy / 0.58 macro-F1, draws unsolved by every one of them):

    domestic + European match corpus
              |
              v
    [ hierarchical Bayesian Poisson backbone ]   <- solves P2, gives uncertainty
              |
              +--> attack/defence/league-strength posteriors
              |
              v
    [ residual arms ]  GNN | GBM | sequence | Elo   <- learn where the backbone errs
              |
              v
    [ joint score matrix ] --> 1X2, O/U any line, BTTS, AH, margin
              |
              v
    [ Monte Carlo over remaining 144 fixtures ] --> top-8 / 9-24 / eliminated
              |
              v
    [ paper ledger, one bankroll per arm ] + [ dashboard + Telegram ]

Arms never blend into a single number until the season's record earns it. A blend
is itself an arm, scored like the rest.

## 4. Model arms (all ship in v1)

| Arm | Family | Rationale |
|---|---|---|
| `hier_bayes` | Partially-pooled Bayesian Poisson (PyMC/NUTS) | Backbone. Shrinks small-league clubs toward their league mean instead of overfitting 8 matches. Real posterior intervals. |
| `dixon_coles` | Cross-league Dixon-Coles + time decay | Low-score correlation correction; the standard football baseline, extended with a league-strength term. |
| `elo_xleague` | Elo fit on cross-border matches only | Control arm. Deliberately simple; if it wins, the complexity was not worth it. |
| `gbm` | LightGBM/XGBoost on engineered features | Descendant of the 635 work, but predicting backbone *residuals*, not raw outcomes. |
| `gnn` | Torch message-passing over the club graph | Cross-border fixtures form a sparse graph; message passing propagates strength across leagues. Genuinely novel here. |
| `sequence` | Torch temporal model over match histories | The "TCN" idea from the 635 Future Improvements section, finally tried. |
| `blend` | Stacked/calibrated combination | Scored as just another arm. |

Every arm implements one interface (`src/ucl/models/base.py`) and returns a full
joint score matrix, not a 1X2 triple. Everything else is derived from that matrix,
which structurally prevents the totals-line bug that hit the EPL lab (a fair value
computed for one line must never be compared against a price for another).

## 5. Data

| Source | Gives | Status |
|---|---|---|
| `openfootball/champions-league` | UCL 2011-12..2025-26 + UCL/EL/ECL qualifiers | Primary cross-border corpus |
| Wikipedia league-phase pages | 2026-27 fixtures + live results, all 144 matches to MD8 | Primary current season |
| `openfootball/europe` | 48 domestic leagues | Historical; **lags current season** |
| football-data.co.uk | 22+ leagues incl. current season | Adapter built, site currently 503, auto-fills on return |
| The Odds API | Prices for settlement + benchmark only | Shared key + credit floor guard |

Cross-border matches (UCL/EL/ECL + qualifiers) are what identify league strength.
Qualifying rounds matter disproportionately: they are where Azerbaijani and
Norwegian clubs meet the rest of Europe often enough to be rated at all.

## 6. The market's role

The model never sees a price. Odds are used only to (a) settle fake bets and
(b) score us, primarily by log-loss against the de-vigged closing line. The
human-facing output shows model probability beside market price and the implied
disagreement; what to do about that gap is Sahil's call, not the program's.

## 7. Paper ledger

One bankroll per arm, plus a `human` arm where Sahil logs his own pre-kickoff
calls. Two staking regimes run in parallel:

- **science arms** bet a flat stake on *every* fixture, so all 189 matches produce
  a labelled, scored decision. With a season this short, statistical power beats
  realism.
- **edge-gated arms** bet only on material model/market disagreement, for realistic
  P&L.

This directly answers the EPL lab's problem, where disciplined thresholds produced
excellent science and almost no data. Here we get both, separately.

**No real money and no order placement exists anywhere in this repo.**

## 8. Evaluation

Primary: log-loss and Brier vs outcomes, and vs the de-vigged closing line.
Secondary: calibration curves per arm, draw recall (the 635 project's unsolved
frontier), qualification-forecast sharpness over time, paper P&L per arm.

A losing arm is a result. Never tune a threshold to make an arm bet more —
that measures the threshold, not the strategy.

## 9. Operational rules

- Predictions are timestamped and committed **before** kickoff by CI; a prediction
  that cannot be proven pre-kickoff is not scored.
- A price seen at or after kickoff is never stamped as a closing price.
- Fail closed: a broken source yields zero bets and a loud exit code, never a guess.
- Force UTF-8 everywhere. Windows cp1252 cannot encode Qarabag, Bodo/Glimt, or
  Sparta Praha and will crash the console mid-run.
