# UCL Predictor

A machine-learning laboratory for the 2026/27 UEFA Champions League league phase.

It produces calibrated probabilistic forecasts for every league-phase fixture,
keeps an honest scoreboard by placing **fake** bets, and leaves every real
decision to a human. There is no order placement anywhere in this repository and
no path to add one by configuration.

**Live dashboard: [sahildayal.me/UCL_PREDICTOR](https://sahildayal.me/UCL_PREDICTOR/)**
— season scoreboard, the current matchday, and a permanent archive of every
matchday as it was forecast before kickoff. Rebuilt automatically by
`predict.yml` and `settle.yml`; nothing on it is hand-edited.

See [DESIGN.md](DESIGN.md) for why it is built the way it is.

## Season so far

Matchday 1 (2026-09-08 to 2026-09-10) is complete: 6 fixtures forecast, 53 paper
bets settled, 0 left open. The market beat every arm on log-loss (0.7966), which
replicates the EPL/La Liga lab's finding rather than embarrassing this one; among
the arms, `gnn` led (0.8419) and `sequence` trailed (0.9704) - six matches is not
enough to read much into the ordering yet, but it is the season's first honest
data point, on the record on the dashboard's scoreboard.

**Matchday 2 is 2026-10-13/14** - the league phase has real gaps between
matchdays, not a weekly rhythm. `predict.yml` and `settle.yml` keep firing on
their Tue/Wed/Thu and Wed/Thu/Fri schedules regardless; between matchdays they
now exit as a fast no-op (see [Known limitations](#known-limitations) for the
bug this replaced).

## What makes this different from a league predictor

The league phase gives you 189 matches a season, and every club plays eight
*different* opponents. There are no home-and-away pairs and no 380-match rhythm
to learn from. So the interesting question is not "how is Bayern playing" but
**how many goals is an Eliteserien attack worth against a Bundesliga defence** -
a latent-variable problem that domestic form data cannot answer on its own.

Everything here follows from that. Cross-border matches (Champions League, plus
Europa and Conference qualifying rounds) are the only evidence that ties leagues
to a common scale, so they are flagged, weighted and modelled explicitly.
Qualifying rounds matter far more than their glamour suggests: they are where
Azerbaijani, Norwegian and Kazakh clubs meet the rest of Europe often enough to
be rated at all.

## Model arms

Every arm implements one interface and returns a full joint score matrix, from
which every market is derived. No arm is blended into another until the season's
record earns it.

| Arm | What it is | What it tests |
|---|---|---|
| `hier_bayes` | Hierarchical Bayesian Poisson (PyMC, ADVI) | Does integrating over rating uncertainty beat conditioning on its mode? |
| `dixon_coles` | Cross-league Dixon-Coles with partial pooling | The MAP version of the same idea, fitted in under a second |
| `elo_xleague` | Plain Elo on all matches | Control. Was any of the complexity worth it? |
| `gbm` | LightGBM with Poisson objective on backbone residuals | Do form, rest and fatigue add anything to ratings? |
| `gnn` | Message passing over the club-versus-club graph | Does propagating strength along the fixture graph help? |
| `sequence` | GRU over each club's recent match history | Does form *trajectory* carry signal? |

Two results were visible from matchday 1 alone and are worth stating up front,
because both are informative rather than embarrassing:

- **Elo collapses on cross-league mismatches.** It rated Manchester United at 64%
  against Sabah where the market said 86%, because the Azerbaijani league is a
  nearly-isolated cluster in the fixture graph and its clubs never leave their
  1500 starting rating. That is exactly the failure the hierarchical model is
  designed to avoid, and it is the cleanest possible argument for league-strength
  modelling.
- **The sequence arm cannot tell Manchester United from Sabah** (55%), because it
  sees only form trajectory and no club identity. An honest ablation of how far
  "recent form" gets you on its own: not far.

## Data

| Source | Provides |
|---|---|
| `openfootball/champions-league` | UCL 2011-12 to 2025-26, plus UCL/UEL/UECL qualifiers |
| `openfootball/europe` + country repos | 48 domestic leagues |
| Wikipedia league-phase pages | All 144 fixtures of 2026-27 and results as they land |
| The Odds API | Settlement prices and the de-vigged benchmark **only** |

The corpus is roughly 53,000 matches, of which about 2,800 are cross-border.

Two parsing traps are handled explicitly and covered by tests, because both fail
*silently* rather than loudly:

- openfootball has **two format generations**. Older files write
  `20:00 Liverpool FC 4-1 (4-0) Norwich City` with the score between the teams and
  the year only in a header comment; newer ones write `Home v Away 1-2 (0-1)`.
  Reading only the modern shape returns zero matches for about fifteen seasons of
  every major league, which is indistinguishable from "not published yet".
- Knockout ties write `5-0 a.e.t. (3-0, 1-0)` where the leading pair is the
  **two-leg aggregate**, not a scoreline. Taking it naively records thrashings
  that never happened.

## The market's role

The model never sees a price. Odds are used only to settle fake bets and to score
each arm against the de-vigged closing line - the hardest honest benchmark in
football forecasting. The dashboard shows model and market side by side with the
gap between them; what to do about that gap is a human decision.

The Odds API key is shared with `EPL_LALIGA_PREDICTOR`. A **credit floor** guard
refuses to spend the shared quota below a reserve, so a heavy UCL week cannot
starve that lab into failing closed with zero bets - which would look exactly
like a real finding and would not be one.

## Paper ledger

One bankroll per arm, in two regimes that answer different questions:

- **science** stakes a flat amount on every fixture, so all 144 league-phase
  matches produce a scored decision per arm. With a season this short,
  statistical power beats realism.
- **edge_gated** stakes only on material disagreement with the line, at quarter
  Kelly, for realistic P&L.

A `human` arm records Sahil's own pre-kickoff calls, so the season measures
whether human judgement adds to the model or subtracts from it.

An arm placing zero bets is a finding. Never lower a threshold to make an arm bet
more - that measures the threshold, not the strategy.

## Usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# full matchday brief: fit every arm, price fixtures, stake, simulate, render
python scripts/predict.py --as-of 2026-09-10 --horizon 1

# fast smoke run without the expensive arms or the ledger
python scripts/predict.py --fast --no-bets

# settle finished matches and score every arm against the closing line
python scripts/settle.py

# check the club registry for silently duplicated clubs
python scripts/audit_teams.py

# rebuild the GitHub Pages site from the archived briefs
python scripts/build_site.py

# log your own pre-kickoff call, so the human arm has something in it
python scripts/log_pick.py --home "manchester united" --away sabah   --date 2026-09-10 --pick home --confidence 0.88 --note "why"
```

Exit codes follow the EPL lab's convention: `0` clean, `1` failed, `2` completed
but a human should look.

## Telegram

`@ucl_sd_bot` pushes a brief on the morning of each matchday (fixtures, model
versus market, disagreements flagged at 8+ points) and a results message the
next morning. Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`;
unconfigured, the push is silently skipped and the matchday run still succeeds.

## Known limitations

- **Leagues disconnected from European competition are not identified.** Russian
  and Belarusian clubs have not played cross-border matches since 2022, so their
  ratings are anchored to pre-2022 evidence and then drift on domestic results
  alone. Russia consequently ranks implausibly high in the league table. No
  Russian club is in the competition, so predictions are unaffected, but the
  league-strength figures should not be read as current.
- **Domestic coverage lags for smaller leagues.** openfootball's files for
  Azerbaijan stop at 2024-25 and Norway at 2025. Long-run strength is well
  identified; current-season domestic form for those countries is not.
  football-data.co.uk would fill the gap and an adapter is written, but the site
  was returning HTTP 503 throughout the build.
- **Travel distance is not yet a feature**, despite being a real effect in this
  competition. It needs club coordinates.
- **No player-level modelling.** Injuries, suspensions and rotation are invisible
  to every arm. In a competition where a qualified club rests six starters on
  matchday 8, this is the largest single gap.

Fixed since launch, worth recording because it was silent and spent a shared,
limited resource for nothing: for the first week after matchday 1, `predict.yml`
kept doing full work - fitting all six arms, running the 20k-simulation
qualification model, and (the actual damage) **calling the Odds API** - on every
scheduled firing even when the horizon held no fixtures. Three empty runs
(2026-09-15/16/17) burned the pool shared with `EPL_LALIGA_PREDICTOR` from ~180
credits toward its 150-credit floor for briefs with zero fixtures, and would
have kept bleeding it for the five weeks until matchday 2. It also pushed an
empty "brief" to Telegram each time and flagged its own exit code 2 ("a human
should look") purely for having nothing to price. Fixed 2026-09-22: the odds
fetch, the Telegram push and the exit-code check are now all gated on
`fixtures` being non-empty; the qualification simulation still runs between
matchdays, since keeping the season odds current has no API cost.

## Safety

No real money. No order placement. The Odds API key is read-only in practice and
used solely for settlement and scoring. Forecasts are decision support for a
human, not instructions.
