"""Matchday orchestration: fit, predict, price, stake, simulate, report.

Ordering matters and is deliberate:

1. Build the corpus and fit every arm with `as_of` set to the matchday, so no arm
   can see a result it is about to predict.
2. Predict, and record predictions BEFORE looking at any market price. Prices are
   fetched afterwards purely to settle and to score.
3. Stake the paper ledger.
4. Simulate the rest of the league phase.
5. Write the brief.

Failure is loud at every step. A missing source produces zero bets and a non-zero
exit code, never a guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED, REPORTS
from .data import corpus as corpus_module
from .data import odds as odds_module
from .eval import metrics
from .ledger.ledger import EDGE_GATED, SCIENCE, Ledger
from .models.base import ArmRegistry
from .models.dixon_coles import DixonColesXLeague
from .models.elo_xleague import CrossLeagueElo
from .models.gbm import GradientBoostedGoals
from .models.gnn import ClubGraphNet
from .models.hier_bayes import HierarchicalBayes
from .models.sequence import SequenceForm
from .sim import qualification
from .teams import TeamRegistry


@dataclass
class MatchdayBrief:
    generated_at: str
    as_of: str
    fixtures: list[dict]
    arm_summary: list[dict]
    qualification: list[dict]
    ledger_summary: list[dict]
    market_available: bool
    notes: list[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=str)


def build_arms(*, fast: bool = False) -> tuple[ArmRegistry, DixonColesXLeague]:
    """Construct the arm roster.

    `fast` drops the expensive arms for a smoke run. The roster is defined in one
    place so a scheduled run and a local run cannot silently disagree about which
    models are in the experiment.
    """
    registry = ArmRegistry()
    backbone = DixonColesXLeague()
    registry.register(backbone)
    registry.register(CrossLeagueElo())
    if not fast:
        registry.register(HierarchicalBayes(advi_iterations=40_000, draws=500))
    registry.register(GradientBoostedGoals(backbone=backbone))
    if not fast:
        registry.register(ClubGraphNet(epochs=250))
        registry.register(SequenceForm(epochs=25))
    return registry, backbone


def fit_all(registry: ArmRegistry, frame: pd.DataFrame, as_of: date,
            notes: list[str]) -> list:
    """Fit every arm, keeping the ones that succeed. A failed arm is reported."""
    fitted = []
    for arm in registry:
        try:
            arm.fit(frame, as_of=as_of)
            fitted.append(arm)
        except Exception as exc:                      # noqa: BLE001
            notes.append(f"arm {arm.name} failed to fit: {type(exc).__name__}: {exc}")
    return fitted


def match_market(prices: list, registry: TeamRegistry) -> dict[tuple[str, str], object]:
    """Key market prices by canonical club pair.

    Bookmaker spellings differ from every other source ("Bodø/Glimt", "Sabah FK",
    "RC Lens"). Unmatched prices are dropped rather than guessed - a price matched
    to the wrong fixture would settle bets at the wrong number.
    """
    out = {}
    for price in prices:
        home = registry.resolve(price.home)
        away = registry.resolve(price.away)
        if home and away:
            out[(home, away)] = price
    return out


def run_matchday(
    *,
    as_of: date | None = None,
    horizon_days: int = 3,
    simulations: int = 20_000,
    fast: bool = False,
    place_bets: bool = True,
    rebuild_corpus: bool = False,
) -> MatchdayBrief:
    as_of = as_of or date.today()
    notes: list[str] = []

    # 1. corpus -------------------------------------------------------------
    if rebuild_corpus:
        frame, team_registry, _ = corpus_module.build(as_of=as_of)
        corpus_module.save(frame, team_registry)
    else:
        frame, team_registry = corpus_module.load()

    upcoming = frame[
        (frame["competition"] == "UCL")
        & (~frame["played"])
        & (frame["date"] >= pd.Timestamp(as_of))
        & (frame["date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=horizon_days))
    ].sort_values(["date", "home"])

    if upcoming.empty:
        notes.append(f"No UCL fixtures within {horizon_days} days of {as_of}")

    # 2. arms ---------------------------------------------------------------
    registry, _backbone = build_arms(fast=fast)
    arms = fit_all(registry, frame, as_of, notes)
    if not arms:
        raise RuntimeError("No arms fitted; refusing to produce a brief")

    # 3. predictions BEFORE prices -----------------------------------------
    fixture_rows: list[dict] = []
    per_arm_matrices: dict[str, dict[tuple[str, str], np.ndarray]] = {a.name: {} for a in arms}

    for row in upcoming.itertuples(index=False):
        entry = {
            "date": str(row.date.date()),
            "kickoff": getattr(row, "kickoff", ""),
            "matchday": int(getattr(row, "matchday", 0) or 0),
            "home": row.home,
            "away": row.away,
            "home_display": team_registry.display_of(row.home),
            "away_display": team_registry.display_of(row.away),
            "arms": {},
        }
        for arm in arms:
            try:
                prediction = arm.predict(row.home, row.away, when=row.date.date())
            except Exception as exc:                  # noqa: BLE001
                notes.append(f"{arm.name} failed on {row.home} v {row.away}: {exc}")
                prediction = None
            if prediction is None:
                entry["arms"][arm.name] = None
                continue
            markets = prediction.markets()
            entry["arms"][arm.name] = {
                "home": markets["home"], "draw": markets["draw"], "away": markets["away"],
                "home_xg": markets["home_xg"], "away_xg": markets["away_xg"],
                "over_2.5": markets["totals"]["over_2.5"],
                "under_2.5": markets["totals"]["under_2.5"],
                "btts": markets["btts"],
                "top_scores": [[int(i), int(j), float(p)] for i, j, p in markets["top_scores"]],
            }
            per_arm_matrices[arm.name][(row.home, row.away)] = prediction.matrix
        fixture_rows.append(entry)

    # 4. market, for settlement and scoring only ---------------------------
    market_by_pair: dict = {}
    market_available = False
    try:
        prices, remaining = odds_module.fetch_odds()
        market_by_pair = match_market(prices, team_registry)
        market_available = True
        notes.append(f"Odds API credits remaining: {remaining}")
        unmatched = len(prices) - len(market_by_pair)
        if unmatched:
            notes.append(f"{unmatched} market events could not be matched to fixtures")
    except odds_module.OddsUnavailable as exc:
        notes.append(f"market unavailable ({type(exc).__name__}): {exc}")

    for entry in fixture_rows:
        entry.setdefault("market", None)
        entry.setdefault("market_in_play_withheld", False)
        price = market_by_pair.get((entry["home"], entry["away"]))
        status, resolved = resolve_market(price, team_registry,
                                          entry["home"], entry["away"])
        if status == "in_play":
            entry["market_in_play_withheld"] = True
            notes.append(
                f"{entry['home_display']} v {entry['away_display']}: market "
                f"withheld (kicked off {price.commence_time.isoformat()}); "
                f"model shown without a benchmark"
            )
        elif status == "ok":
            entry["market"] = resolved

    # 5. paper ledger -------------------------------------------------------
    ledger = Ledger.load()
    if place_bets:
        _stake(ledger, fixture_rows, arms, notes)
        ledger.save()

    # 6. season simulation --------------------------------------------------
    qualification_rows: list[dict] = []
    league_phase = frame[(frame["competition"] == "UCL")
                         & (frame["season"] == "2026-27")]
    if not league_phase.empty and arms:
        primary = arms[0].name
        matrices = dict(per_arm_matrices[primary])
        # Fixtures beyond the horizon still need a price for the simulation.
        remaining_fixtures = league_phase[~league_phase["played"]]
        for row in remaining_fixtures.itertuples(index=False):
            key = (row.home, row.away)
            if key in matrices:
                continue
            prediction = arms[0].predict(row.home, row.away, when=row.date.date())
            if prediction is not None:
                matrices[key] = prediction.matrix
        result = qualification.simulate(league_phase, matrices, simulations=simulations)
        missing = result.table.attrs.get("missing_predictions", [])
        if missing:
            notes.append(f"{len(missing)} fixtures had no prediction in the simulation")
        qualification_rows = result.table.to_dict("records")

    # 7. arm summary --------------------------------------------------------
    arm_summary = [{
        "name": arm.name,
        "hypothesis": arm.hypothesis,
        "priced": sum(1 for e in fixture_rows if e["arms"].get(arm.name)),
        "fixtures": len(fixture_rows),
    } for arm in arms]

    brief = MatchdayBrief(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        as_of=str(as_of),
        fixtures=fixture_rows,
        arm_summary=arm_summary,
        qualification=qualification_rows,
        ledger_summary=ledger.summary(),
        market_available=market_available,
        notes=notes,
    )

    out = PROCESSED / f"brief_{as_of}.json"
    out.write_text(brief.to_json(), encoding="utf-8")
    (PROCESSED / "brief_latest.json").write_text(brief.to_json(), encoding="utf-8")
    return brief


def resolve_market(price, registry: TeamRegistry, home_key: str, away_key: str):
    """Classify a matched market price for one fixture.

    Returns one of:
        ("none", None)     - no price was matched to this fixture
        ("in_play", None)  - a price exists but the match has kicked off. A price
                             seen at or after kickoff reflects the score, not the
                             market's pre-match opinion; using it manufactures a
                             phantom edge and, worse, corrupts the season
                             scoreboard when every arm is later scored against it.
                             The EPL lab logged a 14.88% "edge" from exactly this.
        ("ok", {...})      - a usable pre-kickoff price, de-vigged and mapped to
                             home/draw/away.

    Kept as a pure function with no I/O so the in-play guard is unit-testable
    without a corpus or a live Odds API call.
    """
    if price is None:
        return "none", None
    if not odds_module.is_pre_kickoff(price):
        return "in_play", None
    devigged = price.devigged()
    mapped = _map_market_outcomes(devigged, price, registry, home_key, away_key)
    return "ok", {
        "home": mapped.get("home"), "draw": mapped.get("draw"),
        "away": mapped.get("away"),
        "overround": price.overround,
        "bookmakers": price.bookmaker_count,
        "commence_time": price.commence_time.isoformat(),
        "pre_kickoff": True,
    }


def _map_market_outcomes(devigged: dict, price, registry: TeamRegistry,
                         home_key: str, away_key: str) -> dict:
    """Translate bookmaker outcome labels into home/draw/away."""
    out: dict[str, float] = {}
    for name, probability in devigged.items():
        if name.lower() == "draw":
            out["draw"] = probability
            continue
        resolved = registry.resolve(name)
        if resolved == home_key:
            out["home"] = probability
        elif resolved == away_key:
            out["away"] = probability
    return out


def _stake(ledger: Ledger, fixture_rows: list[dict], arms: list, notes: list[str]) -> None:
    """Place fake bets: flat on every fixture for science, gated for realism."""
    for entry in fixture_rows:
        market = entry.get("market")
        for arm in arms:
            view = entry["arms"].get(arm.name)
            if not view:
                continue
            selection = max(("home", "draw", "away"), key=lambda k: view[k])
            model_probability = view[selection]
            market_probability = market.get(selection) if market else None
            decimal_odds = (1.0 / market_probability
                            if market_probability and market_probability > 0 else None)

            ledger.ensure_arm(arm.name, SCIENCE, arm.hypothesis)
            ledger.place(
                arm=arm.name, regime=SCIENCE, match_date=entry["date"],
                home=entry["home"], away=entry["away"], market="1x2",
                selection=selection, line=None,
                model_probability=model_probability,
                market_probability=market_probability,
                decimal_odds=decimal_odds,
            )

            if market_probability is not None:
                ledger.ensure_arm(arm.name, EDGE_GATED, arm.hypothesis)
                ledger.place(
                    arm=arm.name, regime=EDGE_GATED, match_date=entry["date"],
                    home=entry["home"], away=entry["away"], market="1x2",
                    selection=selection, line=None,
                    model_probability=model_probability,
                    market_probability=market_probability,
                    decimal_odds=decimal_odds,
                )
