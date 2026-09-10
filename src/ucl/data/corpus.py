"""Assemble every source into one match table under canonical club keys.

The corpus has two jobs, and they pull in different directions:

  * Domestic matches separate clubs *within* a league. There are a lot of them.
  * Cross-border matches (UCL, UEL/UECL qualifiers) are the only thing that ties
    leagues to each other. There are far fewer, and they carry all the weight for
    the question this project actually asks.

So cross-border matches are flagged explicitly and the models are free to weight
them differently. Recency is handled with an exponential half-life rather than a
hard cutoff, because a hard cutoff throws away the only Azerbaijani or Kazakh
matches we have.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from ..config import PROCESSED
from ..teams import TeamRegistry
from . import openfootball as of
from . import wikipedia as wiki

# Half-life for match weighting. Two years is long by domestic-form standards and
# deliberately so: this model is about durable club and league strength, not last
# month's form, and small-league clubs need their old matches to be rated at all.
DEFAULT_HALF_LIFE_DAYS = 730.0


def build_registry(
    european: list[of.RawMatch],
    fixtures: list[wiki.Fixture],
    domestic: list[of.RawMatch] | None = None,
) -> TeamRegistry:
    """Seed the club registry from sources that state a club's country.

    European files and the Wikipedia fixtures both tag country, so the registry
    learns club -> country without a hand-maintained table. Domestic matches are
    added afterwards and resolved against what already exists.
    """
    registry = TeamRegistry()
    for match in european:
        registry.add(match.home, match.home_country)
        registry.add(match.away, match.away_country)
    for fixture in fixtures:
        registry.add(fixture.home, fixture.home_country)
        registry.add(fixture.away, fixture.away_country)
    if domestic:
        for match in domestic:
            registry.add(match.home, match.home_country)
            registry.add(match.away, match.away_country)
    return registry


def _rows_from_raw(matches: list[of.RawMatch], registry: TeamRegistry) -> list[dict]:
    rows = []
    for match in matches:
        home = registry.resolve(match.home)
        away = registry.resolve(match.away)
        if not home or not away or home == away:
            continue
        rows.append({
            "date": match.date,
            "season": match.season,
            "competition": match.competition,
            "stage": match.stage,
            "home": home,
            "away": away,
            "home_country": match.home_country or registry.country_of(home),
            "away_country": match.away_country or registry.country_of(away),
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
        })
    return rows


def _rows_from_fixtures(fixtures: list[wiki.Fixture], registry: TeamRegistry) -> list[dict]:
    rows = []
    for fixture in fixtures:
        home = registry.resolve(fixture.home)
        away = registry.resolve(fixture.away)
        if not home or not away or home == away:
            continue
        rows.append({
            "date": fixture.date,
            "season": "2026-27",
            "competition": "UCL",
            "stage": f"League, Matchday {fixture.matchday}",
            "matchday": fixture.matchday,
            "kickoff": fixture.time,
            "home": home,
            "away": away,
            "home_country": fixture.home_country or registry.country_of(home),
            "away_country": fixture.away_country or registry.country_of(away),
            "home_goals": fixture.home_goals,
            "away_goals": fixture.away_goals,
        })
    return rows


def build(
    *,
    countries: list[str] | None = None,
    start_year: int = 2011,
    end_year: int = 2025,
    include_domestic: bool = True,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, TeamRegistry, list[wiki.Fixture]]:
    """Build the full match corpus.

    Returns (matches, registry, current_season_fixtures).
    """
    as_of = as_of or date.today()

    european = of.load_european(of.euro_seasons(start_year, end_year))
    fixtures = wiki.load_current_season()

    domestic: list[of.RawMatch] = []
    if include_domestic:
        for country in (countries or of.PRIORITY_COUNTRIES):
            domestic.extend(of.load_domestic(country))

    registry = build_registry(european, fixtures, domestic)

    rows = _rows_from_raw(european, registry)
    rows += _rows_from_raw(domestic, registry)
    rows += _rows_from_fixtures(fixtures, registry)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, registry, fixtures

    frame = frame.drop_duplicates(subset=["date", "home", "away"], keep="first")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["played"] = frame["home_goals"].notna() & frame["away_goals"].notna()
    frame["cross_border"] = frame["home_country"] != frame["away_country"]
    frame["is_ucl"] = frame["competition"] == "UCL"

    as_of_ts = pd.Timestamp(as_of)
    frame["days_ago"] = (as_of_ts - frame["date"]).dt.days
    # Future fixtures get zero recency weight; they are targets, not evidence.
    frame["weight"] = frame["days_ago"].clip(lower=0).apply(
        lambda d: math.exp(-math.log(2) * d / half_life_days)
    )
    frame.loc[frame["days_ago"] < 0, "weight"] = 0.0

    return frame, registry, fixtures


def save(frame: pd.DataFrame, registry: TeamRegistry) -> None:
    frame.to_parquet(PROCESSED / "corpus.parquet", index=False)
    registry.save()


def load() -> tuple[pd.DataFrame, TeamRegistry]:
    frame = pd.read_parquet(PROCESSED / "corpus.parquet")
    return frame, TeamRegistry.load()


def summarise(frame: pd.DataFrame) -> str:
    played = frame[frame["played"]]
    cross = played[played["cross_border"]]
    lines = [
        f"matches total      : {len(frame):,}",
        f"  played           : {len(played):,}",
        f"  future fixtures  : {len(frame) - len(played):,}",
        f"cross-border played: {len(cross):,}",
        f"date range         : {frame['date'].min().date()} .. {frame['date'].max().date()}",
        f"competitions       : {frame['competition'].nunique()}",
        f"countries          : {frame['home_country'].nunique()}",
    ]
    return "\n".join(lines)
