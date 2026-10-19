"""Final scores for the current season, keyed the way briefs and the ledger are.

Results come from the same Wikipedia templates that supply fixtures, so
settlement needs no extra source and spends no Odds API credits.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..kickoff import kickoff_utc
from ..teams import TeamRegistry
from . import wikipedia


def results_map(registry: TeamRegistry | None = None,
                *, max_age_hours: float = 0.25) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Map (home_key, away_key, 'YYYY-MM-DD') -> (home_goals, away_goals).

    Only finished matches appear. A fixture whose clubs cannot be resolved is
    skipped rather than guessed - settling the wrong fixture is worse than
    leaving a bet open.
    """
    registry = registry or TeamRegistry.load()
    out: dict[tuple[str, str, str], tuple[int, int]] = {}
    for fixture in wikipedia.load_current_season(max_age_hours=max_age_hours):
        if not fixture.played:
            continue
        home = registry.resolve(fixture.home)
        away = registry.resolve(fixture.away)
        if not home or not away:
            continue
        out[(home, away, str(fixture.date))] = (fixture.home_goals, fixture.away_goals)
    return out


def overdue_fixtures(*, now: datetime | None = None, grace_hours: float = 3.0,
                     max_age_hours: float = 0.25) -> list[wikipedia.Fixture]:
    """Matches that kicked off more than `grace_hours` ago but have no score.

    A parser that stops reading scores does not fail - it just reports every
    match as unplayed, and the bets sit open forever. Matchday 2 lost six days
    that way. This is the tripwire: settlement surfaces anything overdue instead
    of quietly settling nothing.
    """
    now = now or datetime.now(timezone.utc)
    overdue = []
    for fixture in wikipedia.load_current_season(max_age_hours=max_age_hours):
        if fixture.played:
            continue
        kickoff = kickoff_utc(fixture.date, fixture.time)
        if kickoff and now - kickoff > timedelta(hours=grace_hours):
            overdue.append(fixture)
    return overdue
