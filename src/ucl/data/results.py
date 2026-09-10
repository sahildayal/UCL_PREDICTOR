"""Final scores for the current season, keyed the way briefs and the ledger are.

Results come from the same Wikipedia templates that supply fixtures, so
settlement needs no extra source and spends no Odds API credits.
"""
from __future__ import annotations

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
