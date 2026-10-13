"""Kickoff times in UTC, from the fixture list's local times.

UEFA - and the Wikipedia pages built from its listings - publish kickoffs in
Central European time: CEST (UTC+2) up to the late-October clock change, CET
(UTC+1) after it. Converting through the Europe/Paris zone gets both halves of
the league phase right without hardcoding which matchdays fall on which side.

This exists because a prediction is only worth scoring if it was provably made
before the match started (DESIGN.md section 9). On 2026-10-13 the scheduled run
fired at 16:52 UTC, seven minutes after two 18:45 CEST kickoffs, and the
pipeline staked those fixtures anyway. A missing market price was the only
symptom. The pipeline needs to know kickoff itself, not infer it from whether
a bookmaker still lists the match.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

UEFA_ZONE = ZoneInfo("Europe/Paris")


def kickoff_utc(match_date: str | date, local_time: str | None) -> datetime | None:
    """Return the UTC kickoff for a fixture listed at `local_time` CET/CEST.

    Returns None when the time is missing or unreadable. Callers treat an
    unknown kickoff as "cannot prove pre-kickoff" only where that matters.
    """
    if not local_time:
        return None
    try:
        hours, minutes = (int(part) for part in local_time.strip()[:5].split(":"))
        day = (match_date if isinstance(match_date, date)
               else date.fromisoformat(str(match_date)))
        local = datetime.combine(day, time(hours, minutes), tzinfo=UEFA_ZONE)
    except (ValueError, TypeError):
        return None
    return local.astimezone(timezone.utc)


def has_kicked_off(match_date: str | date, local_time: str | None,
                   *, now: datetime | None = None) -> bool:
    """True once the fixture's kickoff time has passed.

    An unknown kickoff returns False. The market guard and the bookmaker's own
    commence time still apply in that case, so this check adds protection and
    never removes any.
    """
    kickoff = kickoff_utc(match_date, local_time)
    if kickoff is None:
        return False
    return (now or datetime.now(timezone.utc)) >= kickoff
