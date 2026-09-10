"""Market prices - for settlement and scoring ONLY.

The model never sees a price. This module exists so that fake bets can be
settled at a realistic number and so that every arm can be scored against the
de-vigged closing line, which is the hardest honest benchmark in football
forecasting and the one the EPL lab found nothing beat.

Two operational rules:

**Credit floor.** The Odds API key is shared with EPL_LALIGA_PREDICTOR, which
runs on its own schedule. A UCL run that exhausted the monthly quota would make
that lab fail closed with zero bets - which looks exactly like a real finding and
is not one. Every response reports remaining credits; if a call would take the
shared pool below the reserve, this module refuses to spend and says so.

**Kickoff discipline.** A price observed at or after kickoff reflects the score,
not the market's pre-match opinion. Such a price is never stamped as a closing
line. The EPL lab logged a phantom 14.88% edge from exactly this mistake.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from ..config import (CACHE, ODDS_API_KEY, ODDS_CREDIT_FLOOR, ODDS_SPORT,
                      USER_AGENT)

BASE = "https://api.the-odds-api.com/v4"


class OddsUnavailable(RuntimeError):
    """No usable prices. Callers fail closed - never substitute a guess."""


class CreditFloorReached(OddsUnavailable):
    """Refused to spend shared quota reserved for the EPL/La Liga lab."""


@dataclass
class MarketPrice:
    home: str
    away: str
    commence_time: datetime
    outcomes: dict[str, float] = field(default_factory=dict)   # name -> decimal odds
    bookmaker_count: int = 0

    def implied(self) -> dict[str, float]:
        return {k: 1.0 / v for k, v in self.outcomes.items() if v > 1.0}

    def devigged(self) -> dict[str, float]:
        """Remove the bookmaker margin proportionally.

        Proportional (multiplicative) de-vigging is the standard choice and is
        what the EPL lab used, so the two projects' benchmarks stay comparable.
        It slightly over-corrects long shots; the alternatives (Shin, additive)
        disagree by well under a point on three-way football markets.
        """
        implied = self.implied()
        total = sum(implied.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in implied.items()}

    @property
    def overround(self) -> float:
        implied = self.implied()
        return sum(implied.values()) - 1.0 if implied else 0.0


def _credits_remaining(response: requests.Response) -> int | None:
    value = response.headers.get("x-requests-remaining")
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def fetch_odds(
    *,
    markets: str = "h2h,totals",
    regions: str = "eu,uk",
    api_key: str | None = None,
    credit_floor: int | None = None,
) -> tuple[list[MarketPrice], int | None]:
    """Fetch current UCL prices. Returns (prices, credits_remaining)."""
    key = api_key or ODDS_API_KEY
    if not key:
        raise OddsUnavailable("ODDS_API_KEY is not set")
    floor = ODDS_CREDIT_FLOOR if credit_floor is None else credit_floor

    # Check the pool before spending on the real call. /sports is free.
    probe = requests.get(f"{BASE}/sports", params={"apiKey": key},
                         headers={"User-Agent": USER_AGENT}, timeout=30)
    remaining = _credits_remaining(probe)
    if remaining is not None and remaining <= floor:
        raise CreditFloorReached(
            f"Odds API credits remaining ({remaining}) at or below the reserve "
            f"({floor}) held for EPL_LALIGA_PREDICTOR. Refusing to spend."
        )

    response = requests.get(
        f"{BASE}/sports/{ODDS_SPORT}/odds",
        params={"apiKey": key, "regions": regions, "markets": markets,
                "oddsFormat": "decimal"},
        headers={"User-Agent": USER_AGENT}, timeout=30,
    )
    if response.status_code != 200:
        raise OddsUnavailable(f"Odds API HTTP {response.status_code}: {response.text[:200]}")

    remaining = _credits_remaining(response) or remaining
    payload = response.json()
    (CACHE / "odds_latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    prices: list[MarketPrice] = []
    for event in payload:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        consensus: dict[str, list[float]] = {}
        books = 0
        for bookmaker in event.get("bookmakers", []):
            books += 1
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    consensus.setdefault(outcome["name"], []).append(float(outcome["price"]))
        if not consensus:
            continue
        # Median across books is more robust than the mean to one stale quote.
        import statistics
        prices.append(MarketPrice(
            home=event.get("home_team", ""),
            away=event.get("away_team", ""),
            commence_time=commence,
            outcomes={k: statistics.median(v) for k, v in consensus.items()},
            bookmaker_count=books,
        ))
    return prices, remaining


def is_pre_kickoff(price: MarketPrice, *, now: datetime | None = None) -> bool:
    """A price is only a pre-match price before kickoff. See module docstring."""
    now = now or datetime.now(timezone.utc)
    return now < price.commence_time
