"""The 2026/27 league phase, from Wikipedia's structured match templates.

openfootball has no 2026-27 directory yet, so Wikipedia is the only machine-
readable source for the current season. It is unusually good for this purpose:
every match is a `{{#invoke:Football box|main ...}}` template with named fields,
and the page carries all 144 league-phase fixtures through matchday 8 on
27 January 2027 - which the season-long Monte Carlo needs from day one.

Results appear in the same templates within minutes of full time, so this doubles
as the settlement source.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from .http import FetchError, fetch

API = "https://en.wikipedia.org/w/api.php"
PAGE_2627 = "2026–27 UEFA Champions League league phase"

_BOX_RE = re.compile(r"\{\{#invoke:Football box\|main(.*?)\n\}\}", re.S)
_FIELD_RE = {
    key: re.compile(r"\|\s*" + key + r"\s*=\s*(.*)")
    for key in ("date", "time", "team1", "team2", "score", "stadium")
}
_START_DATE_RE = re.compile(r"Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")
_SCORE_RE = re.compile(r"^\s*(\d{1,2})\s*[–\-]\s*(\d{1,2})")
_MATCHDAY_RE = re.compile(r"^=+\s*Matchday\s+(\d+)\s*=+\s*$", re.M)
_FBAICON_RE = re.compile(r"\{\{fbaicon\|([A-Za-z]{3})\}\}")
_PIPED_LINK_RE = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_PLAIN_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")


@dataclass
class Fixture:
    """One league-phase match. `home_goals is None` means not yet played."""
    date: date
    time: str
    matchday: int
    home: str
    away: str
    home_country: str
    away_country: str
    home_goals: int | None
    away_goals: int | None
    stadium: str = ""

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


def _clean(value: str) -> tuple[str, str]:
    """Strip wiki markup from a team field, returning (name, country code)."""
    country_match = _FBAICON_RE.search(value)
    country = country_match.group(1).upper() if country_match else ""
    text = _FBAICON_RE.sub(" ", value)
    text = _PIPED_LINK_RE.sub(r"\1", text)      # [[Bayern Munich|Bayern]] -> Bayern
    text = _PLAIN_LINK_RE.sub(r"\1", text)      # [[Roma]] -> Roma
    text = _TEMPLATE_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |"), country


def fetch_wikitext(page: str = PAGE_2627, *, max_age_hours: float = 2.0) -> str:
    url = (f"{API}?action=parse&page={requests_quote(page)}"
           f"&prop=wikitext&format=json&formatversion=2")
    body = fetch(url, max_age_hours=max_age_hours, required_marker="wikitext")
    payload = json.loads(body)
    if "parse" not in payload:
        raise FetchError(f"Wikipedia returned no parse section for {page!r}")
    return payload["parse"]["wikitext"]


def requests_quote(text: str) -> str:
    from urllib.parse import quote
    return quote(text.replace(" ", "_"), safe="")


def parse_league_phase(wikitext: str) -> list[Fixture]:
    """Parse all league-phase fixtures, tagging each with its matchday.

    Matchday is taken from the section heading each match falls under, so a
    fixture is never assigned by guessing from its date - matchday 1 spans three
    calendar days this season (8, 9 and 10 September 2026).
    """
    # Map each character offset to the matchday section it belongs to.
    boundaries: list[tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in _MATCHDAY_RE.finditer(wikitext)
    ]

    def matchday_at(offset: int) -> int:
        current = 0
        for start, number in boundaries:
            if start <= offset:
                current = number
            else:
                break
        return current

    fixtures: list[Fixture] = []
    for box in _BOX_RE.finditer(wikitext):
        body = box.group(1)
        fields: dict[str, str] = {}
        for key, pattern in _FIELD_RE.items():
            found = pattern.search(body)
            fields[key] = found.group(1).strip() if found else ""

        date_match = _START_DATE_RE.search(fields["date"])
        if not date_match:
            continue
        year, month, day = (int(g) for g in date_match.groups())

        home, home_country = _clean(fields["team1"])
        away, away_country = _clean(fields["team2"])
        if not home or not away:
            continue

        score_match = _SCORE_RE.match(fields["score"])
        home_goals = int(score_match.group(1)) if score_match else None
        away_goals = int(score_match.group(2)) if score_match else None

        stadium, _ = _clean(fields["stadium"])
        time_text = re.sub(r"&nbsp;.*$", "", fields["time"]).strip()

        fixtures.append(Fixture(
            date=date(year, month, day),
            time=time_text[:5],
            matchday=matchday_at(box.start()),
            home=home,
            away=away,
            home_country=home_country,
            away_country=away_country,
            home_goals=home_goals,
            away_goals=away_goals,
            stadium=stadium,
        ))
    return fixtures


def load_current_season(*, max_age_hours: float = 2.0) -> list[Fixture]:
    return parse_league_phase(fetch_wikitext(max_age_hours=max_age_hours))


def load_season(year: int, *, max_age_hours: float = 24 * 30) -> list[Fixture]:
    """Load an arbitrary season's league phase, e.g. 2025 for 2025-26.

    Used to validate the parser against a completed season before trusting it on
    the live one.
    """
    page = f"{year}–{str(year + 1)[-2:]} UEFA Champions League league phase"
    return parse_league_phase(fetch_wikitext(page, max_age_hours=max_age_hours))
