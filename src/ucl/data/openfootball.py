"""Parser and loaders for the openfootball plain-text match format.

The format is identical for European competitions and domestic leagues; the only
difference is that European files tag each club with a country:

    (UCL)       18:45  Athletic Club (ESP)  v Arsenal FC (ENG)   0-2 (0-0)
    (domestic)  16:00  Stromsgodset IF      v Rosenborg BK       1-2 (0-1)

Date headers carry the year only on their first appearance in a block:

      Tue Sep 16 2025
        ...matches...
      Wed Sep 17
        ...matches...

so the year has to be carried forward. Matches with no score are future fixtures
and are kept - the season-long Monte Carlo needs them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .http import FetchError, fetch, try_fetch

RAW_BASE = "https://raw.githubusercontent.com/openfootball"

# Countries whose leagues live in their own top-level repo rather than in `europe`.
_OWN_REPO = {
    "england": "england",
    "espana": "espana",
    "deutschland": "deutschland",
    "italy": "italy",
    "austria": "austria",
    "belgium": "belgium",
}

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# "  Tue Sep 16 2025" / "  Wed Sep 17" / "Fri Aug 9" (older files are unindented)
_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
)
# "▪ League, Matchday 1" / "▪ 1. Round"
_STAGE_RE = re.compile(r"^\s*[▪●\-]\s*(.+?)\s*$")
# Optional time, then "Home (XXX) v Away (YYY)  2-1 (1-0)", with the score tail
# left unparsed here because knockout ties use several shapes (see _read_score).
_MATCH_RE = re.compile(
    r"^\s+(?:(\d{1,2}:\d{2})\s+)?"           # optional kickoff time
    r"(.+?)\s+v\s+(.+?)"                      # home v away (non-greedy)
    r"(?:\s{2,}(\S.*?))?"                     # optional score tail
    r"\s*$"
)
_SCORE_PAIR = re.compile(r"(\d{1,2})\s*[-–]\s*(\d{1,2})")

# Older openfootball files put the score BETWEEN the teams and omit the " v ":
#     "  20:00  Liverpool FC             4-1 (4-0)  Norwich City"
# Missing this shape silently yields zero matches for roughly fifteen seasons of
# every big league, which looks exactly like "the season is not published yet".
_MATCH_RE_V1 = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?"                    # optional kickoff time
    r"(\S.*?)\s{2,}"                                   # home team
    r"(\d{1,2})\s*[-–]\s*(\d{1,2})"                    # full-time score
    r"(?:\s*\(\s*\d{1,2}\s*[-–]\s*\d{1,2}\s*\))?"      # optional half-time score
    r"(?:\s*(?:a\.e\.t\.|pen\.)[^ ]*)?"                # optional ET/pen marker
    r"\s{2,}(\S.*?)\s*$"                               # away team
)
_COUNTRY_RE = re.compile(r"\(([A-Z]{3})\)\s*$")


@dataclass
class RawMatch:
    """One match exactly as the source states it, before canonicalisation."""
    date: date
    home: str
    away: str
    home_country: str
    away_country: str
    home_goals: int | None
    away_goals: int | None
    competition: str
    season: str
    stage: str

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


def _read_score(tail: str | None) -> tuple[int | None, int | None]:
    """Extract THIS match's 90-minute score from an openfootball score tail.

    Shapes seen in the wild:
        "0-2 (0-0)"                     -> 0-2 at full time, 0-0 at half time
        "3-2 a.e.t. (3-0, 1-0)"         -> 3-2 is the TWO-LEG AGGREGATE;
                                           this match finished 3-0, 1-0 at half
        "4-3 pen. 1-1 a.e.t. (1-1, 0-1)"-> shootout 4-3, 1-1 after extra time,
                                           1-1 at 90 minutes, 0-1 at half time

    Taking the leading pair naively would record a 5-0 match that never happened
    and hand the rating model a fabricated thrashing. Whenever extra time or
    penalties are flagged, the first parenthetical pair is this match's real
    90-minute score, so prefer it.
    """
    if not tail:
        return None, None
    extra_time = "a.e.t" in tail.lower() or "pen." in tail.lower()
    if extra_time:
        paren = re.search(r"\(([^)]*)\)", tail)
        if paren:
            inner = _SCORE_PAIR.search(paren.group(1))
            if inner:
                return int(inner.group(1)), int(inner.group(2))
        return None, None          # flagged as ET but unreadable: refuse to guess
    leading = _SCORE_PAIR.search(tail)
    if not leading:
        return None, None
    return int(leading.group(1)), int(leading.group(2))


def _split_country(name: str) -> tuple[str, str]:
    match = _COUNTRY_RE.search(name)
    if not match:
        return name.strip(), ""
    return _COUNTRY_RE.sub("", name).strip(), match.group(1)


_HEADER_YEAR_RE = re.compile(r"^#\s*Date\b.*?(\d{4})", re.M)


def _seed_year(text: str, season: str) -> int | None:
    """Find the season's starting year.

    Legacy files state the year only in the `# Date Fri Aug 9 2019 - ...` header
    comment; their date lines are bare ("Fri Aug 9"). Without a seed, every match
    in those files is skipped for want of a year - which is indistinguishable
    from the season not being published.
    """
    header = _HEADER_YEAR_RE.search(text)
    if header:
        return int(header.group(1))
    label = re.match(r"(\d{4})", season)
    return int(label.group(1)) if label else None


def parse(text: str, *, competition: str, season: str,
          default_country: str = "") -> list[RawMatch]:
    """Parse an openfootball text file into matches, in either format generation."""
    matches: list[RawMatch] = []
    current_date: date | None = None
    seed_year: int | None = _seed_year(text, season)
    current_year: int | None = seed_year
    previous_month: int | None = None
    stage = ""
    # "2019-20" spans two calendar years; "2025" (Norway, Sweden) spans one and
    # must never roll over. Without this, a rescheduled fixture whose month steps
    # backwards silently advances the year - Norway 2025 ran to 2032.
    spans_two_years = bool(re.match(r"^\d{4}-\d{2}$", season))
    rolled_over = False

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "=")):
            continue

        date_match = _DATE_RE.match(line)
        if date_match:
            month_name, day, year = date_match.groups()
            month = _MONTHS.get(month_name)
            if not month:
                continue
            if year:
                current_year = int(year)
                rolled_over = seed_year is not None and current_year > seed_year
            elif (spans_two_years and not rolled_over and current_year is not None
                  and previous_month is not None
                  and previous_month >= 7 and month <= 6):
                # A split season runs roughly August to May, so exactly one
                # year boundary can occur, and only late-month -> early-month.
                current_year += 1
                rolled_over = True
            if current_year is None:
                continue
            previous_month = month
            try:
                current_date = date(current_year, month, int(day))
            except ValueError:
                current_date = None
            continue

        stripped = line.strip()
        if stripped.startswith(("▪", "●")):
            stage_match = _STAGE_RE.match(line)
            if stage_match:
                stage = stage_match.group(1)
            continue

        if current_date is None:
            continue

        if " v " in line:                      # modern layout: home v away score
            match = _MATCH_RE.match(line)
            if not match:
                continue
            _time, home_raw, away_raw, score_tail = match.groups()
            hg, ag = _read_score(score_tail)
        else:                                  # legacy layout: home score away
            match = _MATCH_RE_V1.match(line)
            if not match:
                continue
            _time, home_raw, hg_text, ag_text, away_raw = match.groups()
            hg, ag = int(hg_text), int(ag_text)

        home, home_country = _split_country(home_raw)
        away, away_country = _split_country(away_raw)
        if not home or not away:
            continue

        matches.append(RawMatch(
            date=current_date,
            home=home,
            away=away,
            home_country=home_country or default_country,
            away_country=away_country or default_country,
            home_goals=hg,
            away_goals=ag,
            competition=competition,
            season=season,
            stage=stage,
        ))
    return matches


# --- European competitions ---------------------------------------------------

# Every file in the champions-league repo. The qualifiers matter more than their
# glamour suggests: they are where Azerbaijani, Norwegian and Kazakh clubs play
# the rest of Europe often enough to be rated at all.
_EURO_FILES = {
    "cl.txt": "UCL",
    "clq.txt": "UCL-Q",
    "elq.txt": "UEL-Q",
    "confq.txt": "UECL-Q",
}


def euro_seasons(start_year: int = 2011, end_year: int = 2025) -> list[str]:
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]


def load_european(seasons: list[str] | None = None) -> list[RawMatch]:
    """Load UCL and European qualifying matches - the cross-border corpus."""
    seasons = seasons or euro_seasons()
    out: list[RawMatch] = []
    for season in seasons:
        for filename, competition in _EURO_FILES.items():
            url = f"{RAW_BASE}/champions-league/master/{season}/{filename}"
            body = try_fetch(url, max_age_hours=24 * 30)
            if not body:
                continue
            out.extend(parse(body, competition=competition, season=season))
    return out


# --- Domestic leagues --------------------------------------------------------

# The big leagues live in their own repos and use a per-season DIRECTORY layout
# (`2026-27/1-premierleague.txt`) rather than the flat `2026-27_fr1.txt` files in
# the `europe` repo. Missing this silently drops England, Spain, Germany and Italy
# entirely - which leaves Manchester United rated on ~68 European matches and no
# domestic ones at all, and drops newly promoted clubs like Como from the model.
# Tier-1 filenames are hardcoded so seasons can be addressed directly, without
# spending GitHub's 60-calls-per-hour unauthenticated budget on directory listings.
_OWN_REPO_TIER1 = {
    "england": ["1-premierleague.txt"],
    "espana": ["1-liga.txt"],
    "deutschland": ["1-bundesliga.txt"],
    "italy": ["1-seriea.txt"],
    "austria": ["1-bundesliga.txt"],
    "belgium": ["be1.txt", "1-proleague.txt"],
}


def _season_labels(start_year: int = 2011, end_year: int = 2026) -> list[str]:
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]


def list_domestic_files(country: str) -> list[str]:
    """List flat-layout season files for a country via the GitHub contents API."""
    api = f"https://api.github.com/repos/openfootball/europe/contents/{country}"
    body = try_fetch(api, max_age_hours=24 * 7)
    if not body:
        return []
    return re.findall(r'"name":\s*"([^"]+\.txt)"', body)


def load_domestic(country: str, *, tier1_only: bool = True,
                  start_year: int = 2011, end_year: int = 2026) -> list[RawMatch]:
    """Load a country's top-flight league seasons, handling both repo layouts.

    `tier1_only` keeps top-flight files and drops second tiers and cups, named
    like `2024-25_no2.txt` and `2024-25_nocup.txt`. Mixing tiers without modelling
    the tier would corrupt the league-strength estimate: a second-division club
    would be rated as though its results came against top-flight opposition.
    """
    country_code = COUNTRY_CODES.get(country, country[:3].upper())
    out: list[RawMatch] = []

    if country in _OWN_REPO:
        repo = _OWN_REPO[country]
        for season in _season_labels(start_year, end_year):
            for filename in _OWN_REPO_TIER1.get(country, []):
                url = f"{RAW_BASE}/{repo}/master/{season}/{filename}"
                body = try_fetch(url, max_age_hours=24 * 30)
                if body:
                    out.extend(parse(body, competition=f"DOM-{country_code}",
                                     season=season, default_country=country_code))
                    break          # first matching tier-1 filename wins
        return out

    for filename in list_domestic_files(country):
        if tier1_only and not re.search(r"_[a-z]{2}1\.txt$", filename):
            continue
        season = filename.split("_")[0]
        url = f"{RAW_BASE}/europe/master/{country}/{filename}"
        body = try_fetch(url, max_age_hours=24 * 30)
        if not body:
            continue
        out.extend(parse(body, competition=f"DOM-{country_code}", season=season,
                         default_country=country_code))
    return out


# Map openfootball directory names to the three-letter codes used in the
# European files, so a club's domestic and European rows agree on its country.
COUNTRY_CODES = {
    "england": "ENG", "espana": "ESP", "deutschland": "GER", "italy": "ITA",
    "france": "FRA", "netherlands": "NED", "portugal": "POR", "belgium": "BEL",
    "austria": "AUT", "turkey": "TUR", "ukraine": "UKR", "norway": "NOR",
    "czech-republic": "CZE", "greece": "GRE", "cyprus": "CYP", "scotland": "SCO",
    "switzerland": "SUI", "denmark": "DEN", "sweden": "SWE", "poland": "POL",
    "croatia": "CRO", "serbia": "SRB", "romania": "ROU", "bulgaria": "BUL",
    "azerbaijan": "AZE", "kazakhstan": "KAZ", "israel": "ISR", "hungary": "HUN",
    "slovakia": "SVK", "slovenia": "SVN", "russia": "RUS", "belarus": "BLR",
    "finland": "FIN", "iceland": "ISL", "ireland": "IRL", "wales": "WAL",
    "north-macedonia": "MKD", "bosnia-herzegovina": "BIH", "albania": "ALB",
    "armenia": "ARM", "georgia": "GEO", "moldova": "MDA", "montenegro": "MNE",
    "kosovo": "KOS", "latvia": "LVA", "lithuania": "LTU", "estonia": "EST",
    "luxembourg": "LUX", "malta": "MLT", "faroe-islands": "FRO",
    "gibraltar": "GIB", "andorra": "AND", "san-marino": "SMR",
    "northern-ireland": "NIR", "liechtenstein": "LIE",
}

# Countries with a club in the 2026/27 league phase, plus the larger leagues
# that supply most cross-border opposition.
PRIORITY_COUNTRIES = [
    "england", "espana", "deutschland", "italy", "france", "netherlands",
    "portugal", "belgium", "austria", "turkey", "ukraine", "norway",
    "czech-republic", "greece", "cyprus", "scotland", "switzerland", "denmark",
    "sweden", "croatia", "serbia", "azerbaijan", "kazakhstan", "poland",
    "romania", "slovakia", "slovenia", "hungary", "bulgaria", "israel",
]
