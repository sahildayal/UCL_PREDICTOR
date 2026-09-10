"""Canonical club identity across sources.

Three sources spell the same club three ways:
    openfootball : "FC Bayern Munchen"      (with country tag "(GER)")
    Wikipedia    : "Bayern Munich"
    Odds API     : "Bayern Munich"

A silent mismatch does not raise - it invents a second club with a handful of
matches and a garbage rating. That is the single most dangerous failure mode in a
cross-league model, so resolution is explicit, logged, and fails loud on request.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .config import PROCESSED

# Tokens that carry no identity: legal forms, club-type prefixes, founding years.
_NOISE = {
    "fc", "cf", "afc", "sc", "ac", "as", "sk", "fk", "kv", "bc", "bk", "if", "ff",
    "sv", "vfb", "vfl", "tsg", "cd", "ud", "rcd", "sd", "cp", "scp", "sl", "ogc",
    # Club-type prefixes that bookmakers keep and other sources drop, e.g. the
    # Odds API says "RC Lens" where every other source says "Lens". An unmatched
    # price means a fixture silently loses its benchmark and settlement number.
    "rc", "rsc", "ssc", "acf", "losc", "aj", "kaa", "krc", "stade", "gnk",
    "club", "futbol", "football", "fussball", "calcio", "sport", "sportif",
    "sportive", "spor", "kulubu", "de", "del", "di", "the", "ii", "team",
    "1899", "1900", "1901", "1902", "1903", "1904", "1905", "1906", "1907",
    "1908", "1909", "1910", "1911", "1912", "1913", "1999", "04", "05", "96",
    "79", "03",
}

# Explicit aliases where normalisation alone cannot bridge the gap.
# Left side is the *normalised* form; right side is the canonical key.
_ALIASES: dict[str, str] = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "bayern munchen": "bayern munich",
    "bayern": "bayern munich",
    "internazionale milano": "inter",
    "internazionale": "inter",
    "inter milan": "inter",
    "atletico madrid": "atletico madrid",
    "atletico": "atletico madrid",
    "atleti": "atletico madrid",
    "paris saint germain": "psg",
    "paris sg": "psg",
    "sport lisboa e benfica": "benfica",
    "slavia praha": "slavia prague",
    "sparta praha": "sparta prague",
    "bodo glimt": "bodo/glimt",
    "bodoglimt": "bodo/glimt",
    "qarabag agdam": "qarabag",
    "royale union saint gilloise": "union saint-gilloise",
    "union saint gilloise": "union saint-gilloise",
    "union sg": "union saint-gilloise",
    "royal antwerp": "antwerp",
    "olympiakos": "olympiacos",
    "olympiacos piraeus": "olympiacos",
    "dortmund": "borussia dortmund",
    "leverkusen": "bayer leverkusen",
    "leipzig": "rb leipzig",
    "red bull salzburg": "salzburg",
    "psv eindhoven": "psv",
    "eindhoven": "psv",
    "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
    "sporting lisbon": "sporting",
    "athletic": "athletic bilbao",
    "athletic bilbao": "athletic bilbao",
    "real betis balompie": "real betis",
    "betis": "real betis",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "club brugge": "club brugge",
    "brugge": "club brugge",
    "newcastle united": "newcastle",
    "monaco": "monaco",
    "bayern munich": "bayern munich",
    # Found by the near-duplicate scan in scripts/audit_teams.py. Each of these
    # was two clubs in the registry, splitting one club's match history in half.
    "1 union berlin": "union berlin",
    "aek athen": "aek athens",
    "bor monchengladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
}

# Pairs the scan flags that must NOT be merged - they are genuinely different
# clubs whose names are nearly identical. Recorded so a future pass at the alias
# table does not "fix" them:
#   CS Universitatea Craiova / FC Universitatea Craiova  (rival Romanian clubs)
#   FC Santa Coloma / UE Santa Coloma                     (both Andorran)
#   B36 Torshavn / HB Torshavn                            (both Faroese)
#   Hibernian (SCO) / Hibernians (MLT)
#   Partizan (SRB) / Partizani (ALB)
DO_NOT_MERGE = {
    ("cs universitatea craiova", "universitatea craiova"),
    ("santa coloma", "ue santa coloma"),
    ("b36 torshavn", "hb torshavn"),
    ("hibernian", "hibernians"),
    ("partizan", "partizani"),
}

# Characters that appear inside club names and must be folded for matching.
_PUNCT = re.compile(r"[.\-_'`‘’“”]")
_COUNTRY_TAG = re.compile(r"\(([A-Za-z]{3})\)")


def strip_accents(text: str) -> str:
    """Bodo/Glimt, Qarabag, Fenerbahce: fold to ASCII for matching only."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalise(name: str) -> str:
    """Reduce a club name to a comparable key. Never shown to a user."""
    if not name:
        return ""
    s = strip_accents(name).lower()
    s = _COUNTRY_TAG.sub(" ", s)                    # drop "(ESP)" country tags
    # NFKD does not decompose these, so "Sumgayit" spelled with a dotless i
    # became a separate club from "Sumgayit" spelled with a normal one.
    for src, dst in (("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("đ", "d"),
                     ("ł", "l"), ("ı", "i"), ("ǝ", "e"), ("ə", "e")):
        s = s.replace(src, dst)
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"[^a-z0-9/ ]", " ", s)
    tokens = [t for t in s.split() if t and t not in _NOISE]
    if not tokens:                                  # name was entirely noise
        tokens = [t for t in s.split() if t]
    return " ".join(tokens).strip()


@dataclass
class Club:
    """A club with a stable id and the country whose league it plays in."""
    key: str
    display: str
    country: str = ""
    aliases: set[str] = field(default_factory=set)


class TeamRegistry:
    """Resolves any spelling to a canonical club key.

    Countries come from the openfootball "(GER)" tags, which makes the registry
    self-populating: the league-strength model needs club -> country, and the
    match corpus already states it on every row.
    """

    def __init__(self) -> None:
        self._clubs: dict[str, Club] = {}
        self._index: dict[str, str] = {}
        self.unresolved: dict[str, int] = {}

    # -- construction ---------------------------------------------------------
    def add(self, raw_name: str, country: str = "") -> str:
        key = self._canonical_key(raw_name)
        club = self._clubs.get(key)
        if club is None:
            club = Club(key=key, display=self._display(raw_name), country=country)
            self._clubs[key] = club
        if country and not club.country:
            club.country = country
        club.aliases.add(raw_name)
        self._index[normalise(raw_name)] = key
        self._index[key] = key
        return key

    @staticmethod
    def _display(raw_name: str) -> str:
        return _COUNTRY_TAG.sub("", raw_name).strip()

    @staticmethod
    def _canonical_key(raw_name: str) -> str:
        n = normalise(raw_name)
        return _ALIASES.get(n, n)

    # -- resolution -----------------------------------------------------------
    def resolve(self, raw_name: str, *, strict: bool = False) -> str | None:
        """Return a canonical key, or None if the name cannot be matched.

        Fuzzy matching is deliberately tight (0.90). A loose threshold merges
        distinct clubs - "Milan" vs "Inter Milan", "Sporting" vs "Sporting Gijon" -
        which is far worse than an honest miss that shows up in `unresolved`.
        """
        n = normalise(raw_name)
        if not n:
            return None
        key = _ALIASES.get(n, n)
        if key in self._index:
            return self._index[key]
        if n in self._index:
            return self._index[n]

        candidates = difflib.get_close_matches(key, list(self._clubs.keys()), n=1, cutoff=0.90)
        if candidates:
            self._index[n] = candidates[0]
            return candidates[0]

        self.unresolved[raw_name] = self.unresolved.get(raw_name, 0) + 1
        if strict:
            raise KeyError(f"Unresolvable club name: {raw_name!r} (normalised {n!r})")
        return None

    # -- access ---------------------------------------------------------------
    def get(self, key: str) -> Club | None:
        return self._clubs.get(key)

    def country_of(self, key: str) -> str:
        club = self._clubs.get(key)
        return club.country if club else ""

    def display_of(self, key: str) -> str:
        club = self._clubs.get(key)
        return club.display if club else key

    def keys(self) -> list[str]:
        return sorted(self._clubs)

    def __len__(self) -> int:
        return len(self._clubs)

    def __contains__(self, key: str) -> bool:
        return key in self._clubs

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or PROCESSED / "team_registry.json"
        payload = {
            k: {"display": c.display, "country": c.country, "aliases": sorted(c.aliases)}
            for k, c in sorted(self._clubs.items())
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "TeamRegistry":
        path = path or PROCESSED / "team_registry.json"
        reg = cls()
        if not path.exists():
            return reg
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, rec in data.items():
            club = Club(key=key, display=rec["display"], country=rec.get("country", ""),
                        aliases=set(rec.get("aliases", [])))
            reg._clubs[key] = club
            reg._index[key] = key
            for alias in club.aliases:
                reg._index[normalise(alias)] = key
        return reg
