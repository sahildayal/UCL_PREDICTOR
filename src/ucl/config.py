"""Central configuration. Import this before anything that prints."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- UTF-8 enforcement -------------------------------------------------------
# Windows consoles default to cp1252, which cannot encode Qarabag, Bodo/Glimt or
# Sparta Praha. Any print() of a club name would raise UnicodeEncodeError and kill
# a matchday run. Fix it at import time, once, for every entry point.
def force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


force_utf8()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"
LEDGER_DIR = DATA / "ledger"
REPORTS = ROOT / "reports"

for _d in (RAW, CACHE, PROCESSED, LEDGER_DIR, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

SEASON = "2026-27"
COMPETITION = "UCL"

# The league phase: 36 teams, 8 matches each, 144 matches total.
N_TEAMS = 36
N_MATCHDAYS = 8
N_LEAGUE_MATCHES = 144

# Qualification thresholds under the league-phase format.
AUTO_R16_CUTOFF = 8      # positions 1-8 advance directly
PLAYOFF_CUTOFF = 24      # positions 9-24 enter the knockout playoff

USER_AGENT = "ucl-predictor/0.1 (personal research; contact via github.com/sahildayal)"

# --- Odds API ----------------------------------------------------------------
# The key is shared with EPL_LALIGA_PREDICTOR. Never let a UCL run starve that
# lab: abort if a call would push remaining credits below this reserve.
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_CREDIT_FLOOR = int(os.getenv("ODDS_CREDIT_FLOOR", "150"))
ODDS_SPORT = "soccer_uefa_champs_league"

# --- Paper ledger ------------------------------------------------------------
STARTING_BANKROLL = 10_000.0
FLAT_STAKE = 100.0          # science arms: every fixture, same stake
EDGE_THRESHOLD = 0.03       # edge-gated arms: 3 percentage points vs de-vigged line
KELLY_FRACTION = 0.25

# No order placement exists in this repository. Prices are read-only.
LIVE_TRADING_IMPOSSIBLE = True
