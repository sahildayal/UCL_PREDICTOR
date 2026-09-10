"""Produce a matchday brief. Entry point for CI and local runs.

Exit codes follow the EPL lab's convention so a scheduled run is legible at a
glance:
    0  completed, nothing needs a human
    1  failed
    2  completed, but a human should look (an arm failed, market missing, etc.)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from ucl.config import force_utf8  # noqa: E402
from ucl.pipeline import run_matchday  # noqa: E402
from ucl.report.dashboard import write_dashboard  # noqa: E402

force_utf8()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a UCL matchday brief")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--horizon", type=int, default=3, help="days ahead to price")
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--fast", action="store_true", help="skip expensive arms")
    parser.add_argument("--no-bets", action="store_true", help="do not touch the ledger")
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").date()
             if args.as_of else date.today())

    brief = run_matchday(
        as_of=as_of,
        horizon_days=args.horizon,
        simulations=args.simulations,
        fast=args.fast,
        place_bets=not args.no_bets,
        rebuild_corpus=args.rebuild_corpus,
    )

    print(f"Brief for {brief.as_of}: {len(brief.fixtures)} fixtures, "
          f"{len(brief.arm_summary)} arms, market={'yes' if brief.market_available else 'NO'}")
    for entry in brief.fixtures:
        market = entry.get("market")
        market_text = f"{market['home']:.1%}" if market and market.get("home") else "  --  "
        line = f"  {entry['date']} {entry['home_display'][:20]:<20} v {entry['away_display'][:20]:<20} market={market_text}"
        for name, view in entry["arms"].items():
            line += f" {name[:4]}={view['home']:.0%}" if view else f" {name[:4]}=--"
        print(line)

    for note in brief.notes:
        print(f"  note: {note}")

    if not args.no_dashboard:
        path = write_dashboard(brief)
        print(f"dashboard: {path}")

    problems = [n for n in brief.notes if "failed" in n or "unavailable" in n]
    if problems or not brief.market_available:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
