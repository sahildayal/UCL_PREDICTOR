"""Settle finished matches and score every arm against the closing line.

Run after a matchday. Results come from the same Wikipedia templates that supply
fixtures, so settlement needs no extra source and no Odds API credits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from ucl.config import PROCESSED, force_utf8  # noqa: E402
from ucl.data.results import results_map  # noqa: E402
from ucl.eval import season  # noqa: E402
from ucl.ledger.ledger import Ledger  # noqa: E402
from ucl.report import telegram  # noqa: E402
from ucl.report.site import build_site  # noqa: E402
from ucl.teams import TeamRegistry  # noqa: E402

force_utf8()


def main() -> int:
    parser = argparse.ArgumentParser(description="Settle bets and score arms")
    parser.add_argument("--brief", default=None, help="brief JSON to score against")
    args = parser.parse_args()

    registry = TeamRegistry.load()
    played = results_map(registry)

    ledger = Ledger.load()
    settled = 0
    for (home, away, match_date), (hg, ag) in played.items():
        settled += ledger.settle_match(home=home, away=away, match_date=match_date,
                                       home_goals=hg, away_goals=ag)
    ledger.save()
    print(f"settled {settled} bets across {len(ledger.arms)} arms")
    for row in ledger.summary():
        print(f"  {row['arm']:<14} {row['regime']:<11} bankroll={row['bankroll']:>9,.2f} "
              f"P&L={row['profit']:>+9,.2f} {row['won']}W-{row['lost']}L open={row['open']}")

    # --- score every arm against the closing line -------------------------
    # season.build() is the authoritative scoreboard the public site uses. It
    # reads the archived briefs (immutable, committed pre-kickoff), joins them to
    # results, de-duplicates any re-forecast fixture, and compares each arm to the
    # market only on fixtures where BOTH priced - so an in-play-withheld market on
    # one fixture never poisons the comparison on the rest.
    board, scored = season.build(played)
    if not scored:
        print("\nno archived forecast lines up with a finished match yet")
    else:
        print(f"\nseason scoreboard ({len(scored)} scored fixtures):")
        for entry in board:
            edge = entry.get("edge_vs_market")
            edge_text = ("" if entry.get("is_market")
                         else f"  vs market {edge:+.4f}" if edge is not None
                         else "  vs market n/a")
            print(f"  {entry['arm']:<14} log_loss={entry['log_loss']:.4f} "
                  f"rps={entry['rps']:.4f}{edge_text}")

    brief_path = Path(args.brief) if args.brief else PROCESSED / "brief_latest.json"
    brief = (json.loads(brief_path.read_text(encoding="utf-8"))
             if brief_path.exists() else None)

    site = build_site(played)
    print(f"\nsite rebuilt: {site}")

    if brief is not None and telegram.send_results(brief, played, ledger):
        print("results pushed to Telegram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
