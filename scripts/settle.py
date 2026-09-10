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

import numpy as np  # noqa: E402

from ucl.config import PROCESSED, force_utf8  # noqa: E402
from ucl.data import wikipedia  # noqa: E402
from ucl.eval import metrics  # noqa: E402
from ucl.ledger.ledger import Ledger  # noqa: E402
from ucl.teams import TeamRegistry  # noqa: E402

force_utf8()


def main() -> int:
    parser = argparse.ArgumentParser(description="Settle bets and score arms")
    parser.add_argument("--brief", default=None, help="brief JSON to score against")
    args = parser.parse_args()

    registry = TeamRegistry.load()
    fixtures = wikipedia.load_current_season(max_age_hours=0.25)
    played = {}
    for fixture in fixtures:
        if not fixture.played:
            continue
        home = registry.resolve(fixture.home)
        away = registry.resolve(fixture.away)
        if home and away:
            played[(home, away, str(fixture.date))] = (fixture.home_goals, fixture.away_goals)

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

    # --- score arms against outcomes and the closing line -------------------
    brief_path = Path(args.brief) if args.brief else PROCESSED / "brief_latest.json"
    if not brief_path.exists():
        print("no brief to score")
        return 0
    brief = json.loads(brief_path.read_text(encoding="utf-8"))

    rows = []
    for entry in brief["fixtures"]:
        key = (entry["home"], entry["away"], entry["date"])
        if key not in played:
            continue
        hg, ag = played[key]
        rows.append((entry, metrics.outcome_index([hg], [ag])[0]))

    if not rows:
        print("\nno finished fixtures in the brief yet")
        return 0

    outcomes = np.array([o for _, o in rows])
    print(f"\nscoring {len(rows)} finished fixtures")
    market_probs = None
    if all(e.get("market") and e["market"].get("home") for e, _ in rows):
        market_probs = np.array([[e["market"]["home"], e["market"]["draw"],
                                  e["market"]["away"]] for e, _ in rows])
        summary = metrics.summarise(market_probs, outcomes)
        print(f"  {'market':<14} log_loss={summary['log_loss']:.4f} rps={summary['rps']:.4f}")

    arm_names = [n for n, v in rows[0][0]["arms"].items() if v]
    for name in arm_names:
        probs = np.array([[e["arms"][name]["home"], e["arms"][name]["draw"],
                           e["arms"][name]["away"]] for e, _ in rows
                          if e["arms"].get(name)])
        if len(probs) != len(outcomes):
            continue
        summary = metrics.summarise(probs, outcomes)
        line = f"  {name:<14} log_loss={summary['log_loss']:.4f} rps={summary['rps']:.4f}"
        if market_probs is not None:
            comparison = metrics.compare_to_market(probs, market_probs, outcomes)
            edge = comparison["edge_vs_market"]
            line += f"  vs market: {edge:+.4f} ({'better' if edge > 0 else 'worse'})"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
