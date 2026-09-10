"""Record your own pre-kickoff call, so the season measures whether you help.

The point of the project is a human making the final decision with model support.
That claim is only testable if the human's calls are recorded on the same terms
as every model's: same fixtures, same stake, same settlement, same scoreboard.

    python scripts/log_pick.py --home "manchester united" --away "sabah" \
        --date 2026-09-10 --pick home --confidence 0.88 --note "Sabah keeper out"

Fuzzy club names are fine - they resolve through the same registry the models
use. A pick logged after kickoff is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl.config import PROCESSED, force_utf8  # noqa: E402
from ucl.ledger.ledger import HUMAN, Ledger  # noqa: E402
from ucl.teams import TeamRegistry  # noqa: E402

force_utf8()


def main() -> int:
    parser = argparse.ArgumentParser(description="Log a human pick")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--pick", required=True, choices=["home", "draw", "away"])
    parser.add_argument("--confidence", type=float, required=True,
                        help="your probability for the pick, 0-1")
    parser.add_argument("--note", default="", help="why - worth writing down")
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        print("confidence must be strictly between 0 and 1")
        return 1

    registry = TeamRegistry.load()
    home = registry.resolve(args.home)
    away = registry.resolve(args.away)
    if not home or not away:
        print(f"could not resolve: {args.home if not home else args.away!r}")
        return 1

    match_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    if match_date < date.today():
        print(f"{match_date} is in the past; a pick logged after the fact is not a pick")
        return 1

    # Reuse the market price from the latest brief so the human arm is settled at
    # the same number the model arms are.
    market_probability = None
    decimal_odds = None
    brief_path = PROCESSED / "brief_latest.json"
    if brief_path.exists():
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        for entry in brief.get("fixtures", []):
            if (entry["home"], entry["away"], entry["date"]) == (home, away, args.date):
                market = entry.get("market") or {}
                market_probability = market.get(args.pick)
                if market_probability:
                    decimal_odds = 1.0 / market_probability
                break

    ledger = Ledger.load()
    ledger.ensure_arm("human", HUMAN, "Sahil's own pre-kickoff call")
    bet = ledger.place(
        arm="human", regime=HUMAN, match_date=args.date, home=home, away=away,
        market="1x2", selection=args.pick, model_probability=args.confidence,
        market_probability=market_probability, decimal_odds=decimal_odds,
        note=args.note,
    )
    if bet is None:
        print("pick not recorded (insufficient bankroll?)")
        return 1
    ledger.save()

    gap = ("" if market_probability is None
           else f" | market {market_probability:.1%} | gap {args.confidence - market_probability:+.1%}")
    print(f"logged {registry.display_of(home)} v {registry.display_of(away)}: "
          f"{args.pick} @ {args.confidence:.0%}{gap}")
    if args.note:
        print(f"  note: {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
