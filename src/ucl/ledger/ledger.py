"""Paper ledger. One bankroll per arm. No real money, no order placement.

There is no code path in this repository that places a bet. The Odds API key is
used read-only, for settlement prices and scoring. That absence is the design, not
an oversight, and it should stay that way.

Two staking regimes run side by side, which is the direct answer to what went
wrong in the EPL lab:

  **science arms** stake a flat amount on every fixture. Over a league phase of
  144 matches, disciplined thresholds would produce a handful of bets and no
  statistical power to separate six model families. Betting everything produces
  144 scored decisions per arm.

  **edge-gated arms** stake only on material disagreement with the de-vigged
  line, which is what realistic P&L looks like.

The two answer different questions and are never mixed. And a `human` arm records
Sahil's own pre-kickoff calls, so the season measures whether human judgement adds
to the model or subtracts from it - the question the whole project is for.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import (EDGE_THRESHOLD, FLAT_STAKE, KELLY_FRACTION, LEDGER_DIR,
                      STARTING_BANKROLL)

SCIENCE = "science"
EDGE_GATED = "edge_gated"
HUMAN = "human"


@dataclass
class Bet:
    bet_id: str
    arm: str
    regime: str
    match_date: str
    home: str
    away: str
    market: str            # "1x2", "totals", "btts"
    selection: str         # "home" / "draw" / "away" / "over" / "under"
    line: float | None     # goals line for totals; None for 1x2. Never implicit.
    stake: float
    model_probability: float
    market_probability: float | None
    decimal_odds: float | None
    edge: float | None
    placed_at: str
    status: str = "open"   # open | won | lost | void | push
    payout: float = 0.0
    settled_at: str | None = None
    result: str | None = None
    note: str = ""


@dataclass
class Arm:
    name: str
    regime: str
    bankroll: float = STARTING_BANKROLL
    starting_bankroll: float = STARTING_BANKROLL
    hypothesis: str = ""
    bets: list[Bet] = field(default_factory=list)

    @property
    def staked(self) -> float:
        return sum(b.stake for b in self.bets if b.status != "void")

    @property
    def settled(self) -> list[Bet]:
        return [b for b in self.bets if b.status in {"won", "lost", "push"}]

    @property
    def open_bets(self) -> list[Bet]:
        return [b for b in self.bets if b.status == "open"]

    @property
    def profit(self) -> float:
        return self.bankroll - self.starting_bankroll

    @property
    def roi(self) -> float:
        staked = sum(b.stake for b in self.settled)
        return (self.profit / staked) if staked > 0 else 0.0

    @property
    def record(self) -> tuple[int, int, int]:
        won = sum(1 for b in self.settled if b.status == "won")
        lost = sum(1 for b in self.settled if b.status == "lost")
        push = sum(1 for b in self.settled if b.status == "push")
        return won, lost, push


class Ledger:
    """Persistent record of every fake bet, per arm."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or LEDGER_DIR / "ledger.json"
        self.arms: dict[str, Arm] = {}

    # -- arm management -------------------------------------------------------
    def ensure_arm(self, name: str, regime: str, hypothesis: str = "") -> Arm:
        key = f"{name}:{regime}"
        if key not in self.arms:
            self.arms[key] = Arm(name=name, regime=regime, hypothesis=hypothesis)
        return self.arms[key]

    # -- staking --------------------------------------------------------------
    def place(
        self,
        *,
        arm: str,
        regime: str,
        match_date: str,
        home: str,
        away: str,
        market: str,
        selection: str,
        model_probability: float,
        line: float | None = None,
        market_probability: float | None = None,
        decimal_odds: float | None = None,
        stake: float | None = None,
        note: str = "",
    ) -> Bet | None:
        """Record a fake bet. Returns None when the arm declines to bet.

        An arm declining is a legitimate outcome and is never worked around by
        lowering a threshold - that measures the threshold, not the strategy.
        """
        target = self.ensure_arm(arm, regime)

        # Refuse a duplicate. A matchday run can happen more than once - a CI
        # retry, a manual dispatch, a local run before the scheduled one - and
        # without this each pass stakes the same fixture again, quietly inflating
        # every arm's exposure and corrupting the season's P&L.
        if self._already_held(target, match_date, home, away, market, selection, line):
            return None

        edge = None
        if market_probability is not None:
            edge = model_probability - market_probability

        if regime == EDGE_GATED:
            if edge is None or edge < EDGE_THRESHOLD:
                return None
            if decimal_odds is None or decimal_odds <= 1.0:
                return None
            # Fractional Kelly on the model's own probability.
            b = decimal_odds - 1.0
            kelly = max((model_probability * b - (1 - model_probability)) / b, 0.0)
            stake = round(target.bankroll * kelly * KELLY_FRACTION, 2)
            if stake < 1.0:
                return None
        elif stake is None:
            stake = FLAT_STAKE

        if stake > target.bankroll:
            return None                     # cannot stake what the arm does not have

        bet = Bet(
            bet_id=uuid.uuid4().hex[:12],
            arm=arm,
            regime=regime,
            match_date=match_date,
            home=home,
            away=away,
            market=market,
            selection=selection,
            line=line,
            stake=float(stake),
            model_probability=float(model_probability),
            market_probability=market_probability,
            decimal_odds=decimal_odds,
            edge=edge,
            placed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=note,
        )
        target.bets.append(bet)
        target.bankroll -= bet.stake
        return bet

    @staticmethod
    def _already_held(arm: Arm, match_date: str, home: str, away: str,
                      market: str, selection: str, line: float | None) -> bool:
        """Has this arm already staked this exact selection on this fixture?

        Voided bets do not count as held - a bet voided for a data error should
        be replaceable once the error is fixed.
        """
        for bet in arm.bets:
            if bet.status == "void":
                continue
            if (bet.match_date, bet.home, bet.away, bet.market, bet.selection,
                    bet.line) == (match_date, home, away, market, selection, line):
                return True
        return False

    # -- settlement -----------------------------------------------------------
    def settle_match(self, *, home: str, away: str, match_date: str,
                     home_goals: int, away_goals: int) -> int:
        """Settle every open bet on a finished match. Returns how many settled."""
        settled = 0
        for arm in self.arms.values():
            for bet in arm.open_bets:
                if (bet.home, bet.away, bet.match_date) != (home, away, match_date):
                    continue
                won = self._evaluate(bet, home_goals, away_goals)
                if won is None:
                    bet.status = "push"
                    arm.bankroll += bet.stake
                elif won:
                    bet.status = "won"
                    odds = bet.decimal_odds or (1.0 / max(bet.model_probability, 1e-6))
                    bet.payout = round(bet.stake * odds, 2)
                    arm.bankroll += bet.payout
                else:
                    bet.status = "lost"
                    bet.payout = 0.0
                bet.result = f"{home_goals}-{away_goals}"
                bet.settled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                settled += 1
        return settled

    @staticmethod
    def _evaluate(bet: Bet, home_goals: int, away_goals: int) -> bool | None:
        """True won, False lost, None push. Totals ALWAYS check the stored line."""
        if bet.market == "1x2":
            if home_goals > away_goals:
                actual = "home"
            elif home_goals == away_goals:
                actual = "draw"
            else:
                actual = "away"
            return bet.selection == actual

        if bet.market == "totals":
            if bet.line is None:
                # A totals bet without its line cannot be settled correctly. The
                # EPL lab booked a ~45-point phantom edge by comparing an Over 2.5
                # fair value against an Over 5.5 price; refusing here is the fix.
                return None
            total = home_goals + away_goals
            if total == bet.line:
                return None                 # whole-number line: push
            over = total > bet.line
            return over if bet.selection == "over" else not over

        if bet.market == "btts":
            both = home_goals > 0 and away_goals > 0
            return both if bet.selection == "yes" else not both

        return None

    def void_bet(self, bet_id: str, reason: str) -> bool:
        for arm in self.arms.values():
            for bet in arm.bets:
                if bet.bet_id == bet_id and bet.status == "open":
                    bet.status = "void"
                    bet.note = f"VOID: {reason}"
                    arm.bankroll += bet.stake
                    return True
        return False

    # -- persistence ----------------------------------------------------------
    def save(self) -> Path:
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arms": {
                key: {
                    "name": arm.name,
                    "regime": arm.regime,
                    "bankroll": round(arm.bankroll, 2),
                    "starting_bankroll": arm.starting_bankroll,
                    "hypothesis": arm.hypothesis,
                    "bets": [asdict(b) for b in arm.bets],
                }
                for key, arm in self.arms.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, path: Path | None = None) -> "Ledger":
        ledger = cls(path)
        if not ledger.path.exists():
            return ledger
        data = json.loads(ledger.path.read_text(encoding="utf-8"))
        for key, rec in data.get("arms", {}).items():
            arm = Arm(name=rec["name"], regime=rec["regime"],
                      bankroll=rec["bankroll"],
                      starting_bankroll=rec.get("starting_bankroll", STARTING_BANKROLL),
                      hypothesis=rec.get("hypothesis", ""))
            arm.bets = [Bet(**b) for b in rec.get("bets", [])]
            ledger.arms[key] = arm
        return ledger

    def summary(self) -> list[dict]:
        rows = []
        for arm in self.arms.values():
            won, lost, push = arm.record
            rows.append({
                "arm": arm.name,
                "regime": arm.regime,
                "bankroll": round(arm.bankroll, 2),
                "profit": round(arm.profit, 2),
                "roi": round(arm.roi, 4),
                "open": len(arm.open_bets),
                "won": won, "lost": lost, "push": push,
            })
        return sorted(rows, key=lambda r: (-r["profit"], r["arm"]))
