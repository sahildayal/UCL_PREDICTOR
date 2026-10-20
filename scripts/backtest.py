"""Walk-forward backtest of arms on past league phases.

For every matchday of the chosen seasons, the corpus is rewound to that
morning: later results are hidden and recency weights are recomputed from that
date, so no arm sees a score it is about to predict. Every arm is refit, then
scored on the matchday's fixtures. This is the bar a new arm version has to
clear before it joins the live roster.

    python scripts/backtest.py                       # elo/sequence v1 vs v2
    python scripts/backtest.py --arms dixon_coles gbm --seasons 2025-26
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ucl.config import force_utf8  # noqa: E402
from ucl.data import corpus as corpus_module  # noqa: E402
from ucl.eval import metrics  # noqa: E402
from ucl.models.dixon_coles import DixonColesXLeague  # noqa: E402
from ucl.models.elo_xleague import CrossLeagueElo, LeagueOffsetElo  # noqa: E402
from ucl.models.gbm import GradientBoostedGoals  # noqa: E402
from ucl.models.sequence import SequenceForm, SequenceResidual  # noqa: E402

force_utf8()

DEFAULT_ARMS = ["dixon_coles", "elo_xleague", "elo_xleague_v2", "sequence", "sequence_v2"]


def make_arms(names: list[str]):
    """Fresh arms per matchday. The backbone is always fitted first."""
    backbone = DixonColesXLeague()
    factories = {
        "elo_xleague": lambda: CrossLeagueElo(),
        "elo_xleague_v2": lambda: LeagueOffsetElo(),
        "sequence": lambda: SequenceForm(epochs=25),
        "sequence_v2": lambda: SequenceResidual(backbone=backbone, epochs=25),
        "gbm": lambda: GradientBoostedGoals(backbone=backbone),
    }
    arms = [backbone] + [factories[n]() for n in names if n != "dixon_coles"]
    return arms, ("dixon_coles" in names)


def rewind(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The corpus as it stood on the morning of `as_of`."""
    view = frame.copy()
    future = view["date"] >= as_of
    view.loc[future, ["home_goals", "away_goals"]] = np.nan
    view.loc[future, "played"] = False
    view["days_ago"] = (as_of - view["date"]).dt.days
    half_life = corpus_module.DEFAULT_HALF_LIFE_DAYS
    view["weight"] = view["days_ago"].clip(lower=0).map(
        lambda d: math.exp(-math.log(2) * d / half_life))
    view.loc[view["days_ago"] < 0, "weight"] = 0.0
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    args = parser.parse_args()

    frame, _ = corpus_module.load()
    league = frame[frame["is_ucl"] & frame["season"].isin(args.seasons)
                   & frame["stage"].str.startswith("League") & frame["played"]]
    if league.empty:
        print("no league-phase matches for those seasons")
        return 1

    probabilities: dict[str, list] = {}
    outcomes: dict[str, list] = {}
    for (season, stage), fixtures in league.groupby(["season", "stage"], sort=False):
        as_of = fixtures["date"].min()
        view = rewind(frame, as_of)
        arms, keep_backbone = make_arms(args.arms)
        started = time.time()
        fitted = []
        for arm in arms:
            try:
                arm.fit(view, as_of=as_of.date())
                fitted.append(arm)
            except Exception as exc:                  # noqa: BLE001
                print(f"  {arm.name} failed: {type(exc).__name__}: {exc}")
        for arm in fitted:
            if arm.name == "dixon_coles" and not keep_backbone:
                continue
            for row in fixtures.itertuples(index=False):
                prediction = arm.predict(row.home, row.away, when=row.date.date())
                if prediction is None:
                    continue
                view_1x2 = prediction.markets()
                probabilities.setdefault(arm.name, []).append(
                    [view_1x2["home"], view_1x2["draw"], view_1x2["away"]])
                outcomes.setdefault(arm.name, []).append(
                    int(metrics.outcome_index([row.home_goals], [row.away_goals])[0]))
        print(f"{season} {stage}: {len(fixtures)} fixtures, fitted in "
              f"{time.time() - started:.0f}s", flush=True)

    print(f"\n{'arm':<16}{'n':>5}{'log_loss':>10}{'rps':>8}{'brier':>8}")
    for name in args.arms:
        if name not in probabilities:
            continue
        p = np.array(probabilities[name])
        y = np.array(outcomes[name])
        print(f"{name:<16}{len(y):>5}{metrics.log_loss(p, y):>10.4f}"
              f"{metrics.ranked_probability_score(p, y):>8.4f}"
              f"{metrics.brier_score(p, y):>8.4f}")
    print(f"{'uniform':<16}{'':>5}{math.log(3):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
