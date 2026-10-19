"""Cumulative season scoreboard, rebuilt from the archived briefs.

Each matchday's brief is committed before kickoff, so the season record is
reconstructed from what was actually claimed at the time rather than from
anything refitted later. That is the difference between a track record and a
backtest, and it only holds because the briefs are timestamped and immutable in
git history.

Arms are scored against outcomes and against the de-vigged closing line. An arm
that beats the line over a season would be a genuinely surprising result; the
EPL lab found nothing that did over sixteen seasons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PROCESSED
from . import metrics


def load_briefs(directory: Path | None = None) -> list[dict]:
    """Every archived brief, oldest first, excluding the rolling 'latest' copy."""
    directory = directory or PROCESSED
    briefs = []
    for path in sorted(directory.glob("brief_*.json")):
        if path.name == "brief_latest.json":
            continue
        try:
            briefs.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return briefs


def collect_scored_fixtures(briefs: list[dict], results: dict) -> list[dict]:
    """Join archived predictions to final scores.

    `results` maps (home, away, date) -> (home_goals, away_goals).
    A fixture predicted twice (a re-run) contributes once: the earliest brief
    wins, because that is the prediction that was genuinely made first.

    An entry flagged `predicted_after_kickoff` is skipped. A forecast can only
    be scored if it was provably made before the match started (DESIGN.md
    section 9). An earlier brief that did predict the fixture in time still
    counts.
    """
    seen: set[tuple] = set()
    scored = []
    for brief in briefs:
        for entry in brief.get("fixtures", []):
            key = (entry["home"], entry["away"], entry["date"])
            if key in seen or key not in results:
                continue
            if entry.get("predicted_after_kickoff"):
                continue
            seen.add(key)
            home_goals, away_goals = results[key]
            scored.append({
                "key": key,
                "entry": entry,
                "outcome": int(metrics.outcome_index([home_goals], [away_goals])[0]),
                "score": f"{home_goals}-{away_goals}",
                "date": entry["date"],
            })
    return scored


def scoreboard(scored: list[dict]) -> list[dict]:
    """Per-arm cumulative metrics, plus the market as a reference row."""
    if not scored:
        return []

    outcomes = np.array([row["outcome"] for row in scored])

    arm_names: list[str] = []
    for row in scored:
        for name, view in row["entry"].get("arms", {}).items():
            if view and name not in arm_names:
                arm_names.append(name)

    rows: list[dict] = []

    def probabilities_for(getter) -> tuple[np.ndarray, np.ndarray]:
        values, mask = [], []
        for row in scored:
            view = getter(row)
            if view and view.get("home") is not None:
                values.append([view["home"], view["draw"], view["away"]])
                mask.append(True)
            else:
                mask.append(False)
        return np.array(values, dtype=float), np.array(mask, dtype=bool)

    market_probs, market_mask = probabilities_for(lambda r: r["entry"].get("market"))
    if market_mask.any():
        summary = metrics.summarise(market_probs, outcomes[market_mask])
        rows.append({"arm": "market", "is_market": True, **summary,
                     "edge_vs_market": 0.0})

    for name in arm_names:
        probs, mask = probabilities_for(lambda r, n=name: r["entry"]["arms"].get(n))
        if not mask.any():
            continue
        summary = metrics.summarise(probs, outcomes[mask])
        entry = {"arm": name, "is_market": False, **summary}
        # Compare only on fixtures where BOTH the arm and the market priced,
        # otherwise the two log-losses are computed over different fixtures and
        # the difference means nothing.
        both = mask & market_mask if market_mask.any() else None
        if both is not None and both.any():
            arm_subset = np.array(
                [[r["entry"]["arms"][name]["home"], r["entry"]["arms"][name]["draw"],
                  r["entry"]["arms"][name]["away"]]
                 for r, keep in zip(scored, both) if keep], dtype=float)
            market_subset = np.array(
                [[r["entry"]["market"]["home"], r["entry"]["market"]["draw"],
                  r["entry"]["market"]["away"]]
                 for r, keep in zip(scored, both) if keep], dtype=float)
            comparison = metrics.compare_to_market(arm_subset, market_subset,
                                                   outcomes[both])
            entry["edge_vs_market"] = comparison["edge_vs_market"]
            entry["compared_on"] = int(both.sum())
        else:
            entry["edge_vs_market"] = None
            entry["compared_on"] = 0
        rows.append(entry)

    market_rows = [r for r in rows if r["is_market"]]
    arm_rows = sorted((r for r in rows if not r["is_market"]),
                      key=lambda r: r["log_loss"])
    return market_rows + arm_rows


def announce_key(entry: dict) -> str:
    """Stable id for a fixture in the announced-results record."""
    return f"{entry['home']}|{entry['away']}|{entry['date']}"


def unannounced(briefs: list[dict], played: dict, announced: set[str]) -> list[dict]:
    """Finished fixtures not yet sent to Telegram, each with its forecast.

    The forecast shown is the earliest on-time one across every archived brief,
    falling back to a late one (which the message marks as not scored). Reading
    brief_latest instead silently sent nothing: by settle time a later, empty
    between-matchday brief had usually replaced it, so no results message went
    out for matchday 1 or 2.
    """
    chosen: dict[tuple, dict] = {}
    for brief in briefs:
        for entry in brief.get("fixtures", []):
            key = (entry["home"], entry["away"], entry["date"])
            if key not in played or announce_key(entry) in announced:
                continue
            held = chosen.get(key)
            if held is None or (held.get("predicted_after_kickoff")
                                and not entry.get("predicted_after_kickoff")):
                chosen[key] = entry
    return sorted(chosen.values(),
                  key=lambda e: (e["date"], e.get("kickoff_utc") or "", e["home"]))


def build(results: dict, directory: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Return (scoreboard rows, scored fixtures)."""
    briefs = load_briefs(directory)
    scored = collect_scored_fixtures(briefs, results)
    return scoreboard(scored), scored
