"""Push a short matchday brief to Telegram.

The phone is a glance surface, not a study surface. It carries the headline for
each fixture and, crucially, where the arms disagree with the market - because
that is the only part that needs a decision. The dashboard carries the rest.

Silent when unconfigured: a missing token is a skipped notification, never a
failed matchday run.
"""
from __future__ import annotations

import os

import requests

API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LENGTH = 4000


def _escape(text: str) -> str:
    for char in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-",
                 "=", "|", "{", "}", ".", "!"):
        text = text.replace(char, f"\\{char}")
    return text


def format_brief(brief) -> str:
    data = brief.__dict__ if hasattr(brief, "__dict__") else brief
    lines = [f"*UCL brief — {_escape(str(data['as_of']))}*", ""]

    for entry in data["fixtures"]:
        market = entry.get("market") or {}
        arms = {k: v for k, v in entry["arms"].items() if v}
        if not arms:
            continue
        consensus = sum(v["home"] for v in arms.values()) / len(arms)
        market_home = market.get("home")

        header = (f"*{_escape(entry['home_display'])}* v "
                  f"*{_escape(entry['away_display'])}*")
        kickoff = _escape(entry.get("kickoff") or "")
        lines.append(f"{header}  `{kickoff}`")

        if market_home is not None:
            gap = consensus - market_home
            flag = " ⚑" if abs(gap) >= 0.08 else ""
            # MarkdownV2 treats '-' and '+' as reserved, so a signed number has
            # to be escaped or Telegram rejects the whole message with HTTP 400.
            gap_text = _escape(format(gap * 100, "+.0f"))
            lines.append(
                f"  model {consensus * 100:.0f}% \\| market {market_home * 100:.0f}% "
                f"\\| gap {gap_text}{flag}")
        elif entry.get("market_in_play_withheld"):
            lines.append(f"  model {consensus * 100:.0f}% \\| in\\-play, market withheld")
        else:
            lines.append(f"  model {consensus * 100:.0f}% \\| no market price")

        spread = [v["home"] for v in arms.values()]
        if len(spread) > 1:
            span = _escape(f"{min(spread) * 100:.0f}-{max(spread) * 100:.0f}")
            lines.append(f"  arms {span}%")
        lines.append("")

    flagged = [e for e in data["fixtures"] if e.get("market") and e["market"].get("home")
               and any(v for v in e["arms"].values())]
    if flagged:
        lines.append("_⚑ marks an 8\\+ point gap to the de\\-vigged line\\._")
    lines.append("_Paper only\\. Your call\\._")
    return "\n".join(lines)[:MAX_LENGTH]


def format_results(brief, results: dict, ledger=None) -> str:
    """Post-matchday message: what happened and how each model did.

    The headline number is the probability each arm gave to the result that
    actually occurred, which is the only per-match summary that is both honest
    and readable on a phone.
    """
    data = brief.__dict__ if hasattr(brief, "__dict__") else brief
    lines = ["*UCL results*", ""]

    scored: list[tuple[str, dict, str]] = []
    for entry in data.get("fixtures", []):
        key = (entry["home"], entry["away"], entry["date"])
        if key not in results:
            continue
        home_goals, away_goals = results[key]
        actual = ("home" if home_goals > away_goals
                  else "draw" if home_goals == away_goals else "away")
        scored.append((f"{home_goals}-{away_goals}", entry, actual))

    if not scored:
        return ""

    for score, entry, actual in scored:
        market = (entry.get("market") or {}).get(actual)
        arms = {k: v for k, v in entry["arms"].items() if v}
        consensus = (sum(v[actual] for v in arms.values()) / len(arms)) if arms else None
        line = (f"*{_escape(entry['home_display'])}* {_escape(score)} "
                f"*{_escape(entry['away_display'])}*")
        lines.append(line)
        detail = []
        if consensus is not None:
            detail.append(f"models {consensus * 100:.0f}%")
        if market is not None:
            detail.append(f"market {market * 100:.0f}%")
        if detail:
            lines.append("  " + _escape(" | ".join(detail)) + " on the actual result")
        lines.append("")

    if ledger is not None:
        rows = [r for r in ledger.summary() if r["regime"] == "science"]
        if rows:
            best = max(rows, key=lambda row: row["profit"])
            profit = format(best["profit"], "+,.0f")
            lines.append(f"_Best paper arm: {_escape(best['arm'])} "
                         f"{_escape(profit)}_")
    return "\n".join(lines)[:MAX_LENGTH]


def _post(text: str, token: str | None, chat_id: str | None) -> bool:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id or not text:
        return False
    try:
        response = requests.post(
            API.format(token=token),
            json={"chat_id": chat_id, "text": text,
                  "parse_mode": "MarkdownV2", "disable_web_page_preview": True},
            timeout=20,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def send(brief, *, token: str | None = None, chat_id: str | None = None) -> bool:
    """Send the pre-matchday brief. False when unconfigured or refused."""
    return _post(format_brief(brief), token, chat_id)


def send_results(brief, results: dict, ledger=None, *,
                 token: str | None = None, chat_id: str | None = None) -> bool:
    """Send the post-matchday results message."""
    return _post(format_results(brief, results, ledger), token, chat_id)
