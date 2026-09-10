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
            lines.append(
                f"  model {consensus * 100:.0f}% \\| market {market_home * 100:.0f}% "
                f"\\| gap {gap * 100:+.0f}{flag}")
        else:
            lines.append(f"  model {consensus * 100:.0f}% \\| no market price")

        spread = [v["home"] for v in arms.values()]
        if len(spread) > 1:
            lines.append(f"  arms {min(spread) * 100:.0f}–{max(spread) * 100:.0f}%")
        lines.append("")

    flagged = [e for e in data["fixtures"] if e.get("market") and e["market"].get("home")
               and any(v for v in e["arms"].values())]
    if flagged:
        lines.append("_⚑ marks an 8\\+ point gap to the de\\-vigged line\\._")
    lines.append("_Paper only\\. Your call\\._")
    return "\n".join(lines)[:MAX_LENGTH]


def send(brief, *, token: str | None = None, chat_id: str | None = None) -> bool:
    """Send the brief. Returns False when unconfigured or the API refuses."""
    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        response = requests.post(
            API.format(token=token),
            json={"chat_id": chat_id, "text": format_brief(brief),
                  "parse_mode": "MarkdownV2", "disable_web_page_preview": True},
            timeout=20,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False
