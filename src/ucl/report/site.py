"""Build the GitHub Pages site: landing page plus a permanent page per matchday.

The archive is the point. A forecast is only worth anything if you can go back in
May and see exactly what each arm said before matchday 3 and how it turned out,
without taking anyone's word for it. Each matchday page is written once, from the
brief that was committed before kickoff, and never rewritten afterwards except to
attach the results.

Shares its design language with `dashboard.py`, which owns the CSS.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import ROOT
from . import dashboard

SITE_DIR = ROOT / "docs"

EXTRA_CSS = """
.hero{display:grid; gap:6px; margin-bottom:8px}
.lede{color:var(--ink-soft); max-width:62ch; font-size:15.5px; margin:10px 0 0}
.grid{display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.stat{background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px}
.stat__k{font-size:11px; text-transform:uppercase; letter-spacing:.1em;
  color:var(--ink-faint); font-weight:600}
.stat__v{font-family:"Bricolage Grotesque",Georgia,serif; font-size:27px;
  font-weight:700; font-variant-numeric:tabular-nums; margin-top:3px}
.stat__n{font-size:12px; color:var(--ink-faint); margin-top:2px}
.archive{display:flex; flex-direction:column; gap:0;
  border:1px solid var(--line); border-radius:10px; background:var(--panel);
  overflow:hidden}
.archive a{display:flex; flex-wrap:wrap; gap:6px 14px; align-items:baseline;
  padding:12px 16px; text-decoration:none; color:inherit;
  border-bottom:1px solid var(--line)}
.archive a:last-child{border-bottom:none}
.archive a:hover{background:var(--panel-2)}
.archive a:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}
.archive__md{font-family:"Bricolage Grotesque",Georgia,serif; font-weight:700;
  font-size:15px; min-width:44px}
.archive__date{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:12px;
  color:var(--ink-faint)}
.archive__n{font-size:12.5px; color:var(--ink-soft); margin-left:auto}
.market-row td{font-weight:700; background:var(--panel-2)}
.backlink{display:inline-block; margin-bottom:20px; font-size:13.5px;
  color:var(--accent); text-decoration:none}
.backlink:hover{text-decoration:underline}
.pending{color:var(--ink-faint); font-style:italic; font-size:13px}
.res{font-family:"JetBrains Mono",ui-monospace,monospace; font-weight:500}
"""

ARM_LABELS = dashboard.ARM_LABELS


def _head(title: str, description: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{dashboard.CSS}{EXTRA_CSS}</style>
</head><body>"""


def _scoreboard_table(rows: list[dict]) -> str:
    if not rows:
        return ('<p class="pending">No settled fixtures yet. The scoreboard '
                'fills in from the first settlement run.</p>')
    body = []
    for row in rows:
        is_market = row.get("is_market")
        label = "Market (de-vigged)" if is_market else ARM_LABELS.get(row["arm"], row["arm"])
        edge = row.get("edge_vs_market")
        if is_market:
            edge_cell = '<td class="num">—</td>'
        elif edge is None:
            edge_cell = '<td class="num">—</td>'
        else:
            tone = "up" if edge > 0 else ("down" if edge < 0 else "flat")
            edge_cell = f'<td class="num delta--{tone}">{edge:+.4f}</td>'
        draw = row.get("draw_recall")
        draw_cell = "—" if draw is None or draw != draw else f"{draw * 100:.0f}%"
        body.append(f"""
        <tr class="{'market-row' if is_market else ''}">
          <td>{html.escape(label)}</td>
          <td class="num">{row['n']}</td>
          <td class="num">{row['log_loss']:.4f}</td>
          <td class="num">{row['rps']:.4f}</td>
          <td class="num">{row['accuracy'] * 100:.0f}%</td>
          <td class="num">{draw_cell}</td>
          {edge_cell}
        </tr>""")
    return f"""
    <div class="scroller">
      <table class="qtable">
        <thead><tr>
          <th>Arm</th><th class="num">N</th><th class="num">Log loss</th>
          <th class="num">RPS</th><th class="num">Acc</th>
          <th class="num">Draw rec.</th><th class="num">vs market</th>
        </tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>"""


def _archive_list(entries: list[dict]) -> str:
    if not entries:
        return '<p class="pending">No archived matchdays yet.</p>'
    items = []
    for entry in entries:
        settled = entry.get("settled", 0)
        total = entry.get("fixtures", 0)
        state = (f"{settled}/{total} settled" if settled
                 else f"{total} fixtures · pending")
        items.append(f"""
        <a href="{html.escape(entry['file'])}">
          <span class="archive__md">MD{entry.get('matchday') or '?'}</span>
          <span class="archive__date">{html.escape(entry['date'])}</span>
          <span class="archive__n">{html.escape(state)}</span>
        </a>""")
    return f'<div class="archive">{"".join(items)}</div>'


def _matchday_of(brief: dict) -> int | None:
    for entry in brief.get("fixtures", []):
        if entry.get("matchday"):
            return int(entry["matchday"])
    return None


def render_landing(briefs: list[dict], board: list[dict],
                   archive: list[dict], latest: dict | None) -> str:
    total_fixtures = sum(len(b.get("fixtures", [])) for b in briefs)
    settled = sum(a.get("settled", 0) for a in archive)
    best = next((r for r in board if not r.get("is_market")), None)
    market = next((r for r in board if r.get("is_market")), None)

    beating = "—"
    if best and market and best.get("log_loss") is not None:
        beating = "yes" if best["log_loss"] < market["log_loss"] else "no"

    stats = f"""
    <div class="grid">
      <div class="stat"><div class="stat__k">Matchdays archived</div>
        <div class="stat__v">{len(archive)}</div>
        <div class="stat__n">{total_fixtures} fixtures forecast</div></div>
      <div class="stat"><div class="stat__k">Fixtures settled</div>
        <div class="stat__v">{settled}</div>
        <div class="stat__n">scored against the closing line</div></div>
      <div class="stat"><div class="stat__k">Leading arm</div>
        <div class="stat__v">{html.escape(ARM_LABELS.get(best['arm'], best['arm'])) if best else '—'}</div>
        <div class="stat__n">{f"log loss {best['log_loss']:.4f}" if best else 'awaiting results'}</div></div>
      <div class="stat"><div class="stat__k">Beating the market?</div>
        <div class="stat__v">{beating}</div>
        <div class="stat__n">the only question that matters</div></div>
    </div>"""

    latest_link = ""
    if latest:
        matchday = _matchday_of(latest)
        latest_link = (f'<p class="lede"><a class="backlink" '
                       f'href="matchday-{html.escape(str(latest["as_of"]))}.html">'
                       f'Latest brief — matchday {matchday or "?"}, '
                       f'{html.escape(str(latest["as_of"]))} →</a></p>')

    return _head(
        "Champions League Forecast Lab",
        "Six machine-learning models forecasting the 2026/27 Champions League "
        "league phase, scored against the closing line."
    ) + f"""
<div class="wrap">
  <header class="masthead">
    <div>
      <div class="eyebrow">2026/27 League Phase</div>
      <h1>Champions League Forecast Lab</h1>
    </div>
  </header>

  <p class="lede">
    Six models forecast every league-phase fixture. None of them ever sees a
    betting price: odds are used only to settle fake bets and to score each model
    against the de-vigged closing line. Every brief is committed before kickoff,
    so the record below is what was actually claimed at the time, not a backtest.
  </p>
  {latest_link}

  <section>{stats}</section>

  <section>
    <div class="section__head">
      <h2>Season scoreboard</h2>
      <span class="section__note">Lower log loss is better. The
      <em>vs market</em> column is the market's log loss minus the arm's:
      positive means the arm beat the closing line.</span>
    </div>
    {_scoreboard_table(board)}
  </section>

  <section>
    <div class="section__head">
      <h2>Archive</h2>
      <span class="section__note">Every matchday as it was forecast, before
      kickoff.</span>
    </div>
    {_archive_list(archive)}
  </section>

  <p class="disclaimer">
    No real money and no order placement exists in this system. Prices are read
    only. Forecasts are decision support for a human, not instructions.
    Source: <a href="https://github.com/sahildayal/UCL_PREDICTOR">github.com/sahildayal/UCL_PREDICTOR</a>
  </p>
</div>
</body></html>"""


def render_matchday(brief: dict, results: dict) -> str:
    """A matchday page: the brief as forecast, with results attached if known."""
    matchday = _matchday_of(brief)
    fragment = dashboard.render(brief)
    # dashboard.render emits a bare <title> + content; strip its title so the
    # archive page can carry its own document head.
    fragment = fragment.split("</style>", 1)[-1]

    result_rows = []
    for entry in brief.get("fixtures", []):
        key = (entry["home"], entry["away"], entry["date"])
        if key not in results:
            continue
        home_goals, away_goals = results[key]
        actual = ("home" if home_goals > away_goals
                  else "draw" if home_goals == away_goals else "away")
        cells = []
        market = entry.get("market") or {}
        if market.get(actual) is not None:
            cells.append(f'<td class="num">{market[actual] * 100:.0f}%</td>')
        else:
            cells.append('<td class="num">—</td>')
        for name in ARM_LABELS:
            view = entry["arms"].get(name)
            cells.append(f'<td class="num">{view[actual] * 100:.0f}%</td>'
                         if view else '<td class="num">—</td>')
        result_rows.append(f"""
        <tr>
          <td>{html.escape(entry['home_display'])} v {html.escape(entry['away_display'])}</td>
          <td class="res">{home_goals}–{away_goals}</td>
          {''.join(cells)}
        </tr>""")

    results_section = ""
    if result_rows:
        headers = "".join(f'<th class="num">{html.escape(ARM_LABELS[n])}</th>'
                          for n in ARM_LABELS)
        results_section = f"""
  <section>
    <div class="section__head">
      <h2>How it turned out</h2>
      <span class="section__note">Probability each model assigned to the result
      that actually happened. Higher is better.</span>
    </div>
    <div class="scroller">
      <table class="qtable">
        <thead><tr><th>Fixture</th><th>Result</th>
        <th class="num">Market</th>{headers}</tr></thead>
        <tbody>{''.join(result_rows)}</tbody>
      </table>
    </div>
  </section>"""

    head = _head(
        f"Matchday {matchday or '?'} — {brief['as_of']}",
        f"Champions League matchday {matchday or ''} forecasts, {brief['as_of']}."
    )
    return (head + '<div class="wrap"><a class="backlink" href="index.html">'
            '← Season scoreboard</a></div>'
            + fragment + results_section + "</body></html>")


def build_site(results: dict | None = None, out_dir: Path | None = None) -> Path:
    """Regenerate the whole site from archived briefs. Idempotent."""
    from ..eval import season

    out_dir = out_dir or SITE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results = results or {}

    briefs = season.load_briefs()
    board, _scored = season.build(results)

    archive: list[dict] = []
    for brief in briefs:
        as_of = str(brief.get("as_of"))
        filename = f"matchday-{as_of}.html"
        settled = sum(1 for e in brief.get("fixtures", [])
                      if (e["home"], e["away"], e["date"]) in results)
        (out_dir / filename).write_text(render_matchday(brief, results),
                                        encoding="utf-8")
        archive.append({
            "file": filename, "date": as_of, "matchday": _matchday_of(brief),
            "fixtures": len(brief.get("fixtures", [])), "settled": settled,
        })

    archive.sort(key=lambda a: a["date"], reverse=True)
    latest = briefs[-1] if briefs else None
    (out_dir / "index.html").write_text(
        render_landing(briefs, board, archive, latest), encoding="utf-8")

    # Tell GitHub Pages not to run the files through Jekyll, which would
    # otherwise ignore anything it considers a special filename.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out_dir
