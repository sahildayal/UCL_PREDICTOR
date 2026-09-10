"""Render a matchday brief as a self-contained HTML page.

The page is a working surface, not a report: Sahil reads it before kickoff and
makes his own call. So the information design leads with disagreement - where the
arms diverge from the de-vigged line, and from each other - because agreement
needs no thought and disagreement is the whole decision.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import REPORTS

ARM_LABELS = {
    "dixon_coles": "Dixon-Coles",
    "hier_bayes": "Bayesian",
    "elo_xleague": "Elo",
    "gbm": "LightGBM",
    "gnn": "Graph net",
    "sequence": "Sequence",
}


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _fixture_card(entry: dict) -> str:
    market = entry.get("market") or {}
    arms = entry.get("arms") or {}
    market_home = market.get("home")

    rows = []
    if market_home is not None:
        rows.append(_probability_row("Market (de-vigged)", market.get("home"),
                                     market.get("draw"), market.get("away"),
                                     kind="market", delta=None))
    for name, view in arms.items():
        if not view:
            rows.append(
                f'<div class="prow prow--absent"><span class="prow__name">'
                f'{html.escape(ARM_LABELS.get(name, name))}</span>'
                f'<span class="prow__absent">no price — club not in model</span></div>'
            )
            continue
        delta = None if market_home is None else view["home"] - market_home
        rows.append(_probability_row(ARM_LABELS.get(name, name), view["home"],
                                     view["draw"], view["away"],
                                     kind="arm", delta=delta))

    spread = [v["home"] for v in arms.values() if v]
    spread_text = ""
    if len(spread) > 1:
        width = (max(spread) - min(spread)) * 100
        tone = "wide" if width >= 15 else ("some" if width >= 7 else "tight")
        spread_text = (f'<span class="chip chip--{tone}">arm spread '
                       f'{width:.0f} pts</span>')

    consensus = arms.get("hier_bayes") or next((v for v in arms.values() if v), None)
    scoreline = ""
    if consensus and consensus.get("top_scores"):
        i, j, p = consensus["top_scores"][0]
        scoreline = (f'<span class="chip">most likely {int(i)}–{int(j)} '
                     f'({p * 100:.0f}%)</span>')

    goals = ""
    if consensus:
        goals = (f'<span class="chip">xG {consensus["home_xg"]:.2f} – '
                 f'{consensus["away_xg"]:.2f}</span>'
                 f'<span class="chip">O2.5 {_pct(consensus["over_2.5"])}</span>'
                 f'<span class="chip">BTTS {_pct(consensus["btts"])}</span>')

    books = market.get("bookmakers")
    market_note = (f'{books} books · {market.get("overround", 0) * 100:.1f}% vig'
                   if books else "no market price")

    return f"""
    <article class="fixture">
      <header class="fixture__head">
        <div class="fixture__teams">
          <span class="team team--home">{html.escape(entry['home_display'])}</span>
          <span class="fixture__v">v</span>
          <span class="team team--away">{html.escape(entry['away_display'])}</span>
        </div>
        <div class="fixture__meta">
          <span class="kick">{html.escape(entry.get('kickoff') or '')} · MD{entry.get('matchday', 0)}</span>
          <span class="books">{html.escape(market_note)}</span>
        </div>
      </header>
      <div class="chips">{scoreline}{goals}{spread_text}</div>
      <div class="prows">{''.join(rows)}</div>
    </article>"""


def _probability_row(label: str, home, draw, away, *, kind: str, delta) -> str:
    home = home or 0.0
    draw = draw or 0.0
    away = away or 0.0
    total = max(home + draw + away, 1e-9)
    h, d, a = home / total * 100, draw / total * 100, away / total * 100

    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else "−"
        tone = "up" if delta > 0.03 else ("down" if delta < -0.03 else "flat")
        delta_html = (f'<span class="delta delta--{tone}">{sign}'
                      f'{abs(delta) * 100:.0f}</span>')

    return f"""
      <div class="prow prow--{kind}">
        <span class="prow__name">{html.escape(label)}</span>
        <div class="bar" role="img" aria-label="home {h:.0f}%, draw {d:.0f}%, away {a:.0f}%">
          <span class="bar__seg bar__seg--home" style="width:{h:.2f}%"></span>
          <span class="bar__seg bar__seg--draw" style="width:{d:.2f}%"></span>
          <span class="bar__seg bar__seg--away" style="width:{a:.2f}%"></span>
        </div>
        <span class="prow__nums">
          <b>{h:.0f}</b><i>{d:.0f}</i><s>{a:.0f}</s>
        </span>
        {delta_html}
      </div>"""


def _qualification_table(rows: list[dict], display) -> str:
    if not rows:
        return '<p class="empty">No simulation available.</p>'
    body = []
    for position, row in enumerate(rows[:36], start=1):
        band = "top8" if position <= 8 else ("playoff" if position <= 24 else "out")
        body.append(f"""
        <tr class="q q--{band}">
          <td class="q__pos">{position}</td>
          <td class="q__club">{html.escape(display(row['club']))}</td>
          <td class="num">{row['mean_points']:.1f}</td>
          <td class="num">{row['top8'] * 100:.0f}%</td>
          <td class="num">{row['playoff'] * 100:.0f}%</td>
          <td class="num q__out">{row['eliminated'] * 100:.0f}%</td>
        </tr>""")
    return f"""
    <div class="scroller">
      <table class="qtable">
        <thead><tr>
          <th>#</th><th>Club</th><th class="num">Pts</th>
          <th class="num">Top 8</th><th class="num">Playoff</th><th class="num">Out</th>
        </tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>"""


def _ledger_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No bets recorded yet.</p>'
    body = []
    for row in rows:
        profit = row["profit"]
        tone = "up" if profit > 0 else ("down" if profit < 0 else "flat")
        body.append(f"""
        <tr>
          <td>{html.escape(ARM_LABELS.get(row['arm'], row['arm']))}</td>
          <td><span class="regime regime--{row['regime']}">{html.escape(row['regime'].replace('_', ' '))}</span></td>
          <td class="num">{row['bankroll']:,.0f}</td>
          <td class="num delta--{tone}">{profit:+,.0f}</td>
          <td class="num">{row['won']}–{row['lost']}</td>
          <td class="num">{row['open']}</td>
        </tr>""")
    return f"""
    <div class="scroller">
      <table class="qtable">
        <thead><tr><th>Arm</th><th>Regime</th><th class="num">Bankroll</th>
        <th class="num">P&amp;L</th><th class="num">W–L</th><th class="num">Open</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>"""


CSS = """
:root{
  --ground:#f4f5f9; --panel:#ffffff; --panel-2:#eceef6;
  --ink:#141830; --ink-soft:#4a5070; --ink-faint:#767ea0;
  --line:#dcdfeb;
  --accent:#3a5fd9; --accent-soft:#dbe2fb;
  --home:#3a5fd9; --draw:#9aa2c4; --away:#c2683f;
  --up:#1f8a5c; --down:#c0405c; --flat:#767ea0;
  --shadow:0 1px 2px rgba(20,24,48,.06),0 8px 24px rgba(20,24,48,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0c0f1e; --panel:#141930; --panel-2:#1b2140;
    --ink:#e8eaf4; --ink-soft:#a8b0cf; --ink-faint:#767ea0;
    --line:#252c4c;
    --accent:#7d9bff; --accent-soft:#232b52;
    --home:#7d9bff; --draw:#6a7299; --away:#e08b5e;
    --up:#3fc98a; --down:#f0708a; --flat:#767ea0;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0c0f1e; --panel:#141930; --panel-2:#1b2140;
  --ink:#e8eaf4; --ink-soft:#a8b0cf; --ink-faint:#767ea0;
  --line:#252c4c;
  --accent:#7d9bff; --accent-soft:#232b52;
  --home:#7d9bff; --draw:#6a7299; --away:#e08b5e;
  --up:#3fc98a; --down:#f0708a; --flat:#767ea0;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Source Sans 3",ui-sans-serif,system-ui,-apple-system,sans-serif;
  font-size:15px; line-height:1.5;
}
.wrap{max-width:1080px; margin:0 auto; padding-inline:20px; padding-block:36px 64px}
.masthead{display:flex; flex-wrap:wrap; gap:16px; align-items:baseline;
  justify-content:space-between; border-bottom:2px solid var(--ink);
  padding-bottom:14px; margin-bottom:8px}
h1{font-family:"Bricolage Grotesque",Georgia,serif; font-weight:700;
  font-size:clamp(26px,4.4vw,40px); letter-spacing:-.02em; margin:0; text-wrap:balance}
.masthead__sub{color:var(--ink-soft); font-size:14px;
  font-variant-numeric:tabular-nums}
.eyebrow{font-size:11px; text-transform:uppercase; letter-spacing:.14em;
  color:var(--ink-faint); font-weight:600}
h2{font-family:"Bricolage Grotesque",Georgia,serif; font-size:21px; margin:0;
  letter-spacing:-.01em}
section{margin-top:40px}
.section__head{display:flex; align-items:baseline; gap:12px; margin-bottom:16px}
.section__note{color:var(--ink-faint); font-size:13px}

.fixtures{display:flex; flex-direction:column; gap:14px}
.fixture{background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; box-shadow:var(--shadow)}
.fixture__head{display:flex; flex-wrap:wrap; gap:8px 16px;
  justify-content:space-between; align-items:baseline}
.fixture__teams{display:flex; flex-wrap:wrap; align-items:baseline; gap:9px;
  font-family:"Bricolage Grotesque",Georgia,serif; font-size:19px; font-weight:600}
.fixture__v{color:var(--ink-faint); font-size:13px; font-weight:400;
  font-family:"Source Sans 3",sans-serif}
.fixture__meta{display:flex; gap:12px; flex-wrap:wrap;
  font-family:"JetBrains Mono",ui-monospace,monospace; font-size:11.5px;
  color:var(--ink-faint)}
.chips{display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 14px}
.chip{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:11px;
  padding:3px 8px; border-radius:5px; background:var(--panel-2);
  color:var(--ink-soft); white-space:nowrap}
.chip--wide{background:var(--down); color:#fff}
.chip--some{background:var(--accent-soft); color:var(--accent)}

.prows{display:flex; flex-direction:column; gap:7px}
.prow{display:grid; grid-template-columns:112px 1fr 74px 34px; gap:10px;
  align-items:center}
.prow--market .prow__name{font-weight:700; color:var(--ink)}
.prow__name{font-size:12.5px; color:var(--ink-soft); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis}
.prow--absent{grid-template-columns:112px 1fr}
.prow__absent{font-size:12px; color:var(--ink-faint); font-style:italic}
.bar{display:flex; height:15px; border-radius:3px; overflow:hidden;
  background:var(--panel-2)}
.prow--market .bar{height:19px}
.bar__seg--home{background:var(--home)}
.bar__seg--draw{background:var(--draw)}
.bar__seg--away{background:var(--away)}
.prow__nums{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:11.5px;
  font-variant-numeric:tabular-nums; display:flex; gap:7px}
.prow__nums b{color:var(--home)} .prow__nums i{color:var(--draw); font-style:normal}
.prow__nums s{color:var(--away); text-decoration:none}
.delta{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:11.5px;
  text-align:right; font-variant-numeric:tabular-nums}
.delta--up{color:var(--up)} .delta--down{color:var(--down)} .delta--flat{color:var(--flat)}

.scroller{overflow-x:auto; border:1px solid var(--line); border-radius:10px;
  background:var(--panel)}
.qtable{border-collapse:collapse; width:100%; font-size:13.5px}
.qtable th{text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.08em; color:var(--ink-faint); font-weight:600;
  padding:11px 12px; border-bottom:1px solid var(--line); white-space:nowrap}
.qtable td{padding:8px 12px; border-bottom:1px solid var(--line); white-space:nowrap}
.qtable tbody tr:last-child td{border-bottom:none}
.num{text-align:right; font-family:"JetBrains Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
.q__pos{color:var(--ink-faint); font-family:"JetBrains Mono",monospace; width:34px}
.q--top8 .q__pos{color:var(--up); font-weight:700}
.q--out .q__club{color:var(--ink-faint)}
.q__out{color:var(--ink-faint)}
.q--top8{background:linear-gradient(90deg,color-mix(in srgb,var(--up) 9%,transparent),transparent 40%)}
.regime{font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
  padding:2px 7px; border-radius:4px; background:var(--panel-2); color:var(--ink-soft)}
.regime--science{background:var(--accent-soft); color:var(--accent)}

.notes{background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:8px; padding:14px 16px; font-size:13px; color:var(--ink-soft)}
.notes ul{margin:8px 0 0; padding-left:18px} .notes li{margin:3px 0}
.disclaimer{margin-top:44px; padding-top:16px; border-top:1px solid var(--line);
  font-size:12.5px; color:var(--ink-faint)}
.empty{color:var(--ink-faint); font-style:italic}
@media (max-width:560px){
  .prow{grid-template-columns:88px 1fr 62px 30px; gap:7px}
  .prow__name{font-size:11.5px}
}
"""


def render(brief) -> str:
    data = brief.__dict__ if hasattr(brief, "__dict__") else brief
    fixtures = data["fixtures"]
    display_map = {f["home"]: f["home_display"] for f in fixtures}
    display_map.update({f["away"]: f["away_display"] for f in fixtures})

    def display(key: str) -> str:
        return display_map.get(key, key.replace("-", " ").title())

    cards = "".join(_fixture_card(f) for f in fixtures) or \
        '<p class="empty">No fixtures inside the horizon.</p>'
    notes = "".join(f"<li>{html.escape(str(n))}</li>" for n in data.get("notes", []))
    market_state = "live" if data.get("market_available") else "unavailable"

    return f"""<title>Champions League Forecast Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header class="masthead">
    <div>
      <div class="eyebrow">2026/27 League Phase · Matchday brief</div>
      <h1>Champions League Forecast Lab</h1>
    </div>
    <div class="masthead__sub">
      as of {html.escape(str(data.get('as_of')))}<br>
      market {market_state} · {len(data.get('arm_summary', []))} arms
    </div>
  </header>

  <section>
    <div class="section__head">
      <h2>Fixtures</h2>
      <span class="section__note">Each arm's home/draw/away, against the de-vigged
      line. The right-hand number is the arm's gap to the market in points.</span>
    </div>
    <div class="fixtures">{cards}</div>
  </section>

  <section>
    <div class="section__head">
      <h2>Qualification</h2>
      <span class="section__note">Monte Carlo over the remaining league phase.
      Top 8 go straight to the round of 16; 9–24 play a knockout playoff.</span>
    </div>
    {_qualification_table(data.get('qualification', []), display)}
  </section>

  <section>
    <div class="section__head">
      <h2>Paper ledger</h2>
      <span class="section__note">Fake money. Science arms stake every fixture;
      edge-gated arms stake only on material disagreement.</span>
    </div>
    {_ledger_table(data.get('ledger_summary', []))}
  </section>

  <section>
    <div class="section__head"><h2>Run notes</h2></div>
    <div class="notes"><ul>{notes or '<li>Clean run.</li>'}</ul></div>
  </section>

  <p class="disclaimer">
    No real money and no order placement exists in this system. Prices are read
    only, used to settle fake bets and to score each arm against the closing line.
    Forecasts are decision support for a human, not instructions.
  </p>
</div>"""


def write_dashboard(brief, path: Path | None = None) -> Path:
    path = path or REPORTS / "matchday.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(brief), encoding="utf-8")
    return path
