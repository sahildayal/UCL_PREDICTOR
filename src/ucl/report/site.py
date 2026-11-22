"""Build the GitHub Pages site from the committed briefs, results and ledger.

The archive is the point. A forecast is only worth anything if you can go back
in May and see exactly what each model said before a matchday and how it turned
out. Every page is a pure function of what is committed: the briefs (written
before kickoff), Wikipedia results and the paper ledger. Rebuilding twice gives
the same site.

Pages: Overview, Matchdays (+ one page per matchday), Models, Bankrolls,
Table, How it works. Chrome and CSS live in `theme.py`.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import ROOT
from . import dashboard
from .theme import band, esc, foot, head, nav

SITE_DIR = ROOT / "docs"

ARM_LABELS = dashboard.ARM_LABELS
ARMS = list(ARM_LABELS)
OUTCOMES = ("home", "draw", "away")

# label, nickname, plain-English description, optional tag
PROFILES = {
    "dixon_coles": ("Dixon–Coles", "The veteran scout",
                    "The classic football model. Every club gets an attack and a "
                    "defence rating, every league gets a strength, and goals follow a "
                    "Poisson curve with a fix for low scores. The backbone two of the "
                    "other models build on.", ""),
    "hier_bayes": ("Bayesian", "The professor",
                   "The same idea done fully Bayesian: club ratings sit inside league "
                   "ratings, so a club with little European history borrows strength "
                   "from its league until its own results say otherwise.", ""),
    "elo_xleague": ("Elo", "The chess master",
                    "One number per club, nudged up or down after every match, like "
                    "chess ratings. It is the control: if Elo keeps up, the fancier "
                    "models aren't earning their keep.", ""),
    "elo_xleague_v2": ("Elo v2", "The chess master, upgraded",
                       "Elo again, but every country's league gets its own offset that "
                       "only moves when clubs from different countries meet. It beat "
                       "the original on a replay of last season before joining.",
                       "Joined at matchday 3"),
    "gbm": ("LightGBM", "The tree surgeon",
            "Starts from the Dixon–Coles forecast and lets a forest of decision trees "
            "correct what it tends to get wrong, using form, rest days and home "
            "advantage.", ""),
    "gnn": ("Graph net", "The networker",
            "Treats every club as a point in a web of who has played whom, and passes "
            "strength along the links. Built for clubs that never meet each other at "
            "home.", ""),
    "sequence": ("Sequence", "The film analyst",
                 "A neural network that reads each club's recent matches as a sequence. "
                 "It only knows form, not who a club is, which is how it once gave "
                 "Sabah a real chance against Manchester United.", ""),
    "sequence_v2": ("Sequence v2", "The film analyst, upgraded",
                    "The same network, now starting from the Dixon–Coles rating and "
                    "learning only a small correction from how teams have done against "
                    "expectation lately.", "Joined at matchday 3"),
}
MARKET_PROFILE = ("The market", "The bookmaker",
                  "Not a model: the bookmakers' prices with their margin removed, "
                  "averaged across many books. Thousands of people with money on the "
                  "line. The one everyone here is trying to beat.")

CODES = {
    "aek athens": "AEK", "arsenal": "ARS", "aston villa": "AVL", "atletico madrid": "ATM",
    "barcelona": "BAR", "bayern munich": "BAY", "bodo/glimt": "BOD",
    "borussia dortmund": "BVB", "club brugge": "BRU", "como": "COM", "fenerbahce": "FEN",
    "feyenoord": "FEY", "galatasaray": "GAL", "inter": "INT", "lask": "LASK", "lens": "RCL",
    "lille": "LIL", "liverpool": "LIV", "manchester city": "MCI",
    "manchester united": "MUN", "napoli": "NAP", "porto": "POR", "psg": "PSG", "psv": "PSV",
    "rb leipzig": "RBL", "real betis": "BET", "real madrid": "RMA", "roma": "ROM",
    "sabah": "SAB", "shakhtar donetsk": "SHA", "slavia prague": "SLA",
    "slovan bratislava": "SLO", "sporting": "SCP", "stuttgart": "VFB", "viking": "VIK",
    "villarreal": "VIL",
}

REGIME_LABELS = {"science": "Flat bet", "edge_gated": "Edge only"}

# What went wrong, newest first. Each was a silent failure: plausible numbers,
# no error. They stay on the site because the record is only honest with them.
INCIDENTS = [
    ("4 Nov", "A model bet on both teams",
     "Wednesday games appear in both Tuesday's and Wednesday's forecasts. Elo v2 "
     "changed its mind overnight on Lens v Como and backed both sides.",
     "The second bet was voided, and a model can now hold only one bet per game."),
    ("20 Oct", "Matchday 3 had no odds",
     "The odds feed shared a free quota with another project, and it ran dry. "
     "Matchday 3 was still forecast and scored, but nobody placed a bet.",
     "This project now has its own quota. No price means no bet, never a guessed one."),
    ("13 Oct", "Nine results the code couldn't read",
     "Wikipedia wrote matchday 2's scores with a maths minus sign (0−2) instead of a "
     "hyphen. Nine finished games looked unplayed, and 47 bets sat open for six days.",
     "Any dash now counts, and a missing score more than three hours after kickoff "
     "sends an alert."),
    ("13 Oct", "Forecasts after kickoff",
     "GitHub ran the morning job seven hours late, after two games had started. "
     "The models still bet on them.",
     "Those two forecasts are shown but never scored, and their 12 bets were voided. "
     "The job now runs twice, early, with a kickoff check."),
    ("Sep", "Lens existed twice",
     "One data source called the club RC Lens, another Lens, so the models rated two "
     "half-clubs instead of one.",
     "Names are merged through one registry, and an audit runs after every data rebuild."),
]


# --- data ----------------------------------------------------------------------

def _load_schedule() -> list[dict]:
    """Every league-phase fixture from Wikipedia, keyed like briefs. [] if offline."""
    try:
        from ..data import wikipedia
        from ..teams import TeamRegistry
        registry = TeamRegistry.load()
        rows = []
        for f in wikipedia.load_current_season(max_age_hours=6):
            home, away = registry.resolve(f.home), registry.resolve(f.away)
            if not home or not away:
                continue
            rows.append({"key": (home, away, str(f.date)), "md": f.matchday,
                         "date": str(f.date), "time": f.time,
                         "home_name": f.home, "away_name": f.away})
        return rows
    except Exception:                                   # noqa: BLE001
        return []


def _load_ledger():
    try:
        from ..ledger.ledger import Ledger
        return Ledger.load()
    except Exception:                                   # noqa: BLE001
        return None


def _matchday_of(brief: dict) -> int | None:
    for entry in brief.get("fixtures", []):
        if entry.get("matchday"):
            return int(entry["matchday"])
    return None


def chosen_entries(briefs: list[dict]) -> dict[tuple, dict]:
    """One forecast per fixture: the earliest on time, else the earliest at all.

    Matches how the scoreboard scores (season.collect_scored_fixtures), so a
    matchday page never shows a different forecast from the one that was scored.
    """
    on_time: dict[tuple, dict] = {}
    late: dict[tuple, dict] = {}
    for brief in briefs:
        for entry in brief.get("fixtures", []):
            key = (entry["home"], entry["away"], entry["date"])
            bucket = late if entry.get("predicted_after_kickoff") else on_time
            bucket.setdefault(key, entry)
    return {**late, **on_time}


def _kick_iso(day: str, local: str | None) -> str | None:
    try:
        from ..kickoff import kickoff_utc
        moment = kickoff_utc(day, local)
        return moment.isoformat() if moment else None
    except Exception:                                   # noqa: BLE001
        return None


def build_matchdays(schedule: list[dict], entries: dict, results: dict,
                    *, now: datetime | None = None) -> dict[int, dict]:
    now = now or datetime.now(timezone.utc)
    rows: dict[tuple, dict] = {}
    for fx in schedule:
        rows[fx["key"]] = dict(fx)
    for key, entry in entries.items():
        row = rows.setdefault(key, {"key": key, "md": entry.get("matchday"),
                                    "date": entry["date"], "time": entry.get("kickoff"),
                                    "home_name": entry.get("home_display"),
                                    "away_name": entry.get("away_display")})
        row["entry"] = entry
        if row.get("md") is None:
            row["md"] = entry.get("matchday")
    out: dict[int, dict] = {}
    for row in rows.values():
        if not row.get("md"):
            continue
        row.setdefault("entry", None)
        row["result"] = results.get(row["key"])
        row["kick"] = _kick_iso(row["date"], row.get("time"))
        md = out.setdefault(int(row["md"]), {"md": int(row["md"]), "rows": []})
        md["rows"].append(row)
    for md in out.values():
        md["rows"].sort(key=lambda r: (r["date"], r.get("time") or "", r["key"][0]))
        md["dates"] = sorted({r["date"] for r in md["rows"]})
        played = sum(1 for r in md["rows"] if r["result"])
        started = any(r["kick"] and datetime.fromisoformat(r["kick"]) <= now
                      for r in md["rows"])
        md["played"] = played
        md["forecast"] = sum(1 for r in md["rows"] if r["entry"])
        if played == len(md["rows"]):
            md["status"] = "played"
        elif played or started:
            md["status"] = "live"
        else:
            md["status"] = "upcoming"
    upcoming = [m for m in sorted(out) if out[m]["status"] == "upcoming"]
    if upcoming and not any(m["status"] == "live" for m in out.values()):
        out[upcoming[0]]["status"] = "next"
    return dict(sorted(out.items()))


def _outcome(result) -> str | None:
    if not result:
        return None
    h, a = result
    return "home" if h > a else ("draw" if h == a else "away")


def _consensus(entry: dict) -> dict | None:
    views = [v for v in (entry.get("arms") or {}).values() if v]
    if not views:
        return None
    return {k: sum(v[k] for v in views) / len(views) for k in OUTCOMES}


def _market(entry: dict) -> dict | None:
    market = entry.get("market") or {}
    if market.get("home") is None:
        return None
    return {k: market[k] for k in OUTCOMES}


def _pick(view: dict) -> str:
    return max(OUTCOMES, key=lambda k: view[k])


def _log_loss(p: float) -> float:
    return -math.log(max(p, 1e-12))


def matchday_summary(md: dict) -> dict:
    """Market hit rate and the best model on one matchday's scored games."""
    scored = [r for r in md["rows"] if r["result"] and r["entry"]
              and not r["entry"].get("predicted_after_kickoff")]
    market_hits = market_n = 0
    losses: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        actual = _outcome(r["result"])
        market = _market(r["entry"])
        if market:
            market_n += 1
            market_hits += _pick(market) == actual
        for name, view in (r["entry"].get("arms") or {}).items():
            if view:
                losses[name].append(_log_loss(view[actual]))
    best = None
    eligible = {n: sum(v) / len(v) for n, v in losses.items()
                if len(v) >= max(1, len(scored) // 2)}
    if eligible:
        name = min(eligible, key=eligible.get)
        best = {"arm": name, "log_loss": eligible[name]}
    return {"scored": len(scored), "market_hits": market_hits, "market_n": market_n,
            "best": best}


def upsets(matchdays: dict[int, dict], limit: int = 6) -> list[dict]:
    out = []
    for md in matchdays.values():
        for r in md["rows"]:
            entry = r["entry"]
            if not (r["result"] and entry) or entry.get("predicted_after_kickoff"):
                continue
            actual = _outcome(r["result"])
            market = _market(entry)
            source, view = ("market", market) if market else ("models", _consensus(entry))
            if not view:
                continue
            out.append({"row": r, "md": md["md"], "actual": actual,
                        "p": view[actual], "source": source})
    out.sort(key=lambda u: u["p"])
    return out[:limit]


def headline(board: list[dict]) -> dict | None:
    """Market vs the best model, on the games both were scored on."""
    market = next((r for r in board if r.get("is_market")), None)
    models = [r for r in board if not r.get("is_market")]
    if not models:
        return None
    if market:
        full = [r for r in models if r.get("compared_on") == market["n"]
                and r.get("edge_vs_market") is not None]
        if full:
            best = max(full, key=lambda r: r["edge_vs_market"])
            return {"arm": best["arm"], "model_ll": market["log_loss"] - best["edge_vs_market"],
                    "market_ll": market["log_loss"], "n": market["n"],
                    "beating": best["edge_vs_market"] > 0}
    best = min(models, key=lambda r: r["log_loss"])
    return {"arm": best["arm"], "model_ll": best["log_loss"], "market_ll": None,
            "n": best["n"], "beating": None}


def standings(matchdays: dict[int, dict]) -> list[dict]:
    table: dict[str, dict] = {}
    for md in matchdays.values():
        for r in md["rows"]:
            home, away, _ = r["key"]
            for club, name in ((home, r.get("home_name")), (away, r.get("away_name"))):
                table.setdefault(club, {"club": club, "name": name or club.title(), "p": 0,
                                        "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
            if not r["result"]:
                continue
            hg, ag = r["result"]
            for club, gf, ga in ((home, hg, ag), (away, ag, hg)):
                t = table[club]
                t["p"] += 1
                t["gf"] += gf
                t["ga"] += ga
                t["w" if gf > ga else ("d" if gf == ga else "l")] += 1
    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]
        t["pts"] = 3 * t["w"] + t["d"]
    return sorted(table.values(), key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["name"]))


def bankroll_series(ledger) -> list[dict]:
    """Realised equity per arm after each matchday date, from the paper ledger."""
    if ledger is None:
        return []
    out = []
    for arm in ledger.arms.values():
        settled = sorted((b for b in arm.bets if b.status in ("won", "lost", "push")),
                         key=lambda b: (b.match_date, b.placed_at))
        equity = arm.starting_bankroll
        points: dict[str, float] = {}
        won = lost = 0
        for bet in settled:
            equity += (bet.payout or 0.0) - bet.stake
            points[bet.match_date] = equity
            won += bet.status == "won"
            lost += bet.status == "lost"
        out.append({"arm": arm.name, "regime": arm.regime, "start": arm.starting_bankroll,
                    "equity": equity, "won": won, "lost": lost,
                    "open": sum(1 for b in arm.bets if b.status == "open"),
                    "points": sorted(points.items()), "bets": arm.bets})
    order = {n: i for i, n in enumerate(ARMS)}
    out.sort(key=lambda s: (s["regime"] != "science", order.get(s["arm"], 99)))
    return out


# --- small renderers -------------------------------------------------------------

def _name(arm: str) -> str:
    return PROFILES.get(arm, (ARM_LABELS.get(arm, arm),))[0]


def _avatar(arm: str, cls: str = "", alt: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    return (f'<img{klass} src="assets/avatar-{esc(arm)}.webp" alt="{esc(alt)}" '
            f'width="180" height="180" loading="lazy">')


def _code(club: str) -> str:
    return CODES.get(club, club.replace(" ", "")[:3].upper())


def _day(iso: str, fmt: str = "%a %d %b") -> str:
    try:
        return date.fromisoformat(iso).strftime(fmt).replace(" 0", " ")
    except ValueError:
        return iso


def _money(value: float, signed: bool = False) -> str:
    if signed:
        sign = "+" if value > 0 else ("−" if value < 0 else "±")
        return f"{sign}${abs(value):,.0f}"
    return f"${value:,.0f}"


def _bar(view: dict, *, large: bool = False, label: str = "") -> str:
    total = sum(view[k] for k in OUTCOMES) or 1
    h, d, a = (view[k] / total * 100 for k in OUTCOMES)
    aria = f"{label} home {h:.0f}%, draw {d:.0f}%, away {a:.0f}%".strip()
    return (f'<div class="bar{" bar--lg" if large else ""}" role="img" aria-label="{esc(aria)}">'
            f'<span class="h" style="width:{h:.1f}%"></span>'
            f'<span class="d" style="width:{d:.1f}%"></span>'
            f'<span class="a" style="width:{a:.1f}%"></span></div>')


def _trio(view: dict, actual: str | None = None) -> str:
    total = sum(view[k] for k in OUTCOMES) or 1
    parts = []
    for k, word in zip(OUTCOMES, ("home", "draw", "away")):
        pct = f"{view[k] / total * 100:.0f}%"
        parts.append(f"<span>{word} <b>{pct}</b></span>" if k == actual
                     else f"<span>{word} {pct}</span>")
    return f'<div class="trio">{"".join(parts)}</div>'


def _kick_html(row: dict) -> str:
    if row.get("kick"):
        utc = datetime.fromisoformat(row["kick"]).strftime("%H:%M")
        return f'<time data-kick datetime="{esc(row["kick"])}">{utc} UTC</time>'
    return esc(row.get("time") or "")


def match_card(row: dict) -> str:
    home, away, _ = row["key"]
    entry = row["entry"]
    actual = _outcome(row["result"])
    if row["result"]:
        mid = f'<span class="bug__score">{row["result"][0]}–{row["result"][1]}</span>'
        mid += '<span class="bug__note">full time</span>'
    else:
        mid = f'<span class="bug__time">{_kick_html(row)}</span><span class="bug__note">{esc(_day(row["date"]))}</span>'
    parts = [f"""
      <div class="bug">
        <div class="bug__team"><span class="bug__code">{esc(_code(home))}</span>
          <span class="bug__name">{esc(row.get('home_name') or home)}</span></div>
        <div class="bug__mid">{mid}</div>
        <div class="bug__team bug__team--away"><span class="bug__code">{esc(_code(away))}</span>
          <span class="bug__name">{esc(row.get('away_name') or away)}</span></div>
      </div>"""]
    if not entry:
        note = ("Forecasts are published the morning of the match."
                if not row["result"] else "Not forecast: played before the lab went live.")
        parts.append(f'<p class="flag">{note}</p>')
        return f'<article class="match">{"".join(parts)}</article>'

    market, consensus = _market(entry), _consensus(entry)
    rows_html = []
    if market:
        rows_html.append(f'<div class="mrow"><span class="mrow__k">Market</span>'
                         f'<div>{_bar(market, large=True, label="market")}{_trio(market, actual)}</div></div>')
    if consensus:
        rows_html.append(f'<div class="mrow"><span class="mrow__k">Models</span>'
                         f'<div>{_bar(consensus, label="average of models")}{_trio(consensus, actual)}</div></div>')
    parts.append(f'<div class="match__rows">{"".join(rows_html)}</div>')

    views = [(n, v) for n, v in (entry.get("arms") or {}).items() if v]
    crowd = []
    for name, view in views:
        state = "" if not actual else (" hit" if _pick(view) == actual else " miss")
        crowd.append(f'<img class="{state.strip()}" src="assets/avatar-{esc(name)}.webp" '
                     f'alt="{esc(_name(name))}" title="{esc(_name(name))}: '
                     f'{esc(_pick_word(_pick(view), row))}" width="28" height="28" loading="lazy">')
    if actual:
        hits = sum(1 for _, v in views if _pick(v) == actual)
        crowd_txt = f"{hits} of {len(views)} called it"
    elif consensus:
        fav = _pick(consensus)
        backers = sum(1 for _, v in views if _pick(v) == fav)
        crowd_txt = f"{backers} of {len(views)} pick {_pick_word(fav, row)}"
    else:
        crowd_txt = ""
    parts.append(f'<div class="crowd">{"".join(crowd)}<span class="crowd__txt">{esc(crowd_txt)}</span></div>')

    flags = []
    if entry.get("predicted_after_kickoff"):
        flags.append("Forecast arrived after kickoff, so it isn't scored.")
    elif not market:
        flags.append("No market price, so no bets were placed.")
    if flags:
        parts.append(f'<p class="flag">{" ".join(flags)}</p>')

    arm_rows = "".join(
        f'<div class="arm">{_avatar(n)}<span class="arm__n">{esc(_name(n))}</span>'
        f'{_bar(v, label=_name(n))}<span class="arm__p">{v["home"]*100:.0f} · {v["draw"]*100:.0f} · {v["away"]*100:.0f}</span></div>'
        for n, v in sorted(views, key=lambda nv: ARMS.index(nv[0]) if nv[0] in ARMS else 99))
    parts.append(f'<details><summary>Every model</summary><div class="arms">{arm_rows}</div></details>')
    return f'<article class="match">{"".join(parts)}</article>'


def _pick_word(pick: str, row: dict) -> str:
    if pick == "draw":
        return "a draw"
    home, away, _ = row["key"]
    return row.get("home_name") or home if pick == "home" else row.get("away_name") or away


def _page(title: str, description: str, path: str, current: str, ctx: dict, body: str) -> str:
    return (head(title, description, path=path) + nav(current, ctx["pill"])
            + f'<main id="main">{body}</main>' + foot(ctx["generated"]))


# --- pages -------------------------------------------------------------------------

def render_overview(ctx: dict) -> str:
    hl = ctx["headline"]
    if hl and hl["market_ll"] is not None:
        verdict = ("A model is ahead of the market." if hl["beating"]
                   else "The market is still unbeaten.")
        board = f"""
      <div class="board" role="img" aria-label="Best model {esc(_name(hl['arm']))} {hl['model_ll']:.3f}, market {hl['market_ll']:.3f}. Lower is better.">
        <div class="board__side">{_avatar(hl['arm'], 'board__avatar')}
          <div class="board__who"><span class="board__label">{esc(_name(hl['arm']))}</span>
          <span class="board__sub">best model</span></div></div>
        <div class="board__score"><span class="board__num">{hl['model_ll']:.3f}</span>
          <span class="board__sep">v</span><span class="board__num">{hl['market_ll']:.3f}</span></div>
        <div class="board__side board__side--away">{_avatar('market', 'board__avatar')}
          <div class="board__who"><span class="board__label">Market</span>
          <span class="board__sub">the bookmakers</span></div></div>
      </div>
      <p class="board__caption"><b>{verdict}</b> Scores are log loss over the {hl['n']}
      games with a market price: how surprised each side was by the results. Lower wins.</p>"""
    else:
        board = '<p class="board__caption">The scoreboard fills in after the first results.</p>'

    hero = f"""
<section class="hero">
  <picture class="hero__img">
    <source media="(max-width:700px)" srcset="assets/hero-900.webp">
    <img src="assets/hero.webp" alt="" width="1672" height="941" fetchpriority="high">
  </picture>
  <div class="wrap hero__inner">
    <p class="hero__kicker">Champions League 2026/27 · league phase</p>
    <h1>Can a model beat the market?</h1>
    <p class="hero__lede">Eight machine-learning models forecast every Champions League
    game without ever seeing the betting odds. The bookmakers are the opponent.</p>
    {board}
  </div>
</section>"""

    return _page("Forecast Lab", "Eight models forecast every Champions League game "
                 "without seeing the odds. Can any of them beat the market?",
                 "", "overview", ctx,
                 hero + _next_up(ctx) + _leaderboard_section(ctx) + _money_section(ctx)
                 + _upsets_section(ctx) + _incidents_teaser())


def _next_up(ctx: dict) -> str:
    mds = ctx["matchdays"]
    focus = next((m for m in mds.values() if m["status"] in ("live", "next")), None)
    last = next((m for m in reversed(list(mds.values())) if m["status"] == "played"), None)
    left = ""
    if focus:
        label = "Live now" if focus["status"] == "live" else "Next up"
        dates = " & ".join(_day(d) for d in focus["dates"])
        items = []
        for r in focus["rows"][:8]:
            home, away, _ = r["key"]
            pick = ""
            if r["entry"] and _consensus(r["entry"]):
                c = _consensus(r["entry"])
                fav = _pick(c)
                pick = f"{_pick_word(fav, r)} {c[fav]*100:.0f}%"
            elif r["result"]:
                pick = f"{r['result'][0]}–{r['result'][1]}"
            items.append(f"""<li><span class="fx__when">{esc(_day(r['date'], '%a'))} {_kick_html(r)}</span>
              <span class="fx__teams"><b>{esc(_code(home))}</b> v <b>{esc(_code(away))}</b>
              <span class="muted">{esc(r.get('home_name') or home)} v {esc(r.get('away_name') or away)}</span></span>
              <span class="fx__pick">{esc(pick) or '<span class="muted">forecast soon</span>'}</span></li>""")
        more = len(focus["rows"]) - 8
        note = ("The models' favourite for each game." if focus["forecast"]
                else "Forecasts land the morning of each game and go out on Telegram.")
        left = f"""
      <div class="panel nextup">
        <div class="nextup__head"><div><p class="muted">{label}</p>
          <h3 class="nextup__title">Matchday {focus['md']}</h3></div>
          <p class="muted">{esc(dates)}</p></div>
        <p class="nextup__note">{note}</p>
        <ul class="fx">{''.join(items)}</ul>
        <a class="more" href="matchday-{focus['md']}.html">All {len(focus['rows'])} games{f' (+{more} more)' if more > 0 else ''}</a>
      </div>"""
    if last:
        s = matchday_summary(last)
        best = s["best"]
        best_html = (f'<div class="lastmd__best">{_avatar(best["arm"])}<div><p class="muted">Model of the matchday</p>'
                     f'<p class="lastmd__name">{esc(_name(best["arm"]))}</p></div></div>') if best else ""
        market_line = (f'The market\'s favourite won <b>{s["market_hits"]} of {s["market_n"]}</b>.'
                       if s["market_n"] else "No market prices this matchday.")
        left += f"""
      <div class="panel lastmd">
        <div><p class="muted">Last time out</p><h3>Matchday {last['md']}</h3></div>
        {best_html}
        <p class="lastmd__txt">{market_line}</p>
        <a class="more" href="matchday-{last['md']}.html">Results and forecasts</a>
      </div>"""
    return f"""
<section class="section"><div class="wrap split">
  <div class="stack">{left or '<p class="empty">No fixtures loaded.</p>'}</div>
  <div>{_leaderboard_panel(ctx, compact=True)}</div>
</div></section>"""


def _leaderboard_panel(ctx: dict, compact: bool = False) -> str:
    board = ctx["board"]
    if not board:
        return '<div class="panel"><p class="empty">No scored games yet.</p></div>'
    market = next((r for r in board if r.get("is_market")), None)
    models = sorted((r for r in board if not r.get("is_market")), key=lambda r: r["log_loss"])
    items = []
    if market:
        items.append(f"""<li class="is-market"><span class="lb__rank" aria-hidden="true">★</span>
          {_avatar('market')}<div><p class="lb__name">The market</p><p class="lb__nick">The one to beat · {market['n']} games</p></div>
          <div class="lb__val"><p class="lb__ll">{market['log_loss']:.3f}</p></div></li>""")
    for i, r in enumerate(models, 1):
        edge = r.get("edge_vs_market")
        if edge is None:
            gap = f"{r['n']} games"
        else:
            word = "ahead" if edge > 0 else "behind"
            gap = f"{abs(edge):.3f} {word} on {r['compared_on']}"
        items.append(f"""<li><span class="lb__rank">{i}</span>{_avatar(r['arm'])}
          <div><p class="lb__name">{esc(_name(r['arm']))}</p><p class="lb__nick">{esc(PROFILES.get(r['arm'], ('', ''))[1])}</p></div>
          <div class="lb__val"><p class="lb__ll">{r['log_loss']:.3f}</p><p class="lb__gap">{esc(gap)}</p></div></li>""")
    title = '<div class="lbhead"><h3>Leaderboard</h3><p class="muted">Log loss, lower is better</p></div>' if compact else ""
    return f'<div class="panel">{title}<ol class="lb">{"".join(items)}</ol></div>'


def _leaderboard_section(ctx: dict) -> str:
    return ""                 # the overview carries the leaderboard beside "next up"


def _money_rows(series: list[dict]) -> str:
    if not series:
        return '<p class="empty">No bets settled yet.</p>'
    rows = sorted(series, key=lambda s: -s["equity"])
    scale = max(abs(s["equity"] - s["start"]) for s in rows) or 1
    out = []
    for s in rows:
        diff = s["equity"] - s["start"]
        tone = "up" if diff >= 0 else "down"
        width = abs(diff) / scale * 50
        out.append(f"""<div class="money__row">{_avatar(s['arm'])}<span>{esc(_name(s['arm']))}</span>
          <div class="money__track" role="img" aria-label="{esc(_money(diff, True))}"><span class="money__fill {tone}" style="width:{width:.1f}%"></span></div>
          <span class="money__val {tone}">{_money(diff, True)}</span></div>""")
    axis = (f'<div class="money__axis"><span></span><span></span><span>started at {_money(rows[0]["start"])}</span>'
            f'<span></span></div>')
    return f'<div class="panel money">{"".join(out)}{axis}</div>'


def _money_section(ctx: dict) -> str:
    flat = [s for s in ctx["series"] if s["regime"] == "science"]
    return f"""
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Who's up on paper</h2>
    <p>Each model bets $100 of fake money on its favourite in every game with a price.
    Profit or loss so far.</p></div>
    <a class="more" href="bankrolls.html">Both strategies, every bet</a></div>
  {_money_rows(flat)}
</div></section>"""


def _upsets_section(ctx: dict) -> str:
    cards = []
    for u in ctx["upsets"]:
        r = u["row"]
        home, away, _ = r["key"]
        hg, ag = r["result"]
        hn, an = r.get("home_name") or home, r.get("away_name") or away
        hs = f"<em>{esc(hn)} {hg}</em>" if u["actual"] == "home" else f"{esc(hn)} {hg}"
        as_ = f"<em>{ag} {esc(an)}</em>" if u["actual"] == "away" else f"{ag} {esc(an)}"
        what = "a draw" if u["actual"] == "draw" else (hn if u["actual"] == "home" else an)
        who = "The market" if u["source"] == "market" else "The models"
        cards.append(f"""<a class="panel upset" href="matchday-{u['md']}.html">
          <p class="upset__score">{hs} – {as_}</p>
          <p class="upset__p">{who} gave {esc(what)} <b>{u['p']*100:.0f}%</b></p>
          <p class="upset__md">Matchday {u['md']} · {esc(_day(r['date']))}</p></a>""")
    if not cards:
        return ""
    return f"""
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Results nobody saw coming</h2>
    <p>The biggest shocks so far, ranked by how unlikely the market thought they were.</p></div></div>
  <div class="upsets">{''.join(cards)}</div>
</div></section>"""


def _incidents_teaser() -> str:
    items = "".join(f"""<li><span class="log__date">{esc(d)}</span><div><h3>{esc(t)}</h3>
      <p>{esc(what)}</p></div></li>""" for d, t, what, _ in INCIDENTS[:3])
    return f"""
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>What went wrong</h2>
    <p>Every bug that touched a forecast or a bet, and what changed because of it.</p></div>
    <a class="more" href="how-it-works.html#what-went-wrong">All {len(INCIDENTS)} and the fixes</a></div>
  <div class="panel" style="padding:4px 20px"><ul class="log">{items}</ul></div>
</div></section>"""


def render_matchdays(ctx: dict) -> str:
    tiles = []
    for md in ctx["matchdays"].values():
        dates = " & ".join(_day(d, "%d %b") for d in md["dates"])
        state = {"played": "Played", "live": "Live", "next": "Next", "upcoming": "Upcoming"}[md["status"]]
        if md["status"] == "played":
            s = matchday_summary(md)
            body = f"<b>{md['played']}</b> games"
            if s["market_n"]:
                body += f" · market favourite won <b>{s['market_hits']}/{s['market_n']}</b>"
            if s["best"]:
                body += f"<br>Model of the matchday: <b>{esc(_name(s['best']['arm']))}</b>"
        elif md["status"] == "live":
            body = f"<b>{md['played']}</b> of {len(md['rows'])} played"
        else:
            body = f"{len(md['rows'])} games" + (" · forecasts in" if md["forecast"] else "")
        quiet = " tile--quiet" if md["status"] == "upcoming" else ""
        tiles.append(f"""<a class="panel tile{quiet}" href="matchday-{md['md']}.html">
          <span class="state state--{md['status']}">{state}</span>
          <span class="tile__md"><small>MD</small>{md['md']}</span>
          <span class="tile__dates">{esc(dates)}</span>
          <span class="tile__body">{body}</span></a>""")
    body = band("Matchdays", "Eight matchdays in the league phase. Each page shows the "
                "forecasts as they were published before kickoff, then the results.",
                "matchdays") + f"""
<section class="section"><div class="wrap"><div class="tiles">{''.join(tiles)}</div></div></section>"""
    return _page("Matchdays", "Every Champions League league-phase matchday: forecasts "
                 "before kickoff, results after.", "matchdays.html", "matchdays", ctx, body)


def render_matchday_page(ctx: dict, md: dict) -> str:
    mds = list(ctx["matchdays"])
    i = mds.index(md["md"])
    prev_link = f'<a class="more" href="matchday-{mds[i-1]}.html">Matchday {mds[i-1]}</a>' if i else ""
    next_link = (f'<a class="more" href="matchday-{mds[i+1]}.html">Matchday {mds[i+1]}</a>'
                 if i + 1 < len(mds) else "")
    dates = " and ".join(_day(d, "%A %d %B") for d in md["dates"])
    s = matchday_summary(md)
    stats = []
    if md["status"] in ("played", "live"):
        stats.append((f"{md['played']}/{len(md['rows'])}", "games played"))
    if s["market_n"]:
        stats.append((f"{s['market_hits']}/{s['market_n']}", "market favourites won"))
    if s["best"]:
        stats.append((_name(s["best"]["arm"]), "model of the matchday"))
    stats.append((f"{md['forecast']}/{len(md['rows'])}", "games forecast"))
    stats_html = "".join(f'<div class="stat"><p class="stat__v">{esc(v)}</p><p class="stat__k">{esc(k)}</p></div>'
                         for v, k in stats)
    groups = []
    for day in md["dates"]:
        cards = "".join(match_card(r) for r in md["rows"] if r["date"] == day)
        groups.append(f'<div class="daygroup"><h3>{esc(_day(day, "%A %d %B"))}</h3><div class="matches">{cards}</div></div>')
    legend = """<div class="legend"><span><i style="background:var(--home)"></i>home win</span>
      <span><i style="background:var(--draw)"></i>draw</span><span><i style="background:var(--away)"></i>away win</span>
      <span><i style="background:var(--turf);border-radius:50%"></i>ring = model called it</span></div>"""
    body = band(f"Matchday {md['md']}", esc(dates) + ".", "matchdays") + f"""
<section class="section"><div class="wrap">
  <div class="stats">{stats_html}</div>
  {legend}
</div></section>
<section class="section"><div class="wrap">{''.join(groups)}
  <div class="mdnav">{prev_link}<a class="more" href="matchdays.html">All matchdays</a>{next_link}</div>
</div></section>"""
    return _page(f"Matchday {md['md']}", f"Champions League matchday {md['md']}: every "
                 "model's forecast and the results.", f"matchday-{md['md']}.html",
                 "matchdays", ctx, body)


def render_models(ctx: dict) -> str:
    board = {r["arm"]: r for r in ctx["board"] if not r.get("is_market")}
    market = next((r for r in ctx["board"] if r.get("is_market")), None)
    ranked = sorted(board.values(), key=lambda r: r["log_loss"])
    rank = {r["arm"]: i for i, r in enumerate(ranked, 1)}
    money = {(s["arm"], s["regime"]): s for s in ctx["series"]}
    hypotheses = ctx["hypotheses"]
    cards = []
    if market:
        label, nick, desc = MARKET_PROFILE
        cards.append(f"""<article class="panel profile profile--market">
          <div class="profile__top">{_avatar('market')}<div><p class="profile__nick">{nick}</p>
          <h2 class="profile__name">{label}</h2><span class="profile__tag">The benchmark</span></div></div>
          <div class="profile__body"><p>{desc}</p></div>
          <div class="profile__stats"><div><b>{market['log_loss']:.3f}</b><span>log loss</span></div>
          <div><b>{market['accuracy']*100:.0f}%</b><span>favourite won</span></div>
          <div><b>{market['n']}</b><span>games priced</span></div></div></article>""")
    for arm in sorted(PROFILES, key=lambda a: rank.get(a, 99)):
        label, nick, desc, tag = PROFILES[arm]
        r = board.get(arm)
        flat = money.get((arm, "science"))
        gated = money.get((arm, "edge_gated"))

        def pl(s):
            if not s:
                return "—", ""
            diff = s["equity"] - s["start"]
            return _money(diff, True), ("up-t" if diff >= 0 else "down-t")
        flat_v, flat_c = pl(flat)
        gated_v, gated_c = pl(gated)
        idea = hypotheses.get(arm)
        cards.append(f"""<article class="panel profile">
          <div class="profile__top">{_avatar(arm)}<div><p class="profile__nick">{esc(nick)}</p>
          <h2 class="profile__name">{esc(label)}</h2>
          {f'<span class="profile__tag">{esc(tag)}</span>' if tag else ''}</div></div>
          <div class="profile__body"><p>{esc(desc)}</p>
          {f'<p class="profile__idea">Testing: {esc(idea)}</p>' if idea else ''}</div>
          <div class="profile__stats">
            <div><b>{'#' + str(rank[arm]) if arm in rank else '—'}</b><span>{f"log loss {r['log_loss']:.3f}" if r else 'not scored yet'}</span></div>
            <div><b class="{flat_c}">{flat_v}</b><span>flat bet</span></div>
            <div><b class="{gated_c}">{gated_v}</b><span>edge only</span></div></div></article>""")
    body = band("The models", "Eight models, one question each. Ranked by how well "
                "they've forecast so far. The market sits on top as the benchmark.",
                "models") + f"""
<section class="section"><div class="wrap"><div class="profiles">{''.join(cards)}</div></div></section>"""
    return _page("Models", "The eight models in the Champions League forecast lab, "
                 "what each one tests, and how they're doing.", "models.html",
                 "models", ctx, body)


def _spark(s: dict, dates: list[str]) -> str:
    diff = s["equity"] - s["start"]
    tone = "up" if diff >= 0 else "down"
    pts = s["points"]
    svg = ""
    if pts:
        values = [s["start"]] + [v for _, v in pts]
        lo, hi = min(values + [s["start"]]), max(values + [s["start"]])
        pad = (hi - lo) * 0.12 or 50
        lo, hi = lo - pad, hi + pad
        idx = {d: i + 1 for i, d in enumerate(dates)}
        span = max(len(dates), 1)
        # The line starts one step before the arm's first settled matchday, so
        # an arm that joined late (the v2s) doesn't look like it bet all season.
        first = idx[pts[0][0]] - 1
        coords = [(first / span * 100, s["start"], "start")] + [(idx[d] / span * 100, v, d) for d, v in pts]

        def y(v):
            return 60 - (v - lo) / (hi - lo) * 60
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.2f},{y(v):.2f}" for i, (x, v, _) in enumerate(coords))
        data = json.dumps([[round(x, 2), round(y(v), 2), _money(v),
                            "Start" if d == "start" else _day(d, "%d %b")] for x, v, d in coords])
        lx, lv, _ = coords[-1]
        svg = f"""<svg viewBox="0 0 100 60" preserveAspectRatio="none" data-pts='{esc(data)}'
          role="img" aria-label="{esc(_name(s['arm']))} bankroll from {_money(s['start'])} to {_money(s['equity'])}">
          <line class="base" x1="0" x2="100" y1="{y(s['start']):.2f}" y2="{y(s['start']):.2f}" vector-effect="non-scaling-stroke"/>
          <path class="line {tone}" d="{path}" vector-effect="non-scaling-stroke"/>
        </svg><span class="tip" aria-hidden="true"></span>"""
    return f"""<div class="panel spark">
      <div class="spark__head">{_avatar(s['arm'])}<div><p class="spark__name">{esc(_name(s['arm']))}</p>
        <p class="spark__rec">{s['won']} won · {s['lost']} lost{f" · {s['open']} open" if s['open'] else ''}</p></div>
        <p class="spark__val">{_money(s['equity'])}<small class="{tone}-t">{_money(diff, True)}</small></p></div>
      {svg or '<p class="empty" style="padding:12px 0 0">No settled bets yet.</p>'}</div>"""


SPARK_JS = """<script>
document.querySelectorAll('.spark svg[data-pts]').forEach(function(svg){
  var pts=JSON.parse(svg.getAttribute('data-pts')), card=svg.parentNode, tip=card.querySelector('.tip');
  var dot=document.createElement('span'); dot.className='sdot'; card.appendChild(dot);
  function show(ev){
    var r=svg.getBoundingClientRect(), x=(ev.clientX-r.left)/r.width*100, best=pts[0];
    pts.forEach(function(p){ if(Math.abs(p[0]-x)<Math.abs(best[0]-x)) best=p; });
    var cr=card.getBoundingClientRect();
    var px=r.left-cr.left+best[0]/100*r.width, py=r.top-cr.top+best[1]/60*r.height;
    dot.style.left=px+'px'; dot.style.top=py+'px'; dot.classList.add('on');
    tip.style.left=px+'px'; tip.style.top=(py-8)+'px';
    tip.textContent=best[3]+': '+best[2]; tip.classList.add('on');
  }
  svg.addEventListener('pointermove',show);
  svg.addEventListener('pointerleave',function(){dot.classList.remove('on'); tip.classList.remove('on');});
});
var fm=document.getElementById('f-model'), fr=document.getElementById('f-regime');
function filt(){ document.querySelectorAll('#bets tbody tr').forEach(function(tr){
  var ok=(!fm.value||tr.dataset.arm===fm.value)&&(!fr.value||tr.dataset.regime===fr.value);
  tr.hidden=!ok; }); }
if(fm){ fm.addEventListener('change',filt); fr.addEventListener('change',filt); }
</script>"""


def render_bankrolls(ctx: dict) -> str:
    series = ctx["series"]
    dates = sorted({d for s in series for d, _ in s["points"]})
    flat = "".join(_spark(s, dates) for s in sorted(
        (s for s in series if s["regime"] == "science"), key=lambda s: -s["equity"]))
    gated = "".join(_spark(s, dates) for s in sorted(
        (s for s in series if s["regime"] == "edge_gated"), key=lambda s: -s["equity"]))
    names = ctx["names"]
    rows = []
    for s in series:
        for b in s["bets"]:
            home_n, away_n = names.get(b.home, b.home.title()), names.get(b.away, b.away.title())
            pick = "Draw" if b.selection == "draw" else (home_n if b.selection == "home" else away_n)
            if b.status == "won":
                pl, cls = _money((b.payout or 0) - b.stake, True), "res-won"
            elif b.status == "lost":
                pl, cls = _money(-b.stake, True), "res-lost"
            elif b.status == "void":
                pl, cls = "void", "res-void"
            else:
                pl, cls = "open", ""
            rows.append((b.match_date, f"""<tr data-arm="{esc(s['arm'])}" data-regime="{esc(s['regime'])}">
              <td>{esc(_day(b.match_date, '%d %b'))}</td><td>{esc(home_n)} v {esc(away_n)}</td>
              <td>{esc(_name(s['arm']))}</td><td>{esc(REGIME_LABELS.get(s['regime'], s['regime']))}</td>
              <td>{esc(pick)}</td><td class="n">{f"{b.decimal_odds:.2f}" if b.decimal_odds else '—'}</td>
              <td class="n">{_money(b.stake)}</td><td class="n {cls}">{pl}</td></tr>"""))
    rows.sort(key=lambda r: r[0], reverse=True)
    options = "".join(f'<option value="{a}">{esc(_name(a))}</option>' for a in ARMS)
    body = band("Bankrolls", "Every model started with $10,000 of fake money in two "
                "accounts. Nothing here is real, and no bet is ever placed anywhere.",
                "bankrolls") + f"""
<section class="section"><div class="wrap">
  <div class="rules">
    <div class="panel rule"><h3>Flat bet</h3><p>$100 on the model's favourite in every
    game that has a market price. Lots of bets, so luck evens out faster. This is
    the fair test.</p></div>
    <div class="panel rule"><h3>Edge only</h3><p>Bets only when the model rates a
    result at least a few points more likely than the market does, and stakes more
    the bigger the gap. Closer to how a real bettor would play.</p></div>
  </div>
</section>
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Flat bet</h2><p>Bankroll after each matchday. Hover a line for the value.</p></div></div>
  <div class="sparks">{flat or '<p class="empty">No bets yet.</p>'}</div>
</div></section>
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Edge only</h2><p>Fewer, bigger bets, so expect bigger swings.</p></div></div>
  <div class="sparks">{gated or '<p class="empty">No bets yet.</p>'}</div>
</div></section>
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Every bet</h2><p>{len(rows)} bets, newest first. Voided bets are crossed out.</p></div></div>
  <div class="filters">
    <label>Model <select id="f-model"><option value="">All models</option>{options}</select></label>
    <label>Strategy <select id="f-regime"><option value="">Both</option>
      <option value="science">Flat bet</option><option value="edge_gated">Edge only</option></select></label>
  </div>
  <div class="tablewrap"><table class="t" id="bets"><thead><tr><th>Date</th><th>Game</th><th>Model</th>
    <th>Strategy</th><th>Pick</th><th class="n">Odds</th><th class="n">Stake</th><th class="n">Result</th></tr></thead>
    <tbody>{''.join(r[1] for r in rows)}</tbody></table></div>
</div></section>""" + SPARK_JS
    return _page("Bankrolls", "Paper bankrolls for every model in the Champions League "
                 "forecast lab. Fake money only.", "bankrolls.html", "bankrolls", ctx, body)


def render_table(ctx: dict) -> str:
    table = ctx["standings"]
    quali = {q["club"]: q for q in ctx["qualification"]}
    rows = []
    for pos, t in enumerate(table, 1):
        zone = "top8" if pos <= 8 else ("playoff" if pos <= 24 else "out")
        cut = " cut" if pos in (9, 25) else ""
        q = quali.get(t["club"])

        def prob(v):
            if v is None:
                return '<td class="n muted">—</td>'
            return (f'<td class="n"><span class="prob">{v*100:.0f}%'
                    f'<i style="width:{max(v*40, 1):.0f}px" aria-hidden="true"></i></span></td>')
        rows.append(f"""<tr class="zone-{zone}{cut}"><td class="pos">{pos}</td>
          <td><span class="club"><span class="code">{esc(_code(t['club']))}</span>{esc(t['name'])}</span></td>
          <td class="n">{t['p']}</td><td class="n">{t['w']}</td><td class="n">{t['d']}</td><td class="n">{t['l']}</td>
          <td class="n">{t['gd']:+d}</td><td class="n pts">{t['pts']}</td>
          <td class="n">{f"{q['mean_points']:.1f}" if q else '—'}</td>
          {prob(q['top8'] if q else None)}{prob(_top24(q) if q else None)}</tr>""")
    sim_note = ("Projections come from simulating the remaining games 20,000 times with "
                "the Dixon–Coles model." if quali else "Projections appear with the next forecast.")
    body = band("The table", "The real league-phase table so far, with the models' "
                "projection of where each club finishes.", "table") + f"""
<section class="section"><div class="wrap">
  <div class="tablewrap"><table class="t"><thead><tr><th>#</th><th>Club</th><th class="n">P</th>
    <th class="n">W</th><th class="n">D</th><th class="n">L</th><th class="n">GD</th><th class="n">Pts</th>
    <th class="n">Projected pts</th><th class="n">Top 8</th><th class="n">Top 24</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
  <div class="legend"><span>1–8 go straight to the round of 16</span><span>9–24 go to a playoff</span>
    <span>25–36 are out</span><span>{sim_note}</span></div>
</div></section>"""
    return _page("Table", "The Champions League league-phase table with each club's "
                 "projected finish.", "table.html", "table", ctx, body)


def _top24(q: dict) -> float:
    """Chance of finishing in the top 24: straight through plus the playoff."""
    if q.get("advance") is not None:
        return q["advance"]
    return q.get("top8", 0.0) + q.get("playoff", 0.0)


def render_how(ctx: dict) -> str:
    log = "".join(f"""<li><span class="log__date">{esc(d)}</span><div><h3>{esc(t)}</h3>
      <p>{esc(what)}</p><p class="fix">Fix: {esc(fix)}</p></div></li>""" for d, t, what, fix in INCIDENTS)
    body = band("How it works", "From raw match data to a Telegram message before "
                "kickoff, then a scorecard the morning after.", "how-it-works") + f"""
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>Every matchday</h2></div></div>
  <ol class="steps">
    <li class="panel"><h3>Collect</h3><p>About 53,000 past matches across Europe's leagues and
    cups, plus this season's fixtures and results from Wikipedia.</p></li>
    <li class="panel"><h3>Refit</h3><p>All eight models retrain on the morning of each
    matchday, using only what was known that morning.</p></li>
    <li class="panel"><h3>Publish</h3><p>The forecasts are committed to GitHub before kickoff
    and sent to Telegram, so nothing can be edited after the fact.</p></li>
    <li class="panel"><h3>Price</h3><p>Only now are the bookmakers' odds fetched: to compare
    against, and to settle the fake bets.</p></li>
    <li class="panel"><h3>Settle</h3><p>The next morning, results settle every bet and update
    the scoreboard, and a results message goes out.</p></li>
  </ol>
</div></section>
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>The rules</h2></div></div>
  <div class="rules">
    <div class="panel rule"><h3>The models never see the odds</h3><p>If a model could read
    the market's prices it would just copy them. The odds are used only to score and
    settle, never as an input.</p></div>
    <div class="panel rule"><h3>Before kickoff or it doesn't count</h3><p>A forecast made
    after a game starts is shown but never scored, and its bets are voided.</p></div>
    <div class="panel rule"><h3>Fake money only</h3><p>There is no way for this system to
    place a real bet. That isn't a setting. The code to do it doesn't exist.</p></div>
  </div>
</div></section>
<section class="section"><div class="wrap">
  <div class="section__head"><div><h2>How the scoring works</h2></div></div>
  <div class="prose">
    <p><b>Log loss</b> measures how surprised a forecaster was by what actually happened.
    Give the winner 80% and you lose very little. Give them 5% and you lose a lot.
    Lower is better. Guessing a third each way scores 1.099, and the market usually
    lands around 0.8 to 0.9.</p>
    <p>A season of 144 games is small. Differences of a few hundredths between models
    can be luck, so the honest reading of the leaderboard is in tiers, not ranks.</p>
  </div>
</div></section>
<section class="section" id="what-went-wrong"><div class="wrap">
  <div class="section__head"><div><h2>What went wrong</h2>
    <p>Every one of these produced believable numbers rather than an error. They stay
    here because a track record is only honest with its mistakes in it.</p></div></div>
  <div class="panel" style="padding:4px 20px"><ul class="log">{log}</ul></div>
</div></section>"""
    return _page("How it works", "How the Champions League forecast lab collects data, "
                 "fits models, publishes forecasts and settles fake bets.",
                 "how-it-works.html", "how", ctx, body)


def _redirect(target: str) -> str:
    return (f'<!doctype html><meta charset="utf-8"><title>Moved</title>'
            f'<meta http-equiv="refresh" content="0; url={target}">'
            f'<link rel="canonical" href="{target}">'
            f'<p>This page moved to <a href="{target}">{target}</a>.</p>')


# --- build ---------------------------------------------------------------------------

def build_context(briefs: list[dict], results: dict, *, schedule: list[dict] | None = None,
                  ledger=None, now: datetime | None = None) -> dict:
    entries = chosen_entries(briefs)
    matchdays = build_matchdays(schedule or [], entries, results, now=now)
    board, _ = _board(briefs, results)

    names: dict[str, str] = {}
    for md in matchdays.values():
        for r in md["rows"]:
            names.setdefault(r["key"][0], r.get("home_name") or r["key"][0].title())
            names.setdefault(r["key"][1], r.get("away_name") or r["key"][1].title())

    qualification: list[dict] = []
    hypotheses: dict[str, str] = {}
    for brief in reversed(briefs):
        if not qualification and brief.get("qualification"):
            qualification = brief["qualification"]
        for arm in brief.get("arm_summary", []):
            hypotheses.setdefault(arm.get("name"), arm.get("hypothesis"))

    focus = next((m for m in matchdays.values() if m["status"] in ("live", "next")), None)
    pill = None
    if focus:
        live = focus["status"] == "live"
        pill = {"href": f"matchday-{focus['md']}.html", "live": live,
                "short": f"Matchday {focus['md']}",
                "long": " live" if live else f" · {_day(focus['dates'][0])}"}

    stamps = [str(b.get("generated_at", "")) for b in briefs if b.get("generated_at")]
    generated = ""
    if stamps:
        try:
            moment = datetime.fromisoformat(max(stamps).replace("Z", "+00:00"))
            generated = moment.strftime("%d %b %Y, %H:%M UTC").lstrip("0")
        except ValueError:
            generated = max(stamps)

    return {"briefs": briefs, "results": results, "matchdays": matchdays, "board": board,
            "headline": headline(board), "upsets": upsets(matchdays),
            "standings": standings(matchdays), "series": bankroll_series(ledger),
            "names": names, "qualification": qualification, "hypotheses": hypotheses,
            "pill": pill, "generated": generated}


def _board(briefs: list[dict], results: dict):
    from ..eval import season
    scored = season.collect_scored_fixtures(briefs, results)
    return season.scoreboard(scored), scored


def build_site(results: dict | None = None, out_dir: Path | None = None, *,
               briefs: list[dict] | None = None, schedule: list[dict] | None = None,
               ledger=None) -> Path:
    """Regenerate the whole site. Idempotent for the same inputs."""
    from ..eval import season

    out_dir = out_dir or SITE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results = results or {}
    briefs = briefs if briefs is not None else season.load_briefs()
    schedule = schedule if schedule is not None else _load_schedule()
    ledger = ledger if ledger is not None else _load_ledger()

    ctx = build_context(briefs, results, schedule=schedule, ledger=ledger)
    pages = {
        "index.html": render_overview(ctx),
        "matchdays.html": render_matchdays(ctx),
        "models.html": render_models(ctx),
        "bankrolls.html": render_bankrolls(ctx),
        "table.html": render_table(ctx),
        "how-it-works.html": render_how(ctx),
    }
    for md in ctx["matchdays"].values():
        pages[f"matchday-{md['md']}.html"] = render_matchday_page(ctx, md)
    # Old per-brief URLs (matchday-YYYY-MM-DD.html) were shared before the
    # redesign, so they keep working as redirects.
    for brief in briefs:
        number = _matchday_of(brief)
        target = f"matchday-{number}.html" if number in ctx["matchdays"] else "matchdays.html"
        pages.setdefault(f"matchday-{brief.get('as_of')}.html", _redirect(target))
    for name, text in pages.items():
        (out_dir / name).write_text(text, encoding="utf-8")

    # Tell GitHub Pages not to run the files through Jekyll.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out_dir
