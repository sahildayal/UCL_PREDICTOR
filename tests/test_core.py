"""Tests for the failure modes that actually bit during the build.

Each test here corresponds to a bug that was real, silent, and would have
produced plausible-looking wrong numbers rather than an error.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl.data.openfootball import parse  # noqa: E402
from ucl.ledger.ledger import EDGE_GATED, SCIENCE, Ledger  # noqa: E402
from ucl.scoring import matrix as M  # noqa: E402
from ucl.teams import TeamRegistry, normalise  # noqa: E402


# --- openfootball parsing ----------------------------------------------------

def test_modern_format_parses():
    text = """= UEFA Champions League 2025/26

# Date  Tue Sep 16 2025 - Sat May 30 2026

▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)
"""
    matches = parse(text, competition="UCL", season="2025-26")
    assert len(matches) == 1
    match = matches[0]
    assert match.home_goals == 0 and match.away_goals == 2
    assert match.home_country == "ESP" and match.away_country == "ENG"
    assert match.date == date(2025, 9, 16)


def test_legacy_format_parses_with_header_year():
    """Older files put the score between the teams and the year only in a comment."""
    text = """= English Premier League 2019/20

# Date       Fri Aug 9 2019 - Sun Jul 26 2020 (352d)

▪ Matchday 1

Fri Aug 9

  20:00  Liverpool FC             4-1 (4-0)  Norwich City
"""
    matches = parse(text, competition="DOM-ENG", season="2019-20", default_country="ENG")
    assert len(matches) == 1
    assert matches[0].home == "Liverpool FC"
    assert matches[0].away == "Norwich City"
    assert (matches[0].home_goals, matches[0].away_goals) == (4, 1)
    assert matches[0].date == date(2019, 8, 9)


def test_aggregate_score_is_not_taken_as_the_match_score():
    """"5-0 a.e.t. (3-0, 1-0)" is a two-leg aggregate; this match finished 3-0."""
    text = """= UEFA Champions League 2025/26

# Date  Tue Mar 17 2026 - Tue Mar 17 2026

▪ Finals, Round of 16
  Tue Mar 17 2026
    18:45  Sporting Clube de Portugal (POR) v FK Bodo/Glimt (NOR)      5-0 a.e.t. (3-0, 1-0)
"""
    matches = parse(text, competition="UCL", season="2025-26")
    assert (matches[0].home_goals, matches[0].away_goals) == (3, 0)


def test_calendar_year_season_never_rolls_over():
    """A rescheduled fixture stepping backwards must not advance the year."""
    text = """= Norway Eliteserien 2025

# Date       Sat Mar 29 - Sun Nov 30 2025

▪ Matchday 1
  Sat Nov 29 2025
    16:00  Stromsgodset IF         v Rosenborg BK             1-2 (0-1)
  Sun Mar 30
    14:30  Valerenga IF            v Viking FK                3-1 (1-0)
"""
    matches = parse(text, competition="DOM-NOR", season="2025", default_country="NOR")
    assert {m.date.year for m in matches} == {2025}


# --- team resolution ---------------------------------------------------------

def test_club_type_prefixes_are_stripped():
    assert normalise("RC Lens") == normalise("Lens")
    assert normalise("FC Bayern Munchen") == normalise("Bayern Munchen")
    assert normalise("GNK Dinamo Zagreb") == normalise("Dinamo Zagreb")


def test_dotless_i_folds_to_i():
    assert normalise("Sumgayıt") == normalise("Sumgayit")


def test_registry_resolves_variants_and_refuses_strangers():
    registry = TeamRegistry()
    registry.add("FC Bayern Munchen (GER)", "GER")
    registry.add("RC Lens (FRA)", "FRA")
    assert registry.resolve("Bayern Munich") == registry.resolve("FC Bayern Munchen")
    assert registry.resolve("Lens") == registry.resolve("RC Lens")
    # An unknown club must return None, never a fuzzy neighbour.
    assert registry.resolve("Real Madrid") is None
    assert "Real Madrid" in registry.unresolved


def test_distinct_clubs_are_not_merged():
    registry = TeamRegistry()
    registry.add("Hibernian FC", "SCO")
    registry.add("Hibernians FC", "MLT")
    assert registry.resolve("Hibernian FC") != registry.resolve("Hibernians FC")


# --- score matrix ------------------------------------------------------------

def test_matrix_sums_to_one_and_1x2_partitions():
    m = M.poisson_matrix(1.6, 1.1)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)
    result = M.result_1x2(m)
    assert sum(result.as_tuple()) == pytest.approx(1.0, abs=1e-6)


def test_over_under_is_line_specific():
    """Every totals probability is produced together with its line."""
    m = M.poisson_matrix(1.5, 1.2)
    over_25, under_25 = M.over_under(m, 2.5)
    over_55, _ = M.over_under(m, 5.5)
    assert over_25 + under_25 == pytest.approx(1.0, abs=1e-9)
    assert over_55 < over_25          # a higher line is strictly less likely
    assert over_55 < 0.15


def test_whole_line_leaves_push_mass():
    m = M.poisson_matrix(1.4, 1.3)
    over, under = M.over_under(m, 3.0)
    assert over + under < 1.0         # exactly three goals pushes


def test_dixon_coles_moves_low_scores_only():
    lam_h, lam_a = 1.3, 1.1
    base = M.poisson_matrix(lam_h, lam_a)
    adjusted = M.apply_dixon_coles(base, lam_h, lam_a, -0.05)
    assert not np.allclose(base[:2, :2], adjusted[:2, :2])
    assert np.allclose(base[4:, 4:] / base[4:, 4:].sum(),
                       adjusted[4:, 4:] / adjusted[4:, 4:].sum(), atol=1e-6)


# --- ledger ------------------------------------------------------------------

def test_science_arm_bets_every_fixture(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    for i in range(5):
        bet = ledger.place(arm="test", regime=SCIENCE, match_date="2026-09-10",
                           home=f"h{i}", away=f"a{i}", market="1x2",
                           selection="home", model_probability=0.4)
        assert bet is not None


def test_edge_gated_arm_declines_without_edge(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    declined = ledger.place(arm="test", regime=EDGE_GATED, match_date="2026-09-10",
                            home="h", away="a", market="1x2", selection="home",
                            model_probability=0.50, market_probability=0.50,
                            decimal_odds=2.0)
    assert declined is None            # no edge, no bet - this is correct


def test_totals_without_a_line_cannot_be_settled(tmp_path):
    """The EPL lab's 45-point phantom edge came from a line-less totals match."""
    ledger = Ledger(tmp_path / "l.json")
    ledger.place(arm="test", regime=SCIENCE, match_date="2026-09-10", home="h",
                 away="a", market="totals", selection="over", line=None,
                 model_probability=0.6)
    ledger.settle_match(home="h", away="a", match_date="2026-09-10",
                        home_goals=3, away_goals=1)
    bet = ledger.arms["test:science"].bets[0]
    assert bet.status == "push"        # refused, not guessed


def test_totals_settle_against_their_own_line(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.place(arm="t", regime=SCIENCE, match_date="2026-09-10", home="h", away="a",
                 market="totals", selection="over", line=2.5, model_probability=0.6)
    ledger.place(arm="t", regime=SCIENCE, match_date="2026-09-10", home="h", away="a",
                 market="totals", selection="over", line=5.5, model_probability=0.1)
    ledger.settle_match(home="h", away="a", match_date="2026-09-10",
                        home_goals=3, away_goals=1)
    statuses = [b.status for b in ledger.arms["t:science"].bets]
    assert statuses == ["won", "lost"]  # 4 goals: over 2.5 wins, over 5.5 loses


def test_ledger_round_trips(tmp_path):
    path = tmp_path / "l.json"
    ledger = Ledger(path)
    ledger.place(arm="a", regime=SCIENCE, match_date="2026-09-10", home="h",
                 away="a", market="1x2", selection="home", model_probability=0.5)
    ledger.save()
    reloaded = Ledger.load(path)
    assert len(reloaded.arms["a:science"].bets) == 1
    assert reloaded.arms["a:science"].bankroll == ledger.arms["a:science"].bankroll


def test_rerunning_a_matchday_does_not_double_stake(tmp_path):
    """A CI retry or a manual dispatch must not stake the same fixture twice."""
    ledger = Ledger(tmp_path / "l.json")
    first = ledger.place(arm="a", regime=SCIENCE, match_date="2026-09-10",
                         home="h", away="a", market="1x2", selection="home",
                         model_probability=0.6)
    second = ledger.place(arm="a", regime=SCIENCE, match_date="2026-09-10",
                          home="h", away="a", market="1x2", selection="home",
                          model_probability=0.6)
    assert first is not None
    assert second is None
    assert len(ledger.arms["a:science"].bets) == 1


def test_different_totals_lines_are_not_duplicates(tmp_path):
    """Over 2.5 and Over 5.5 are different bets, not a repeat of one."""
    ledger = Ledger(tmp_path / "l.json")
    a = ledger.place(arm="a", regime=SCIENCE, match_date="2026-09-10", home="h",
                     away="a", market="totals", selection="over", line=2.5,
                     model_probability=0.6)
    b = ledger.place(arm="a", regime=SCIENCE, match_date="2026-09-10", home="h",
                     away="a", market="totals", selection="over", line=5.5,
                     model_probability=0.1)
    assert a is not None and b is not None


# --- telegram formatting -----------------------------------------------------

def test_markdownv2_reserved_characters_are_escaped():
    """Telegram rejects the whole message with HTTP 400 on an unescaped '-'."""
    from ucl.report import telegram

    brief = {
        "as_of": "2026-09-10",
        "fixtures": [{
            "date": "2026-09-10", "kickoff": "21:00", "matchday": 1,
            "home": "h", "away": "a",
            "home_display": "Bayern", "away_display": "Bodo/Glimt",
            "arms": {"dixon_coles": {"home": 0.74, "draw": 0.14, "away": 0.12},
                     "gnn": {"home": 0.70, "draw": 0.16, "away": 0.14}},
            "market": {"home": 0.87, "draw": 0.08, "away": 0.05},
        }],
    }
    text = telegram.format_brief(brief)
    # Every '-' and '+' outside a code span must carry a backslash.
    for index, char in enumerate(text):
        if char in "-+" and not text.startswith("`", max(0, index - 1)):
            assert text[index - 1] == "\\", f"unescaped {char!r} at {index}: {text[max(0,index-25):index+10]!r}"


# --- in-play market contamination ------------------------------------------

def test_in_play_price_is_withheld_from_the_brief(monkeypatch, tmp_path):
    """A price seen at/after kickoff must never reach staking or scoring.

    The EPL lab booked a phantom 14.88% "edge" by diffing a fair value against
    an in-play price. Here the whole season scoreboard would be corrupted, since
    every arm is scored against entry["market"].
    """
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    from ucl import pipeline
    from ucl.data import corpus as corpus_module
    from ucl.data import odds as odds_module

    # run_matchday writes brief_<date>.json into PROCESSED; redirect it at a
    # tmp dir so the test never clobbers the committed archive.
    monkeypatch.setattr(pipeline, "PROCESSED", tmp_path)

    frame, registry = corpus_module.load()
    started = frame[(frame["competition"] == "UCL") & (~frame["played"])].head(2).copy()
    if len(started) < 2:
        import pytest
        pytest.skip("no upcoming UCL fixtures in the cached corpus")

    now = datetime.now(timezone.utc)

    class FakePrice:
        def __init__(self, home, away, minutes):
            self.home, self.away = home, away
            self.commence_time = now + timedelta(minutes=minutes)
            self.bookmaker_count = 30
            self.overround = 0.05

        def devigged(self):
            return {"Home": 0.5, "Draw": 0.3, "Away": 0.2}

    rows = list(started.itertuples(index=False))
    pre = FakePrice(rows[0].home, rows[0].away, minutes=120)     # not yet kicked off
    live = FakePrice(rows[1].home, rows[1].away, minutes=-30)    # in progress

    monkeypatch.setattr(odds_module, "fetch_odds",
                        lambda *a, **k: ([pre, live], 999))
    monkeypatch.setattr(pipeline, "match_market",
                        lambda prices, reg: {(p.home, p.away): p for p in prices})
    monkeypatch.setattr(pipeline, "_map_market_outcomes",
                        lambda dv, price, reg, h, a: {"home": 0.5, "draw": 0.3, "away": 0.2})

    brief = pipeline.run_matchday(as_of=now.date(), horizon_days=30,
                                  simulations=200, fast=True, place_bets=False)

    by_pair = {(f["home"], f["away"]): f for f in brief.fixtures}
    pre_entry = by_pair.get((rows[0].home, rows[0].away))
    live_entry = by_pair.get((rows[1].home, rows[1].away))
    assert pre_entry is not None and pre_entry["market"] is not None
    assert live_entry is not None
    assert live_entry["market"] is None
    assert live_entry["market_in_play_withheld"] is True
