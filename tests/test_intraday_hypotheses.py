"""
Внутридневные гипотезы H1–H4 (research/intraday_hypotheses.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import intraday_hypotheses as ih           # noqa: E402

D = dt.date(2025, 3, 4)


def bars_of(tk, d, rows):
    """rows: [(HH:MM, open, high, low, close, volume)] → DataFrame баров."""
    out = []
    for hhmm, o, h, lo, c, v in rows:
        tm = pd.Timestamp(dt.datetime.combine(d, dt.datetime.strptime(hhmm, "%H:%M").time()))
        out.append({"ticker": tk, "tm": tm, "d": d, "open": o, "high": h, "low": lo,
                    "close": c, "volume": v})
    return pd.DataFrame(out)


def flat(tk, d, px, start="10:00", end="18:15", vol=100.0):
    t = dt.datetime.combine(d, dt.datetime.strptime(start, "%H:%M").time())
    stop = dt.datetime.combine(d, dt.datetime.strptime(end, "%H:%M").time())
    rows = []
    while t <= stop:
        rows.append((t.strftime("%H:%M"), px, px, px, px, vol))
        t += dt.timedelta(minutes=5)
    return bars_of(tk, d, rows)


class TestH1(unittest.TestCase):

    def panic(self, idx_drop=False):
        s = flat("SBER", D, 100.0)
        # 12:05–12:30 пролив до 97 (−3 % за 30 минут), бар 12:30 — объём ×10
        # (до сигнала ≥ 20 баров — для среднего объёма Vol20)
        for i, hhmm in enumerate(("12:05", "12:10", "12:15", "12:20", "12:25", "12:30")):
            px = 100.0 - 0.5 * (i + 1)
            s.loc[s["tm"].dt.strftime("%H:%M") == hhmm, ["open", "high", "low", "close"]] = px
        s.loc[s["tm"].dt.strftime("%H:%M") == "12:30", "volume"] = 1000.0
        # 12:35 вход по 97,2; дальше цена возвращается к 99,9 — выше VWAP
        s.loc[s["tm"].dt.strftime("%H:%M") == "12:35", ["open", "high", "low", "close"]] = [97.2, 97.5, 97.0, 97.4]
        after = s["tm"].dt.strftime("%H:%M") >= "12:40"
        s.loc[after, ["open", "high", "low", "close"]] = [97.5, 99.9, 97.4, 99.8]
        idx = flat(ih.INDEX, D, 3000.0)
        if idx_drop:
            idx.loc[idx["tm"].dt.strftime("%H:%M") == "12:30", ["open", "high", "low", "close"]] = 2970.0
        return pd.concat([s, idx], ignore_index=True)

    def test_panic_long_exits_at_vwap(self):
        bars = self.panic()
        tr = ih.h1_trades(bars, ih._series(bars, ih.INDEX), {D: {"SBER"}})
        self.assertEqual(len(tr), 1)
        t = tr[0]
        self.assertEqual(t["t_in"].strftime("%H:%M"), "12:35")         # следующий бар после сигнала
        self.assertEqual(t["exit"], "vwap")
        self.assertGreater(t["gross"], 0)
        self.assertEqual(t["dir"], 1)

    def test_market_crash_blocks_signal(self):
        bars = self.panic(idx_drop=True)
        self.assertEqual(ih.h1_trades(bars, ih._series(bars, ih.INDEX), {D: {"SBER"}}), [])

    def test_only_top15(self):
        bars = self.panic()
        self.assertEqual(ih.h1_trades(bars, ih._series(bars, ih.INDEX), {D: {"GAZP"}}), [])


class TestH2(unittest.TestCase):

    def setUp(self):
        d0 = D - dt.timedelta(days=1)
        self.days = [d0, D]
        s = pd.concat([flat("SBER", d0, 100.0), flat("SBER", D, 102.0)], ignore_index=True)
        # после гэпа +2 % к 11:15 цена возвращается к 101
        late = (s["d"] == D) & (s["tm"].dt.strftime("%H:%M") >= "10:30")
        s.loc[late, ["open", "high", "low", "close"]] = 101.0
        a = pd.concat([flat("AKRN", d0, 100.0), flat("AKRN", D, 102.0)], ignore_index=True)
        idx = pd.concat([flat(ih.INDEX, d0, 3000.0), flat(ih.INDEX, D, 3000.0)], ignore_index=True)
        self.bars = pd.concat([s, a, idx], ignore_index=True)
        self.pc = {("SBER", D): 100.0, ("AKRN", D): 100.0}

    def run_h2(self, news):
        return ih.h2_trades(self.bars, ih._series(self.bars, ih.INDEX), self.pc, self.days,
                            news, {"AKRN"})

    def test_gap_up_short_without_news(self):
        tr = self.run_h2({})
        self.assertEqual([t["ticker"] for t in tr], ["SBER"])          # AKRN не шортуется
        t = tr[0]
        self.assertEqual((t["dir"], t["t_in"].strftime("%H:%M"), t["t_out"].strftime("%H:%M")),
                         (-1, "10:05", "11:15"))
        self.assertAlmostEqual(t["gross"], -(101.0 / 102.0 - 1) * 100)

    def test_news_blocks(self):
        news = {"SBER": [dt.datetime.combine(D, dt.time(8, 0))]}
        self.assertEqual(self.run_h2(news), [])
        stale = {"SBER": [dt.datetime.combine(D - dt.timedelta(days=1), dt.time(12, 0))]}
        self.assertEqual(len(self.run_h2(stale)), 1)                  # до 18:50 вчера — не считается

    def test_has_news_bounds(self):
        ts = [dt.datetime(2025, 3, 4, 9, 0)]
        self.assertTrue(ih.has_news(ts, dt.datetime(2025, 3, 3, 18, 50), dt.datetime(2025, 3, 4, 10, 5)))
        self.assertFalse(ih.has_news(ts, dt.datetime(2025, 3, 4, 9, 0), dt.datetime(2025, 3, 4, 10, 5)))


class TestH3(unittest.TestCase):

    def frame(self, z):
        tms = pd.date_range(dt.datetime.combine(D, dt.time(10, 0)), periods=len(z), freq="5min")
        return pd.DataFrame({"open_a": np.linspace(100, 101, len(z)), "close_a": 100.0,
                             "open_b": 50.0, "close_b": 50.0, "z": z, "d": D}, index=tms)

    def test_rich_leg_shorted_exit_on_zero(self):
        m = self.frame([0.5, 2.3, 1.5, -0.1, 0.2, 0.3])
        tr = ih.h3_pair_trades(m, "LKOH", "ROSN", pd.DataFrame(), set())
        self.assertEqual(len(tr), 1)
        t = tr[0]
        self.assertEqual((t["exit"], t["legs"], t["dir"]), ("zero", 2, 0))
        self.assertEqual(t["t_in"], m.index[2])
        self.assertEqual(t["t_out"], m.index[4])                       # open бара после пересечения
        ea, xa = m["open_a"].iloc[2], m["open_a"].iloc[4]
        self.assertAlmostEqual(t["gross"], 0.0 - (xa / ea - 1) * 100)  # шорт a, лонг b (b стоит)

    def test_stop_and_blocked(self):
        m = self.frame([2.1, 2.5, 3.6, 3.0, 2.0])
        self.assertEqual(ih.h3_pair_trades(m, "LKOH", "ROSN", pd.DataFrame(), set())[0]["exit"], "stop")
        self.assertEqual(ih.h3_pair_trades(m, "LKOH", "ROSN", pd.DataFrame(), {"LKOH"}), [])

    def test_pair_frame_z_uses_past_only(self):
        n = 400
        tms = pd.date_range(dt.datetime.combine(D, dt.time(10, 0)), periods=n, freq="5min")
        a = pd.DataFrame({"ticker": "A", "tm": tms, "d": [t.date() for t in tms], "open": 100.0,
                          "high": 100.0, "low": 100.0, "close": 100 + np.sin(np.arange(n)), "volume": 1.0})
        b = a.assign(ticker="B", close=100.0)
        m = ih.pair_frame(pd.concat([a, b]), "A", "B")
        self.assertTrue(np.isnan(m["z"].iloc[ih.H3_MIN - 1]))
        self.assertTrue(np.isfinite(m["z"].iloc[ih.H3_MIN + 1]))


class TestH4(unittest.TestCase):

    def test_breakdown_shorts_three_weakest(self):
        days = [dt.date(2025, 3, 3) + dt.timedelta(days=i) for i in range(16) if
                (dt.date(2025, 3, 3) + dt.timedelta(days=i)).weekday() < 5]
        frames, last = [], []
        for k, d in enumerate(days):
            sig = d == days[-1]
            ix = 2950.0 if sig else 3000.0
            frames.append(bars_of(ih.INDEX, d, [("10:00", 3000, 3000, 2990, 3000, 0),
                                                ("13:55", ix, ix, ix, ix, 0),
                                                ("14:00", ix, ix, ix, ix, 0),
                                                ("18:15", ix, ix, ix, ix, 0)]))
            last.append({"ticker": ih.INDEX, "d": d, "last_close": 3000.0, "day_low": 2990.0})
            for j, tk in enumerate(("SBER", "GAZP", "LKOH", "ROSN", "AKRN")):
                px = 100.0 - (j + 1) * (1.0 if sig else 0.0)
                vol = 10_000.0 if sig else 100.0
                frames.append(bars_of(tk, d, [("10:00", 100, 100, 100, 100, vol),
                                              ("13:55", px, px, px, px, vol),
                                              ("14:00", px, px, px, px, vol),
                                              ("18:15", px - 1, px, px - 1, px - 1, vol)]))
                last.append({"ticker": tk, "d": d, "last_close": 100.0, "day_low": 99.0})
        bars = pd.concat(frames, ignore_index=True)
        last = pd.DataFrame(last)
        tr, log_days = ih.h4_trades(bars, ih._series(bars, ih.INDEX), last,
                                    ih.prev_close_map(last), days, {}, {"AKRN"})
        self.assertEqual(sorted(t["ticker"] for t in tr), ["GAZP", "LKOH", "ROSN"])  # AKRN самая слабая, но не шортуется
        self.assertTrue(all(t["gross"] > 0 for t in tr))
        self.assertEqual(sum(1 for r in log_days if r["signal"]), 1)


class TestStats(unittest.TestCase):

    def test_pairs_pay_two_legs_and_excess(self):
        tr = pd.DataFrame({"hyp": "H3", "date": [D, D + dt.timedelta(days=1)], "gross": [1.0, 1.0],
                           "legs": [2, 2], "dir": [0, 0], "idx_move": [0.5, -0.5], "status": "ok"})
        st = ih.stats(tr, 0.128)
        self.assertAlmostEqual(st["mean_trade"], 1.0 - 0.256)
        self.assertAlmostEqual(st["excess_idx"], 1.0)                 # нейтральная: индекс не вычитается
        lo = tr.assign(hyp="H2", legs=1, dir=1)
        self.assertAlmostEqual(ih.stats(lo, 0.128)["excess_idx"], 1.0)  # (1−0,5 + 1+0,5)/2

    def test_prev_close_map(self):
        last = pd.DataFrame({"ticker": "A", "d": [dt.date(2025, 3, 1), dt.date(2025, 3, 3)],
                             "last_close": [10.0, 11.0], "day_low": [9.0, 10.0]})
        m = ih.prev_close_map(last)
        self.assertEqual(m[("A", dt.date(2025, 3, 3))], 10.0)          # суббота — последняя цена до понедельника

    def test_rub_per_year(self):
        self.assertAlmostEqual(ih.rub_per_year({"n": 252, "mean_trade": 0.1}, 252), 2520.0)


if __name__ == "__main__":
    unittest.main()
