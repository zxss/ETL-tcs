"""
Event study новостей (research/event_study_news.py). Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import news_classify as ncl                # noqa: E402

DAYS = [dt.date(2025, 3, 3), dt.date(2025, 3, 4), dt.date(2025, 3, 5), dt.date(2025, 3, 6)]


def make_bars(price_of):
    """Бары 10:00–18:45 и 19:05–20:00 по DAYS; price_of(day_idx, time) → (open, close)."""
    tm, o, c = [], [], []
    for k, d in enumerate(DAYS):
        for start, end in ((dt.time(10, 0), dt.time(18, 45)), (dt.time(19, 5), dt.time(20, 0))):
            t = dt.datetime.combine(d, start)
            while t.time() <= end:
                oo, cc = price_of(k, t.time())
                tm.append(t); o.append(oo); c.append(cc)
                t += dt.timedelta(minutes=5)
    return es.Bars(tm, o, c)


def stock_price(k, t):
    base = 100.0 + k
    if k == 0 and t == dt.time(11, 5):
        return 100.0, 101.0
    if k == 0 and t == dt.time(11, 30):
        return 101.0, 102.0
    if t == dt.time(18, 45):
        return base, 103.0 + k                              # close основной сессии: 103, 104, 105, 106
    if k == 0 and dt.time(11, 30) < t < dt.time(18, 45):
        return 102.0, 102.0
    return base, base


IDX = make_bars(lambda k, t: (1000.0, 1000.0))
STOCK = make_bars(stock_price)


class TestOutcome(unittest.TestCase):

    def test_windows_during_session(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 11, 5))          # первый бар после публикации
        got = [round(o[f"raw_{w}"], 6) for w in es.WINDOWS]
        self.assertEqual(got, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(o["ar_2d"], o["raw_2d"])                          # индекс стоит

    def test_evening_news_counts_from_next_day(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 19, 7), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 19, 10))
        self.assertEqual((o["end_eod"], o["end_1d"], o["end_2d"]), (DAYS[1], DAYS[2], DAYS[3]))

    def test_no_bar_within_30_minutes(self):
        self.assertIsNone(es.outcome(dt.datetime(2025, 3, 3, 23, 55), es.LAG_REACTION, STOCK, IDX, DAYS))

    def test_trade_lag(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_TRADE, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 11, 15))

    def test_abnormal_subtracts_index(self):
        idx = make_bars(lambda k, t: (1000.0, 1000.0 + (10.0 if t == dt.time(18, 45) else 0.0)))
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, idx, DAYS)
        self.assertAlmostEqual(o["ar_eod"], 3.0 - 1.0)


class TestEconomics(unittest.TestCase):

    def test_long_pays_cost_and_fund_short_pays_carry(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("fund,date,close\nTMON@,2025-01-01,99\nTMON@,2025-03-03,100\nTMON@,2025-03-05,100.1\n")
        h = es.Hurdle(f.name)
        os.unlink(f.name)
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS)
        e = es.economics(o, "ZZZZ", h, {"__fallback__": (0.05, 0.1, 0.0)})
        self.assertAlmostEqual(e["cost"], 0.13)
        self.assertAlmostEqual(e["net_long_2d"], 5.0 - 0.13 - 0.1, places=9)
        self.assertAlmostEqual(e["net_short_2d"], -5.0 - 0.13 - 2 * 0.45, places=9)
        self.assertAlmostEqual(e["net_long_30m"], 2.0 - 0.13, places=9)   # внутри дня фонд не дорожает
        self.assertAlmostEqual(e["net_short_30m"], -2.0 - 0.13, places=9) # шорт внутри дня без переноса

    def test_hurdle_falls_back_to_lqdt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("fund,date,close\nLQDT,2022-01-01,1.0\nLQDT,2022-01-11,1.01\nTMON@,2025-02-25,128\n")
        h = es.Hurdle(f.name)
        os.unlink(f.name)
        self.assertAlmostEqual(h.growth(dt.date(2022, 1, 1), dt.date(2022, 1, 11)), 1.0)
        self.assertEqual(h.growth(dt.date(2022, 1, 5), dt.date(2022, 1, 5)), 0.0)


class TestEvents(unittest.TestCase):

    def test_dedupe_and_filters(self):
        posts = pd.DataFrame({
            "message_id": [1, 2, 3],
            "msk": pd.to_datetime(["2025-03-03 11:00", "2025-03-03 12:00", "2025-03-03 12:30"]),
            "text": ["❗️🇷🇺#PLZL #дивиденд\nСД ПОЛЮС: ДИВИДЕНДЫ = 730 РУБ/АКЦ",
                     "❗️🇷🇺#PLZL #дивиденд\nГОСА ПОЛЮС одобрило дивиденды",
                     "⚠️🇷🇺#GAZP = мин за 5 мес"]})
        ev = es.build_events(posts, ncl.Classifier())
        self.assertEqual(list(ev["message_id"]), [1])                     # повтор и отчёт о цене отброшены
        self.assertEqual((ev.iloc[0]["category"], ev.iloc[0]["bucket"]), ("DIVIDEND", "pos"))


class TestRules(unittest.TestCase):

    def synthetic(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(60):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=i)
            for cat in ncl.CATEGORIES:
                for b, s in (("neg", -0.5), ("pos", 0.5)):
                    row = {"date": d, "category": cat, "bucket": b, "sentiment": 1.0 if b == "pos" else s}
                    for w in es.WINDOWS:
                        eff = 1.0 if (cat == "DIVIDEND" and b == "pos" and w == "1d") else 0.0
                        row[f"ar_{w}"] = eff + rng.normal(0, 0.5)
                        row[f"net_long_{w}"] = eff - 0.3 + rng.normal(0, 0.5)
                        row[f"net_short_{w}"] = -eff - 0.3 + rng.normal(0, 0.5)
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_selection_and_tz_rule(self):
        rules, trials = es.select_rules(self.synthetic())
        self.assertEqual(trials, len(ncl.CATEGORIES) * 2 * len(es.RULE_WINDOWS) + 1)
        picked = [(r["category"], r.get("bucket"), r["window"], r["direction"]) for r in rules
                  if r["source"] == "отбор на dev"]
        self.assertIn(("DIVIDEND", "pos", "1d", 1), picked)
        self.assertEqual(len(picked), 1)
        tz = [r for r in rules if r["source"].startswith("ТЗ")]
        self.assertEqual((tz[0]["window"], tz[0]["direction"]), ("2d", 1))

    def test_daily_series_ir_dsr(self):
        ev = self.synthetic()
        days = sorted(ev["date"].unique())
        rule = {"category": "DIVIDEND", "bucket": "pos", "window": "1d", "direction": 1}
        s = es.rule_daily_series(ev, rule, days)
        self.assertEqual(len(s), len(days))
        self.assertGreater(es.information_ratio(s), 0)
        d1, d100 = es.deflated_sharpe(s, 2), es.deflated_sharpe(s, 100)
        self.assertTrue(0 <= d100 <= d1 <= 1)

    def test_trials_registry_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "trials.jsonl")
            self.assertEqual(es.register_trials("dev", 41, "x", p), 41)
            self.assertEqual(es.register_trials("holdout", 0, "x", p), 41)
            self.assertEqual(es.register_trials("dev", 9, "y", p), 50)

    def test_bucket(self):
        self.assertEqual([es.bucket(x) for x in (-1, -0.5, -0.2, 0, 0.4, 0.5, 1)],
                         ["neg", "neg", "neu", "neu", "neu", "pos", "pos"])


if __name__ == "__main__":
    unittest.main()
