"""
Проверка конфаундера новостей (research/news_confound.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import news_confound as nc                 # noqa: E402
from research import news_event_study as ns              # noqa: E402


class TestCarry(unittest.TestCase):

    def test_tariff(self):
        self.assertAlmostEqual(nc.carry_pct(10_000, 1), 0.40)       # 40 ₽ в день
        self.assertAlmostEqual(nc.carry_pct(10_000, 3), 1.20)       # выходные — календарные дни
        self.assertEqual(nc.carry_pct(5_000, 1), 0.0)               # до 5 000 ₽ бесплатно
        self.assertAlmostEqual(nc.carry_pct(200_000, 1), 0.095)
        with self.assertRaises(ValueError):
            nc.carry_pct(5_000_000, 1)


class TestSources(unittest.TestCase):

    def test_hashtag_vs_name(self):
        self.assertEqual(ns.ticker_sources("#SBER отчёт"), {"SBER": {"hashtag"}})
        self.assertEqual(ns.ticker_sources("Сбербанк отчитался"), {"SBER": {"name"}})
        self.assertEqual(ns.ticker_sources("#SBER Сбербанк"), {"SBER": {"hashtag", "name"}})
        self.assertEqual(ns.ticker_sources("#SNGS"), {"SNGS": {"hashtag"}, "SNGSP": {"hashtag"}})
        self.assertEqual(ns.tickers_in("#YNDX и ВТБ"), {"YDEX", "VTBR"})


class TestFeOls(unittest.TestCase):

    def test_equals_ols_with_date_dummies(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(30):
            a = rng.normal(0, 2)                                    # шок даты
            for k in range(8):
                news = float(rng.random() < 0.3)
                x = rng.normal()
                rows.append({"date": i, "News": news, "x": x,
                             "y": a + 0.5 * news - 0.3 * x + rng.normal(0, 0.5)})
        df = pd.DataFrame(rows)
        r = nc.fe_cluster_ols(df, "y", ["News", "x"])
        D = pd.get_dummies(df["date"]).to_numpy(float)
        X = np.column_stack([df["News"], df["x"], D])
        beta = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)[0]
        self.assertAlmostEqual(r["coef"]["News"]["b"], beta[0], places=8)
        self.assertAlmostEqual(r["coef"]["x"]["b"], beta[1], places=8)
        self.assertEqual((r["n"], r["clusters"]), (240, 30))
        self.assertTrue(np.isfinite(r["coef"]["News"]["t"]))

    def test_too_few_clusters(self):
        df = pd.DataFrame({"date": [1, 1, 2, 2], "y": [1.0, 2, 3, 4], "News": [0.0, 1, 0, 1]})
        self.assertNotIn("coef", nc.fe_cluster_ols(df, "y", ["News"]))


D1, D2, D3, D4 = (dt.date(2025, 3, 3), dt.date(2025, 3, 4), dt.date(2025, 3, 5),
                  dt.date(2025, 3, 12))


def agg_frame():
    rows = []
    for d, px in ((D1, 1.00), (D2, 1.02), (D3, 1.01), (D4, 1.05)):
        for tk, m in (("T1", 100), ("T2", 50), ("T3", 10), ("PENNY", 0.006)):
            rows.append({"ticker": tk, "d": d, "o_main": m * px, "c_eve": m * px * 1.01,
                         "vol_day": 1000.0})
    return pd.DataFrame(rows)


def ev(tk, when, src, report=False):
    return {"ticker": tk, "avail": when, "sources": {src}, "price_report": report}


class TestPanel(unittest.TestCase):

    def setUp(self):
        events = [ev("T1", dt.datetime(2025, 3, 3, 12, 0), "hashtag"),
                  ev("T2", dt.datetime(2025, 3, 3, 12, 0), "name"),
                  ev("T3", dt.datetime(2025, 3, 3, 12, 0), "hashtag", report=True),
                  ev("T3", dt.datetime(2025, 3, 3, 20, 0), "hashtag")]
        self.p = nc.build_panel(agg_frame(), events, min_tickers=2)
        self.by = {(r.ticker, r.date): r for r in self.p.itertuples()}

    def test_targets_and_gap(self):
        r = self.by[("T1", D1)]
        self.assertAlmostEqual(r.raw_gap, (1.02 / 1.01 - 1) * 100)
        self.assertNotIn(("T1", D3), self.by)                       # D3 → D4: 7 дней, NaN
        self.assertEqual(r.nights, 1)
        # leave-one-out: у всех бумаг одинаковый ход → аномальный 0
        self.assertAlmostEqual(r.ar_gap, 0.0, places=9)

    def test_penny_stock_excluded(self):
        self.assertNotIn("PENNY", set(self.p["ticker"]))

    def test_cohorts_and_hold(self):
        self.assertTrue(self.by[("T1", D1)].k1)
        self.assertFalse(self.by[("T1", D1)].k2)
        self.assertTrue(self.by[("T2", D1)].k2)
        self.assertFalse(self.by[("T3", D1)].news)                  # отчёт о цене не событие
        self.assertTrue(self.by[("T3", D1)].hold)                   # 20:00 — во время удержания
        self.assertTrue(self.by[("T3", D2)].k1)                     # и известна к решению D2

    def test_cohort_sample_excludes_other_cohort(self):
        s = nc.cohort_sample(self.p, "k1")
        self.assertNotIn(("T2", D1), set(zip(s["ticker"], s["date"])))
        self.assertEqual(s["News"].sum(), self.p["k1"].sum())


class TestVerdict(unittest.TestCase):

    def m(self, b, t):
        return {"coef": {"News": {"b": b, "t": t}}}

    def test_rules(self):
        halves = {"a": self.m(-0.1, -2.0), "b": self.m(-0.2, -2.5)}
        go = nc.c4_verdict(self.m(-0.15, -3.5), halves, {"net_day": 0.05})
        self.assertTrue(go["verdict"].startswith("ПРОДОЛЖАТЬ"))
        weak = nc.c4_verdict(self.m(-0.15, -1.5), halves, {"net_day": 0.05})
        self.assertTrue(weak["verdict"].startswith("ЗАКРЫТЬ"))
        flip = nc.c4_verdict(self.m(-0.15, -3.5), {"a": self.m(0.1, 1), "b": self.m(-0.2, -3)},
                             {"net_day": 0.05})
        self.assertTrue(flip["verdict"].startswith("ЗАКРЫТЬ"))
        costly = nc.c4_verdict(self.m(-0.15, -3.5), halves, {"net_day": -0.3})
        self.assertTrue(costly["verdict"].startswith("НЕ ПРОДОЛЖАТЬ"))
        mid = nc.c4_verdict(self.m(-0.15, -2.5), halves, {"net_day": 0.05})
        self.assertTrue(mid["verdict"].startswith("НЕ ПРОДОЛЖАТЬ"))  # t между −3 и −2

    def test_short_economics_carry_and_blacklist(self):
        p = pd.DataFrame({"date": [D1, D1, D2], "ticker": ["T1", "AKRN", "T1"],
                          "k1": [True, True, True], "raw_gap": [-1.0, -5.0, -1.0],
                          "nights": [1, 1, 3]})
        e = nc.short_economics(p, "k1", 10_000)
        self.assertEqual(e["n"], 2)                                  # AKRN не шортуется
        self.assertAlmostEqual(e["net_day"], ((1 - 0.128 - 0.4) + (1 - 0.128 - 1.2)) / 2)


if __name__ == "__main__":
    unittest.main()
