"""
Спринт 4: фильтры допуска intraday_short (research/sprint4_gatekeeper.py). Синтетика.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import sprint4_gatekeeper as g              # noqa: E402

RULES = g.load_rules()
DAYS = [d.date() for d in pd.bdate_range("2023-01-02", periods=80)]


class TestFilters(unittest.TestCase):

    def test_flags(self):
        c = pd.DataFrame({
            "ticker": ["LKOH", "LKOH", "GMKN", "SBER", "SBER"],
            "rs30": [-1.0, -1.0, -2.0, -1.0, 1.0],
            "below_ema50": [True, True, True, True, True],
            "cs_narrow": [True, False, np.nan, True, True],
            "brent_day": [-0.6, -0.4, -0.6, -0.6, -0.6],
            "cny_day": [0.1, 0.1, -0.2, -0.2, -0.2],
            "neg_news": [False, True, False, True, True]})
        f = g.apply_filters(c, RULES)
        self.assertEqual(list(f["F1"]), [True, True, True, True, False])
        self.assertEqual(list(f["F2"]), [True, False, False, True, False])      # NaN спреда — не допуск
        self.assertEqual(list(f["F3"]), [True, False, True, False, False])     # SBER — не экспортёр
        self.assertEqual(list(f["F4"]), [False, True, False, True, False])

    def test_news_window_before_entry(self):
        at = dt.datetime(2023, 3, 6, 10, 5)
        times = np.sort(np.array([np.datetime64(at - dt.timedelta(hours=47), "ns")]))
        self.assertTrue(g.had_news(times, at, 48))
        self.assertFalse(g.had_news(times, at - dt.timedelta(hours=48), 48))
        late = np.array([np.datetime64(at + dt.timedelta(minutes=1), "ns")])
        self.assertFalse(g.had_news(late, at, 48))
        ev = pd.DataFrame({"ticker": ["A", "A", "B"], "category": ["FINANCIAL", "CORPORATE", "SANCTIONS_MACRO"],
                           "sentiment": [-0.5, -1.0, 0.5], "posted": [at, at, at]})
        n = g.negative_news(ev, RULES["filters"]["F4"]["categories"])
        self.assertEqual(set(n), {"A"})
        self.assertEqual(len(n["A"]), 1)


class TestFeatures(unittest.TestCase):

    def frame(self):
        rng = np.random.default_rng(5)
        rows = []
        for tk, drift in (("IMOEX", 0.0), ("AAA", -0.003)):
            c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(DAYS))))
            for d, x in zip(DAYS, c):
                rows.append({"ticker": tk, "date": d, "open": x, "high": x * 1.01, "low": x * 0.99, "close": x,
                             "volume": 1000.0})
        return pd.DataFrame(rows)

    def test_split_and_no_lookahead(self):
        df = self.frame()
        m = (df["ticker"] == "AAA") & (df["date"] >= DAYS[50])
        df.loc[m, ["open", "high", "low", "close"]] /= 10.0
        adj = g.adjust_daily_splits(df)
        a = adj[adj["ticker"] == "AAA"].set_index("date")["close"]
        self.assertLess(abs(a[DAYS[50]] / a[DAYS[49]] - 1), 0.05)
        f0 = g.filter_features(adj).set_index(["ticker", "date"])
        adj2 = adj.copy()
        adj2.loc[adj2["date"] >= DAYS[70], ["open", "high", "low", "close"]] *= 1.3
        f1 = g.filter_features(adj2).set_index(["ticker", "date"])
        early = [("AAA", d) for d in DAYS[:70]]
        pd.testing.assert_frame_equal(f0.loc[early], f1.loc[early])
        self.assertTrue(np.isnan(f0.loc[("AAA", DAYS[10]), "below_ema50"]))    # меньше 50 баров

    def test_acceptance(self):
        acc = RULES["acceptance"]
        self.assertTrue(g.accepted({"trades": 50, "net": 0.41, "win": 0.54}, acc))
        self.assertFalse(g.accepted({"trades": 50, "net": 0.41, "win": 0.52}, acc))
        self.assertFalse(g.accepted({"trades": 0}, acc))

    def test_rules_frozen(self):
        self.assertEqual(sorted(RULES["filters"]), ["F1", "F2", "F3", "F4"])
        self.assertEqual((RULES["trials"], RULES["acceptance"]["net_min_pct"]), (4, 0.40))


if __name__ == "__main__":
    unittest.main()
