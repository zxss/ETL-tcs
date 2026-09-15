"""
Спринт 2, товарный lead-lag (research/sprint2_leadlag.py). Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402

FRI, MON, TUE = dt.date(2025, 3, 7), dt.date(2025, 3, 10), dt.date(2025, 3, 11)
SAT = dt.date(2025, 3, 8)


def bars(spec):
    """spec: {день: [(время, open, close), ...]}"""
    tm, o, c = [], [], []
    for d, xs in spec.items():
        for t, oo, cc in xs:
            tm.append(dt.datetime.combine(d, t)); o.append(oo); c.append(cc)
    return es.Bars(tm, o, c)


def T(h, m=0):
    return dt.time(h, m)


FUT = bars({FRI: [(T(9), 80, 80), (T(23, 45), 80, 80)],
            SAT: [(T(10), 90, 90)],                                  # выходная сессия не в счёт
            MON: [(T(9), 80.5, 80.8), (T(9, 45), 80.8, 81.0), (T(10), 81.0, 81.0), (T(18, 25), 81.5, 82.0)]})
STOCK = bars({MON: [(T(9, 55), 99, 99), (T(10), 100, 100), (T(10, 55), 100, 101), (T(11, 55), 101, 102),
                    (T(18, 25), 102, 100), (T(18, 30), 100, 100.5)],
              TUE: [(T(6, 55), 103, 103), (T(10), 104, 104)]})
IDX = bars({MON: [(T(10), 1000, 1000), (T(10, 55), 1000, 1005), (T(11, 55), 1005, 1000)]})
SIG_2A = {"to_bar": "09:45", "abs_threshold_pct": 1.0}


class TestFront(unittest.TestCase):

    def test_roll_two_business_days_before_expiration(self):
        until = ll.eligible_until({"BRJ5": dt.date(2025, 3, 12), "CNY": None}, 2)   # экспирация в среду
        self.assertEqual(until["BRJ5"], FRI)          # Пн и Вт уже следующий контракт
        self.assertIsNone(until["CNY"])


class Test2A(unittest.TestCase):

    def test_signal_from_previous_weekday_evening_close(self):
        s = ll.signal_2a(FUT, MON, SIG_2A)
        self.assertEqual(s["prev_day"], FRI)          # не суббота
        self.assertAlmostEqual(s["chg"], 1.25)
        self.assertEqual(s["dir"], 1)
        self.assertEqual(ll.signal_2a(FUT, MON, {**SIG_2A, "abs_threshold_pct": 2.0})["dir"], 0)

    def test_no_morning_futures_no_signal(self):
        self.assertIsNone(ll.signal_2a(FUT, TUE, SIG_2A))

    def test_intraday_outcome_and_rows(self):
        o = ll.outcome_intraday(STOCK, IDX, MON, T(10), T(10, 55))
        self.assertAlmostEqual(o["move"], 1.0)
        self.assertAlmostEqual(o["ar"], 0.5)
        h = {"family": "2A", "stocks": ["LKOH"], "signal": SIG_2A, "entry_time": "10:00", "exit_bar": "11:55"}
        rows, cnt = ll.rows_2a(h, lambda d: FUT, {"LKOH": STOCK}, IDX, [MON], cost=lambda tk: 0.2)
        self.assertEqual(cnt["signal_days"], 1)
        self.assertAlmostEqual(rows[0]["signed_ar"], 2.0)      # 2 % бумаги при плоском индексе
        self.assertAlmostEqual(rows[0]["net"], 1.8)


class Test2B(unittest.TestCase):

    def test_signal_and_overnight(self):
        self.assertAlmostEqual(ll.signal_2b(FUT, MON, {"from_time": "10:00", "to_bar": "18:25"}), 100 * (82 / 81 - 1))
        o = ll.outcome_overnight(STOCK, MON, TUE, T(18, 25), [(TUE, 1.0)])
        self.assertAlmostEqual(o["move"], 4.0)                  # (103 + 1) / 100
        self.assertAlmostEqual(o["move_1835"], 100 * (104 / 100.5 - 1))

    def test_rows_and_summary_gate(self):
        h = {"family": "2B", "stocks": ["PLZL"], "signal": {"from_time": "10:00", "to_bar": "18:25"},
             "entry_bar": "18:25"}
        rows, cnt = ll.rows_2b(h, lambda d: FUT, {"PLZL": STOCK}, [MON], [FRI, MON, TUE], {},
                               cost=lambda tk: 0.3, hurdle=lambda a, b: 0.05, carry=lambda n: 0.45)
        self.assertEqual(cnt["up_days"], 1)
        self.assertAlmostEqual(rows[0]["net_long"], 3.0 - 0.35)
        self.assertAlmostEqual(rows[0]["net_short"], -3.0 - 0.75)


class TestStats(unittest.TestCase):

    def test_holm(self):
        adj = ll.holm({"a": 0.01, "b": 0.04, "c": None})
        self.assertAlmostEqual(adj["a"], 0.02)
        self.assertAlmostEqual(adj["b"], 0.04)

    def test_gate(self):
        g = {"t_min": 2.0, "net_min": 0.0, "dates_min": 30}
        self.assertTrue(ll.gate_ok({"t": 2.1, "dates": 40}, {"mean": 0.1}, g))
        self.assertFalse(ll.gate_ok({"t": 2.1, "dates": 20}, {"mean": 0.1}, g))
        self.assertFalse(ll.gate_ok({"t": 2.1, "dates": 40}, {"mean": -0.1}, g))

    def test_frozen_rules(self):
        with open(ll.RULES_PATH, encoding="utf-8") as f:
            r = json.load(f)
        ids = [h["id"] for h in r["hypotheses"]]
        self.assertEqual(ids, ["2A_11", "2A_12", "2B_oil", "2B_gold"])
        self.assertEqual((r["samples"]["holdout"]["runs"], r["futures"]["roll_business_days_before_expiration"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
