"""
Тесты идей повышения доходности (research/boost_tests.py). Синтетика, без сети и БД.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import boost_tests as bt                   # noqa: E402


class TestShortExit(unittest.TestCase):

    def test_time_exit(self):
        px, why = bt.short_exit([100, 100], [100.5, 100.5], [99.5, 99.5], [100, 99], stop_pct=1.0)
        self.assertEqual((px, why), (99.0, "time"))

    def test_stop_gap_through(self):
        px, why = bt.short_exit([100, 102], [100.5, 102.5], [99.5, 101.5], [100, 102], stop_pct=1.0)
        self.assertEqual((px, why), (102.0, "stop"))              # open бара выше стопа — по open

    def test_take_profit_and_both_in_bar(self):
        px, why = bt.short_exit([100, 99], [100.2, 99.2], [99.5, 96.0], [100, 97], stop_pct=2.0, tp_pct=2.5)
        self.assertEqual((px, why), (97.5, "tp"))
        px, why = bt.short_exit([100], [102.5], [97.0], [100], stop_pct=2.0, tp_pct=2.5)
        self.assertEqual(why, "stop")                              # оба в баре — стоп


class TestSizing(unittest.TestCase):

    def test_risk_position_caps(self):
        self.assertAlmostEqual(bt.risk_position(5e6, 0.5, 5.0, 700000, None, 0.01), 500000)   # 25k / 5 %
        self.assertAlmostEqual(bt.risk_position(5e6, 0.5, 2.0, 700000, None, 0.01), 700000)   # потолок
        self.assertAlmostEqual(bt.risk_position(5e6, 0.5, 5.0, 700000, 20e6, 0.01), 200000)   # 1 % оборота
        self.assertEqual(bt.risk_position(5e6, 0.5, 0.0, 700000, None, 0.01), 0.0)

    def test_lots(self):
        self.assertAlmostEqual(bt.lots_notional(100000, 330.0, 10), 99000.0)
        self.assertEqual(bt.lots_notional(100000, 150000.0, 1), 0.0)


class TestStats(unittest.TestCase):

    def test_rsi_bounds_and_direction(self):
        up = bt.rsi(pd.Series(np.arange(1, 40, dtype=float)))
        self.assertAlmostEqual(up.iloc[-1], 100.0)
        down = bt.rsi(pd.Series(np.arange(40, 1, -1, dtype=float)))
        self.assertLess(down.iloc[-1], 1.0)
        self.assertTrue(np.isnan(up.iloc[5]))

    def test_max_drawdown(self):
        self.assertAlmostEqual(bt.max_drawdown(pd.Series([100, -50, -80, 60])), -130.0)
        self.assertAlmostEqual(bt.max_drawdown(pd.Series([-30, 10])), -30.0)
        self.assertEqual(bt.max_drawdown(pd.Series([], dtype=float)), 0.0)

    def test_exposure_overlapping_holds(self):
        import datetime as dt
        days = [dt.date(2025, 3, 3) + dt.timedelta(days=i) for i in range(5)]
        tr = pd.DataFrame({"date": [days[0], days[1]], "exit_date": [days[2], days[1]], "notional": [100.0, 50.0]})
        e = bt.exposure_by_day(tr, days)
        self.assertEqual(list(e), [100.0, 150.0, 100.0, 0.0, 0.0])

    def test_margin_rules(self):
        m = bt.load_rules()["account"]["margin"]
        self.assertEqual(m["buying_power_rub"], 5000000 * (100 - m["collateral_discount_pct"]) / 100)

    def test_rules_frozen(self):
        r = bt.load_rules()
        self.assertEqual(sorted(r["tests"]), sorted(bt.TESTS))
        self.assertEqual((r["tests"]["T1"]["risk_pct_of_capital"], r["tests"]["T2"]["tp_atr_k"],
                          r["tests"]["T4"]["rsi_max"]), (0.5, 2.5, 30))


if __name__ == "__main__":
    unittest.main()
