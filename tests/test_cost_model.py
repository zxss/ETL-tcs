"""
Издержки по тарифу пользователя (research/cost_model.py). Без сети и БД.
"""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import cost_model as cm                    # noqa: E402


class TestCostModel(unittest.TestCase):

    def setUp(self):
        self.sp = cm.load_spreads()

    def test_premium_fee(self):
        self.assertEqual(cm.FEE_SIDE_PCT, 0.04)
        self.assertAlmostEqual(cm.round_trip("SBER", "fee", self.sp), 0.08)

    def test_base_matches_cost_matrix(self):
        """Матрица построена при той же комиссии 0,04 % — base совпадает с cost_rt_base."""
        m = pd.read_csv(cm.MATRIX_PATH)
        for r in m.itertuples(index=False):
            self.assertAlmostEqual(cm.round_trip(r.ticker, "base", self.sp), r.cost_rt_base, places=9)
            self.assertAlmostEqual(cm.round_trip(r.ticker, "stress", self.sp), r.cost_rt_stress, places=9)

    def test_fallback_and_pairs(self):
        t = cm.round_trip("T", "base", self.sp)
        self.assertGreater(t, 0.08)
        self.assertAlmostEqual(cm.trade_cost("SBER/T", "base", self.sp),
                               cm.round_trip("SBER", "base", self.sp) + t)

    def test_premium_carry(self):
        self.assertEqual(cm.carry_pct(5_000, 1), 0.0)              # до 5 000 ₽ бесплатно
        self.assertAlmostEqual(cm.carry_pct(10_000, 1), 0.45)      # «от 45 ₽ в день»
        self.assertAlmostEqual(cm.carry_pct(10_000, 3), 1.35)      # выходные — календарные дни
        self.assertAlmostEqual(cm.carry_pct(200_000, 1), 0.095)
        with self.assertRaises(ValueError):
            cm.carry_pct(5_000_000, 1)


if __name__ == "__main__":
    unittest.main()
