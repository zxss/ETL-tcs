"""
Вход ночных лонгов по текущей цене (решение пользователя 22.09.2026):
в песочнице покупки и продажи должны проходить каждый день.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                              # noqa: E402
from services import stage2_demo as s2                     # noqa: E402
from tft_forecast.combined import Order                    # noqa: E402


def order(tk="LENT", entry=1603.57, anchor=1622.0, lots=62, lot=1, total=None, direction="LONG"):
    total = total if total is not None else lots * lot * entry
    return Order(ticker=tk, strategy="long_overnight", direction=direction, anchor_price=anchor,
                 f_low=1500.0, f_high=1700.0, down_pct=-1.0, entry_price=entry, better_pct=1.1,
                 stop_price=entry * 0.99, stop_pct=1.0, tp_price=entry * 1.039, tp_pct=3.9,
                 lot_size=lot, lot_known=True, quantity_lots=lots, total_rub=total, unavailable=False)


class FakeBroker:
    def __init__(self, prices):
        self.prices = prices

    def find_instrument(self, tk):
        if tk not in self.prices:
            raise RuntimeError("нет инструмента")
        return SimpleNamespace(instrument_uid=tk)

    def get_last_price(self, uid):
        return self.prices[uid]


class TestRepriceMarketable(unittest.TestCase):

    def test_default_mode_is_marketable(self):
        self.assertEqual(config.OVERNIGHT_ENTRY_MODE, "marketable")
        self.assertIn("OVERNIGHT_ENTRY_MODE", s2._CONFIG_KEYS)

    def test_long_goes_above_last_and_keeps_risk_percent(self):
        out, notes = s2.reprice_marketable(FakeBroker({"LENT": 1650.0}), [order()],
                                           slip_pct=0.1, position_rub=100_000)
        o = out[0]
        self.assertEqual(notes, [])
        self.assertAlmostEqual(o.entry_price, 1650.0 * 1.001)
        self.assertAlmostEqual(o.stop_price, o.entry_price * 0.99)          # тот же стоп 1 %
        self.assertAlmostEqual(o.tp_price, o.entry_price * 1.039)           # тот же тейк 3,9 %
        self.assertEqual(o.quantity_lots, int(100_000 / o.entry_price))
        self.assertLessEqual(o.total_rub, 100_000)
        self.assertTrue(o.is_placeable)

    def test_maxpos_cut_keeps_the_smaller_sum(self):
        cut = order(tk="AKRN", entry=18114.35, anchor=18212.0, lots=3, total=54_343.05)
        out, _ = s2.reprice_marketable(FakeBroker({"AKRN": 18250.0}), [cut],
                                       slip_pct=0.1, position_rub=100_000)
        self.assertLessEqual(out[0].total_rub, 54_343.05)
        self.assertEqual(out[0].quantity_lots, 2)

    def test_no_price_leaves_forecast_order(self):
        o = order(tk="ETLN")
        out, notes = s2.reprice_marketable(FakeBroker({}), [o], slip_pct=0.1, position_rub=100_000)
        self.assertIs(out[0], o)
        self.assertTrue(notes and "цена недоступна" in notes[0])

    def test_lot_too_expensive_is_skipped(self):
        o = order(tk="LKOH", entry=6000.0, anchor=6000.0, lots=1, lot=10, total=60_000.0)
        out, notes = s2.reprice_marketable(FakeBroker({"LKOH": 12_000.0}), [o],
                                           slip_pct=0.1, position_rub=100_000)
        self.assertFalse(out[0].is_placeable)
        self.assertTrue(any("пропуск" in n for n in notes))

    def test_short_goes_below_last(self):
        o = order(direction="SHORT")
        out, _ = s2.reprice_marketable(FakeBroker({"LENT": 1650.0}), [o], slip_pct=0.1,
                                       position_rub=100_000)
        self.assertAlmostEqual(out[0].entry_price, 1650.0 * 0.999)
        self.assertGreater(out[0].stop_price, out[0].entry_price)


if __name__ == "__main__":
    unittest.main()
