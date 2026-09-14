"""
Шорт-правило без TFT (research/short_rule.py). Без сети и БД.

Главное — признаки правила совпадают с продом (tft_forecast/market) бар в бар:
правило взято из кода, а не из пересказа.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import short_rule as sr                    # noqa: E402
from tft_forecast import market                          # noqa: E402


def synthetic(n=400, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    h = c * (1 + rng.uniform(0, 0.03, n))
    lo = c * (1 - rng.uniform(0, 0.03, n))
    o = c * (1 + rng.normal(0, 0.005, n))
    dates = pd.bdate_range("2024-01-01", periods=n).date
    return pd.DataFrame({"date": dates, "open": o, "high": h, "low": lo, "close": c,
                         "volume": rng.integers(1000, 5000, n)})


class TestFeaturesMatchProd(unittest.TestCase):

    def test_ret1_atr_pctl_atr_pct_bar_by_bar(self):
        df = synthetic()
        f = sr.ticker_features(df)
        for i in (29, 30, 45, 120, 300, 399):
            win = df.iloc[: i + 1].tail(sr.LOOKBACK).reset_index(drop=True)
            if i + 1 < sr.MIN_ROWS:
                self.assertTrue(np.isnan(f["atr_pctl"].iloc[i]))
                continue
            self.assertAlmostEqual(f["ret1"].iloc[i], market._ret1(win["close"]), places=9)
            self.assertAlmostEqual(f["atr_pctl"].iloc[i], market._atr_pctl(win), places=9)
            self.assertAlmostEqual(f["atr_pct"].iloc[i], market._atr_pct(win), places=9)
        self.assertTrue(np.isnan(f["atr_pctl"].iloc[28]))       # < 30 баров — прод не берёт

    def test_index_above_ema_matches_prod_window(self):
        df = synthetic(n=420, seed=3)
        flags = sr.index_above_ema(df["close"])
        for i in (48, 49, 60, 330, 419):
            win = df["close"].iloc[: i + 1].tail(sr.LOOKBACK).reset_index(drop=True)
            self.assertEqual(flags.iloc[i], market._above_ema(win, 50))

    def test_adv_uses_lot(self):
        df = synthetic(n=40)
        f1, f10 = sr.ticker_features(df, 1), sr.ticker_features(df, 10)
        self.assertAlmostEqual(f10["adv_rub"].iloc[-1], 10 * f1["adv_rub"].iloc[-1])
        self.assertTrue(np.isnan(f1["adv_rub"].iloc[sr.LIQ_WIN - 2]))


class TestRule(unittest.TestCase):

    def test_gates(self):
        self.assertTrue(sr.gates_open(60.0, False))
        self.assertFalse(sr.gates_open(50.0, False))           # строго > 50
        self.assertFalse(sr.gates_open(70.0, True))            # IMOEX выше EMA50
        self.assertIsNone(sr.gates_open(70.0, None))           # предохранитель без данных — день не оценивается
        self.assertTrue(sr.gates_open(None, False))            # нет оценки ATR — как прод, не блокирует

    def test_quantum(self):
        self.assertFalse(sr.quantum_ok(0.0062))                # TGKA
        self.assertTrue(sr.quantum_ok(0.5))                    # HYDR: 0,02 %
        self.assertFalse(sr.quantum_ok(0.0))

    def test_short_outcome(self):
        o = np.array([100.0, 101, 99]); h = np.array([100.5, 102, 99.5]); c = np.array([100.8, 99.2, 98])
        r = sr.short_outcome(o, h, c)
        self.assertAlmostEqual(r["gross"], 2.0)
        self.assertAlmostEqual(r["mae"], 2.0)
        s = sr.short_outcome(o, h, c, stop_pct=1.5)             # 101,5 пробит во 2-м баре
        self.assertTrue(s["stopped"])
        self.assertAlmostEqual(s["gross"], -1.5)
        g = sr.short_outcome(np.array([100.0, 104]), np.array([100.2, 105]),
                             np.array([100.1, 104.5]), stop_pct=3.5)
        self.assertAlmostEqual(g["gross"], -4.0)               # гэп сквозь стоп: по open бара

    def test_select(self):
        c = pd.DataFrame({"ticker": list("ABCDEFG"), "ret1": [-1, -5, -3, -2, -7, -0.5, -4],
                          "adv_rub": [7, 6, 5, 4, 3, 2, 1]})
        self.assertEqual(list(sr.select(c, "drop5")["ticker"]), ["E", "B", "G", "C", "D"])
        self.assertEqual(list(sr.select(c, "liq5")["ticker"]), ["A", "B", "C", "D", "E"])
        self.assertEqual(len(sr.select(c, "all")), 7)

    def test_run_rule_order_of_filters(self):
        A, Ap = dt.date(2025, 3, 4), dt.date(2025, 3, 3)
        tks = ["T1", "T2", "T3", "T4", "T5", "T6", "BLK"]
        feats = pd.DataFrame({"date": [Ap] * 7, "ticker": tks,
                              "ret1": [-6, -5, -4, -3, -2, -1, -9], "atr_pct": [2.0] * 7,
                              "atr_pctl": [60] * 7, "adv_rub": [1e6] * 7})
        path = (np.array([100.0, 99]), np.array([100.5, 99.5]), np.array([99.5, 99.0]))
        paths = {(tk, A): path for tk in tks if tk != "T4"}           # T4 — нет 5-минуток
        paths[(sr.INDEX, A)] = (np.array([3000.0]), np.array([3001.0]), np.array([2970.0]))
        lots = {"T2": 1000}                                           # лот 100 000 ₽ > позиции
        tr, days = sr.run_rule(feats, pd.Series({Ap: 60.0}), pd.Series({Ap: False}),
                               [Ap, A], paths, lots, {"BLK"}, [A])
        d5 = tr[tr["selection"] == "drop5"]
        self.assertNotIn("BLK", set(tr["ticker"]))                   # нешортуемые — до отбора
        self.assertEqual(list(d5["ticker"]), ["T1", "T2", "T3", "T4", "T5"])  # слот T2 не замещается
        self.assertEqual(dict(zip(d5["ticker"], d5["status"]))["T2"], "lot_over_position")
        self.assertEqual(dict(zip(d5["ticker"], d5["status"]))["T4"], "no_data")
        self.assertEqual(days.iloc[0]["candidates"], 6)
        self.assertAlmostEqual(days.iloc[0]["imoex_move"], -1.0)
        self.assertAlmostEqual(d5[d5.status == "ok"]["gross_none"].iloc[0], 1.0)

    def test_closed_gate_no_trades(self):
        A, Ap = dt.date(2025, 3, 4), dt.date(2025, 3, 3)
        feats = pd.DataFrame({"date": [Ap], "ticker": ["T1"], "ret1": [-1.0], "atr_pct": [2.0],
                              "atr_pctl": [60], "adv_rub": [1.0]})
        tr, days = sr.run_rule(feats, pd.Series({Ap: 60.0}), pd.Series({Ap: True}),
                               [Ap, A], {}, {}, set(), [A])
        self.assertTrue(tr.empty)
        self.assertFalse(days.iloc[0]["gate"])


class TestStats(unittest.TestCase):

    def test_holm(self):
        adj = sr.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)                        # монотонность

    def test_variant_stats_by_day_and_concentration(self):
        rows = []
        for i in range(12):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=i)
            g = 5.0 if i < 2 else -0.1
            for tk in ("X", "Y"):
                rows.append({"date": d, "ticker": tk, "status": "ok", "gross_none": g,
                             "mae_none": 1.0, "gross_atr": g, "mae_atr": 1.0, "stopped_atr": False})
        tr = pd.DataFrame(rows)
        days = pd.DataFrame({"date": sorted(tr["date"].unique()),
                             "imoex_move": np.linspace(-1, 1, 12)})
        st = sr.variant_stats(tr, days, "none", 0.128)
        self.assertEqual((st["n"], st["days"]), (24, 12))           # t — по 12 дням, не по 24 сделкам
        self.assertAlmostEqual(st["sum_wo_top5"], st["sum_days"] - (2 * 4.872 + 3 * -0.228))
        self.assertGreater(st["top5_share"], 1.0)                   # весь результат — два дня
        self.assertIn("beta", st)

    def test_prev_date(self):
        ds = [dt.date(2025, 3, 3), dt.date(2025, 3, 5)]
        self.assertEqual(sr.prev_date(ds, dt.date(2025, 3, 5)), dt.date(2025, 3, 3))
        self.assertEqual(sr.prev_date(ds, dt.date(2025, 3, 4)), dt.date(2025, 3, 3))
        self.assertIsNone(sr.prev_date(ds, dt.date(2025, 3, 3)))

    def test_window_entry_tolerance(self):
        d = dt.date(2025, 3, 3)
        bars = pd.DataFrame({"ticker": ["A", "A", "B"],
                             "tm": pd.to_datetime(["2025-03-03 10:05", "2025-03-03 18:15",
                                                   "2025-03-03 10:20"]),
                             "open": [1.0, 2, 3], "high": [1.0, 2, 3], "close": [1.0, 2, 3]})
        bars["d"] = d
        p = sr.window_paths(bars)
        self.assertIn(("A", d), p)
        self.assertNotIn(("B", d), p)                              # первый бар позже 10:15


if __name__ == "__main__":
    unittest.main()
