"""
Стратегическое исследование (ТЗ 18.09.2026): модель издержек, CPCV/PBO, барьеры.
Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.strategic import costs as sc                 # noqa: E402
from research.strategic import panel as pn                 # noqa: E402
from research.strategic import validation as va            # noqa: E402
from research.strategic import h3_meta_labeling as h3      # noqa: E402
from research.strategic import h4_totm_inelastic as h4     # noqa: E402

SPREADS = {"SBER": (0.02, 0.05, 0.01), "TGKA": (0.30, 0.60, 0.15), "__fallback__": (0.05, 0.10, 0.02)}


def model(y=1.0):
    return sc.CostModel(impact_y=y, spread_floor=dict(SPREADS))


class TestCosts(unittest.TestCase):

    def test_commission_both_sides(self):
        self.assertAlmostEqual(model().fee_round_trip_pct(), 0.14, places=9)

    def test_impact_follows_square_root(self):
        m = model()
        one = m.impact_pct(2.0, 100_000, 50_000_000)
        four = m.impact_pct(2.0, 400_000, 50_000_000)
        self.assertAlmostEqual(four / one, 2.0, places=9)          # ×4 объёма → ×2 воздействия
        self.assertAlmostEqual(one, 2.0 * math.sqrt(0.002), places=9)
        self.assertEqual(m.impact_pct(2.0, 100_000, 0), float("inf"))

    def test_spread_floor_protects_from_optimistic_cs(self):
        m = model()
        self.assertAlmostEqual(m.spread_pct("TGKA", cs_pct=0.01), 0.30)     # оценка ниже измеренной
        self.assertAlmostEqual(m.spread_pct("TGKA", cs_pct=0.55), 0.55)
        self.assertAlmostEqual(m.spread_pct("NEIZVESTEN", cs_pct=None), 0.05)

    def test_capacity_matches_formula(self):
        m = model()
        alpha, sigma, adv = 0.5, 2.0, 100_000_000.0
        self.assertAlmostEqual(m.capacity_rub(alpha, sigma, adv),
                               4.0 / 9.0 * (alpha / sigma) ** 2 * adv, places=6)
        self.assertEqual(m.capacity_rub(-0.1, sigma, adv), 0.0)            # нет перевеса — нет ёмкости

    def test_capacity_is_the_optimum_of_net_profit(self):
        """Q_opt действительно максимизирует (α − Y·σ·√(Q/V))·Q."""
        m, alpha, sigma, adv = model(), 0.5, 2.0, 100_000_000.0
        q = m.capacity_rub(alpha, sigma, adv)
        profit = lambda x: (alpha - sigma * math.sqrt(x / adv)) * x       # noqa: E731
        self.assertGreater(profit(q), profit(q * 0.5))
        self.assertGreater(profit(q), profit(q * 1.5))

    def test_round_trip_includes_two_impacts(self):
        m = model()
        rt = m.round_trip_pct("SBER", 2.0, 100_000, 50_000_000, cs_pct=0.02)
        self.assertAlmostEqual(rt, 0.14 + 0.02 + 2.0 * m.impact_pct(2.0, 100_000, 50_000_000))

    def test_corwin_schultz_non_negative(self):
        h = pd.Series([101.0, 102.0, 103.0, 101.5])
        l = pd.Series([99.0, 100.0, 101.0, 100.0])
        s = sc.corwin_schultz(h, l).dropna()
        self.assertTrue((s >= 0).all())


class TestCPCV(unittest.TestCase):

    def days(self, n=200):
        d = dt.date(2024, 1, 1)
        return [d + dt.timedelta(days=i) for i in range(n) if (d + dt.timedelta(days=i)).weekday() < 5]

    def test_splits_are_disjoint_and_purged(self):
        days = self.days()
        splits = va.cpcv_splits(days, n_blocks=5, k_test=2, embargo_days=5, label_days=5)
        self.assertEqual(len(splits), math.comb(5, 2))
        self.assertEqual(va.n_paths(5, 2), 4)
        for sp in splits:
            self.assertFalse(sp["train"] & sp["test"])                       # пересечения нет
            for d in sp["train"]:
                # метка обучающего дня не заходит в тест, и карантин соблюдён
                self.assertFalse(any(0 <= (t - d).days <= 5 for t in sp["test"]))
                self.assertFalse(any(0 < (d - t).days <= 5 for t in sp["test"]))

    def test_too_few_dates(self):
        with self.assertRaises(ValueError):
            va.date_blocks([dt.date(2024, 1, 1)], 5)

    def test_pbo_high_for_pure_noise(self):
        rng = np.random.default_rng(7)
        perf = pd.DataFrame(rng.normal(size=(40, 8)))
        res = va.cscv_pbo(perf, s=8)
        self.assertGreater(res["pbo"], 0.3)

    def test_pbo_low_when_one_config_is_really_better(self):
        rng = np.random.default_rng(7)
        perf = pd.DataFrame(rng.normal(scale=0.2, size=(40, 8)))
        perf[3] += 3.0                                                    # устойчиво лучшая
        res = va.cscv_pbo(perf, s=8)
        self.assertLess(res["pbo"], 0.1)

    def test_pbo_needs_grid(self):
        self.assertIsNone(va.cscv_pbo(pd.DataFrame({"a": [1.0] * 20}))["pbo"])


class TestMetrics(unittest.TestCase):

    def trades(self, net, n=30):
        rows = []
        for i in range(n):
            d0 = dt.date(2024, 6, 3) + dt.timedelta(days=7 * i)
            rows.append({"entry_day": d0, "exit_day": d0 + dt.timedelta(days=5), "notional": 100_000.0,
                         "net_excess_pct": net[i % len(net)],
                         "pnl_excess_rub": 100_000.0 * net[i % len(net)] / 100.0})
        return pd.DataFrame(rows)

    def test_summary_fields(self):
        tr = self.trades([0.5, -0.1, 0.3])
        s = va.summarize(tr, dt.date(2024, 6, 3), dt.date(2026, 1, 1), 5_000_000.0)
        self.assertEqual(s["trades"], 30)
        self.assertGreater(s["t"], 0)
        self.assertGreater(s["excess_annual_pct"], 0)
        self.assertLessEqual(s["mdd_rub"], 0)

    def test_max_drawdown(self):
        self.assertAlmostEqual(va.max_drawdown(pd.Series([1.0, -3.0, 1.0])), -3.0)

    def test_dev_gate_lists_every_failure(self):
        ok, fails = va.dev_gate({"excess_annual_pct": 1.0, "t": 2.0}, pbo=0.6, capacity_rub=1e6,
                                gates={"excess_annual_min_pct": 2.0, "t_min": 3.01, "pbo_max": 0.4,
                                       "capacity_min_rub": 5e6})
        self.assertFalse(ok)
        self.assertEqual(len(fails), 4)

    def test_dev_gate_passes_when_all_conditions_hold(self):
        ok, fails = va.dev_gate({"excess_annual_pct": 5.0, "t": 3.5}, pbo=0.1, capacity_rub=9e6,
                                gates={"excess_annual_min_pct": 2.0, "t_min": 3.01, "pbo_max": 0.4,
                                       "capacity_min_rub": 5e6})
        self.assertTrue(ok)
        self.assertEqual(fails, [])

    def test_holdout_gate(self):
        ok, fails = va.holdout_gate({"t": 1.9, "excess_annual_pct": 1.44, "dsr": None},
                                    {"t_min": 2.5, "net_excess_min_pct": 0.0, "dsr_min": 0.95})
        self.assertFalse(ok)
        self.assertTrue(any("DSR" in f for f in fails))


class TestTripleBarrier(unittest.TestCase):

    def frame(self, highs, lows, closes):
        n = len(highs)
        return pd.DataFrame({"date": [dt.date(2024, 6, 3) + dt.timedelta(days=i) for i in range(n)],
                             "high": highs, "low": lows, "close": closes})

    def test_take_profit(self):
        g = self.frame([100, 104, 104], [99, 100, 100], [100, 103, 103])
        r = h3.triple_barrier(g, 0, atr=2.0)                 # тейк 103, стоп 98
        self.assertEqual((r["label"], r["reason"]), (1, "тейк"))

    def test_stop_loss(self):
        g = self.frame([100, 101, 101], [99, 97, 97], [100, 97.5, 97.5])
        r = h3.triple_barrier(g, 0, atr=2.0)
        self.assertEqual((r["label"], r["reason"]), (0, "стоп"))

    def test_stop_wins_when_both_barriers_in_one_day(self):
        g = self.frame([100, 110, 110], [99, 90, 90], [100, 100, 100])
        r = h3.triple_barrier(g, 0, atr=2.0)
        self.assertEqual(r["reason"], "стоп")                # консервативно: пути внутри дня нет

    def test_timeout_exit_at_close(self):
        g = self.frame([100] * 8, [99] * 8, [100, 100.2, 100.1, 100.3, 100.2, 100.4, 100.1, 100.0])
        r = h3.triple_barrier(g, 0, atr=2.0)
        self.assertEqual(r["reason"], "время")
        self.assertEqual(r["exit_day"], g["date"].iloc[h3.MAX_DAYS])

    def test_bad_atr(self):
        g = self.frame([100, 101], [99, 100], [100, 100.5])
        self.assertIsNone(h3.triple_barrier(g, 0, atr=float("nan")))


class TestCalendarAndPanel(unittest.TestCase):

    def test_month_windows_skip_gaps(self):
        days = ([dt.date(2024, 1, d) for d in (29, 30, 31)] + [dt.date(2024, 2, d) for d in (1, 2, 5)]
                + [dt.date(2024, 4, d) for d in (1, 2, 3)])
        w = h4.month_windows(days)
        self.assertEqual(len(w), 1)                          # февраль→апрель — не подряд
        self.assertEqual((w[0]["t_m1"], w[0]["t_p3"]), (dt.date(2024, 1, 31), dt.date(2024, 2, 5)))

    def test_cross_section_z_is_per_date(self):
        f = pd.DataFrame({"date": [dt.date(2024, 6, 3)] * 3 + [dt.date(2024, 6, 4)] * 3,
                          "ticker": list("ABC") * 2,
                          "resid_mom_30": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        z = pn.cross_section_z(f, cols=("resid_mom_30",))
        self.assertAlmostEqual(z["resid_mom_30_z"].iloc[0], z["resid_mom_30_z"].iloc[3], places=9)
        self.assertAlmostEqual(z.groupby("date")["resid_mom_30_z"].mean().abs().max(), 0.0, places=9)

    def test_forward_return_does_not_look_further_than_horizon(self):
        d = [dt.date(2024, 6, 3) + dt.timedelta(days=i) for i in range(7)]
        f = pd.DataFrame({"date": d, "ticker": ["A"] * 7, "close": [100.0, 101, 102, 103, 104, 105, 106]})
        out = pn.forward_return(f, horizon=5)
        self.assertAlmostEqual(out["fwd_ret"].iloc[0], 5.0, places=9)
        self.assertEqual(out["fwd_date"].iloc[0], d[5])
        self.assertTrue(pd.isna(out["fwd_ret"].iloc[2]))     # хвост без будущего — пусто


if __name__ == "__main__":
    unittest.main()
