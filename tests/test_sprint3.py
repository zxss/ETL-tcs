"""
Спринт 3: панель признаков и walk-forward (research/sprint3_panel.py, sprint3_wf.py).
Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import sprint3_panel as sp                 # noqa: E402

try:
    import sklearn                                        # noqa: F401
    HAVE_SK = True
except ImportError:                                       # pragma: no cover
    HAVE_SK = False

DAYS = [d.date() for d in pd.bdate_range("2025-01-06", periods=90)]


def frame(seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS))))
    g = pd.DataFrame({"open_main": c * 0.999, "c1830": c, "c1835": c * 1.0005, "hi": c * 1.01, "lo": c * 0.99,
                      "vol_main": rng.integers(1000, 5000, len(DAYS)).astype(float),
                      "vol_evening": 100.0, "vol_day": 6000.0}, index=DAYS)
    return g


class TestStockFeatures(unittest.TestCase):

    def test_no_lookahead(self):
        g = frame()
        imx = pd.Series(1000 * (1 + 0.001 * np.arange(len(DAYS))), index=DAYS)
        f0 = sp.stock_features(g, imx)
        g2 = g.copy()
        g2.iloc[70:, :] = g2.iloc[70:, :] * 1.5
        f1 = sp.stock_features(g2, imx)
        pd.testing.assert_frame_equal(f0.iloc[:70], f1.iloc[:70])

    def test_corwin_schultz_zero_range(self):
        h = pd.Series([10.0, 10.0, 10.0])
        self.assertTrue((sp.corwin_schultz(h, h).dropna() == 0).all())

    def test_split_adjustment_and_blackout(self):
        g = frame()
        g.iloc[40:, :5] = g.iloc[40:, :5] / 100.0                 # сплит 100:1 перед днём 40
        adj, black = sp.adjust_splits(g)
        self.assertAlmostEqual(adj["c1830"].iloc[39] / adj["c1830"].iloc[40], g["c1830"].iloc[39] / g["c1830"].iloc[40] / 100, places=1)
        self.assertTrue(black.iloc[40] and black.iloc[len(DAYS) - 1] and not black.iloc[39])   # 60 дней — до конца ряда
        self.assertAlmostEqual(adj["adj"].iloc[0], g["open_main"].iloc[40] / g["c1835"].iloc[39])


class TestTargets(unittest.TestCase):

    def test_hurdle_cost_margin_and_dividend(self):
        g = pd.DataFrame({"c1835": [100.0, 101.0, 100.5, 102.0]}, index=DAYS[:4])
        t = sp.targets(g, "X", [(DAYS[2], 1.0)], lambda a, b: 0.1, cost=0.2, margin=0.1, horizon=2)
        self.assertAlmostEqual(t["R"].iloc[0], 1.5)                # (100,5 + 1) / 100
        self.assertEqual(t["y"].iloc[0], 1.0)                       # 1,5 > 0,1 + 0,2 + 0,1
        self.assertAlmostEqual(t["excess"].iloc[0], 1.2)
        self.assertEqual(t["exit_d"].iloc[0], DAYS[2])
        self.assertTrue(np.isnan(t["y"].iloc[2]))                  # выхода нет


class TestNews(unittest.TestCase):

    def test_48h_window_before_decision(self):
        d = DAYS[5]
        at = dt.datetime.combine(d, dt.time(18, 30))
        ev = pd.DataFrame({"ticker": ["SBER"] * 4,
                           "posted": [at - dt.timedelta(hours=47), at - dt.timedelta(hours=49),
                                      at + dt.timedelta(minutes=1), at - dt.timedelta(hours=1)],
                           "category": ["FINANCIAL", "FINANCIAL", "FINANCIAL", "DIVIDEND_ANNOUNCE"],
                           "sentiment": [-1.0, -1.0, -1.0, 0.5]})
        nf = sp.news_features(ev, ["SBER"], DAYS).set_index("d")
        self.assertEqual(nf.at[d, "news_fin_48h"], 1)
        self.assertEqual(nf.at[d, "news_div_48h"], 1)
        self.assertAlmostEqual(nf.at[d, "news_sent_48h"], -0.5)


class TestConfig(unittest.TestCase):

    def test_clusters_and_features(self):
        cfg = sp.load_config()
        tks = [t for v in cfg["clusters"].values() for t in v]
        self.assertEqual(len(tks), len(set(tks)))
        self.assertEqual(len(tks), 47)
        f = sp.stock_features(frame(), pd.Series(1000.0, index=DAYS))
        self.assertTrue(set(cfg["features"]["stock"]) <= set(f.columns))
        self.assertTrue(set(cfg["features"]["market"]) <= set(sp.market_features(pd.Series(1000.0, index=DAYS)).columns))
        self.assertTrue(cfg["samples"]["holdout"]["approved"])
        self.assertEqual((cfg["target"]["column"], cfg["evaluation"]["net"], cfg["hedge"]["root"]),
                         ("y_rel", "net_mn", "MX"))


class TestRelative(unittest.TestCase):

    def test_market_legs_same_contract(self):
        from research import event_study_news as es
        d = DAYS[:4]
        b = es.Bars([dt.datetime.combine(x, dt.time(18, 30)) for x in d], [1, 1, 1, 1], [100.0, 101.0, 102.0, 99.0])
        imx = pd.Series([1000.0, 1010.0, 1005.0, 990.0], index=d)
        legs = sp.market_legs(imx, lambda r, day: b if r == "MX" else None, d, 2, "MX")
        self.assertAlmostEqual(legs.at[d[0], "r_idx"], 0.5)
        self.assertAlmostEqual(legs.at[d[0], "r_fut"], 2.0)
        self.assertTrue(np.isnan(legs.at[d[2], "r_fut"]))            # выхода нет

    def test_relative_target_and_bundle(self):
        cfg = sp.load_config()
        p = pd.DataFrame({"R": [1.0, 0.3], "r_idx": [0.4, 0.4], "r_fut": [0.3, 0.3], "hurdle": [0.1, 0.1],
                          "cost": [0.2, 0.2]})
        q = sp.add_relative_targets(p, cfg)
        self.assertEqual(list(q["y_rel"]), [1.0, 0.0])             # альфа 0,6 > 0,2 + 0,1; −0,1 — нет
        self.assertAlmostEqual(q["net_alpha"].iloc[0], 0.4)
        self.assertAlmostEqual(q["net_mn"].iloc[0], 1.0 - 0.3 - 0.1 - 0.2 - 0.07)
        self.assertAlmostEqual(q["net_mn_fee"].iloc[0], 1.0 - 0.3 - 0.1 - 0.08 - 0.07)


@unittest.skipUnless(HAVE_SK, "нет sklearn")
class TestHoldoutGuard(unittest.TestCase):

    def test_guard(self):
        import json
        import tempfile
        from research import sprint3_wf as wf
        cfg = sp.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                wf.holdout_clusters(cfg, tmp)                          # нет результатов разработки
            os.makedirs(os.path.join(tmp, "dev_v1"))
            p = os.path.join(tmp, "dev_v1", "results.json")
            for meta, passed in (({"config": "другая"}, ["FINANCIALS"]), ({"config": cfg["version"]}, [])):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"meta": meta, "passed": passed}, f)
                with self.assertRaises(SystemExit):
                    wf.holdout_clusters(cfg, tmp)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"meta": {"config": cfg["version"]}, "passed": ["FINANCIALS"]}, f)
            self.assertEqual(wf.holdout_clusters(cfg, tmp), ["FINANCIALS"])
            os.makedirs(os.path.join(tmp, "holdout_v1"))
            open(os.path.join(tmp, "holdout_v1", "results.json"), "w").close()
            with self.assertRaises(SystemExit):
                wf.holdout_clusters(cfg, tmp)                          # второй прогон запрещён


@unittest.skipUnless(HAVE_SK, "нет sklearn")
class TestWalkForward(unittest.TestCase):

    def test_purge_and_nw(self):
        from research import sprint3_wf as wf
        rng = np.random.default_rng(3)
        days = pd.bdate_range("2024-01-01", "2025-03-31")
        rows = []
        for tk in ("A", "B", "C", "D"):
            for i, d in enumerate(days[:-2]):
                x = rng.normal()
                rows.append({"ticker": tk, "cluster": "K", "d": d, "exit_d": days[i + 2], "x": x,
                             "y": float(x + rng.normal(0, 0.5) > 0), "R": x, "excess": x, "cost": 0.1, "hurdle": 0.1})
        panel = pd.DataFrame(rows)
        cfg = {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 20,
               "l2_regularization": 0.0, "random_state": 0}
        pred = wf.walk_forward(panel, ["x"], cfg, {"train_months": 12, "min_train_months": 6, "min_train_rows": 100})
        first = pred["fit"].min()
        exp_n = ((panel["exit_d"] < first) & (panel["d"] >= first - pd.DateOffset(months=12))).sum()
        self.assertEqual(int(pred.loc[pred["fit"] == first, "n_train"].iloc[0]), int(exp_n))
        self.assertTrue((pred["d"] >= pred["fit"]).all())
        res = wf.evaluate(pred, 0.5, 10, {"t_min": 2.0, "net_min": 0.0, "dates_min": 30})
        self.assertGreater(res["K"]["auc"], 0.7)
        self.assertTrue(res["K"]["gate"])
        x = pd.Series(rng.normal(0.1, 1, 500))
        self.assertAlmostEqual(wf.newey_west_t(x, 0), x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), places=6)


if __name__ == "__main__":
    unittest.main()
