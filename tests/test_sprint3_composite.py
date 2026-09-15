"""
Спринт 3, контрольная точка без ML (research/sprint3_composite.py). Синтетика.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import sprint3_composite as sc             # noqa: E402

DAYS = pd.bdate_range("2025-01-06", periods=6)


class TestScore(unittest.TestCase):

    def test_cross_z_and_signs(self):
        df = pd.DataFrame({"d": [DAYS[0]] * 3, "rs_30": [1.0, 2.0, 3.0], "z_ema50": [0.0, 0.0, 0.0],
                           "news_sent_48h": [0.0, 0.0, np.nan], "cs_spread": [0.3, 0.2, 0.1]})
        z = sc.cross_z(df, ["rs_30", "z_ema50", "news_sent_48h"])
        self.assertAlmostEqual(z["rs_30"].iloc[2], 1.224744871, places=6)
        self.assertTrue((z["z_ema50"] == 0).all() and (z["news_sent_48h"] == 0).all())
        s = sc.score(df, sc.load_spec()["score"]["terms"])
        self.assertEqual(int(s.idxmax()), 2)                    # сила выше и спред уже — лучший скор

    def test_threshold_uses_past_only(self):
        df = pd.DataFrame({"d": np.repeat(DAYS, 2), "score": np.arange(12, dtype=float)})
        thr = sc.rolling_threshold(df, 1.0, 3, 2)
        self.assertTrue(np.isnan(thr.iloc[0]) and np.isnan(thr.iloc[2]))   # истории < 2 дней
        self.assertEqual(thr.iloc[4], 3.0)                      # день 3: максимум дней 1–2
        self.assertEqual(thr.iloc[10], 9.0)                     # день 6: окно дней 3–5

    def test_spec_frozen(self):
        s = sc.load_spec()
        self.assertEqual(len(s["pool"]), 9)
        self.assertEqual(s["score"]["terms"], [["rs_30", 1], ["z_ema50", 1], ["news_sent_48h", 1], ["cs_spread", -1]])
        self.assertEqual((s["tail"]["quantile"], s["tail"]["lookback_days"]), (0.975, 250))

    def test_holdout_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                sc.holdout_allowed(tmp)
            os.makedirs(os.path.join(tmp, "composite_dev"))
            with open(os.path.join(tmp, "composite_dev", "results.json"), "w", encoding="utf-8") as f:
                json.dump({"result": {"gate": False}}, f)
            with self.assertRaises(SystemExit):
                sc.holdout_allowed(tmp)


if __name__ == "__main__":
    unittest.main()
