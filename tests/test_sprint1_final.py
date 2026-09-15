"""
Финальная гипотеза Спринта 1 (research/sprint1_final.py). Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import sprint1_final as sf                 # noqa: E402

DAYS = [dt.date(2023, 3, 6), dt.date(2023, 3, 7)]


def bars(price):
    tm, o, c = [], [], []
    for k, d in enumerate(DAYS):
        for start, end in ((dt.time(10, 0), dt.time(18, 45)), (dt.time(19, 5), dt.time(20, 0))):
            t = dt.datetime.combine(d, start)
            while t.time() <= end:
                oo, cc = price(k, t.time())
                tm.append(t); o.append(oo); c.append(cc)
                t += dt.timedelta(minutes=5)
    return es.Bars(tm, o, c)


STOCK = bars(lambda k, t: (100.0, 98.0) if t == dt.time(18, 15) else (100.0, 100.0))
IDX = bars(lambda k, t: (1000.0, 1000.0))


class TestOutcome(unittest.TestCase):

    def test_in_session_entry_and_1820_exit(self):
        o = sf.intraday_outcome(dt.datetime(2023, 3, 6, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2023, 3, 6, 11, 5))
        self.assertAlmostEqual(o["move"], -2.0)                   # close бара 18:15
        self.assertAlmostEqual(o["ar"], -2.0)

    def test_night_news_enter_at_ten(self):
        o = sf.intraday_outcome(dt.datetime(2023, 3, 6, 21, 0), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2023, 3, 7, 10, 0))

    def test_too_late_to_trade_same_day(self):
        o = sf.intraday_outcome(dt.datetime(2023, 3, 6, 18, 12), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2023, 3, 7, 10, 0))

    def test_rule_filter(self):
        ev = pd.DataFrame({"category": ["FINANCIAL", "FINANCIAL", "CORPORATE"], "sentiment": [-0.5, 0.0, -1.0]})
        self.assertEqual(len(sf.rule_events(ev)), 1)


class TestVerdict(unittest.TestCase):

    def test_criterion(self):
        days = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(40)]
        good = pd.DataFrame({"date": days, "ar": [-1.0 + 0.1 * (i % 3) for i in range(40)],
                             "net_short": [0.8 - 0.1 * (i % 3) for i in range(40)]})
        self.assertEqual(sf.evaluate(good, days, 83)["verdict"], "подтверждено")
        costly = good.assign(net_short=-0.1)
        self.assertEqual(sf.evaluate(costly, days, 83)["verdict"], "не подтверждено")

    def test_frozen_rule_file(self):
        with open(sf.RULE_PATH, encoding="utf-8") as f:
            rule = json.load(f)
        self.assertEqual((rule["event"]["category"], rule["strategy"], rule["sample"]["runs"]),
                         ("FINANCIAL", "intraday_short", 1))


if __name__ == "__main__":
    unittest.main()
