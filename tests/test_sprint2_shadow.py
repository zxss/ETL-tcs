"""
Теневой журнал «золотой ночи» (research/sprint2_shadow.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import sprint1_shadow as sh                # noqa: E402
from research import sprint2_shadow as g                 # noqa: E402

WED, THU = dt.date(2026, 9, 16), dt.date(2026, 9, 17)
H = {"id": "2B_gold", "family": "2B", "driver": "GD", "stocks": ["PLZL"],
     "signal": {"from_time": "10:00", "to_bar": "18:25"}, "entry_bar": "18:25"}


def bars(spec):
    tm, o, c = [], [], []
    for d, xs in spec.items():
        for t, oo, cc in xs:
            tm.append(dt.datetime.combine(d, t)); o.append(oo); c.append(cc)
    return es.Bars(tm, o, c)


FUT = bars({WED: [(dt.time(10), 3000, 3000), (dt.time(18, 25), 3010, 3030)]})
STOCK = bars({WED: [(dt.time(18, 25), 100, 100), (dt.time(18, 30), 100, 100)],
              THU: [(dt.time(6, 55), 101, 101)]})


class TestGoldShadow(unittest.TestCase):

    def test_entry_days_need_known_exit(self):
        self.assertEqual(g.entry_days(dt.date(2026, 9, 16)), [])          # 15.09 — до начала наблюдения
        self.assertEqual(g.entry_days(dt.date(2026, 9, 21)), [WED, THU, dt.date(2026, 9, 18)])

    def test_trade_and_idempotent_journal(self):
        rows = g.trades(H, lambda d: FUT, {"PLZL": STOCK}, [WED], [WED, THU], {}, lambda tk: 0.14,
                        lambda a, b: 0.05, lambda d: "GDZ6")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["date"], r["exit_date"], r["front"]), ("2026-09-16", "2026-09-17", "GDZ6"))
        self.assertAlmostEqual(r["net_long"], 1.0 - 0.19)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "j.csv")
            self.assertEqual(sh.merge_journal(rows, p, g.FIELDS, g._key), 1)
            self.assertEqual(sh.merge_journal(rows, p, g.FIELDS, g._key), 0)
            self.assertIn("| 18:30 | 1 | 1 |", g.summary(p))

    def test_falling_gold_no_trade(self):
        down = bars({WED: [(dt.time(10), 3000, 3000), (dt.time(18, 25), 2990, 2980)]})
        self.assertEqual(g.trades(H, lambda d: down, {"PLZL": STOCK}, [WED], [WED, THU], {},
                                  lambda tk: 0.14, lambda a, b: 0.05, lambda d: "GDZ6"), [])

    def test_fund_extension_and_dividends(self):
        hu = es.Hurdle.__new__(es.Hurdle)
        hu.series = {"TMON@": [(dt.date(2026, 9, 14), 164.26)]}
        g.extend_fund(hu, [("2026-09-16", 164.40), (THU, 164.50)])
        self.assertAlmostEqual(hu.growth(WED, THU), (164.50 / 164.40 - 1) * 100)
        d = g.dividends_by_exdate({"PLZL": [{"last_buy_date": "2026-09-16", "dividend_net": 50.0}]}, [WED, THU])
        self.assertEqual(d["PLZL"], [(THU, 50.0)])


if __name__ == "__main__":
    unittest.main()
