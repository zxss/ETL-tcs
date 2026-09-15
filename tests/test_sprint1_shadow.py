"""
Теневой журнал правила Спринта 1 (research/sprint1_shadow.py). Без сети и БД.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import sprint1_shadow as sh                # noqa: E402

DAYS = [dt.date(2026, 9, 16), dt.date(2026, 9, 17)]        # новое расписание: основная с 09:10


def bars(price):
    tm, o, c = [], [], []
    for d in DAYS:
        t = dt.datetime.combine(d, dt.time(9, 10))
        while t.time() <= dt.time(18, 50):
            oo, cc = price(t.time())
            tm.append(t); o.append(oo); c.append(cc)
            t += dt.timedelta(minutes=5)
    return es.Bars(tm, o, c)


STOCK = bars(lambda t: (100.0, 99.0) if t == dt.time(18, 15) else (100.0, 100.0))
IDX = bars(lambda t: (1000.0, 1000.0))
EV = pd.DataFrame({
    "message_id": [1, 2, 3, 4],
    "posted": [dt.datetime(2026, 9, 15, 21, 0), dt.datetime(2026, 9, 16, 12, 1),
               dt.datetime(2026, 9, 16, 12, 30), dt.datetime(2026, 9, 16, 18, 30)],
    "ticker": ["LKOH"] * 4,
    "category": ["FINANCIAL", "FINANCIAL", "CORPORATE", "FINANCIAL"],
    "sentiment": [-0.5, -1.0, -1.0, -0.5]})


class TestDayTrades(unittest.TestCase):

    def test_only_rule_events_entered_that_day(self):
        rows = sh.day_trades(EV, {"LKOH": STOCK}, IDX, DAYS, DAYS[0], cost=lambda tk: 0.1)
        main = {r["message_id"]: r for r in rows if r["variant"] == "main"}
        self.assertEqual(set(main), {1, 2})                   # CORPORATE — нет; 18:30 — вход 17.09
        self.assertEqual(main[1]["t0"], "2026-09-16T09:10")   # ночная новость — начало основной
        self.assertEqual(main[2]["t0"], "2026-09-16T12:05")
        self.assertAlmostEqual(main[2]["net_short"], 0.9)     # −(−1 %) − 0,1
        lag = {r["message_id"]: r for r in rows if r["variant"] == "lag10"}
        self.assertEqual(lag[2]["t0"], "2026-09-16T12:15")
        nxt = sh.day_trades(EV, {"LKOH": STOCK}, IDX, DAYS, DAYS[1], cost=lambda tk: 0.1)
        self.assertEqual({r["message_id"] for r in nxt if r["variant"] == "main"}, {4})


class TestJournal(unittest.TestCase):

    def test_merge_is_idempotent_and_summary(self):
        rows = sh.day_trades(EV, {"LKOH": STOCK}, IDX, DAYS, DAYS[0], cost=lambda tk: 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shadow", "journal.csv")
            self.assertEqual(sh.merge_journal(rows, path), 4)
            self.assertEqual(sh.merge_journal(rows, path), 0)
            with open(path, encoding="utf-8") as f:
                got = list(csv.DictReader(f))
            self.assertEqual(len(got), 4)
            self.assertNotIn("text", got[0])
            text = sh.summary(path)
            self.assertIn("| main | 2 | 1 |", text)


if __name__ == "__main__":
    unittest.main()
