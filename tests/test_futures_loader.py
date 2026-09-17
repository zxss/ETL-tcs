"""
Загрузчик фьючерсов Спринта 2 (research/futures_loader.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import futures_loader as fl                # noqa: E402
from services import backfill_5m as bf                   # noqa: E402

FUT = [
    {"ticker": "BRG3", "classCode": "SPBFUT", "uid": "u1", "expirationDate": "2023-02-01T00:00:00Z"},
    {"ticker": "BRZ1", "classCode": "SPBFUT", "uid": "u0", "expirationDate": "2021-12-01T00:00:00Z"},
    {"ticker": "SiH2", "classCode": "SPBFUT", "uid": "u2", "expirationDate": "2022-03-17T00:00:00Z"},
    {"ticker": "SVH3", "classCode": "SPBFUT", "uid": "u3", "expirationDate": "2023-03-16T00:00:00Z"},
    {"ticker": "BRX6", "classCode": "SPBFUT", "uid": "u4", "expirationDate": "2026-11-03T00:00:00Z"},
    {"ticker": "BRF2", "classCode": "FORTS", "uid": "u5", "expirationDate": "2022-01-03T00:00:00Z"},
    {"ticker": "BRENTX", "classCode": "SPBFUT", "uid": "u6", "expirationDate": "2023-01-01T00:00:00Z"},
]


class TestSelect(unittest.TestCase):

    def test_roots_classes_and_years(self):
        c = {x["ticker"]: x for x in fl.select_contracts(FUT, dt.date(2022, 1, 1), dt.date(2026, 9, 11))}
        self.assertEqual(set(c), {"BRG3", "SiH2", "BRX6"})          # серебро, истёкший до периода, чужой класс — нет
        self.assertEqual(c["BRG3"]["years"], [2022, 2023])
        self.assertEqual(c["SiH2"]["years"], [2022])                  # 2021 — вне периода
        self.assertEqual(c["BRX6"]["years"], [2025, 2026])            # экспирация после конца периода — год экспирации в периоде

    def test_stock_futures_roots(self):
        fut = FUT + [{"ticker": "SRZ4", "classCode": "SPBFUT", "uid": "u7", "expirationDate": "2024-12-20T00:00:00Z"},
                     {"ticker": "GZH5", "classCode": "SPBFUT", "uid": "u8", "expirationDate": "2025-03-21T00:00:00Z"}]
        got = {c["ticker"] for c in fl.select_contracts(fut, dt.date(2022, 1, 1), dt.date(2026, 9, 11), fl.STOCK_ROOTS)}
        self.assertEqual(got, {"SRZ4", "GZH5"})                     # сырьё и валюта не попадают
        self.assertEqual({c["ticker"] for c in fl.select_contracts(fut, dt.date(2022, 1, 1), dt.date(2026, 9, 11))},
                         {"BRG3", "SiH2", "BRX6"})                   # по умолчанию — прежние корни

    def test_futures_table_whitelisted(self):
        self.assertEqual(bf._table("research_fut_5m"), "research_fut_5m")
        self.assertIn("CREATE TABLE IF NOT EXISTS research_fut_5m",
                      bf.CREATE_RESEARCH_SQL.format(table="research_fut_5m"))


if __name__ == "__main__":
    unittest.main()
