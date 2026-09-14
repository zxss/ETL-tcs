"""
Офлайн-разложение баланса Этапа 2 (audit/stage2_decompose.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import stage2_decompose as sd                 # noqa: E402

MSK = dt.timezone(dt.timedelta(hours=3))


def snap(phase, when, total, etf=0.0, shares=0.0, cash=None):
    t = dt.datetime.fromisoformat(when).replace(tzinfo=MSK)
    return {"phase": phase, "captured": t, "captured_at": t.isoformat(),
            "run_dir": f"{t:%Y%m%d-%H%M}-{phase}", "total_portfolio_rub": total,
            "etf_value_rub": etf, "shares_value_rub": shares,
            "free_cash_rub": cash if cash is not None else total - etf - shares,
            "account_id": "acc"}


def led(when, side, amount, mode="broker"):
    return {"ts": dt.datetime.fromisoformat(when).replace(tzinfo=MSK), "account_id": "acc",
            "mode": mode, "side": side, "amount_rub": amount, "reason": "sweep", "run_id": None}


def two_days():
    """День 1: парковка, +20 интрадей, TMON +800 днём; вечером продажа паёв на
    10 000 под ночную покупку; ночью бумага +50, TMON +800; утром выход и сweep."""
    snaps = [snap("PREP", "2026-09-15 08:46", 2_000_000),
             snap("CLOSE", "2026-09-15 09:11", 2_000_000, etf=1_950_000),
             snap("ORDER", "2026-09-15 09:16", 2_000_000, etf=1_950_000),
             snap("CLEANUP", "2026-09-15 18:21", 2_000_820, etf=1_950_800),
             snap("OVERNIGHT", "2026-09-15 18:36", 2_000_820, etf=1_940_800, shares=10_000),
             snap("PREP", "2026-09-16 08:46", 2_001_670, etf=1_941_600, shares=10_050),
             snap("CLOSE", "2026-09-16 09:11", 2_001_670, etf=1_951_670)]
    ledger = [led("2026-09-15 09:10:30", "BUY", 1_950_000),
              led("2026-09-15 18:35:20", "SELL", 10_000),
              led("2026-09-16 09:10:40", "BUY", 10_070)]
    return snaps, ledger


class TestDecompose(unittest.TestCase):

    def test_treasury_separated_from_trading(self):
        snaps, ledger = two_days()
        iv = sd.intervals(snaps, ledger)
        days = sd.by_day(iv)
        d1 = days[dt.date(2026, 9, 15)]
        self.assertAlmostEqual(d1["интрадей"], 20.0)
        self.assertAlmostEqual(d1["вечер"], 0.0)          # продажа паёв под ночь — не доход
        self.assertAlmostEqual(d1["ночь"], 50.0)          # ночь 15→16 относится к 15.09
        self.assertAlmostEqual(d1["treasury"], 1_600.0)   # 800 днём + 800 ночью
        self.assertEqual(set(days), {dt.date(2026, 9, 15)})

    def test_segments(self):
        snaps, ledger = two_days()
        segs = [(r["from"][-5:].strip("-"), r["segment"]) for r in sd.intervals(snaps, ledger)]
        self.assertEqual([s for _, s in segs],
                         ["утро", "интрадей", "интрадей", "вечер", "ночь", "ночь"])

    def test_identity_and_checks(self):
        snaps, ledger = two_days()
        iv = sd.intervals(snaps, ledger)
        msgs = sd.checks(snaps, iv, ledger)
        self.assertTrue(msgs[0].startswith("✓ сходимость"))
        self.assertTrue(any("все 3 операций" in m for m in msgs))

    def test_misattributed_flow_is_flagged(self):
        snaps, ledger = two_days()
        ledger[0]["ts"] = snaps[1]["captured"] + dt.timedelta(minutes=1)   # парковка «после» снимка CLOSE
        msgs = sd.checks(snaps, sd.intervals(snaps, ledger), ledger)
        self.assertTrue(any("проверить" in m for m in msgs))

    def test_virtual_ledger_warned_and_ignored(self):
        snaps, ledger = two_days()
        ledger.append(led("2026-09-15 12:00", "BUY", 5_000, mode="virtual"))
        iv = sd.intervals(snaps, ledger)
        self.assertAlmostEqual(sd.by_day(iv)[dt.date(2026, 9, 15)]["treasury"], 1_600.0)
        self.assertTrue(any("mode≠broker" in m for m in sd.checks(snaps, iv, ledger)))

    def test_archive_without_close_night_until_next_prep(self):
        """Архив попытки 1: +52,63 ₽ между OVERNIGHT 10.09 и PREP 11.09 — ночь 10.09."""
        snaps = [snap("PREP", "2026-09-10 09:51", 100_000),
                 snap("ORDER", "2026-09-10 10:05", 100_000),
                 snap("CLEANUP", "2026-09-10 18:20", 100_000),
                 snap("OVERNIGHT", "2026-09-10 18:41", 100_000),
                 snap("PREP", "2026-09-11 09:52", 100_052.6304, shares=9_996.8),
                 snap("ORDER", "2026-09-11 10:05", 100_060.6304, shares=10_004.8)]
        days = sd.by_day(sd.intervals(snaps, []))
        self.assertAlmostEqual(days[dt.date(2026, 9, 10)]["ночь"], 52.6304)
        self.assertAlmostEqual(days[dt.date(2026, 9, 11)]["интрадей"], 8.0)

    def test_weekend_night_belongs_to_friday(self):
        snaps = [snap("OVERNIGHT", "2026-09-18 18:36", 1_000, etf=900),
                 snap("PREP", "2026-09-21 08:46", 1_010, etf=905),
                 snap("CLOSE", "2026-09-21 09:11", 1_010, etf=905)]
        days = sd.by_day(sd.intervals(snaps, []))
        self.assertEqual(list(days), [dt.date(2026, 9, 18)])
        self.assertAlmostEqual(days[dt.date(2026, 9, 18)]["treasury"], 5.0)
        self.assertAlmostEqual(days[dt.date(2026, 9, 18)]["ночь"], 5.0)

    def test_load_snapshots_from_run_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            for name, b in (("20260915-091001-CLOSE", {"phase": "CLOSE", "captured_at": "2026-09-15T09:11:00+03:00",
                                                        "total_portfolio_rub": 1.0}),
                            ("20260915-084501-PREP", {"phase": "PREP", "captured_at": "2026-09-15T08:46:00+03:00",
                                                       "total_portfolio_rub": 1.0})):
                os.makedirs(os.path.join(d, "runs", name))
                with open(os.path.join(d, "runs", name, "balance.json"), "w") as f:
                    json.dump(b, f)
            s = sd.load_snapshots(d)
            self.assertEqual([x["phase"] for x in s], ["PREP", "CLOSE"])


if __name__ == "__main__":
    unittest.main()
