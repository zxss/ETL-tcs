"""
Догрузка 5-минуток из годовых архивов (services/backfill_5m.py) и календарь
сессий (research/session_calendar.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import session_calendar as sc              # noqa: E402
from services import backfill_5m as bf                   # noqa: E402

UTC = dt.timezone.utc
MSK = dt.timezone(dt.timedelta(hours=3))
UID = "e6123145-9665-43e0-8413-cd61b8aa9b13"

# Реальный формат архива: uid;time;open;CLOSE;high;low;volume;
DAY1 = (f"{UID};2024-12-19T07:00:00Z;230.9;230;230.9;229.5;30;\n"
        f"{UID};2024-12-19T07:01:00Z;230.15;230.03;230.78;230.01;77;\n"
        f"{UID};2024-12-19T07:04:00Z;230.03;231.00;231.20;230.02;98\n"
        f"{UID};2024-12-19T07:05:00Z;231.0;231.1;231.3;230.9;5;\n"
        f"{UID};2024-12-19T21:02:00Z;232.0;232.1;232.2;231.9;7;\n")   # 00:02 МСК 20.12
DAY0 = f"{UID};2024-12-18T07:00:00Z;229;229.5;229.9;228.8;11;\n"


def make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


class TestArchive(unittest.TestCase):

    def test_field_order_open_close_high_low(self):
        rows = bf.parse_minute_csv(DAY1)
        ts, o, h, lo, c, v = rows[0]
        self.assertEqual(ts, dt.datetime(2024, 12, 19, 7, 0, tzinfo=UTC))
        self.assertEqual((o, h, lo, c, v), (230.9, 230.9, 229.5, 230.0, 30))

    def test_garbage_and_zero_open_skipped(self):
        rows = bf.parse_minute_csv("мусор\n" + f"{UID};2024-12-19T07:00:00Z;0;1;1;1;1;\n"
                                   + f"{UID};;1;1;1;1;1;\n")
        self.assertEqual(rows, [])

    def test_minutes_fold_into_5m(self):
        bars = bf.to_5m(bf.parse_minute_csv(DAY1))
        first = bars[0]
        self.assertEqual(first[0], dt.datetime(2024, 12, 19, 7, 0, tzinfo=UTC))
        # open первой минуты, high/low — экстремумы, close последней, объём — сумма
        self.assertEqual(first[1:], (230.9, 231.2, 229.5, 231.0, 30 + 77 + 98))
        self.assertEqual(bars[1][0], dt.datetime(2024, 12, 19, 7, 5, tzinfo=UTC))

    def test_msk_date_filter_across_utc_files(self):
        blob = make_zip({f"{UID}_20241218.csv": DAY0, f"{UID}_20241219.csv": DAY1})
        only19 = bf.archive_5m(blob, dt.date(2024, 12, 19), dt.date(2024, 12, 19))
        self.assertEqual({b[0].astimezone(MSK).date() for b in only19}, {dt.date(2024, 12, 19)})
        self.assertEqual(len(only19), 2)
        only20 = bf.archive_5m(blob, dt.date(2024, 12, 20), dt.date(2024, 12, 20))
        self.assertEqual(len(only20), 1)            # 21:02Z = 00:02 МСК следующего дня

    def test_file_day(self):
        self.assertEqual(bf._file_day(f"{UID}_20240408.csv"), dt.date(2024, 4, 8))
        self.assertIsNone(bf._file_day("readme.txt"))


class TestStateAndSql(unittest.TestCase):

    def test_state_resets_on_new_range(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s", "state.json")
            st = bf.load_state(p, dt.date(2024, 5, 21), dt.date(2025, 11, 30))
            st["done"]["SBER:2024"] = {"bars": 1}
            bf.save_state(p, st)
            again = bf.load_state(p, dt.date(2024, 5, 21), dt.date(2025, 11, 30))
            self.assertIn("SBER:2024", again["done"])
            other = bf.load_state(p, dt.date(2024, 1, 1), dt.date(2025, 11, 30))
            self.assertEqual(other["done"], {})

    def test_never_overwrites_etl_bars(self):
        self.assertIn("ON CONFLICT (ticker, ts) DO NOTHING", bf.INSERT_5M_SQL)
        self.assertIn("VALUES %s", bf.INSERT_5M_SQL)

    def test_retry_wait_uses_reset_header(self):
        self.assertEqual(bf._retry_wait({"x-ratelimit-reset": "3"}, 1), 4.0)
        self.assertEqual(bf._retry_wait({}, 2), 10.0)

    def test_compare_bars(self):
        t0 = dt.datetime(2025, 12, 1, 7, 0, tzinfo=UTC)
        t1 = t0 + dt.timedelta(minutes=5)
        arch = [(t0, 1.0, 2.0, 0.5, 1.5, 10), (t1, 1.5, 1.6, 1.4, 1.5, 3)]
        db = [(t0, 1.0, 2.0, 0.5, 1.5, 10), (t1, 1.5, 1.7, 1.4, 1.5, 3),
              (t1 + dt.timedelta(minutes=5), 1, 1, 1, 1, 1)]
        r = bf.compare_bars(arch, db)
        self.assertEqual((r["common"], r["only_db"], r["only_archive"]), (2, 1, 0))
        self.assertEqual(r["ohlc_equal_share"], 0.5)
        self.assertEqual(r["volume_equal_share"], 1.0)

    def test_research_table_is_separate(self):
        self.assertIn("INSERT INTO research_bars_5m",
                      bf.INSERT_5M_SQL.format(table=bf._table("research_bars_5m")))
        with self.assertRaises(ValueError):
            bf._table("market_data; DROP TABLE x")
        self.assertNotEqual(bf.state_path_for("research_bars_5m"), bf.STATE_PATH)
        self.assertEqual(bf.state_path_for("market_data_5m"), bf.STATE_PATH)
        self.assertIn("UNIQUE (ticker, ts)", bf.CREATE_RESEARCH_SQL)

    def test_gaps_longer_than_5_days(self):
        d = dt.date(2024, 7, 1)
        g = bf.gaps({"YDEX": {d, d + dt.timedelta(days=1), d + dt.timedelta(days=10)},
                     "SBER": {d, d + dt.timedelta(days=3)}})
        self.assertEqual([(x["ticker"], x["days"]) for x in g], [("YDEX", 9)])


class TestSessionCalendar(unittest.TestCase):

    def test_before_and_after_new_schedule(self):
        old, new = sc.session(dt.date(2026, 9, 11)), sc.session(dt.date(2026, 9, 14))
        self.assertEqual((old.opening_auction, old.main_open, old.short_entry_bar),
                         (dt.time(9, 50), dt.time(10, 0), dt.time(10, 5)))
        self.assertEqual((new.opening_auction, new.main_open, new.short_entry_bar),
                         (dt.time(9, 0), dt.time(9, 10), dt.time(9, 15)))
        self.assertEqual(sc.main_open(dt.date(2025, 3, 3)), dt.time(10, 0))
        self.assertEqual(sc.short_entry(dt.date(2026, 9, 15)), dt.time(9, 15))

    def test_morning_and_weekend_flags(self):
        self.assertFalse(sc.session(dt.date(2024, 6, 3)).morning)
        self.assertTrue(sc.session(dt.date(2025, 3, 3)).morning)
        self.assertFalse(sc.session(dt.date(2026, 9, 14)).morning)
        self.assertTrue(sc.session(dt.date(2025, 3, 1)).weekend)

    def test_expected_main_bars(self):
        # 10:00…18:35 — 104 бара; 09:10…18:50 — 116
        self.assertEqual(sc.session(dt.date(2025, 3, 3)).expected_main_bars, 104)
        self.assertEqual(sc.session(dt.date(2026, 9, 15)).expected_main_bars, 116)

    def test_fixed_phase_bars(self):
        self.assertEqual(sc.SHORT_EXIT_BAR, dt.time(18, 15))
        self.assertEqual(sc.EVENING_ENTRY_BAR, dt.time(18, 30))


if __name__ == "__main__":
    unittest.main()
