"""
Расписание Мосбиржи с 14.09.2026: аукцион открытия 09:00, торги с 09:10
(T-Invest TradingSchedules). Утренние фазы Этапа 2 привязаны к открытию.

Времена живут в трёх местах — крон, .env.example, дефолты config.py — и cron
.env не читает. Тест не даёт им разъехаться.
"""
from __future__ import annotations

import datetime as dt
import importlib
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

from tft_forecast.quotes import is_market_open           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSK = dt.timezone(dt.timedelta(hours=3))
MON = dt.date(2026, 9, 14)
PHASES = ("prep", "close", "order", "cleanup", "overnight")


def at(h, m, day=MON):
    return dt.datetime(day.year, day.month, day.day, h, m, tzinfo=MSK)


def crontab_times() -> dict[str, str]:
    out = {}
    with open(os.path.join(ROOT, "scripts", "stage2_crontab"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+\*\s+\*\s+1-5\s+.*stage2_demo\s+(\w+)", line)
            if m and m.group(3) in PHASES:
                out[m.group(3)] = f"{int(m.group(2)):02d}:{int(m.group(1)):02d}"
    return out


def env_example_times() -> dict[str, str]:
    out = {}
    with open(os.path.join(ROOT, ".env.example"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^STAGE2_(\w+)_TIME=([0-9:]+)", line)
            if m:
                out[m.group(1).lower()] = m.group(2)
    return out


def config_default_times() -> dict[str, str]:
    """Дефолты кода: .env не подгружается и STAGE2_*_TIME из окружения убраны."""
    import config
    clean = {k: v for k, v in os.environ.items() if not re.match(r"STAGE2_\w+_TIME$", k)}
    try:
        with mock.patch.dict(os.environ, clean, clear=True), \
             mock.patch("dotenv.load_dotenv", lambda *a, **k: False):
            c = importlib.reload(config)
            return {ph: getattr(c, f"STAGE2_{ph.upper()}_TIME") for ph in PHASES}
    finally:
        importlib.reload(config)


class TestMarketOpen(unittest.TestCase):

    def test_new_session_hours(self):
        self.assertFalse(is_market_open(at(8, 45)))      # PREP — до аукциона
        self.assertFalse(is_market_open(at(9, 5)))       # аукцион открытия
        self.assertTrue(is_market_open(at(9, 10)))       # торги
        self.assertTrue(is_market_open(at(9, 45)))       # раньше считалось «закрыт»
        self.assertTrue(is_market_open(at(18, 35)))
        self.assertFalse(is_market_open(at(23, 55)))

    def test_weekend_closed(self):
        self.assertFalse(is_market_open(at(12, 0, dt.date(2026, 9, 12))))


class TestPhaseSchedule(unittest.TestCase):

    def test_three_sources_agree(self):
        cron, env, cfg = crontab_times(), env_example_times(), config_default_times()
        self.assertEqual(set(cron), set(PHASES), cron)
        for ph in PHASES:
            self.assertEqual(cron[ph], env[ph], f"{ph}: крон и .env.example разошлись")
            self.assertEqual(cron[ph], cfg[ph], f"{ph}: крон и config.py разошлись")

    def test_morning_follows_the_open(self):
        t = {k: dt.time.fromisoformat(v) for k, v in crontab_times().items()}
        self.assertLess(t["prep"], dt.time(9, 0), "PREP должен успеть до аукциона 09:00")
        self.assertEqual(t["close"], dt.time(9, 10), "выход — первая минута торгов")
        self.assertLess(t["close"], t["order"])
        self.assertFalse(is_market_open(at(t["prep"].hour, t["prep"].minute)),
                         "PREP строит план от закрытия — рынок ещё закрыт")

    def test_evening_unchanged(self):
        t = crontab_times()
        self.assertEqual((t["cleanup"], t["overnight"]), ("18:20", "18:35"))


if __name__ == "__main__":
    unittest.main()
