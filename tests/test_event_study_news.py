"""
Event study новостей v2 (research/event_study_news.py). Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import event_study_news as es              # noqa: E402
from research import fetch_dividends as fd               # noqa: E402
from research import news_classify as ncl                # noqa: E402

DAYS = [dt.date(2025, 3, 3), dt.date(2025, 3, 4), dt.date(2025, 3, 5), dt.date(2025, 3, 6)]


def make_bars(price_of):
    """Бары 10:00–18:45 и 19:05–20:00 по DAYS; price_of(day_idx, time) → (open, close)."""
    tm, o, c = [], [], []
    for k, d in enumerate(DAYS):
        for start, end in ((dt.time(10, 0), dt.time(18, 45)), (dt.time(19, 5), dt.time(20, 0))):
            t = dt.datetime.combine(d, start)
            while t.time() <= end:
                oo, cc = price_of(k, t.time())
                tm.append(t); o.append(oo); c.append(cc)
                t += dt.timedelta(minutes=5)
    return es.Bars(tm, o, c)


def stock_price(k, t):
    base = 100.0 + k
    if k == 0 and t == dt.time(11, 5):
        return 100.0, 101.0
    if k == 0 and t == dt.time(11, 30):
        return 101.0, 102.0
    if t == dt.time(18, 45):
        return base, 103.0 + k                              # close основной: 103, 104, 105, 106
    if k == 0 and dt.time(11, 30) < t < dt.time(18, 45):
        return 102.0, 102.0
    if t == dt.time(10, 0):
        return 99.0 + k, base                               # открытие дня: 99, 100, 101, 102
    return base, base


IDX = make_bars(lambda k, t: (1000.0, 1000.0))
STOCK = make_bars(stock_price)


class TestEntryBySession(unittest.TestCase):

    def test_during_session_next_bar(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 11, 5))
        self.assertEqual([round(o[f"raw_{w}"], 6) for w in es.WINDOWS], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_evening_and_night_news_enter_at_next_open(self):
        for posted in (dt.datetime(2025, 3, 3, 19, 7), dt.datetime(2025, 3, 3, 23, 55),
                       dt.datetime(2025, 3, 4, 7, 30), dt.datetime(2025, 3, 4, 9, 55)):
            o = es.outcome(posted, es.LAG_REACTION, STOCK, IDX, DAYS)
            self.assertEqual(o["t0"], dt.datetime(2025, 3, 4, 10, 0), posted)
            self.assertEqual(o["p0"], 100.0)                                   # первая сделка 10:00
            self.assertEqual((o["end_eod"], o["end_1d"], o["end_2d"]), (DAYS[1], DAYS[2], DAYS[3]))

    def test_weekend_news_enter_monday(self):
        m = es.entry_moment(dt.datetime(2025, 3, 1, 12, 0), DAYS, set(DAYS))  # суббота
        self.assertEqual(m, dt.datetime(2025, 3, 3, 10, 0))

    def test_late_session_news_skip_to_next_open(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 18, 48), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 4, 10, 0))              # бар 18:50 не бывает — вечер

    def test_intraday_window_capped_at_main_close(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 18, 32), es.LAG_REACTION, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 18, 35))
        self.assertAlmostEqual(o["raw_30m"], (103.0 / 102.0 - 1) * 100)       # close 18:45, не вечерний бар

    def test_trade_lag(self):
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_TRADE, STOCK, IDX, DAYS)
        self.assertEqual(o["t0"], dt.datetime(2025, 3, 3, 11, 15))


class TestDividends(unittest.TestCase):

    def test_window_across_ex_date_gets_dividend(self):
        divs = [(DAYS[2], 2.0)]                                   # отсечка 05.03
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS, divs)
        self.assertAlmostEqual(o["raw_1d"], 4.0)                  # 04.03 — до отсечки
        self.assertAlmostEqual(o["raw_2d"], (105.0 + 2.0) / 100.0 * 100 - 100)
        self.assertEqual(o["div_2d"], 2.0)

    def test_load_dividends_ex_date_is_next_trading_day(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("ticker,last_buy_date,record_date,dividend_net,close_price\nSBER,2025-03-04,2025-03-06,3.5,300\n")
        d = es.load_dividends(f.name, DAYS)
        os.unlink(f.name)
        self.assertEqual(d["SBER"], [(DAYS[2], 3.5)])

    def test_parse_get_dividends(self):
        payload = {"dividends": [
            {"dividendNet": {"units": "37", "nano": 640000000}, "lastBuyDate": "2026-07-17T00:00:00Z",
             "recordDate": "2026-07-20T00:00:00Z", "closePrice": {"units": "277", "nano": 0}},
            {"dividendNet": {"units": "0", "nano": 0}, "lastBuyDate": "2025-07-17T00:00:00Z"}]}
        rows = fd.parse_dividends("SBER", payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["last_buy_date"], rows[0]["dividend_net"]), ("2026-07-17", 37.64))


class TestEconomics(unittest.TestCase):

    def test_long_pays_cost_and_fund_short_pays_carry(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("fund,date,close\nTMON@,2025-01-01,99\nTMON@,2025-03-03,100\nTMON@,2025-03-05,100.1\n")
        h = es.Hurdle(f.name)
        os.unlink(f.name)
        o = es.outcome(dt.datetime(2025, 3, 3, 11, 2), es.LAG_REACTION, STOCK, IDX, DAYS)
        e = es.economics(o, "ZZZZ", h, {"__fallback__": (0.05, 0.1, 0.0)})
        self.assertAlmostEqual(e["net_long_2d"], 5.0 - 0.13 - 0.1, places=9)
        self.assertAlmostEqual(e["net_short_2d"], -5.0 - 0.13 - 2 * 0.45, places=9)
        self.assertAlmostEqual(e["net_short_30m"], -2.0 - 0.13, places=9)


class TestEvents(unittest.TestCase):

    def posts(self, rows):
        return pd.DataFrame({"message_id": [r[0] for r in rows], "msk": pd.to_datetime([r[1] for r in rows]),
                             "text": [r[2] for r in rows]})

    def test_digest_does_not_displace_real_news(self):
        ev, cnt = es.build_events(self.posts([
            (1, "2024-09-06 07:10", "🗓КАЛЕНДАРЬ НА СЕГОДНЯ\n🇷🇺#TATN Татнефть — ГОСА"),
            (2, "2024-09-06 11:47", "🛢🇷🇺#TATN\nТатнефть заместила поставки на НПЗ Словакии")]), ncl.Classifier())
        self.assertEqual(list(ev["message_id"]), [2])
        self.assertEqual(cnt["digest"], 1)

    def test_spam_dropped_and_two_hour_cluster(self):
        ev, cnt = es.build_events(self.posts([
            (1, "2025-03-03 11:00", "📌 Моя стратегия. СМОТРЕТЬ ВИДЕО: https://youtu.be/x #SBER"),
            (2, "2025-03-03 12:00", "❗️🇷🇺#PLZL #дивиденд\nСД ПОЛЮС: ДИВИДЕНДЫ = 730 РУБ/АКЦ"),
            (3, "2025-03-03 12:30", "❗️🇷🇺#PLZL #дивиденд\nГОСА ПОЛЮС одобрило дивиденды"),
            (4, "2025-03-03 15:00", "❗️🇷🇺#PLZL #дивиденд\nАкционеры ПОЛЮС одобрили дивиденды")]), ncl.Classifier())
        self.assertEqual(list(ev["message_id"]), [2, 4])                     # 3 — повтор в пределах 2 часов
        self.assertEqual((cnt["spam"], cnt["cluster_dup"]), (1, 1))
        self.assertEqual(ev.iloc[0]["category"], "DIVIDEND_ANNOUNCE")


class TestRules(unittest.TestCase):

    def synthetic(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(60):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=i)
            for cat in ncl.CATEGORIES:
                for b, s in (("neg", -0.5), ("pos", 0.5)):
                    row = {"date": d, "category": cat, "bucket": b, "sentiment": 1.0 if b == "pos" else s}
                    for w in es.WINDOWS:
                        eff = 1.0 if (b == "neg" and w == "1d" and cat in ("DIVIDEND_CALENDAR", "FINANCIAL")) else 0.0
                        row[f"ar_{w}"] = -eff + rng.normal(0, 0.5)
                        row[f"net_long_{w}"] = -eff - 0.3 + rng.normal(0, 0.5)
                        row[f"net_short_{w}"] = eff - 0.3 + rng.normal(0, 0.5)
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_calendar_never_becomes_rule(self):
        rules, trials = es.select_rules(self.synthetic())
        self.assertEqual(trials, len(es.RULE_CATEGORIES) * 2 * len(es.RULE_WINDOWS) + 1)
        picked = [(r["category"], r.get("bucket"), r["window"], r["direction"]) for r in rules
                  if r["source"] == "отбор на dev"]
        self.assertEqual(picked, [("FINANCIAL", "neg", "1d", -1)])             # календарь с тем же эффектом — нет
        tz = [r for r in rules if r["source"].startswith("ТЗ")]
        self.assertEqual((tz[0]["category"], tz[0]["window"]), ("DIVIDEND_ANNOUNCE", "2d"))

    def test_daily_series_ir_dsr_and_registry(self):
        ev = self.synthetic()
        days = sorted(ev["date"].unique())
        s = es.rule_daily_series(ev, {"category": "FINANCIAL", "bucket": "neg", "window": "1d",
                                      "direction": -1}, days)
        self.assertGreater(es.information_ratio(s), 0)
        self.assertTrue(0 <= es.deflated_sharpe(s, 100) <= es.deflated_sharpe(s, 2) <= 1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "trials.jsonl")
            self.assertEqual(es.register_trials("dev", 41, "x", p), 41)
            self.assertEqual(es.register_trials("dev-v2", 41, "y", p), 82)


if __name__ == "__main__":
    unittest.main()
