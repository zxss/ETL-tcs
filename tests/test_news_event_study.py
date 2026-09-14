"""
Исследование «новости → гэпы» (research/news_event_study.py). Без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                      # noqa: E402

from research import news_event_study as ns              # noqa: E402

D = dt.date
T = dt.datetime
# пн 07.09 … пт 11.09, пн 14.09 (2026)
DAYS = [D(2026, 9, 7), D(2026, 9, 8), D(2026, 9, 9), D(2026, 9, 10), D(2026, 9, 11),
        D(2026, 9, 14)]


class TestTickers(unittest.TestCase):

    def test_hashtags_and_aliases(self):
        self.assertEqual(ns.tickers_in("🇷🇺#SBER / Сбер не ждёт роста резервов"), {"SBER"})
        self.assertEqual(ns.tickers_in("#YNDX переименовали"), {"YDEX"})
        self.assertEqual(ns.tickers_in("#SNGSP дивиденды"), {"SNGS", "SNGSP"})
        self.assertEqual(ns.tickers_in("#ORCL #увольнения"), set())      # не наша вселенная
        self.assertEqual(ns.tickers_in("#MOEXX"), set())

    def test_company_names(self):
        self.assertEqual(ns.tickers_in("Газпром будет наращивать добычу на Ямале"), {"GAZP"})
        self.assertEqual(ns.tickers_in("ПАССАЖИРОПОТОК ГРУППЫ \"АЭРОФЛОТ\" СНИЗИЛСЯ"), {"AFLT"})
        self.assertEqual(ns.tickers_in("Сургутнефтегаз отчитался"), {"SNGS", "SNGSP"})
        self.assertEqual(ns.tickers_in("ГК «Самолет» — продажи"), {"SMLT"})
        self.assertEqual(ns.tickers_in("С уходом ПИКа с биржи рынок потеряет ориентир"), {"PIKK"})

    def test_homonyms_do_not_match(self):
        for text in ("Газпромнефть увеличила переработку",
                     "Газпромбанк снизил ставки",
                     "Алжир закроет пространство для самолетов из ОАЭ",
                     "МТС-Банк разместит облигации",
                     "лента новостей за день",
                     "рынок в позитиве, пик активности",
                     "эталон качества",
                     "нет торгов на Мосбирже"):
            self.assertEqual(ns.tickers_in(text), set(), text)

    def test_price_report_detected(self):
        self.assertTrue(ns.is_price_report("⚠️🇷🇺#SMLT = -10%\nаналитики полагают…"))
        self.assertTrue(ns.is_price_report("#GAZP +3,5% на новостях"))
        self.assertFalse(ns.is_price_report("🇷🇺#VTBR\nВТБ сократит расходы на 10%"))


class TestTone(unittest.TestCase):

    def test_positive_negative_neutral(self):
        self.assertEqual(ns.tone("Совет директоров рекомендовал дивиденды 25 руб"), 1)
        self.assertEqual(ns.tone("Компания объявила байбэк на 10 млрд"), 1)
        self.assertEqual(ns.tone("Компания не будет выплачивать дивиденды"), -1)
        self.assertEqual(ns.tone("Против компании введены санкции"), -1)
        self.assertEqual(ns.tone("Совет директоров обсудит стратегию до 2030г"), 0)


class TestWindows(unittest.TestCase):

    def test_open_schedule(self):
        self.assertEqual(ns.open_cutoff(D(2024, 6, 11)), dt.time(9, 50))
        self.assertEqual(ns.open_cutoff(D(2024, 11, 1)), dt.time(7, 0))
        self.assertEqual(ns.open_cutoff(D(2025, 6, 11)), dt.time(6, 50))

    def test_daily_window(self):
        self.assertEqual(ns.daily_window(T(2026, 9, 8, 3, 0), DAYS), ("gap", D(2026, 9, 8)))
        self.assertEqual(ns.daily_window(T(2026, 9, 8, 12, 0), DAYS), ("day", D(2026, 9, 8)))
        self.assertEqual(ns.daily_window(T(2026, 9, 8, 23, 55), DAYS), ("gap", D(2026, 9, 9)))
        self.assertEqual(ns.daily_window(T(2026, 9, 12, 12, 0), DAYS), ("gap", D(2026, 9, 14)))

    def test_daily_window_follows_old_session_hours(self):
        """В 2024 торги начинались около 09:50: новость в 08:00 — ещё окно гэпа."""
        days = [D(2024, 6, 10), D(2024, 6, 11)]
        self.assertEqual(ns.daily_window(T(2024, 6, 11, 8, 0), days), ("gap", D(2024, 6, 11)))
        days25 = [D(2025, 6, 10), D(2025, 6, 11)]
        self.assertEqual(ns.daily_window(T(2025, 6, 11, 8, 0), days25), ("day", D(2025, 6, 11)))

    def test_exec_windows(self):
        tue, wed, fri, mon = D(2026, 9, 8), D(2026, 9, 9), D(2026, 9, 11), D(2026, 9, 14)
        self.assertEqual(ns.exec_windows(T(2026, 9, 8, 12, 0), DAYS), [("known", tue)])
        self.assertEqual(ns.exec_windows(T(2026, 9, 8, 20, 0), DAYS),
                         [("known", wed), ("hold", tue)])
        self.assertEqual(ns.exec_windows(T(2026, 9, 9, 8, 0), DAYS),
                         [("known", wed), ("hold", tue)])
        self.assertEqual(ns.exec_windows(T(2026, 9, 12, 12, 0), DAYS),
                         [("known", mon), ("hold", fri)])


class TestStats(unittest.TestCase):

    def test_abnormal_is_leave_one_out(self):
        f = pd.DataFrame({"A": [0.03], "B": [0.01], "C": [0.02]})
        ar = ns.abnormal(f)
        self.assertAlmostEqual(ar.at[0, "A"], 0.03 - 0.015)
        self.assertAlmostEqual(ar.sum(axis=1)[0], 0.03 - 0.015 + 0.01 - 0.025 + 0.02 - 0.02)

    def test_by_date_averages_within_date_first(self):
        """Десять наблюдений одной даты не должны весить в десять раз больше."""
        vals = [1.0] * 10 + [-1.0]
        dates = ["d1"] * 10 + ["d2"]
        st = ns.by_date(vals, dates)
        self.assertEqual(st["n_dates"], 2)
        self.assertAlmostEqual(st["mean"], 0.0)

    def test_paired_by_date(self):
        df = pd.DataFrame({"date": ["a", "a", "b", "b", "c", "c"],
                           "f": [True, False, True, False, True, False],
                           "v": [2.0, 1.0, 3.0, 1.0, 2.5, 1.0]})
        p = ns.paired_by_date(df, "f", "v")
        self.assertEqual(p["n_dates"], 3)
        self.assertAlmostEqual(p["diff"], 1.5)


class TestSchemes(unittest.TestCase):

    def daily(self):
        rows = []
        for i, d in enumerate(DAYS):
            for tk, base in (("SBER", 100.0), ("GAZP", 50.0), ("LKOH", 10.0)):
                bump = 1.05 if (tk == "SBER" and d == D(2026, 9, 9)) else 1.0
                rows.append({"ticker": tk, "date": d, "open": base * bump, "close": base})
        return pd.DataFrame(rows)

    def test_scheme_a_flags_gap_news_and_next_day(self):
        ev = [{"message_id": 1, "ticker": "SBER", "avail": T(2026, 9, 9, 5, 0),
               "tone": 1, "price_report": False},
              {"message_id": 2, "ticker": "GAZP", "avail": T(2026, 9, 9, 7, 0),
               "tone": 0, "price_report": True}]                # отчёт о цене — исключён
        res = ns.scheme_a(self.daily(), ev)["df"]
        row = res[(res.ticker == "SBER") & (res.date == D(2026, 9, 9))].iloc[0]
        self.assertTrue(row.gap_news and row.known_news)
        self.assertAlmostEqual(row.ar_gap, 5.0)                 # гэп +5% против 0% у остальных
        self.assertEqual(row.tone, 1)
        self.assertFalse(res[(res.ticker == "GAZP")].gap_news.any())
        prev = res[(res.ticker == "SBER") & (res.date == D(2026, 9, 8))].iloc[0]
        self.assertAlmostEqual(prev.ar_next_gap, 5.0)          # D+1 выровнен по дате

    def test_scheme_b_exec_window_return(self):
        rows = []
        for d in DAYS:
            for tk, p in (("SBER", 100.0), ("GAZP", 50.0), ("LKOH", 10.0)):
                ex = p * (1.02 if (tk == "SBER" and d == D(2026, 9, 9)) else 1.0)
                rows.append({"ticker": tk, "date": d, "hhmm": "18:30", "open": p, "close": p})
                rows.append({"ticker": tk, "date": d, "hhmm": "10:00", "open": ex, "close": ex})
        bars = pd.DataFrame(rows)
        ev = [{"message_id": 1, "ticker": "SBER", "avail": T(2026, 9, 8, 21, 0),
               "tone": -1, "price_report": False}]
        res = ns.scheme_b(bars, ev)["df"]
        row = res[(res.ticker == "SBER") & (res.date == D(2026, 9, 8))].iloc[0]
        self.assertTrue(row.hold)                               # пришла во время удержания
        self.assertAlmostEqual(row.ar_exec, 2.0)                # 18:35 вт → 10:00 ср
        nxt = res[(res.ticker == "SBER") & (res.date == D(2026, 9, 9))].iloc[0]
        self.assertTrue(nxt.known)                              # и известна к решению в ср
        self.assertEqual(nxt.tone, -1)


if __name__ == "__main__":
    unittest.main()
