"""
Тесты торгового календаря (Sprint 3.5).

Главное, что проверяется: горизонт H, по которому обучается модель, и окно
прогноза в шапке дашборда считаются ОДНИМ календарём. Модель предсказывает
close.shift(-H) по строкам датасета, то есть H шагов = H строк; шапка обязана
показывать ровно ту же дату.

Запуск:
    python3 -m unittest tests.test_calendar -v
    python3 -m pytest tests/test_calendar.py -q
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                 # noqa: E402
import trading_calendar as tc # noqa: E402

H = 5                          # WEEK_HORIZON_DAYS по умолчанию

# 2026-09-08 — вторник; 2026-09-11 — пятница.
TUE = dt.date(2026, 9, 8)
FRI = dt.date(2026, 9, 11)


class CalendarModeMixin:
    """Переключение режима INCLUDE_WEEKEND_TRADING с восстановлением."""

    def set_mode(self, mode: int) -> None:
        self._saved = getattr(config, "INCLUDE_WEEKEND_TRADING", 0)
        config.INCLUDE_WEEKEND_TRADING = mode
        tc.reset_cache()
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        config.INCLUDE_WEEKEND_TRADING = self._saved
        tc.reset_cache()


class TestStrictCalendar(CalendarModeMixin, unittest.TestCase):
    """Режим 0 — строгий биржевой календарь Пн–Пт."""

    def setUp(self):
        self.set_mode(0)

    def test_active_weekdays_is_mon_fri(self):
        self.assertEqual(tc.active_weekdays(), tc.MON_FRI)

    def test_horizon_from_tuesday(self):
        days = tc.get_next_trading_days(TUE, H)
        self.assertEqual(days, [dt.date(2026, 9, 9),    # ср
                                dt.date(2026, 9, 10),   # чт
                                dt.date(2026, 9, 11),   # пт
                                dt.date(2026, 9, 14),   # пн (выходные пропущены)
                                dt.date(2026, 9, 15)])  # вт
        self.assertEqual(tc.get_target_horizon_date(TUE, H), dt.date(2026, 9, 15))

    def test_horizon_from_friday_skips_weekend(self):
        days = tc.get_next_trading_days(FRI, H)
        self.assertEqual(days[0], dt.date(2026, 9, 14))       # сразу понедельник
        self.assertEqual(tc.get_target_horizon_date(FRI, H), dt.date(2026, 9, 18))

    def test_no_weekend_dates_in_horizon(self):
        for start in (TUE, FRI, dt.date(2026, 9, 12)):        # в т.ч. старт в субботу
            for d in tc.get_next_trading_days(start, 10):
                self.assertLess(d.weekday(), 5, f"выходной в горизонте: {d}")

    def test_sql_filter_excludes_weekend(self):
        self.assertIn("BETWEEN 1 AND 5", tc.sql_session_filter("date"))

    def test_dow_period_is_five(self):
        self.assertEqual(tc.dow_period(), 5.0)

    def test_filter_sessions_drops_weekend_rows(self):
        pd = _pandas_or_skip(self)
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(21)]
        df = pd.DataFrame({"date": pd.to_datetime(days), "close": range(len(days))})
        out = tc.filter_sessions(df)
        self.assertEqual(len(out), 15)                        # 3 недели × 5 будней
        self.assertTrue((pd.to_datetime(out["date"]).dt.weekday < 5).all())


class TestWeekendTradingCalendar(CalendarModeMixin, unittest.TestCase):
    """Режим 1 — торговля 7 дней в неделю."""

    def setUp(self):
        self.set_mode(1)

    def test_horizon_from_tuesday_is_calendar_days(self):
        days = tc.get_next_trading_days(TUE, H, weekdays=tc.ALL_WEEK)
        self.assertEqual(days, [dt.date(2026, 9, 9),    # ср
                                dt.date(2026, 9, 10),   # чт
                                dt.date(2026, 9, 11),   # пт
                                dt.date(2026, 9, 12),   # сб — сессия выходного дня
                                dt.date(2026, 9, 13)])  # вс
        self.assertEqual(
            tc.get_target_horizon_date(TUE, H, weekdays=tc.ALL_WEEK),
            dt.date(2026, 9, 13))

    def test_horizon_from_friday_includes_weekend(self):
        days = tc.get_next_trading_days(FRI, H, weekdays=tc.ALL_WEEK)
        self.assertIn(dt.date(2026, 9, 12), days)             # суббота внутри окна
        self.assertEqual(days[-1], dt.date(2026, 9, 16))

    def test_sql_filter_is_empty(self):
        self.assertEqual(tc.sql_session_filter("date"), "")

    def test_dow_period_is_seven(self):
        self.assertEqual(tc.dow_period(), 7.0)

    def test_filter_sessions_keeps_everything(self):
        pd = _pandas_or_skip(self)
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(21)]
        df = pd.DataFrame({"date": pd.to_datetime(days), "close": range(len(days))})
        self.assertEqual(len(tc.filter_sessions(df, weekdays=tc.ALL_WEEK)), 21)


class TestHorizonMatchesDatasetStep(CalendarModeMixin, unittest.TestCase):
    """Критерий приёмки: дата в шапке дашборда == H-й шаг датасета модели.

    Датасет строится фильтрацией баров тем же календарём, а цель week_total —
    это close.shift(-H), то есть H СТРОК вперёд. Значит H-я строка после asof
    обязана совпасть с get_target_horizon_date(asof, H).
    """

    def _check(self, mode: int, weekdays):
        self.set_mode(mode)
        # «сырые» бары: торги каждый календарный день (как в market_data)
        raw = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(45)]
        # так же, как загрузчик датасета отбирает строки в этом режиме
        dataset = [d for d in raw if d.weekday() in weekdays]
        for asof in dataset[:15]:
            idx = dataset.index(asof)
            if idx + H >= len(dataset):
                break
            model_target = dataset[idx + H]                       # close.shift(-H)
            header_target = tc.get_target_horizon_date(asof, H, weekdays=weekdays)
            self.assertEqual(model_target, header_target,
                             f"режим {mode}, asof={asof}: датасет даёт "
                             f"{model_target}, шапка — {header_target}")

    def test_mode_0_strict(self):
        self._check(0, tc.MON_FRI)

    def test_mode_1_weekend_trading(self):
        self._check(1, tc.ALL_WEEK)

    def test_old_hardcoded_logic_would_diverge_in_mode_1(self):
        """Контрольный тест: прежняя логика (`while weekday >= 5`) в режиме 7/7
        даёт другую дату — ради этого расхождения задача и заводилась."""
        def old_next(d):
            d += dt.timedelta(days=1)
            while d.weekday() >= 5:
                d += dt.timedelta(days=1)
            return d

        end = TUE
        for _ in range(H):
            end = old_next(end)
        new_end = tc.get_target_horizon_date(TUE, H, weekdays=tc.ALL_WEEK)
        self.assertNotEqual(end, new_end)
        self.assertEqual(end, dt.date(2026, 9, 15))      # старое: пропускало выходные
        self.assertEqual(new_end, dt.date(2026, 9, 13))  # новое: 5 реальных сессий


class TestAgainstDatabase(unittest.TestCase):
    """Проверка на фактических датах из БД (пропускается, если БД недоступна)."""

    def setUp(self):
        try:
            import database
            self.database = database
            with database.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM market_data")
                    if not cur.fetchone()[0]:
                        self.skipTest("market_data пуста")
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"БД недоступна: {e}")

    def test_db_sessions_match_calendar_in_strict_mode(self):
        """Все даты, которые датасет берёт в строгом режиме, — рабочие дни."""
        saved = getattr(config, "INCLUDE_WEEKEND_TRADING", 0)
        config.INCLUDE_WEEKEND_TRADING = 0
        tc.reset_cache()
        try:
            sql_filter = tc.sql_session_filter("date")
            with self.database.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""SELECT DISTINCT date FROM market_data
                                    WHERE TRUE{sql_filter} ORDER BY date DESC LIMIT 60""")
                    dates = [r[0] for r in cur.fetchall()]
            self.assertTrue(dates, "фильтр сессий не вернул ни одной даты")
            for d in dates:
                self.assertLess(d.weekday(), 5, f"выходной прошёл фильтр: {d}")
        finally:
            config.INCLUDE_WEEKEND_TRADING = saved
            tc.reset_cache()

    def test_db_has_weekend_bars_unfiltered(self):
        """Контроль: без фильтра выходные бары в базе действительно есть —
        иначе тест выше проходил бы бессодержательно."""
        with self.database.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT count(*) FROM market_data
                               WHERE EXTRACT(DOW FROM date) IN (0, 6)""")
                self.assertGreater(cur.fetchone()[0], 0)


# ═══════════════════════════════════════════════════════════════════════════
# Sprint 3.6 — производственный календарь и праздники Мосбиржи
# ═══════════════════════════════════════════════════════════════════════════

from services.calendar import (  # noqa: E402
    TradingCalendar, MOEX_HOLIDAYS, WORKING_WEEKENDS, parse_schedule,
)


class TestHolidayCalendar(CalendarModeMixin, unittest.TestCase):
    """Праздники и переносы в строгом режиме (Пн–Пт)."""

    def setUp(self):
        self.set_mode(0)
        self.cal = TradingCalendar(weekdays=tc.MON_FRI)

    def test_new_year_gap_is_skipped(self):
        """H=5 от 29 декабря перешагивает новогодние каникулы.

        Ожидание: 30.12 торговый, затем 31.12 / 01.01 / 02.01 — праздники,
        03.01 — воскресенье, значит первая январская дата — 04.01, а 07.01
        (Рождество) тоже пропускается.
        """
        days = self.cal.get_next_trading_days(dt.date(2026, 12, 29), 5)
        self.assertEqual(days, [dt.date(2026, 12, 30),
                                dt.date(2027, 1, 4),
                                dt.date(2027, 1, 5),
                                dt.date(2027, 1, 6),
                                dt.date(2027, 1, 8)])
        self.assertEqual(self.cal.get_target_horizon_date(dt.date(2026, 12, 29), 5),
                         dt.date(2027, 1, 8))

    def test_no_holiday_falls_inside_horizon(self):
        for d in self.cal.get_next_trading_days(dt.date(2026, 12, 29), 5):
            self.assertNotIn(d, self.cal.holidays, f"праздник попал в горизонт: {d}")
            self.assertTrue(self.cal.is_trading_day(d))

    def test_first_january_date_is_working_day(self):
        """Первая январская дата горизонта — рабочая, а не 1–3 января."""
        days = self.cal.get_next_trading_days(dt.date(2026, 12, 29), 5)
        january = [d for d in days if d.year == 2027 and d.month == 1]
        self.assertTrue(january)
        self.assertEqual(january[0], dt.date(2027, 1, 4))
        for bad in (dt.date(2027, 1, 1), dt.date(2027, 1, 2), dt.date(2027, 1, 3)):
            self.assertNotIn(bad, days)

    def test_working_saturday_is_trading_day(self):
        """Официальный перенос выходного: рабочая суббота считается торговой."""
        for sat in WORKING_WEEKENDS:
            self.assertEqual(sat.weekday(), 5, f"{sat} — не суббота")
            self.assertTrue(self.cal.is_trading_day(sat),
                            f"рабочая суббота {sat} не признана торговой")

    def test_ordinary_saturday_is_not_trading_day(self):
        self.assertFalse(self.cal.is_trading_day(dt.date(2024, 11, 9)))

    def test_working_saturday_appears_in_horizon(self):
        """Горизонт от пятницы 2024-11-01 включает рабочую субботу 02.11
        и пропускает праздник 04.11."""
        days = self.cal.get_next_trading_days(dt.date(2024, 11, 1), 3)
        self.assertEqual(days[0], dt.date(2024, 11, 2))   # рабочая суббота
        self.assertNotIn(dt.date(2024, 11, 4), days)      # День народного единства
        self.assertEqual(days[1], dt.date(2024, 11, 5))

    def test_holiday_is_not_trading_day(self):
        for d in (dt.date(2025, 1, 7), dt.date(2025, 5, 9), dt.date(2026, 1, 7)):
            self.assertFalse(self.cal.is_trading_day(d), f"{d} должен быть праздником")

    def test_exchange_trades_on_state_holidays_2026(self):
        """Календарь биржи ≠ производственный календарь РФ: в 2026 биржа
        торгует 23.02, 01.05 и 12.06 — эти дни НЕ должны выпадать из горизонта.
        Проверено по market_data (бары за эти даты есть)."""
        for s in ("2026-02-23", "2026-05-01", "2026-06-12"):
            d = dt.date.fromisoformat(s)
            self.assertTrue(self.cal.is_trading_day(d),
                            f"{s} ошибочно помечен нерабочим")


class TestOfflineFallback(CalendarModeMixin, unittest.TestCase):
    """Критерий приёмки: без сети календарь работает на оффлайн-наборе."""

    def setUp(self):
        self.set_mode(0)

    def test_sync_failure_does_not_raise(self):
        class DeadClient:
            def _post(self, *a, **k):
                raise OSError("сеть недоступна")

        cal = TradingCalendar(weekdays=tc.MON_FRI)
        before = set(cal.holidays)
        self.assertFalse(cal.sync_schedule(client=DeadClient()))
        self.assertEqual(cal.holidays, before, "оффлайн-словарь не должен меняться")

    def test_offline_horizon_still_correct(self):
        class DeadClient:
            def _post(self, *a, **k):
                raise OSError("сеть недоступна")

        cal = TradingCalendar(weekdays=tc.MON_FRI)
        cal.sync_schedule(client=DeadClient())
        self.assertEqual(cal.get_target_horizon_date(dt.date(2026, 12, 29), 5),
                         dt.date(2027, 1, 8))

    def test_empty_api_response_falls_back(self):
        class EmptyClient:
            def _post(self, *a, **k):
                return {"exchanges": []}

        cal = TradingCalendar(weekdays=tc.MON_FRI)
        self.assertFalse(cal.sync_schedule(client=EmptyClient()))
        self.assertTrue(cal.is_trading_day(dt.date(2026, 9, 9)))

    def test_offline_set_is_not_empty(self):
        self.assertTrue(MOEX_HOLIDAYS)
        self.assertTrue(WORKING_WEEKENDS)


class TestScheduleSync(CalendarModeMixin, unittest.TestCase):
    """Разбор ответа API и применение расписания."""

    def setUp(self):
        self.set_mode(0)

    @staticmethod
    def _response(pairs):
        return {"exchanges": [{"exchange": "MOEX", "days": [
            {"date": f"{d}T00:00:00Z", "isTradingDay": ok} for d, ok in pairs]}]}

    def test_parse_schedule(self):
        parsed = parse_schedule(self._response([("2026-09-09", True),
                                                ("2026-09-12", False)]))
        self.assertEqual(parsed, {dt.date(2026, 9, 9): True,
                                  dt.date(2026, 9, 12): False})

    def test_api_marks_new_holiday(self):
        """Неторговый БУДНИЙ день из расписания становится праздником."""
        class Client:
            def __init__(self, resp): self.resp = resp
            def _post(self, *a, **k): return self.resp

        cal = TradingCalendar(weekdays=tc.MON_FRI)
        surprise = dt.date(2026, 9, 10)
        self.assertTrue(cal.is_trading_day(surprise))
        self.assertTrue(cal.sync_schedule(
            client=Client(self._response([(surprise.isoformat(), False)]))))
        self.assertFalse(cal.is_trading_day(surprise))
        self.assertIn(surprise, cal.holidays)

    def test_api_weekend_flag_does_not_override_mode_1(self):
        """На основной доске MOEX выходные всегда isTradingDay=false. В режиме
        7/7 это не должно выключать сессии выходного дня."""
        class Client:
            def __init__(self, resp): self.resp = resp
            def _post(self, *a, **k): return self.resp

        sat = dt.date(2026, 9, 12)
        cal = TradingCalendar(weekdays=tc.ALL_WEEK)
        cal.sync_schedule(client=Client(self._response([(sat.isoformat(), False)])))
        self.assertTrue(cal.is_trading_day(sat),
                        "расписание основной доски погасило сессию выходного дня")


def _pandas_or_skip(case):
    try:
        import pandas as pd
        return pd
    except ImportError:
        case.skipTest("pandas не установлен")


if __name__ == "__main__":
    unittest.main(verbosity=2)
