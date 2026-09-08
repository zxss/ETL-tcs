"""
Производственный календарь торгов Мосбиржи.

ЧТО ЗДЕСЬ ВАЖНО ЗНАТЬ
---------------------
Календарь биржи — это НЕ производственный календарь РФ. По данным market_data
(711 торговых дат, 2024-05 … 2026-09) биржа с 2026 года торгует в большинство
государственных праздников:

    2026-02-23 (Пн)  торги есть      2026-05-01 (Пт)  торги есть
    2026-03-09 (Пн)  торги есть      2026-05-11 (Пн)  торги есть
                                     2026-06-12 (Пт)  торги есть

Если зашить сюда календарь ТК РФ, окно прогноза начнёт перешагивать через дни,
в которые биржа реально работает, — то есть ровно та ошибка, от которой уходим.
Поэтому оффлайн-словарь MOEX_HOLIDAYS содержит ФАКТИЧЕСКИЕ остановки торгов,
сверенные с историей баров, а не список нерабочих дней по ТК.

Источники в порядке приоритета:
  1. sync_schedule() — InstrumentsService/TradingSchedules по бирже MOEX.
     Авторитетный, но узкий: API отдаёт только [сегодня; сегодня+14] —
     `from` в прошлом и `to` дальше 14 дней отклоняются. Горизонт H=5 в это
     окно укладывается всегда.
  2. MOEX_HOLIDAYS / WORKING_WEEKENDS — оффлайн-fallback на случай отсутствия
     сети или отказа API. Работает на любую дату, но требует сопровождения.
  3. Маска дней недели из trading_calendar (режим INCLUDE_WEEKEND_TRADING).

Данные API применяются только к дням Пн–Пт: на основной доске MOEX выходные
помечены isTradingDay=false, тогда как в market_data есть сессии выходного дня
(отдельная доска). Выходными управляет режим, а API сообщает о ПРАЗДНИКАХ.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading

log = logging.getLogger("services.calendar")

EXCHANGE = "MOEX"

# ── Оффлайн-fallback ─────────────────────────────────────────────────────────
# Фактические остановки торгов (будни без баров в market_data). Сверено с
# историей: 2024-06-12, 2024-11-04, новогодние окна и 2025-05-01/05-09/06-12/11-04.
_VERIFIED_HOLIDAYS = {
    # 2024
    "2024-06-12",                                    # День России
    "2024-11-04",                                    # День народного единства
    "2024-12-31",
    # 2025
    "2025-01-01", "2025-01-02", "2025-01-07",        # 03, 06, 08 января — торговали
    "2025-05-01", "2025-05-09",
    "2025-06-12",
    "2025-11-04",
    "2025-12-31",
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-07",        # 05, 06, 08, 09 января — торговали
}

# 2027 — ПРОГНОЗ, а не официальный список: постановление Правительства РФ о
# переносах на 2027 год ещё не публиковалось. Экстраполирован устойчивый
# биржевой шаблон последних лет (31 декабря, 1–2 января, Рождество 7 января).
# Проверяется и вытесняется данными sync_schedule по мере приближения дат.
_PROJECTED_HOLIDAYS_2027 = {
    "2026-12-31",
    "2027-01-01", "2027-01-02", "2027-01-07",
    "2027-12-31",
}

MOEX_HOLIDAYS: frozenset[dt.date] = frozenset(
    dt.date.fromisoformat(s) for s in (_VERIFIED_HOLIDAYS | _PROJECTED_HOLIDAYS_2027)
)

# Рабочие субботы (переносы выходных постановлениями Правительства РФ).
# Подтверждены барами в market_data: это единственные торговые субботы 2024
# года, когда сессий выходного дня ещё не было.
# С 2025 года биржа торгует по выходным штатно, поэтому список актуален прежде
# всего для режима INCLUDE_WEEKEND_TRADING=0 и требует обновления по каждому
# ежегодному постановлению.
WORKING_WEEKENDS: frozenset[dt.date] = frozenset(
    dt.date.fromisoformat(s) for s in {
        "2024-11-02",   # перенос: рабочая суббота
        "2024-12-28",   # перенос: рабочая суббота
    }
)

_MON_FRI = frozenset({0, 1, 2, 3, 4})


class TradingCalendar:
    """Торговый календарь биржи: праздники, переносы, синхронизация с API."""

    def __init__(self,
                 holidays=None,
                 working_weekends=None,
                 weekdays=None):
        self.holidays = set(holidays if holidays is not None else MOEX_HOLIDAYS)
        self.working_weekends = set(
            working_weekends if working_weekends is not None else WORKING_WEEKENDS)
        self._weekdays = frozenset(weekdays) if weekdays is not None else None
        self._api_days: dict[dt.date, bool] = {}   # из sync_schedule
        self.synced_until: dt.date | None = None
        self._lock = threading.Lock()

    # ── Маска дней недели ────────────────────────────────────────────────────

    def _mask(self) -> frozenset[int]:
        """Какие дни недели вообще торговые — из режима проекта."""
        if self._weekdays is not None:
            return self._weekdays
        import trading_calendar
        return trading_calendar.active_weekdays()

    # ── Основной API ─────────────────────────────────────────────────────────

    def is_trading_day(self, d: dt.date) -> bool:
        """Торгует ли биржа в этот день."""
        if isinstance(d, dt.datetime):
            d = d.date()

        # 1. Официальный перенос выходного → рабочий день (сильнее маски).
        if d in self.working_weekends and d not in self.holidays:
            return True

        # 2. Праздник из оффлайн-словаря — торгов нет.
        if d in self.holidays:
            return False

        mask = self._mask()

        # 3. Расписание из API — только для Пн–Пт: выходными управляет режим,
        #    а основная доска MOEX всегда помечает их как неторговые.
        if d.weekday() < 5 and d in self._api_days:
            return self._api_days[d] and d.weekday() in mask

        # 4. Обычная маска дней недели.
        return d.weekday() in mask

    def get_next_trading_days(self, start_date: dt.date, n: int) -> list[dt.date]:
        """n торговых дней СТРОГО после start_date."""
        if isinstance(start_date, dt.datetime):
            start_date = start_date.date()
        out: list[dt.date] = []
        cur = start_date
        # запас хода: до 30 календарных дней на один торговый (новогодние каникулы)
        limit = max(1, int(n)) * 30 + 30
        for _ in range(limit):
            if len(out) >= int(n):
                break
            cur = cur + dt.timedelta(days=1)
            if self.is_trading_day(cur):
                out.append(cur)
        return out

    def get_target_horizon_date(self, start_date: dt.date, n: int) -> dt.date | None:
        """Дата, на которую смотрит модель через n торговых дней."""
        days = self.get_next_trading_days(start_date, n)
        return days[-1] if days else None

    # ── Синхронизация с биржей ───────────────────────────────────────────────

    def sync_schedule(self, client=None, days: int = 14) -> bool:
        """Подтягивает расписание MOEX из T-Invest на ближайшие `days` дней.

        Никогда не бросает наружу: при отсутствии сети, токена или отказе API
        пишет предупреждение и оставляет оффлайн-словарь — конвейер не падает.
        Возвращает True, если расписание получено.

        client — объект с методом _post(method, payload) (например
        services.broker.TinkoffProdClient). Если не задан — используется
        синхронный загрузчик loaders.moex_loader.fetch_trading_schedules_sync.
        """
        today = dt.date.today()
        # API отвергает `to` дальше 14 дней от текущей даты.
        days = max(1, min(int(days), 14))
        to = today + dt.timedelta(days=days)
        try:
            if client is not None:
                payload = {
                    "exchange": EXCHANGE,
                    "from": _iso(today),
                    "to": _iso(to),
                }
                data = client._post("InstrumentsService/TradingSchedules", payload)
            else:
                from loaders.moex_loader import fetch_trading_schedules_sync
                data = fetch_trading_schedules_sync(EXCHANGE, today, to)
            parsed = parse_schedule(data)
        except Exception as e:  # noqa: BLE001 — календарь не должен валить ETL
            log.warning("Расписание MOEX не получено (%s) — работаем на "
                        "встроенном оффлайн-календаре.", e)
            return False

        if not parsed:
            log.warning("API вернул пустое расписание MOEX — остаёмся на "
                        "оффлайн-календаре.")
            return False

        with self._lock:
            self._api_days.update(parsed)
            self.synced_until = max(parsed)
            # Неторговые будни из расписания — это праздники: доучиваем словарь,
            # чтобы они работали и после того, как окно API уедет вперёд.
            new_holidays = {d for d, ok in parsed.items() if not ok and d.weekday() < 5}
            added = new_holidays - self.holidays
            self.holidays |= new_holidays
        log.info("Расписание MOEX синхронизировано: %d дней по %s (новых "
                 "праздников: %d).", len(parsed), self.synced_until, len(added))
        return True


# ── Разбор ответа API ────────────────────────────────────────────────────────

def _iso(d: dt.date) -> str:
    return dt.datetime(d.year, d.month, d.day,
                       tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_schedule(data: dict) -> dict[dt.date, bool]:
    """{date: isTradingDay} из ответа InstrumentsService/TradingSchedules."""
    out: dict[dt.date, bool] = {}
    for ex in (data or {}).get("exchanges", []) or []:
        for day in ex.get("days", []) or []:
            raw = day.get("date")
            if not raw:
                continue
            try:
                d = dt.date.fromisoformat(str(raw)[:10])
            except ValueError:
                continue
            out[d] = bool(day.get("isTradingDay"))
    return out


# ── Синглтон ─────────────────────────────────────────────────────────────────

_calendar: TradingCalendar | None = None
_singleton_lock = threading.Lock()


def get_calendar() -> TradingCalendar:
    global _calendar
    if _calendar is None:
        with _singleton_lock:
            if _calendar is None:
                _calendar = TradingCalendar()
    return _calendar


def reset_calendar() -> None:
    """Сбросить синглтон (нужно тестам)."""
    global _calendar
    with _singleton_lock:
        _calendar = None


def sync_schedule(client=None, days: int = 14) -> bool:
    """Синхронизировать общий календарь проекта. Не бросает исключений."""
    return get_calendar().sync_schedule(client=client, days=days)


def is_trading_day(d: dt.date) -> bool:
    return get_calendar().is_trading_day(d)


def get_next_trading_days(start_date: dt.date, n: int) -> list[dt.date]:
    return get_calendar().get_next_trading_days(start_date, n)


def get_target_horizon_date(start_date: dt.date, n: int):
    return get_calendar().get_target_horizon_date(start_date, n)
