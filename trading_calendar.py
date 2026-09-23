"""
Единый торговый календарь проекта.

Зачем: в market_data лежат бары за субботы и воскресенья (T-Bank/MOEX weekend
sessions) — 5 605 дневных баров, все 47 тикеров. Средний объём выходной сессии
466 тыс. против 3,83 млн в будни, то есть в 8 раз ниже. Раньше календарь был
захардкожен в двух местах и по-разному: датасет модели брал ВСЕ бары подряд
(шаг модели = календарная сессия), а шапка недельного дашборда шагала по
Пн–Пт (`while d.weekday() >= 5`). Из-за этого H=5 шагов модели и «окно
прогноза» в заголовке расходились.

Режимы (config.INCLUDE_WEEKEND_TRADING):

  0 — строгий биржевой календарь (по умолчанию). Бары выходных отбрасываются
      ещё на загрузке датасета, чтобы низколиквидная сессия не искажала ATR,
      EWMA и z-оценки объёма. Шаг модели = рабочий день, дашборд считает
      окно по Пн–Пт. Оба конца синхронны по построению.

  1 — торговля 7 дней в неделю. Бары выходных остаются в датасете, а торговые
      дни определяются ФАКТИЧЕСКИМ наличием торгов в БД (active_weekdays), а не
      наивной проверкой дня недели.

Праздники и переносы живут в services/calendar.py (TradingCalendar): этот
модуль отвечает за РЕЖИМ и маску дней недели, а тот — за фактический календарь
биржи и синхронизацию с InstrumentsService/TradingSchedules. Для дат внутри
истории самый точный источник — sessions_in_db().
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable

import config

log = logging.getLogger("trading_calendar")

# Python weekday(): 0=Пн … 6=Вс
MON_FRI = frozenset({0, 1, 2, 3, 4})
ALL_WEEK = frozenset(range(7))

# Доля дат недели, начиная с которой день считается торговым (режим 1).
_MIN_COVERAGE = 0.5
_LOOKBACK_DAYS = 120

_cache: dict[bool, frozenset[int]] = {}


def weekend_trading_enabled() -> bool:
    """True — режим 1 (торговля 7 дней в неделю)."""
    try:
        return bool(int(getattr(config, "INCLUDE_WEEKEND_TRADING", 0)))
    except (TypeError, ValueError):
        return False


def reset_cache() -> None:
    """Сбросить кэш активных дней недели (нужно тестам и при смене режима)."""
    _cache.clear()


# ── Какие дни недели торговые ────────────────────────────────────────────────

_COVERAGE_SQL = """
    WITH d AS (
        SELECT DISTINCT date
        FROM market_data
        WHERE date > (SELECT MAX(date) FROM market_data) - %(lookback)s
    )
    SELECT EXTRACT(DOW FROM date)::int AS pg_dow, COUNT(*) AS n
    FROM d GROUP BY 1;
"""


def _weekdays_from_db(conn=None, lookback_days: int = _LOOKBACK_DAYS) -> frozenset[int]:
    """Дни недели, в которые реально идут торги (по последним lookback дням БД).

    День считается торговым, если доля дат этого дня недели в окне не ниже
    _MIN_COVERAGE — так разовый праздничный/технический день не выкидывает
    целый день недели из календаря.
    """
    import database

    def _query(c) -> list[tuple[int, int]]:
        with c.cursor() as cur:
            cur.execute(_COVERAGE_SQL, {"lookback": lookback_days})
            return [(int(r[0]), int(r[1])) for r in cur.fetchall()]

    try:
        if conn is not None:
            rows = _query(conn)
        else:
            with database.get_db_connection() as own:
                rows = _query(own)
    except Exception as e:  # noqa: BLE001 — календарь не должен валить расчёт
        log.warning("Не удалось определить торговые дни из БД (%s) — беру все 7 дней.", e)
        return ALL_WEEK

    if not rows:
        return ALL_WEEK

    # ожидаемое число дат каждого дня недели в окне
    expected = max(1, lookback_days // 7)
    active = set()
    for pg_dow, n in rows:
        py_dow = (pg_dow - 1) % 7          # pg: 0=Вс … 6=Сб  →  py: 0=Пн … 6=Вс
        if n / expected >= _MIN_COVERAGE:
            active.add(py_dow)
    if not active:
        return ALL_WEEK
    return frozenset(active)


def active_weekdays(conn=None) -> frozenset[int]:
    """Торговые дни недели для текущего режима. Результат кэшируется."""
    weekend = weekend_trading_enabled()
    if not weekend:
        return MON_FRI                      # режим 0 — строго Пн–Пт, БД не нужна
    if weekend not in _cache:
        _cache[weekend] = _weekdays_from_db(conn)
        log.info("Торговые дни недели (режим INCLUDE_WEEKEND_TRADING=1): %s",
                 sorted(_cache[weekend]))
    return _cache[weekend]


def _resolve(weekdays: Iterable[int] | None, conn=None) -> frozenset[int]:
    if weekdays is not None:
        wd = frozenset(int(w) for w in weekdays)
        return wd or ALL_WEEK
    return active_weekdays(conn)


# ── Шаги по календарю ────────────────────────────────────────────────────────

def _calendar(weekdays=None, conn=None):
    """Календарь биржи (праздники + переносы) поверх маски дней недели."""
    from services.calendar import TradingCalendar, get_calendar

    wd = _resolve(weekdays, conn)
    shared = get_calendar()
    if weekdays is None:
        return shared                      # общий синглтон: в нём накоплен sync
    # явная маска (тесты, ручные сценарии) — тот же список праздников
    return TradingCalendar(holidays=shared.holidays,
                           working_weekends=shared.working_weekends,
                           weekdays=wd)


def is_trading_day(d: dt.date, weekdays=None, conn=None) -> bool:
    return _calendar(weekdays, conn).is_trading_day(d)


def next_trading_day(d: dt.date, weekdays=None, conn=None) -> dt.date:
    """Следующий торговый день СТРОГО после d (с учётом праздников)."""
    days = _calendar(weekdays, conn).get_next_trading_days(d, 1)
    return days[0] if days else d + dt.timedelta(days=1)


def get_next_trading_days(asof_date: dt.date, n_days: int,
                          weekdays=None, conn=None) -> list[dt.date]:
    """Список из n_days торговых дней, следующих за asof_date.

    Это ровно те даты, которые модель покрывает своим горизонтом: шаг датасета
    = одна строка = один торговый день текущего календаря (маска дней недели
    из режима + праздники и переносы из services/calendar.py).
    """
    return _calendar(weekdays, conn).get_next_trading_days(asof_date, n_days)


def get_target_horizon_date(asof_date: dt.date, horizon_steps: int,
                            weekdays=None, conn=None) -> dt.date | None:
    """Дата, на которую реально смотрит модель через horizon_steps шагов."""
    return _calendar(weekdays, conn).get_target_horizon_date(asof_date, horizon_steps)


def sessions_in_db(conn, start: dt.date, end: dt.date) -> list[dt.date]:
    """Фактические торговые даты из БД в интервале [start, end] — без догадок
    о днях недели. Работает только для прошлого (для будущего баров ещё нет)."""
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT date FROM market_data
                       WHERE date BETWEEN %s AND %s ORDER BY date;""", (start, end))
        return [r[0] for r in cur.fetchall()]


# ── Фильтрация датасета ──────────────────────────────────────────────────────

def sql_session_filter(date_col: str = "date") -> str:
    """Кусок WHERE для запросов к market_data.

    Режим 0 → отбрасывает субботы и воскресенья прямо в SQL (pg DOW: 1..5 =
    Пн..Пт). Режим 1 → пустая строка, берём все бары.
    """
    if weekend_trading_enabled():
        return ""
    return f" AND EXTRACT(DOW FROM {date_col}) BETWEEN 1 AND 5"


def filter_sessions(df, date_col: str = "date", weekdays=None):
    """Тот же фильтр для готового DataFrame (когда SQL уже выполнен)."""
    if df is None or len(df) == 0:
        return df
    wd = _resolve(weekdays)
    if wd == ALL_WEEK:
        return df
    import pandas as pd  # локальный импорт: модуль нужен и без pandas

    dates = pd.to_datetime(df[date_col])
    return df[dates.dt.weekday.isin(sorted(wd))].reset_index(drop=True)


def dow_period() -> float:
    """Период циклического кодирования дня недели для признаков модели:
    5 рабочих дней в режиме 0, полная неделя в режиме 1."""
    return 7.0 if weekend_trading_enabled() else 5.0
