"""
Свежесть дневных свечей: догрузка + гард.

Единая точка, которую используют и торговый путь (services/place_orders.py),
и ETL/дашборд (main.py). Причина существования: market_data двигают вперёд
только load_history + backfill_daily; если их не запускать, прогноз считается
на устаревших свечах (исторически last_close залипал на несколько сессий назад).

ensure_fresh_data():
  1) (опц.) догружает свечи инкрементально: load_history + backfill из 5M;
  2) проверяет, что max(date) в market_data догнал последнюю ЗАВЕРШЁННУЮ сессию,
     известную по market_data_5m (день строго < сегодня с 5-мин барами).
  Если не догнал — StaleDataError.

Вызывающий сам решает, что делать с StaleDataError: place_orders БЛОКИРУЕТ
выставление заявок; main.py печатает громкое предупреждение и продолжает.
"""
from __future__ import annotations

import datetime as dt
import logging

import config
from services import backfill_daily, load_history

log = logging.getLogger("data_freshness")

_MSK = dt.timezone(dt.timedelta(hours=3))


class StaleDataError(RuntimeError):
    """Дневные свечи не догнали последнюю завершённую сессию даже после догрузки."""


def ensure_fresh_data(conn, *, refresh: bool = True) -> dt.date:
    """Догрузить дневные свечи и убедиться, что они актуальны.

    refresh=True → load_history (инкрементально) + backfill дневных из 5M.
    Возвращает актуальную last_date (max(date) в market_data) или бросает
    StaleDataError, если свечи отстают от последней завершённой сессии.
    """
    if refresh:
        log.info("Догрузка свежих свечей (load_history + backfill)...")
        try:
            load_history.run(conn)
        except Exception as e:  # noqa: BLE001 — сеть/API; решение примет гард
            log.warning("load_history не отработал (%s) — проверю, что уже есть.", e)
        try:
            backfill_daily.backfill(conn, config.TICKERS)
        except Exception as e:  # noqa: BLE001
            log.warning("backfill дневных из 5M не отработал: %s", e)
        # Бары основной сессии — источник признаков модели (config.MODEL_BARS).
        # В PREP это закрывает вчерашний бар, в OVERNIGHT — собирает сегодняшний
        # частичный: прогноз ночи строится от сегодняшнего закрытия, а не от
        # вчерашнего (аудит 26.09, F1).
        from services import session_bars
        session_bars.run(conn)

    today_msk = dt.datetime.now(_MSK).date()
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM market_data;")
        md_max = cur.fetchone()[0]
        # последняя завершённая сессия, известная 5-минуткам (день строго < сегодня)
        cur.execute(
            "SELECT MAX((ts AT TIME ZONE 'Europe/Moscow')::date) "
            "FROM market_data_5m "
            "WHERE (ts AT TIME ZONE 'Europe/Moscow')::date < %s;",
            (today_msk,))
        expected = cur.fetchone()[0]

    if md_max is None:
        raise StaleDataError("market_data пуста — сначала прогоните main.py.")
    if expected is not None and md_max < expected:
        raise StaleDataError(
            f"дневные свечи устарели: last_date={md_max}, а последняя завершённая "
            f"сессия={expected}. Догрузка не помогла (источник данных недоступен?).")
    if str(getattr(config, "MODEL_BARS", "session")).lower() != "daily":
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT MAX(date) FROM market_data_session;")
                sb_max = cur.fetchone()[0]
            except Exception:  # noqa: BLE001 — таблицы ещё нет
                conn.rollback()
                sb_max = None
        if sb_max is None or (expected is not None and sb_max < expected):
            raise StaleDataError(
                f"бары основной сессии устарели: last={sb_max}, а последняя "
                f"завершённая сессия={expected}. Модель считается на них "
                f"(MODEL_BARS=session) — проверьте services/session_bars.")
    log.info("Данные актуальны: market_data до %s (сессия=%s).", md_max, expected)
    return md_max
