"""
Реконструкция дневных свечей из 5-минутных, когда официальный дневной бар
T-Invest ещё не опубликован (часто — поздним вечером/ночью, до утренней
финализации). Собирает OHLCV за завершённые торговые дни из market_data_5m
и upsert-ит в market_data.

Когда брокер опубликует настоящий дневной бар, обычный ETL перезапишет
реконструкцию (UPSERT ON CONFLICT DO UPDATE) — данные самокорректируются.

Реконструируются ТОЛЬКО завершённые дни (строго раньше текущей московской
даты), чтобы не зафиксировать частичную свечу текущей сессии.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import database

log = logging.getLogger("backfill_daily")

MSK = timezone(timedelta(hours=3))

# Агрегация 5M → дневная OHLCV по московскому календарному дню.
# open  = первый по времени, close = последний по времени, high/low/volume — экстремумы/сумма.
_BACKFILL_SQL = """
INSERT INTO market_data (ticker, date, open, high, low, close, volume)
SELECT
    ticker,
    d AS date,
    (array_agg(open  ORDER BY ts ASC ))[1] AS open,
    MAX(high)                              AS high,
    MIN(low)                               AS low,
    (array_agg(close ORDER BY ts DESC))[1] AS close,
    SUM(volume)                            AS volume
FROM (
    SELECT ticker, ts, open, high, low, close, volume,
           (ts AT TIME ZONE 'Europe/Moscow')::date AS d
    FROM market_data_5m
    WHERE ticker = %(tk)s
      -- PR-8: только основная сессия (МСК). Вечерняя сессия (19:00–23:50) иначе
      -- попадала бы в агрегат и давала close/high/low, не совпадающие с
      -- официальным дневным баром (основная сессия), который позже перезаписывает
      -- реконструкцию. Закрытие основной сессии — аукцион 18:40–18:50.
      AND (ts AT TIME ZONE 'Europe/Moscow')::time >= TIME '09:50'
      AND (ts AT TIME ZONE 'Europe/Moscow')::time <  TIME '18:50'
) s
WHERE d > %(after)s        -- строго новее последней дневной свечи в БД
  AND d < %(today)s        -- только завершённые дни (не сегодняшняя сессия)
GROUP BY ticker, d
ON CONFLICT (ticker, date) DO UPDATE SET
    open   = EXCLUDED.open,
    high   = EXCLUDED.high,
    low    = EXCLUDED.low,
    close  = EXCLUDED.close,
    volume = EXCLUDED.volume
RETURNING date;
"""

_LAST_DAILY_SQL = "SELECT MAX(date) FROM market_data WHERE ticker = %s;"


def backfill(conn, tickers: list[str]) -> int:
    """
    Достраивает дневные свечи из 5M для всех тикеров. Возвращает число
    реконструированных (ticker, date) строк.
    """
    today_msk = datetime.now(MSK).date()
    total = 0
    with conn.cursor() as cur:
        for tk in tickers:
            cur.execute(_LAST_DAILY_SQL, (tk,))
            row = cur.fetchone()
            last = row[0] if row and row[0] else "2000-01-01"
            cur.execute(_BACKFILL_SQL, {"tk": tk, "after": last, "today": today_msk})
            built = cur.fetchall()
            if built:
                days = ", ".join(str(d[0]) for d in built)
                log.info("  [1D←5M] %s: реконструировано из 5-минуток — %s", tk, days)
                total += len(built)
    conn.commit()
    if total:
        log.info("Реконструировано дневных свечей из 5M: %d "
                 "(будут перезаписаны при публикации официального бара).", total)
    return total
