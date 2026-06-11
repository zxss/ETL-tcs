from __future__ import annotations
"""
Асинхронный сервис загрузки исторических данных.

Инкрементальность:
  - Дневные: from_dt = MAX(date)+1 день (или now-INITIAL_MONTHS если пусто).
             Пропуск если MAX(date) >= вчера.
  - 5-мин:   from_dt = MAX(ts)+1 сек (или now-INITIAL_MONTHS_5M если пусто).
             Пропуск если MAX(ts) >= 15 мин назад.

Параллельность:
  - Все тикеры запускаются одновременно, ограничены семафором MAX_CONCURRENT_TICKERS.
  - Синхронные DB-вызовы обёрнуты в asyncio.to_thread().
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta, date

import aiohttp

import config
import database
from loaders.moex_loader import (
    make_headers, find_instrument,
    fetch_daily_candles, fetch_5m_candles,
)

log = logging.getLogger("load_history")

# MOEX торгует по московскому времени (UTC+3, без перехода на летнее время).
# Порог «свежести» дневных свечей нужно считать в MSK, иначе вечером по UTC
# «сегодня» ещё вчерашний московский день и свежая дневная свеча пропускается.
MSK = timezone(timedelta(hours=3))


# --- Инкрементальные границы -------------------------------------------------

def _daily_from_dt(conn, ticker: str, to_dt: datetime) -> datetime | None:
    """None → данные актуальны, пропускаем."""
    last: date | None = database.get_last_date(conn, ticker)
    if last is None:
        return to_dt - timedelta(days=config.INITIAL_MONTHS_DAILY * 31)

    last_dt = datetime(last.year, last.month, last.day, tzinfo=timezone.utc)

    # «Вчера» по московскому календарю: последний заведомо завершённый торговый
    # день. Если последняя свеча в БД его старше — догружаем (API сам вернёт
    # только реально существующие свечи).
    msk_yesterday = (datetime.now(MSK) - timedelta(days=1)).date()
    last_msk_yesterday = datetime(
        msk_yesterday.year, msk_yesterday.month, msk_yesterday.day,
        tzinfo=timezone.utc)

    if last_dt >= last_msk_yesterday:
        return None  # актуально

    return last_dt + timedelta(days=1)


def _5m_from_dt(conn, ticker: str, to_dt: datetime) -> datetime | None:
    """None → данные актуальны, пропускаем."""
    last: datetime | None = database.get_last_ts_5m(conn, ticker)
    if last is None:
        return to_dt - timedelta(days=config.INITIAL_MONTHS_5M * 31)

    if last >= to_dt - timedelta(minutes=15):
        return None  # актуально

    return last + timedelta(seconds=1)


# --- Загрузка одного тикера -------------------------------------------------

async def _load_ticker(
    ticker: str,
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    conn,
    to_dt: datetime,
) -> dict:
    result = {"ticker": ticker, "daily": 0, "5m": 0,
              "skipped_daily": False, "skipped_5m": False}

    async with sem:
        log.info("── %s: поиск инструмента...", ticker)
        try:
            instr = await find_instrument(session, ticker)
        except RuntimeError as e:
            log.error("  %s: инструмент не найден — %s", ticker, e)
            return result

        uid  = instr.get("uid") or instr.get("figi")
        name = instr.get("name", ticker)
        log.info("  %s | %s | UID=%s", ticker, name, uid)

        # ── Дневные ──
        from_daily = await asyncio.to_thread(_daily_from_dt, conn, ticker, to_dt)
        if from_daily is None:
            log.info("  [1D] %s — актуально, пропускаем", ticker)
            result["skipped_daily"] = True
        else:
            log.info("  [1D] %s: %s → %s", ticker, from_daily.date(), to_dt.date())
            try:
                rows = await fetch_daily_candles(session, uid, from_daily, to_dt)
                if rows:
                    db_rows = [(ticker, *r) for r in rows]
                    saved = await asyncio.to_thread(database.upsert_candles, conn, db_rows)
                    result["daily"] = saved
                    log.info("  [1D] %s: +%d строк", ticker, saved)
                else:
                    log.info("  [1D] %s: новых свечей нет", ticker)
            except RuntimeError as e:
                log.error("  [1D] %s: ошибка — %s", ticker, e)

        # ── 5-мин ──
        from_5m = await asyncio.to_thread(_5m_from_dt, conn, ticker, to_dt)
        if from_5m is None:
            log.info("  [5M] %s — актуально, пропускаем", ticker)
            result["skipped_5m"] = True
        else:
            log.info("  [5M] %s: %s → %s", ticker, from_5m.date(), to_dt.date())
            try:
                rows = await fetch_5m_candles(session, uid, from_5m, to_dt)
                if rows:
                    db_rows = [(ticker, *r) for r in rows]
                    saved = await asyncio.to_thread(database.upsert_candles_5m, conn, db_rows)
                    result["5m"] = saved
                    log.info("  [5M] %s: +%d строк", ticker, saved)
                else:
                    log.info("  [5M] %s: новых свечей нет", ticker)
            except RuntimeError as e:
                log.error("  [5M] %s: ошибка — %s", ticker, e)

    return result


# --- Точка входа ------------------------------------------------------------

async def run_async(conn) -> None:
    to_dt = datetime.now(timezone.utc)
    sem   = asyncio.Semaphore(config.MAX_CONCURRENT_TICKERS)

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        headers=make_headers(),
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as session:
        tasks = [
            _load_ticker(ticker, sem, session, conn, to_dt)
            for ticker in config.TICKERS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_daily = total_5m = skipped_daily = skipped_5m = 0
    for r in results:
        if isinstance(r, Exception):
            log.error("Необработанная ошибка задачи: %s", r)
            continue
        total_daily   += r["daily"]
        total_5m      += r["5m"]
        skipped_daily += int(r["skipped_daily"])
        skipped_5m    += int(r["skipped_5m"])

    log.info(
        "Итого: 1D новых=%d пропущено=%d | 5M новых=%d пропущено=%d",
        total_daily, skipped_daily, total_5m, skipped_5m,
    )


def run(conn) -> None:
    asyncio.run(run_async(conn))
