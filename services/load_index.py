"""
Загрузка биржевых индексов (по умолчанию — IMOEX) отдельным шагом ETL.

Индекс нужен слою рыночного контекста (tft_forecast/market.py): по нему
считается режим рынка (EMA50/EMA200 → BULL/BEAR/NEUTRAL) и относительная сила
бумаг (RS). Без него market.py использует синтетический proxy-индекс.

Индекс — это НЕ акция TQBR, поэтому инструмент ищется с instrumentKind=
INSTRUMENT_TYPE_INDEX. Дневные свечи грузятся тем же MarketDataService и
складываются в market_data под тикером индекса ('IMOEX'). Инкрементальность и
порог свежести (MSK) переиспользуются из load_history.

Управление через config: LOAD_INDEX, INDEX_TICKERS.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

import config
import database
from loaders.moex_loader import (
    make_headers, fetch_daily_candles, _api_post,
)
from services.load_history import _daily_from_dt

log = logging.getLogger("load_index")


async def _find_index(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Ищет индекс по тикеру (instrumentKind=INSTRUMENT_TYPE_INDEX)."""
    data = await _api_post(session, "InstrumentsService/FindInstrument", {
        "query": ticker,
        "instrumentKind": "INSTRUMENT_TYPE_INDEX",
    })
    items = data.get("instruments", [])
    for it in items:
        if it.get("ticker") == ticker:
            return it
    if items:
        log.warning("Точного совпадения для индекса %s нет — берём первый результат", ticker)
        return items[0]
    raise RuntimeError(f"Индекс {ticker} не найден в API")


async def _load_one(session, conn, ticker: str, to_dt: datetime) -> int:
    log.info("── индекс %s: поиск инструмента...", ticker)
    try:
        instr = await _find_index(session, ticker)
    except RuntimeError as e:
        log.error("  индекс %s не найден — %s", ticker, e)
        return 0

    uid = instr.get("uid") or instr.get("figi")
    log.info("  %s | %s | UID=%s", ticker, instr.get("name", ticker), uid)

    from_daily = await asyncio.to_thread(_daily_from_dt, conn, ticker, to_dt)
    if from_daily is None:
        log.info("  [1D] %s — актуально, пропускаем", ticker)
        return 0

    log.info("  [1D] %s: %s → %s", ticker, from_daily.date(), to_dt.date())
    try:
        rows = await fetch_daily_candles(session, uid, from_daily, to_dt)
    except RuntimeError as e:
        log.error("  [1D] %s: ошибка — %s", ticker, e)
        return 0
    if not rows:
        log.info("  [1D] %s: новых свечей нет", ticker)
        return 0

    db_rows = [(ticker, *r) for r in rows]
    saved = await asyncio.to_thread(database.upsert_candles, conn, db_rows)
    log.info("  [1D] %s: +%d строк", ticker, saved)
    return saved


async def _run_async(conn, tickers: list[str]) -> int:
    to_dt = datetime.now(timezone.utc)
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        headers=make_headers(),
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as session:
        total = 0
        for tk in tickers:
            total += await _load_one(session, conn, tk, to_dt)
    return total


def run(conn) -> int:
    """
    Загружает индексы из config.INDEX_TICKERS в market_data. Возвращает число
    добавленных строк. Не бросает наружу — шаг не должен валить ETL.
    """
    if not getattr(config, "LOAD_INDEX", True):
        log.info("LOAD_INDEX=0 — загрузка индексов пропущена.")
        return 0
    tickers = getattr(config, "INDEX_TICKERS", ["IMOEX"])
    if not tickers:
        return 0
    try:
        return asyncio.run(_run_async(conn, tickers))
    except Exception as e:  # noqa: BLE001
        log.warning("Загрузка индексов завершилась с ошибкой: %s", e)
        return 0
