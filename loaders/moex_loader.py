"""
Асинхронный загрузчик свечей через T-Invest API (REST).

Использует aiohttp для HTTP-запросов. Каждый тикер грузится параллельно
(контролируется семафором MAX_CONCURRENT_TICKERS из config).
Синхронные DB-вызовы обёрнуты в asyncio.to_thread().
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import aiohttp

import config

log = logging.getLogger("moex_loader")

RETRYABLE = {429, 500, 502, 503, 504}
MSK = timezone(timedelta(hours=3))


# --- Утилиты ----------------------------------------------------------------

def quotation_to_float(q: dict) -> float:
    """Quotation/MoneyValue {units, nano} -> float."""
    if not q:
        return 0.0
    return int(q.get("units", 0)) + int(q.get("nano", 0)) / 1e9


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_headers() -> dict:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return {
        "Authorization": f"Bearer {config.INVEST_TOKEN}",
        "Content-Type": "application/json",
    }


# --- Async HTTP с ретраями --------------------------------------------------

async def _api_post(
    session: aiohttp.ClientSession,
    method: str,
    payload: dict,
) -> dict:
    url = f"{config.API_BASE_URL}/{config.API_SERVICE}.{method}"

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)

                body = await resp.text()
                if resp.status in RETRYABLE and attempt < config.MAX_RETRIES:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else config.BASE_SLEEP * (2 ** attempt)
                    log.warning("HTTP %s %s. Повтор через %.1f с", resp.status, method, wait)
                    await asyncio.sleep(wait)
                    continue

                raise RuntimeError(f"Ошибка {method}: HTTP {resp.status} {body}")

        except aiohttp.ClientError as e:
            if attempt == config.MAX_RETRIES:
                raise RuntimeError(f"Сетевая ошибка {method}: {e}") from e
            backoff = config.BASE_SLEEP * (2 ** attempt)
            log.warning("Сеть: %s. Повтор через %.1f с", e, backoff)
            await asyncio.sleep(backoff)

    raise RuntimeError(f"Не удалось выполнить {method} за {config.MAX_RETRIES} попыток")


# --- Поиск инструмента -------------------------------------------------------

async def find_instrument(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Ищет инструмент по тикеру на TQBR."""
    data = await _api_post(session, "InstrumentsService/FindInstrument", {
        "query": ticker,
        "instrumentKind": "INSTRUMENT_TYPE_SHARE",
    })
    items = data.get("instruments", [])
    for it in items:
        if it.get("ticker") == ticker and it.get("classCode") == "TQBR":
            return it
    if items:
        log.warning("Точного совпадения для %s не найдено, берём первый результат", ticker)
        return items[0]
    raise RuntimeError(f"Инструмент {ticker} не найден в API")


# --- Чанкированная загрузка свечей ------------------------------------------

async def _fetch_chunked(
    session: aiohttp.ClientSession,
    instrument_uid: str,
    interval_enum: str,
    chunk_days: int,
    from_dt: datetime,
    to_dt: datetime,
) -> dict[str, dict]:
    """Грузит свечи чанками, дедуплицирует по времени начала бара."""
    window = timedelta(days=chunk_days)
    by_time: dict[str, dict] = {}
    cur = from_dt
    requests_made = 0

    while cur < to_dt:
        chunk_to = min(cur + window, to_dt)
        data = await _api_post(session, "MarketDataService/GetCandles", {
            "instrumentId": instrument_uid,
            "from": iso_utc(cur),
            "to": iso_utc(chunk_to),
            "interval": interval_enum,
        })
        requests_made += 1

        for c in data.get("candles", []):
            if not c.get("isComplete", True):
                continue
            by_time[c["time"]] = c

        cur = chunk_to
        await asyncio.sleep(config.REQUEST_SLEEP)

    log.debug("Запросов: %d, свечей: %d", requests_made, len(by_time))
    return by_time


# --- Публичные загрузчики ---------------------------------------------------

async def fetch_daily_candles(
    session: aiohttp.ClientSession,
    instrument_uid: str,
    from_dt: datetime,
    to_dt: datetime,
) -> list[tuple]:
    """
    Загружает дневные свечи.
    Возвращает список кортежей (date_str, open, high, low, close, volume).
    """
    by_time = await _fetch_chunked(
        session, instrument_uid,
        "CANDLE_INTERVAL_DAY", config.CHUNK_DAYS_DAILY,
        from_dt, to_dt,
    )
    rows = []
    for key in sorted(by_time):
        c = by_time[key]
        o = quotation_to_float(c["open"])
        if o == 0:
            continue
        rows.append((
            key[:10],                                    # date
            round(o, 4),
            round(quotation_to_float(c["high"]), 4),
            round(quotation_to_float(c["low"]), 4),
            round(quotation_to_float(c["close"]), 4),
            int(c.get("volume", 0)),
        ))
    return rows


async def fetch_5m_candles(
    session: aiohttp.ClientSession,
    instrument_uid: str,
    from_dt: datetime,
    to_dt: datetime,
) -> list[tuple]:
    """
    Загружает 5-минутные свечи (чанки по 1 дню — лимит API).
    Возвращает список кортежей (ts_str, open, high, low, close, volume).
    """
    by_time = await _fetch_chunked(
        session, instrument_uid,
        "CANDLE_INTERVAL_5_MIN", config.CHUNK_DAYS_5M,
        from_dt, to_dt,
    )
    rows = []
    for key in sorted(by_time):
        c = by_time[key]
        o = quotation_to_float(c["open"])
        if o == 0:
            continue
        rows.append((
            key,                                         # ts (ISO UTC)
            round(o, 4),
            round(quotation_to_float(c["high"]), 4),
            round(quotation_to_float(c["low"]), 4),
            round(quotation_to_float(c["close"]), 4),
            int(c.get("volume", 0)),
        ))
    return rows
