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
import tls

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


def bar_date_msk(iso_ts: str) -> str:
    """ISO-метка начала бара (UTC) → торговая дата 'YYYY-MM-DD' по Москве.

    Раньше дата бралась срезом строки iso_ts[:10], то есть по КАЛЕНДАРЮ UTC.
    Сейчас T-Invest отдаёт дневные бары как '2026-09-05T00:00:00Z', и срез
    совпадает с московской датой — но это совпадение, а не гарантия: как только
    брокер вернёт начало бара в 21:00Z (= 00:00 MSK следующего дня), срез
    сдвинет всю историю на день назад. Торговый день MOEX — московский
    календарный день, поэтому переводим явно."""
    ts = iso_ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # запасной путь: дробные секунды переменной длины
        head, _, rest = ts.partition(".")
        tz = rest[-6:] if len(rest) >= 6 else "+00:00"
        dt = datetime.fromisoformat(head + tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MSK).date().isoformat()


def make_headers() -> dict:
    """Заголовки запроса к API брокера. Здесь же — валидация наличия токена:
    config импортируется без него, а сетевой клиент без токена работать не может."""
    return {
        "Authorization": f"Bearer {config.require_invest_token()}",
        "Content-Type": "application/json",
    }


def make_connector() -> aiohttp.TCPConnector:
    """TCPConnector с проверкой TLS по config.INVEST_TLS_VERIFY (по умолч. вкл)."""
    return aiohttp.TCPConnector(ssl=tls.aiohttp_ssl())


# --- Расписание торгов (синхронный запрос) ----------------------------------

def fetch_trading_schedules_sync(exchange: str, from_date, to_date, timeout: float = 20.0) -> dict:
    """InstrumentsService/TradingSchedules по бирже — синхронно, на stdlib.

    Синхронный, потому что вызывается из services/calendar.sync_schedule() вне
    asyncio-контура (шаг ETL в main.py). Ограничения API: `from` не может быть
    в прошлом, `to` — не дальше 14 дней от текущей даты.
    """
    import json
    import urllib.error
    import urllib.request

    import tls

    def _iso(d):
        if isinstance(d, datetime):
            return d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)\
            .isoformat().replace("+00:00", "Z")

    url = f"{config.API_BASE_URL}/{config.API_SERVICE}.InstrumentsService/TradingSchedules"
    body = json.dumps({"exchange": exchange,
                       "from": _iso(from_date),
                       "to": _iso(to_date)}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {config.require_invest_token()}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=tls.ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"TradingSchedules: HTTP {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"TradingSchedules: сетевая ошибка: {e}") from e


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
            bar_date_msk(key),                           # торговая дата (MSK)
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
