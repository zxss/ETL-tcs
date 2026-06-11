"""
Актуальные рыночные котировки на момент запуска анализа.

Назначение: до пересчёта прогнозов получить последнюю доступную цену по каждому
тикеру (Last Price из T-Invest), её время, и понять — рынок открыт (Real-time)
или закрыт (Previous Close). Это позволяет якорить прогноз диапазона и
направленный PnL на ТЕКУЩЕЙ цене, а не на вчерашнем закрытии.

Содержит также реализованную внутридневную/овернайт-доходность (из 5M в БД),
чтобы пересчитывать ОСТАВШУЮСЯ часть движения: если бóльшая часть дневного
хода уже отработана, лучшая дневная стратегия (long/short) может смениться.

Полностью graceful: при любой ошибке API возвращает пустой контекст, и
вызывающий код откатывается на последнюю цену закрытия (source=Previous Close).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import aiohttp

import config
from loaders.moex_loader import (
    make_headers, find_instrument, quotation_to_float, _api_post,
)

log = logging.getLogger("tft.quotes")

MSK = timezone(timedelta(hours=3))

# Возраст котировки, после которого она считается устаревшей (по умолч. 15 мин).
STALE_SECONDS = int(getattr(config, "TFT_STALE_SECONDS", 900))


@dataclass
class Quote:
    ticker: str
    price: float
    ts: datetime           # время последней сделки (MSK)
    age_sec: float         # возраст котировки в секундах
    today_open: float | None = None   # открытие текущей сессии (из 5M)


@dataclass
class MarketContext:
    as_of: datetime                 # время запуска расчёта (MSK)
    source: str                     # "Real-time" | "Previous Close"
    market_open: bool
    stale: bool                     # есть котировки старше STALE_SECONDS
    max_age_sec: float              # максимальный возраст среди тикеров
    quotes: dict                    # ticker -> Quote


# ── Часы торгов MOEX (приближённо) ─────────────────────────────────────────────

def is_market_open(now_msk: datetime) -> bool:
    """MOEX TQBR: будни, основная + вечерняя сессии (~10:00–23:50 MSK)."""
    if now_msk.weekday() >= 5:           # сб/вс
        return False
    t = now_msk.time()
    return (t >= datetime.strptime("09:55", "%H:%M").time()
            and t <= datetime.strptime("23:50", "%H:%M").time())


# ── Открытие текущей сессии из 5M (для реализованного движения) ────────────────

_TODAY_OPEN_SQL = """
    SELECT open
    FROM market_data_5m
    WHERE ticker = %(tk)s
      AND (ts AT TIME ZONE 'Europe/Moscow')::date = %(today)s
    ORDER BY ts ASC
    LIMIT 1;
"""


def _today_open(conn, ticker: str, today_msk) -> float | None:
    try:
        with conn.cursor() as cur:
            cur.execute(_TODAY_OPEN_SQL, {"tk": ticker, "today": today_msk})
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:  # noqa: BLE001
        return None


# ── Получение последних цен через T-Invest ─────────────────────────────────────

async def _fetch_async(tickers: list[str]) -> dict[str, dict]:
    """Возвращает {ticker: {"uid","price","time"}}. Бросает только при фатале."""
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        headers=make_headers(),
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        sem = asyncio.Semaphore(config.MAX_CONCURRENT_TICKERS)

        async def resolve(tk):
            async with sem:
                try:
                    instr = await find_instrument(session, tk)
                    return tk, instr.get("uid") or instr.get("figi")
                except Exception:  # noqa: BLE001
                    return tk, None

        pairs = await asyncio.gather(*(resolve(tk) for tk in tickers))
        uid_by_tk = {tk: uid for tk, uid in pairs if uid}
        if not uid_by_tk:
            return {}

        uids = list(uid_by_tk.values())
        data = await _api_post(session, "MarketDataService/GetLastPrices",
                               {"instrumentId": uids})

    by_uid = {}
    for lp in data.get("lastPrices", []):
        uid = lp.get("instrumentUid") or lp.get("figi")
        by_uid[uid] = lp

    out = {}
    for tk, uid in uid_by_tk.items():
        lp = by_uid.get(uid)
        if not lp:
            continue
        price = quotation_to_float(lp.get("price", {}))
        if price <= 0:
            continue
        out[tk] = {"uid": uid, "price": price, "time": lp.get("time")}
    return out


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(MSK)
    except Exception:  # noqa: BLE001
        return None


# ── Публичная точка входа ──────────────────────────────────────────────────────

def get_market_context(conn, tickers: list[str]) -> MarketContext:
    """
    Собирает актуальные котировки + контекст рынка. Никогда не бросает наружу:
    при ошибке возвращает контекст с source="Previous Close" и пустыми quotes.
    """
    now = datetime.now(MSK)
    market_open = is_market_open(now)
    today = now.date()

    use_live = getattr(config, "TFT_USE_LIVE_PRICE", True)
    raw: dict[str, dict] = {}
    if use_live:
        try:
            raw = asyncio.run(_fetch_async(tickers))
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось получить актуальные котировки (%s) — "
                        "откат на последнюю цену закрытия.", e)
            raw = {}
    else:
        log.info("TFT_USE_LIVE_PRICE=0 — расчёт на последней цене закрытия.")

    quotes: dict[str, Quote] = {}
    max_age = 0.0
    for tk, d in raw.items():
        ts = _parse_ts(d.get("time")) or now
        age = max(0.0, (now - ts).total_seconds())
        max_age = max(max_age, age)
        quotes[tk] = Quote(
            ticker=tk, price=d["price"], ts=ts, age_sec=age,
            today_open=_today_open(conn, tk, today),
        )

    if quotes and market_open:
        source = "Real-time"
    else:
        source = "Previous Close"
    stale = bool(quotes) and market_open and max_age > STALE_SECONDS

    return MarketContext(
        as_of=now, source=source, market_open=market_open,
        stale=stale, max_age_sec=max_age, quotes=quotes,
    )
