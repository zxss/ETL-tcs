"""
Загрузка календаря дивидендных отсечек (InstrumentsService/GetDividends).

Метод отдаёт и уже прошедшие, и ОБЪЯВЛЕННЫЕ будущие выплаты, поэтому один
ежедневный проход даёт и историю для дивидендной поправки обучающей выборки, и
календарь вперёд для торгового контура. Объявляют выплату обычно за 1–2 месяца
до отсечки, поэтому окно запроса берётся с запасом в оба конца.

Ключевое поле — lastBuyDate: последний день покупки с правом на дивиденд. Гэп
приходится на открытие СЛЕДУЮЩЕГО торгового дня. Выплаты без lastBuyDate или с
нулевой суммой отбрасываются: гэпа по ним нет.

UID инструмента берётся из кэша справочника (таблица instruments, шаг
load_instruments в main.py идёт РАНЬШЕ). Если кэш пуст, тикер ищется через
FindInstrument — медленнее, но шаг не должен зависеть от порядка.

Управление через config: LOAD_DIVIDENDS, DIVIDENDS_HISTORY_YEARS.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging

import aiohttp

import config
import database
from loaders.moex_loader import make_headers, make_connector, _api_post

log = logging.getLogger("load_dividends")

# Старые коды бумаг: выплаты до смены тикера лежат в API под прежним кодом, а в
# наших свечах вся история идёт под нынешним. Пишем под нынешний.
OLD_CODES: dict[str, tuple[str, ...]] = {
    "YDEX": ("YNDX",), "X5": ("FIVE",), "FIXR": ("FIXP",),
}


def _quotation(x: dict | None) -> float:
    """Quotation/MoneyValue → float. Отсутствующее значение — 0."""
    if not x:
        return 0.0
    return int(x.get("units", 0) or 0) + int(x.get("nano", 0) or 0) / 1e9


def parse_dividends(ticker: str, payload: dict) -> list[dict]:
    """Ответ GetDividends → строки таблицы dividends.

    Отбрасываются выплаты без даты последней покупки и с неположительной суммой:
    без lastBuyDate неизвестно, к какому дню относится гэп, а нулевая выплата
    гэпа не создаёт. Заглушка 1970 в API означает «дата не заполнена».
    """
    out: list[dict] = []
    for d in payload.get("dividends") or []:
        amount = _quotation(d.get("dividendNet"))
        last_buy = str(d.get("lastBuyDate") or "")[:10]
        if amount <= 0 or not last_buy or last_buy.startswith("1970"):
            continue
        record = str(d.get("recordDate") or "")[:10]
        payment = str(d.get("paymentDate") or "")[:10]
        out.append({
            "ticker": ticker.upper(),
            "last_buy_date": last_buy,
            "record_date": record if record and not record.startswith("1970") else None,
            "payment_date": payment if payment and not payment.startswith("1970") else None,
            "dividend_net": round(amount, 6),
            "close_price": _quotation(d.get("closePrice")) or None,
            "currency": (d.get("dividendNet") or {}).get("currency"),
        })
    return out


async def _uid_for(session: aiohttp.ClientSession, code: str,
                   cache: dict[str, dict]) -> str | None:
    """UID по коду бумаги: сперва кэш справочника, потом FindInstrument."""
    rec = cache.get(code.upper())
    if rec and rec.get("uid"):
        return str(rec["uid"])
    data = await _api_post(session, "InstrumentsService/FindInstrument",
                           {"query": code, "instrumentKind": "INSTRUMENT_TYPE_SHARE"})
    for it in data.get("instruments", []) or []:
        if it.get("ticker") == code and it.get("classCode") in ("TQBR", "TQBRH"):
            return it.get("uid") or it.get("figi")
    return None


async def _load_one(session, ticker: str, frm: str, to: str,
                    cache: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for code in (ticker,) + OLD_CODES.get(ticker.upper(), ()):
        uid = await _uid_for(session, code, cache)
        if not uid:
            log.warning("  %s: инструмент не найден, выплаты не запрошены", code)
            continue
        try:
            data = await _api_post(session, "InstrumentsService/GetDividends",
                                   {"instrumentId": uid, "from": frm, "to": to})
        except RuntimeError as e:
            log.warning("  %s: GetDividends не ответил — %s", code, e)
            continue
        got = parse_dividends(ticker, data)
        if got:
            log.info("  %-6s (%s): выплат %d", ticker, code, len(got))
        rows += got
    # Одна отсечка не может быть записана дважды (старый и новый код бумаги).
    uniq = {(r["ticker"], r["last_buy_date"]): r for r in rows}
    return list(uniq.values())


async def _run_async(tickers: list[str], years: int) -> int:
    today = dt.date.today()
    frm = f"{today.year - years}-01-01T00:00:00Z"
    to = f"{today.year + 1}-12-31T00:00:00Z"
    cache = await asyncio.to_thread(database.get_instruments)
    async with aiohttp.ClientSession(headers=make_headers(),
                                     connector=make_connector(),
                                     timeout=aiohttp.ClientTimeout(total=60)) as session:
        rows: list[dict] = []
        for tk in tickers:
            rows += await _load_one(session, tk, frm, to, cache)
    if not rows:
        return 0
    return await asyncio.to_thread(database.save_dividends, rows)


def run(conn=None) -> int:
    """Обновляет таблицу dividends по config.TICKERS. Не бросает наружу.

    Шаг не критичен для ETL: без свежего календаря контур деградирует до
    прежнего поведения (ночной вход не блокируется, поправка не применяется) —
    громко, через предупреждение, а не падением.
    """
    if not getattr(config, "LOAD_DIVIDENDS", True):
        log.info("LOAD_DIVIDENDS=0 — календарь отсечек не обновляется.")
        return 0
    tickers = list(config.TICKERS or [])
    if not tickers:
        return 0
    years = int(getattr(config, "DIVIDENDS_HISTORY_YEARS", 12))
    try:
        n = asyncio.run(_run_async(tickers, years))
        log.info("Календарь отсечек обновлён: %d выплат по %d бумагам", n, len(tickers))
        return n
    except Exception as e:  # noqa: BLE001 — шаг ETL не должен валить конвейер
        log.warning("Загрузка дивидендов завершилась с ошибкой: %s", e)
        return 0
