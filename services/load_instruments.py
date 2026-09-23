"""
Наполнение кэша справочника инструментов из T-Invest API.

Зачем отдельный шаг: размер лота, шаг цены и доступность шорта — справочные
данные, которые меняются редко, но меняются (сплиты, редомициляции, изменение
маржинального списка). Держать их в хардкоде нельзя: сверка ручной таблицы
_LOT_SIZES с API показала расхождение по 11 тикерам из 46, включая VTBR
(10 000 против 1) и полностью отсутствующий TGKA (лот 100 000).

Источник: InstrumentsService/ShareBy по classCode=TQBR.

Шаг идемпотентен и не обязателен для работы конвейера: если API недоступен,
используется ранее сохранённый кэш. Полностью пустой кэш — повод для
предупреждения, а не для падения: потребители обязаны отличать «лот неизвестен»
от «лот = 1».

Запуск отдельно:
    python3 -m services.load_instruments
    python3 -m services.load_instruments --show
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import config
import database

log = logging.getLogger("load_instruments")

# Кэш считается свежим столько дней; при более старом — обновляем.
MAX_AGE_DAYS = 7


def _quotation(x) -> float | None:
    """Quotation {units, nano} → float."""
    if not isinstance(x, dict):
        return None
    return int(x.get("units", 0)) + int(x.get("nano", 0)) / 1e9


def fetch(tickers: list[str]) -> list[dict]:
    """Запрашивает справочник по тикерам. Ошибка одного не валит остальные."""
    from services.broker import BrokerError, TinkoffProdClient

    broker = TinkoffProdClient()
    rows, failed = [], []
    for tk in tickers:
        try:
            data = broker._post("InstrumentsService/ShareBy", {
                "idType": "INSTRUMENT_ID_TYPE_TICKER",
                "classCode": "TQBR",
                "id": tk.upper(),
            })
        except BrokerError as e:
            failed.append((tk, str(e)[:80]))
            continue
        except Exception as e:  # noqa: BLE001 — сеть/парсинг
            failed.append((tk, f"{type(e).__name__}: {e}"[:80]))
            continue

        inst = data.get("instrument") or {}
        if not inst.get("ticker"):
            failed.append((tk, "пустой ответ"))
            continue
        rows.append({
            "ticker": inst.get("ticker", tk).upper(),
            "figi": inst.get("figi"),
            "uid": inst.get("uid"),
            "name": inst.get("name"),
            "class_code": inst.get("classCode"),
            "lot": inst.get("lot"),
            "min_price_increment": _quotation(inst.get("minPriceIncrement")),
            "short_enabled": inst.get("shortEnabledFlag"),
            "buy_available": inst.get("buyAvailableFlag"),
            "sell_available": inst.get("sellAvailableFlag"),
            "api_trade_available": inst.get("apiTradeAvailableFlag"),
            "for_qual_investor": inst.get("forQualInvestorFlag"),
            "dlong_client": _quotation(inst.get("dlongClient")),
            "dshort_client": _quotation(inst.get("dshortClient")),
            "trading_status": inst.get("tradingStatus"),
            "sector": inst.get("sector"),
        })

    if failed:
        log.warning("Не удалось получить справочник по %d тикерам: %s",
                    len(failed), ", ".join(t for t, _ in failed))
    return rows


def is_stale(conn=None, max_age_days: int = MAX_AGE_DAYS) -> bool:
    """Кэш пуст или старше max_age_days."""
    cache = database.get_instruments(conn)
    if not cache:
        return True
    ages = [r.get("fetched_at") for r in cache.values() if r.get("fetched_at")]
    if not ages:
        return True
    newest = max(ages)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - newest).days >= max_age_days


def run(conn=None, tickers: list[str] | None = None, force: bool = False) -> int:
    """Обновляет кэш справочника. Возвращает число записанных строк.

    Наружу не бросает: расчётный контур должен пережить недоступность API,
    работая на прошлом снимке.
    """
    tickers = tickers or config.TICKERS
    if not force and not is_stale(conn):
        log.info("Кэш инструментов свежий — обновление пропущено.")
        return 0
    try:
        rows = fetch(tickers)
    except Exception as e:  # noqa: BLE001 — нет токена / нет сети
        log.warning("Справочник инструментов не обновлён (%s) — "
                    "используется прошлый снимок кэша.", e)
        return 0
    if not rows:
        log.warning("Справочник инструментов пуст — используется прошлый снимок.")
        return 0
    n = database.save_instruments(rows, conn)
    log.info("Справочник инструментов обновлён: %d бумаг.", n)

    no_short = sorted(r["ticker"] for r in rows if r.get("short_enabled") is False)
    if no_short:
        log.info("  Шорт недоступен: %s", " ".join(no_short))
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Кэш справочника инструментов T-Invest")
    p.add_argument("--force", action="store_true", help="Обновить, даже если кэш свежий")
    p.add_argument("--show", action="store_true", help="Показать содержимое кэша")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    database.init_db()

    if not args.show:
        run(force=args.force)

    cache = database.get_instruments()
    if not cache:
        print("Кэш пуст.")
        return 1
    print(f"{'тикер':8}{'лот':>8}{'шаг цены':>12}{'шорт':>7}  обновлён")
    for tk in sorted(cache):
        r = cache[tk]
        short = "да" if r.get("short_enabled") else ("нет" if r.get("short_enabled") is False else "—")
        step = r.get("min_price_increment")
        fetched = r.get("fetched_at")
        print(f"{tk:8}{r.get('lot'):>8}{(float(step) if step else 0):>12.6f}{short:>7}  "
              f"{fetched:%Y-%m-%d %H:%M}" if fetched else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
