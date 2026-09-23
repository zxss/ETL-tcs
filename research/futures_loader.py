"""
История фьючерсов на сырьё и валюту для Спринта 2 (ТЗ 50D, товарный lead-lag).
Только чтение API; запись — в research_fut_5m (отдельно от прода, прод её не читает).

Инструменты (SPBFUT, включая истёкшие контракты — T-Invest отдаёт их архивы):
  BR — Brent, GD — золото в $, GL — золото в ₽ (с 2023), NG — газ (США),
  Si — USD/RUB, CR — CNY/RUB (с 06.2022), MX — индекс MOEX;
  плюс спот CNYRUB_TOM (CETS).
Источник — годовые архивы минуток history-data по uid контракта (как
services/backfill_5m); минутки → 5-минутки. Для контракта берутся год экспирации
и предыдущий (∩ период): ближний контракт по объёму выбирается при анализе, а не
здесь, чтобы правило склейки не пряталось в загрузчике.
Справочник контрактов — audit/r4_research/sprint2/contracts.csv.
Продолжение с места обрыва — audit/backfill_5m/state_research_fut_5m.json.

Запуск (на сервере): python -m research.futures_loader --from 2022-01-01 --to 2026-09-11
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services import backfill_5m as bf                   # noqa: E402

log = logging.getLogger("research.futures_loader")

TABLE = "research_fut_5m"
ROOTS = {"BR": "Brent", "GD": "золото $", "GL": "золото ₽", "NG": "газ (США)",
         "Si": "USD/RUB", "CR": "CNY/RUB", "MX": "индекс MOEX"}
SPOT = {"CNYRUB_TOM": "CETS"}
# Фьючерсы на отдельные акции — для базисного арбитража (research/structural).
STOCK_ROOTS = {"SR": "SBER", "LK": "LKOH", "GZ": "GAZP", "VB": "VTBR", "GK": "GMKN"}
ALL_ROOTS = {**ROOTS, **STOCK_ROOTS}
CONTRACTS_PATH = os.path.join(ROOT, "audit", "r4_research", "sprint2", "contracts.csv")


def select_contracts(futures: list[dict], d_from: dt.date, d_to: dt.date,
                     roots: dict = ROOTS) -> list[dict]:
    """Контракты нужных корней, живые в периоде; годы архивов на каждый."""
    out = []
    for f in futures:
        tk = str(f.get("ticker", ""))
        root = tk[:2]
        if root not in roots or f.get("classCode") != "SPBFUT" or len(tk) != 4:
            continue
        raw = (f.get("expirationDate") or "")[:10]
        if not raw:
            continue
        exp = dt.date.fromisoformat(raw)
        if exp < d_from or exp > d_to + dt.timedelta(days=400):
            continue
        years = [y for y in (exp.year - 1, exp.year) if d_from.year <= y <= d_to.year]
        if years:
            out.append({"ticker": tk, "root": root, "uid": f["uid"], "expiration": exp,
                        "years": years, "basic_asset": f.get("basicAsset", "")})
    return sorted(out, key=lambda c: (c["root"], c["expiration"]))


def write_contracts(contracts: list[dict], path: str = CONTRACTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "root", "uid", "expiration", "basic_asset"])
        for c in contracts:
            w.writerow([c["ticker"], c["root"], c["uid"], c["expiration"], c["basic_asset"]])


async def load(conn, d_from: dt.date, d_to: dt.date, roots: dict = ROOTS, spot: dict = SPOT,
               contracts_path: str = CONTRACTS_PATH) -> list[dict]:
    from loaders import moex_loader
    bf.ensure_table(conn, TABLE)
    st_path = bf.state_path_for(TABLE)
    st = bf.load_state(st_path, d_from, d_to)
    stats = []
    async with bf._session() as s:
        data = await moex_loader._api_post(s, "InstrumentsService/Futures",
                                           {"instrumentStatus": "INSTRUMENT_STATUS_ALL"})
        contracts = select_contracts(data.get("instruments", []), d_from, d_to, roots)
        for tk, cls in spot.items():
            found = await moex_loader._api_post(s, "InstrumentsService/FindInstrument", {"query": tk})
            for i in found.get("instruments", []):
                if i.get("ticker") == tk and i.get("classCode") == cls:
                    contracts.append({"ticker": tk, "root": tk, "uid": i["uid"], "expiration": "",
                                      "years": list(range(d_from.year, d_to.year + 1)),
                                      "basic_asset": "CNY/RUB спот"})
        write_contracts(contracts, contracts_path)
        log.info("контрактов %d (архивов %d)", len(contracts), sum(len(c["years"]) for c in contracts))
        for c in contracts:
            for year in c["years"]:
                key = f"{c['ticker']}:{year}"
                if key in st["done"]:
                    continue
                blob = await bf.fetch_archive(s, c["uid"], year)
                await asyncio.sleep(bf.ARCHIVE_PAUSE_SEC)
                lo, hi = max(d_from, dt.date(year, 1, 1)), min(d_to, dt.date(year, 12, 31))
                bars = bf.archive_5m(blob, lo, hi) if blob else []
                new = await asyncio.to_thread(bf.insert_bars, conn, c["ticker"], bars, TABLE)
                rec = {"ticker": c["ticker"], "year": year, "bars": len(bars), "inserted": new,
                       "archive": blob is not None}
                st["done"][key] = rec
                bf.save_state(st_path, st)
                stats.append(rec)
                log.info("%-10s %d: баров %d, новых %d%s", c["ticker"], year, len(bars), new,
                         "" if blob else " (архива нет)")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="История фьючерсов на сырьё и валюту (Спринт 2)")
    ap.add_argument("--from", dest="date_from", default="2022-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-09-11")
    ap.add_argument("--roots", default=",".join(ROOTS),
                    help="корни через запятую, например SR,LK,GZ,VB,GK (фьючерсы на акции)")
    ap.add_argument("--no-spot", action="store_true", help="не грузить спот CNYRUB_TOM")
    ap.add_argument("--contracts-out", default=CONTRACTS_PATH,
                    help="куда записать справочник контрактов (не затирать список Спринта 2)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    roots = {r: ALL_ROOTS[r] for r in a.roots.split(",") if r}
    import database
    conn = database.get_connection()
    try:
        stats = asyncio.run(load(conn, dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to),
                                 roots, {} if a.no_spot else SPOT, a.contracts_out))
    finally:
        conn.close()
    log.info("готово: архивов %d, новых баров %d", len(stats), sum(s["inserted"] for s in stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
