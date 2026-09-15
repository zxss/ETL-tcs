"""
Дивиденды бумаг вселенной для поправки доходностей event study (Спринт 1, v2).
Только чтение API: InstrumentsService/GetDividends → CSV.

Доходность в event study ценовая: в день отсечки цена падает на дивиденд, а
держатель его получает. Без поправки окно, пересекающее отсечку, показывает
ложное падение (dev v1: посты «отсечка/последний день» — CAR 1d −2,36 %).
Для старых кодов (YNDX, FIVE, FIXP) дивиденды пишутся под нынешним тикером.

Запуск (на сервере): python -m research.fetch_dividends
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_PATH = os.path.join(ROOT, "audit", "r4_research", "sprint1", "dividends.csv")
OLD_CODES = {"YDEX": ("YNDX",), "X5": ("FIVE",), "FIXR": ("FIXP",)}


def _q(x: dict | None) -> float:
    return (int(x.get("units", 0)) + int(x.get("nano", 0)) / 1e9) if x else 0.0


def parse_dividends(ticker: str, payload: dict) -> list[dict]:
    """Ответ GetDividends → строки CSV (без нулевых и без даты последней покупки)."""
    rows = []
    for d in payload.get("dividends", []) or []:
        amount = _q(d.get("dividendNet"))
        last_buy = (d.get("lastBuyDate") or "")[:10]
        if amount <= 0 or not last_buy or last_buy.startswith("1970"):
            continue
        rows.append({"ticker": ticker, "last_buy_date": last_buy,
                     "record_date": (d.get("recordDate") or "")[:10],
                     "dividend_net": round(amount, 6), "close_price": _q(d.get("closePrice"))})
    return rows


def main() -> int:
    import config
    import tls
    from research import news_classify as ncl
    tok = config.require_invest_token()

    def post(method, body):
        req = urllib.request.Request(f"{config.API_BASE_URL}/{config.API_SERVICE}.{method}",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, context=tls.ssl_context(), timeout=60).read())
        except urllib.error.HTTPError as e:
            return {"error": e.code}

    rows = []
    for tk in ncl.EVENT_UNIVERSE:
        for code in (tk,) + OLD_CODES.get(tk, ()):
            its = [i for i in post("InstrumentsService/FindInstrument", {"query": code}).get("instruments", [])
                   if i.get("ticker") == code and i.get("classCode") == "TQBR"]
            if not its:
                continue
            got = parse_dividends(tk, post("InstrumentsService/GetDividends", {
                "instrumentId": its[0]["uid"], "from": "2021-12-01T00:00:00Z",
                "to": "2026-12-31T00:00:00Z"}))
            rows += got
            print(f"{tk:<6} ({code}): выплат {len(got)}")
    uniq = {(r["ticker"], r["last_buy_date"]): r for r in rows}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "last_buy_date", "record_date", "dividend_net",
                                          "close_price"])
        w.writeheader()
        w.writerows(sorted(uniq.values(), key=lambda r: (r["ticker"], r["last_buy_date"])))
    print(f"записано выплат {len(uniq)} → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
