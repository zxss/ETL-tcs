"""
Форвардный сбор стакана и ленты сделок (ТЗ пользователя 24.09.2026, C2).

Единственный класс данных, которого у программы нет, — микроструктура
(стакан/поток заявок). Истории нет нигде, поэтому копим с сегодняшнего дня;
через 4–6 недель — тест «дисбаланс стакана / знак потока → доходность 1–5 мин»
и оценка лимитного исполнения (A2) по настоящим данным.

Только чтение (MarketDataService), заявок не ставит. Лимиты API общие с
торговым контуром, поэтому нагрузка намеренно маленькая:
  * GetOrderBook depth 20 по каждой бумаге раз в POLL_SEC (10 бумаг / 15 с ≈
    40 запросов/мин при лимите unary MarketData в сотни в минуту);
  * GetLastTrades (лента за последний час, с направлением buy/sell) раз в
    TRADES_EVERY_SEC по каждой бумаге — дубли снимаются по времени сделки.
Храним агрегаты снимка (лучшие цены, суммы объёма по 1/5/20 уровням, спред,
дисбаланс) и сделки — в csv.gz по дням в audit/r4_research/orderflow. В БД
ничего не пишем. Процесс сам завершается в STOP_AT.

Запуск (сервер): python -m research.orderflow.collector [--minutes N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

log = logging.getLogger("research.orderflow.collector")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "orderflow")
TICKERS = ["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "TATN", "VTBR", "YDEX", "PLZL"]
POLL_SEC = 15
TRADES_EVERY_SEC = 20 * 60
STOP_AT = dt.time(23, 50)
MSK = dt.timezone(dt.timedelta(hours=3))


def _q(x: dict | None) -> float:
    return (int(x.get("units", 0)) + int(x.get("nano", 0)) / 1e9) if x else 0.0


class Api:
    def __init__(self):
        import config
        import tls
        self._cfg, self._ctx = config, tls.ssl_context()
        self._tok = config.require_invest_token()

    def post(self, method: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self._cfg.API_BASE_URL}/{self._cfg.API_SERVICE}.{method}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self._tok}", "Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, context=self._ctx, timeout=20).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log.warning("429 — пауза 30 с")
                time.sleep(30)
            return {}
        except Exception as e:
            log.warning("%s: %s", method, e)
            return {}


def resolve(api: Api) -> dict[str, str]:
    out = {}
    for tk in TICKERS:
        its = [i for i in api.post("InstrumentsService/FindInstrument", {"query": tk}).get("instruments", [])
               if i.get("ticker") == tk and i.get("classCode") == "TQBR"]
        if its:
            out[tk] = its[0]["uid"]
    return out


def book_row(tk: str, ob: dict) -> dict | None:
    bids = [(_q(b.get("price")), int(b.get("quantity", 0))) for b in ob.get("bids", [])]
    asks = [(_q(a.get("price")), int(a.get("quantity", 0))) for a in ob.get("asks", [])]
    if not bids or not asks:
        return None
    row = {"ts": dt.datetime.now(MSK).isoformat(timespec="milliseconds"), "ticker": tk,
           "orderbook_ts": ob.get("orderbookTs", ""), "bid1": bids[0][0], "ask1": asks[0][0],
           "last": _q(ob.get("lastPrice"))}
    for n in (1, 5, 20):
        bq, aq = sum(q for _, q in bids[:n]), sum(q for _, q in asks[:n])
        row[f"bid_q{n}"], row[f"ask_q{n}"] = bq, aq
        row[f"imb{n}"] = round((bq - aq) / (bq + aq), 4) if bq + aq else 0.0
    row["spread_bp"] = round((row["ask1"] - row["bid1"]) / ((row["ask1"] + row["bid1"]) / 2) * 1e4, 2)
    return row


class DayWriter:
    def __init__(self, kind: str):
        self.kind, self.day, self.f, self.w = kind, None, None, None

    def write(self, row: dict):
        d = row["ts"][:10]
        if d != self.day:
            if self.f:
                self.f.close()
            os.makedirs(OUT_DIR, exist_ok=True)
            path = os.path.join(OUT_DIR, f"{self.kind}_{d}.csv.gz")
            new = not os.path.exists(path)
            self.f = gzip.open(path, "at", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            if new:
                self.w.writeheader()
            self.day = d
        self.w.writerow(row)

    def flush(self):
        if self.f:
            self.f.flush()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None, help="ограничить длительность (проверка)")
    a = ap.parse_args()
    api = Api()
    uids = resolve(api)
    log.info("инструменты: %s", ", ".join(uids))
    books, trades = DayWriter("book"), DayWriter("trades")
    seen: dict[str, str] = {}
    last_trades = 0.0
    t_end = time.time() + a.minutes * 60 if a.minutes else None
    n_book = n_tr = 0
    while True:
        now = dt.datetime.now(MSK)
        if now.time() >= STOP_AT or (t_end and time.time() >= t_end):
            break
        t0 = time.time()
        for tk, uid in uids.items():
            r = book_row(tk, api.post("MarketDataService/GetOrderBook", {"instrumentId": uid, "depth": 20}))
            if r:
                books.write(r)
                n_book += 1
        if time.time() - last_trades >= TRADES_EVERY_SEC:
            for tk, uid in uids.items():
                frm = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=59)).isoformat().replace("+00:00", "Z")
                resp = api.post("MarketDataService/GetLastTrades", {"instrumentId": uid, "from": frm,
                                "to": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")})
                mark = seen.get(tk, "")
                for t in sorted(resp.get("trades", []), key=lambda x: x.get("time", "")):
                    if t.get("time", "") <= mark:
                        continue
                    trades.write({"ts": dt.datetime.now(MSK).isoformat(timespec="seconds"), "ticker": tk,
                                  "time": t.get("time"), "direction": t.get("direction"),
                                  "price": _q(t.get("price")), "quantity": int(t.get("quantity", 0))})
                    seen[tk] = t.get("time")
                    n_tr += 1
            last_trades = time.time()
        books.flush(); trades.flush()
        time.sleep(max(0.0, POLL_SEC - (time.time() - t0)))
    log.info("готово: снимков стакана %d, сделок %d", n_book, n_tr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
