"""
Длинная дневная история акций с MOEX ISS (публичный API биржи, без токена) —
для гипотезы C1 (ТЗ пользователя 24.09.2026).

Грузим ПОЛНЫЙ срез доски TQBR на каждую дату (а не текущий список тикеров),
поэтому в данных есть и бумаги, позже снятые с торгов, — без ошибки выжившего.
Плюс дивиденды (T-Invest GetDividends, только чтение) — для полной доходности.

Хранение: файлы в audit/r4_research/longhist (по году csv.gz, дивиденды csv) —
в БД ничего не пишем, торговый контур не затрагивается. Повторный запуск
докачивает только недостающие годы.
Запуск (сервер): python -m research.longhist.iss_loader [--from 2014] [--to 2024]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

log = logging.getLogger("research.longhist.iss_loader")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "longhist")
ISS = "https://iss.moex.com/iss"
COLS = "SECID,TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE"
LAST_DAY = dt.date(2024, 5, 20)          # дальше есть market_data


def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:                       # сеть/ISS иногда рвёт — повтор
            log.warning("повтор %d: %s", attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def day_slice(d: dt.date) -> list[list]:
    rows, start = [], 0
    while True:
        j = _get(f"{ISS}/history/engines/stock/markets/shares/boards/TQBR/securities.json"
                 f"?date={d}&start={start}&iss.meta=off&history.columns={COLS}")
        data = j["history"]["data"]
        rows += data
        if len(data) < 100:
            return rows
        start += len(data)


def load_year(y: int) -> pd.DataFrame:
    rows = []
    d = dt.date(y, 1, 1)
    end = min(dt.date(y, 12, 31), LAST_DAY)
    while d <= end:
        if d.weekday() < 5:
            rows += day_slice(d)
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows, columns=COLS.split(","))


def dividends(secids: list[str]) -> pd.DataFrame:
    """Дивиденды — из T-Invest GetDividends (только чтение; у ISS публичного
    эндпоинта дивидендов нет). Снятые с торгов бумаги API часто не находит —
    для них дивидендов не будет (оговорено в отчёте)."""
    import config
    import tls
    from research.fetch_dividends import parse_dividends
    tok = config.require_invest_token()

    def post(method, body):
        req = urllib.request.Request(f"{config.API_BASE_URL}/{config.API_SERVICE}.{method}",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                return json.loads(urllib.request.urlopen(req, context=tls.ssl_context(), timeout=60).read())
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    return {}
                time.sleep(5 * (attempt + 1))
            except Exception:
                time.sleep(3)
        return {}

    rows = []
    for sid in secids:
        its = [i for i in post("InstrumentsService/FindInstrument", {"query": sid}).get("instruments", [])
               if i.get("ticker") == sid and i.get("classCode") == "TQBR"]
        if its:
            rows += parse_dividends(sid, post("InstrumentsService/GetDividends", {
                "instrumentId": its[0]["uid"], "from": "2013-01-01T00:00:00Z",
                "to": "2024-12-31T00:00:00Z"}))
        time.sleep(0.4)
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, default=2014)
    ap.add_argument("--to", dest="y1", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    secids: set[str] = set()
    for y in range(a.y0, a.y1 + 1):
        path = os.path.join(OUT_DIR, f"tqbr_{y}.csv.gz")
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            t0 = time.time()
            df = load_year(y)
            df.to_csv(path, index=False)
            log.info("%d: %d строк, %d бумаг, %.0f с", y, len(df), df["SECID"].nunique(), time.time() - t0)
        secids |= set(df["SECID"].unique())
    dv = dividends(sorted(secids))
    dv.to_csv(os.path.join(OUT_DIR, "dividends.csv"), index=False)
    log.info("дивиденды: %d записей по %d бумагам", len(dv), dv["ticker"].nunique() if len(dv) else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
