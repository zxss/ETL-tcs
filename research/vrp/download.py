"""
Выгрузка дневной истории опционов и фьючерсов RI (RTS) и Si из ISS Мосбиржи.

Зачем: проверка премии за волатильность на реальных ценах опционов
(RESEARCH-VRP-FEASIBILITY-2026-09-26.md). Индекс RVI показал премию +5,3 п.
волатильности, но в дисперсии она неотличима от нуля из-за хвостов — стратегии с
урезанным хвостом нужно мерить на ценах серий, а не на индексе.

Источник (публичный, без токена):
  опционы  /iss/history/engines/futures/markets/options/securities.json
  фьючерсы /iss/history/engines/futures/markets/forts/securities.json
  с параметрами date=YYYY-MM-DD & assetcode=RTS|Si; страница — не больше 100 строк.

Выход: <out>/{options|futures}_{RTS|Si}_{год}.csv (дописывается) и done.txt —
выполненные (вид, актив, дата), чтобы прерванную выгрузку можно было продолжить.
Метаданные серий (страйк, тип, месяц) — из кода серии, research/vrp/secid.py;
экспирация — последний день, когда серия встречается.

Запуск на сервере (фоном, только чтение внешнего API):
    python -m research.vrp.download --out /opt/research-data/vrp --from 2014-01-01
"""
from __future__ import annotations

import argparse
import csv
import http.client
import datetime as dt
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger("vrp.download")

ISS = "https://iss.moex.com/iss"
COLS = ("TRADEDATE", "SECID", "OPEN", "LOW", "HIGH", "CLOSE", "VALUE", "VOLUME",
        "OPENPOSITION", "SETTLEPRICE", "WAPRICE")
MARKETS = {"options": "options", "futures": "forts"}
ASSETS = ("RTS", "Si")

_lock = threading.Lock()


def _get(url: str, tries: int = 6) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        # OSError покрывает обрыв соединения (ConnectionResetError) и ошибки TLS:
        # 27.09.2026 в 01:05 один такой обрыв уронил поток на 12 794-м задании из 12 796.
        except (OSError, ValueError, http.client.HTTPException) as e:
            wait = min(60, 2 ** k)
            log.warning("ISS: %s — повтор через %d с", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"ISS не ответил: {url}")


def trading_days(start: dt.date, end: dt.date) -> list[str]:
    """Торговые дни — даты истории индекса РТС (у FORTS тот же календарь)."""
    out, s = [], 0
    while True:
        j = _get(f"{ISS}/history/engines/stock/markets/index/securities/RTSI.json"
                 f"?iss.meta=off&from={start}&till={end}&start={s}"
                 f"&history.columns=TRADEDATE")
        h = j["history"]["data"]
        out += [r[0] for r in h]
        if len(h) < 100:
            return sorted(set(out))
        s += len(h)


def fetch_day(kind: str, asset: str, day: str) -> list[list]:
    rows, s = [], 0
    cols = ",".join(COLS)
    while True:
        j = _get(f"{ISS}/history/engines/futures/markets/{MARKETS[kind]}/securities.json"
                 f"?iss.meta=off&date={day}&assetcode={asset}&start={s}&limit=100"
                 f"&history.columns={cols}")
        h = j["history"]["data"]
        rows += h
        total = j["history.cursor"]["data"][0][1]
        s += 100
        if s >= total or not h:
            return rows
        time.sleep(0.2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="start", default="2014-01-01")
    ap.add_argument("--till", default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    os.makedirs(a.out, exist_ok=True)
    done_path = os.path.join(a.out, "done.txt")
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = {tuple(line.split()) for line in f if line.strip()}
    till = dt.date.fromisoformat(a.till) if a.till else dt.date.today() - dt.timedelta(days=1)
    days = trading_days(dt.date.fromisoformat(a.start), till)
    tasks = [(k, s, d) for d in days for k in ("futures", "options") for s in ASSETS
             if (k, s, d) not in done]
    log.info("дней %d, заданий %d (уже сделано %d)", len(days), len(tasks), len(done))

    def run(task):
        kind, asset, day = task
        rows = fetch_day(kind, asset, day)
        path = os.path.join(a.out, f"{kind}_{asset}_{day[:4]}.csv")
        with _lock:
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(COLS)
                w.writerows(rows)
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(f"{kind} {asset} {day}\n")
        return task, len(rows)

    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, t) for t in tasks]
        for fu in as_completed(futs):
            (kind, asset, day), cnt = fu.result()
            n += 1
            if n % 50 == 0 or n == len(tasks):
                rate = n / max(1.0, time.time() - t0)
                left = (len(tasks) - n) / rate / 3600 if rate else float("nan")
                log.info("%d/%d (%s %s %s: %d строк), осталось ≈ %.1f ч",
                         n, len(tasks), kind, asset, day, cnt, left)
    log.info("готово: %d заданий за %.1f ч", n, (time.time() - t0) / 3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
