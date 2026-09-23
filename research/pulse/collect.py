"""
Сбор ДНЕВНОЙ активности постов Т-Банк Пульс по тикерам нашего универса —
только для исследования контрариан-сигнала внимания розницы (ТЗ пользователя
23.09.2026). Демо/исследование, в торговый контур не пишет.

Приватность/ToS (см. память tbank-pulse-data-source):
  * API неофициальный (внутренний REST веб-фронта), чтение без авторизации.
  * Сохраняем ТОЛЬКО агрегат по (тикер, дата): число постов и число уникальных
    авторов. НИ текста постов, НИ ников, НИ финансов авторов — ничего
    персонального ни в CSV, ни тем более в git. Файл *.csv и так в .gitignore.
  * Вежливая задержка между запросами; глубина ограничена START_DATE.

Запуск: python -m research.pulse.collect [--from YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import ssl
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import news_event_study as ns                # noqa: E402

log = logging.getLogger("research.pulse.collect")
OUT = os.path.join(ROOT, "audit", "r4_research", "pulse", "daily_counts.csv")
API = "https://www.tbank.ru/api/invest-gw/social/v1/post/instrument/{tk}"
# Песочница делает TLS-перехват (self-signed в цепочке) — для чтения публичного
# эндпоинта проверку отключаем осознанно; это прокси окружения, не MITM.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
DELAY = 0.2
PAGE = 50
MAX_PAGES = 1000                      # на быстрой сети VDS тяжёлые тикеры достаются глубже


def _get(tk: str, cursor: int | None):
    url = API.format(tk=tk) + f"?limit={PAGE}&appName=invest&platform=web"
    if cursor:
        url += f"&cursor={cursor}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                return json.load(r)
        except Exception as e:                              # noqa: BLE001
            if attempt == 3:
                log.warning("%s: сдаюсь на %s", tk, e)
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def collect_ticker(tk: str, d_from: dt.date) -> dict:
    """(date → {posts, authors:set}) для одного тикера, назад до d_from."""
    by_day: dict[dt.date, dict] = {}
    cursor, pages = None, 0
    while pages < MAX_PAGES:
        d = _get(tk, cursor)
        items = (d.get("payload") or {}).get("items") or []
        if not items:
            break
        stop = False
        for it in items:
            ts = it.get("inserted")
            if not ts:
                continue
            day = dt.date.fromisoformat(ts[:10])
            if day < d_from:
                stop = True
                continue
            rec = by_day.setdefault(day, {"posts": 0, "authors": set()})
            rec["posts"] += 1
            pid = it.get("profileId")
            if pid:
                rec["authors"].add(pid)
        pages += 1
        cursor = (d.get("payload") or {}).get("nextCursor")
        if stop or not cursor:
            break
        time.sleep(DELAY)
    return by_day


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2026-04-01")
    args = ap.parse_args()
    d_from = dt.date.fromisoformat(args.d_from)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tickers = sorted(set(ns.UNIVERSE))
    rows = []
    for i, tk in enumerate(tickers, 1):
        t0 = time.time()
        by_day = collect_ticker(tk, d_from)
        for day, rec in sorted(by_day.items()):
            rows.append({"ticker": tk, "date": day.isoformat(),
                         "posts": rec["posts"], "authors": len(rec["authors"])})
        log.info("[%d/%d] %s: дней %d, постов %d (%.1fs)", i, len(tickers), tk,
                 len(by_day), sum(r["posts"] for r in by_day.values()), time.time() - t0)
        # чекпоинт после каждого тикера
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ticker", "date", "posts", "authors"])
            w.writeheader()
            w.writerows(rows)
    log.info("готово: строк %d → %s", len(rows), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
