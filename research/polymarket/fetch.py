"""
Сбор рынков Polymarket по российско-украинской повестке и их часовой истории.

Публичные API без ключа (только чтение):
  gamma-api.polymarket.com/public-search — поиск событий (открытых и закрытых);
  gamma-api.polymarket.com/events?slug=   — рынки события, токены исходов;
  clob.polymarket.com/prices-history      — история цены токена YES (= вероятность),
      fidelity=60 — часовая; interval=max отдаёт только месяц часовых точек,
      поэтому история забирается кусками startTs/endTs по 14 дней.

Отбор и классификация — по формулировке вопроса, котировки рынка РФ не
используются. Список запросов и правила заморожены здесь до анализа.

Запуск (на сервере): python -m research.polymarket.fetch
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "polymarket")
log = logging.getLogger("research.polymarket.fetch")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
QUERIES = ("russia ukraine ceasefire", "russia ukraine peace deal", "ukraine peace agreement",
           "putin zelensky meet", "trump putin meet", "trump putin zelensky", "nato russia",
           "russia invade", "russian strike nato", "ukraine nato", "russia sanctions",
           "russia military action", "donbas", "crimea", "zaporizhzhia")
RELEVANT = re.compile(r"russia|ukrain|putin|zelensk|nato|kremlin|kyiv|kiev|crimea|donbas|donetsk|zaporizh|kursk|moscow", re.I)
PEACE = re.compile(r"ceasefire|peace|agreement|deal|meet|talks|summit|shake hands|seen together|"
                   r"not to join nato|recogni[sz]e|sanctions? (lifted|relief|eased)|end of (the )?war|negotiat", re.I)
WAR = re.compile(r"clash|invade|invasion|strike|attack|military action|article 5|troops|nuclear|drone|"
                 r"missile|mobiliz|capture|offensive|declare war|martial law", re.I)
MIN_VOLUME = 100_000.0
CHUNK_DAYS = 14


def get(url: str, retries: int = 4) -> dict | list:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "etl-tcs-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                    # noqa: BLE001
            if i == retries - 1:
                raise
            log.warning("повтор %s: %s", url[:90], e)
            time.sleep(2 * (i + 1))
    return {}


def classify(question: str) -> str | None:
    p, w = bool(PEACE.search(question)), bool(WAR.search(question))
    if p and not w:
        return "PEACE"
    if w and not p:
        return "WAR"
    return None                                    # смешанные и прочие — не берём


def discover() -> list[dict]:
    slugs = {}
    for q in QUERIES:
        for page in range(1, 6):
            url = (f"{GAMMA}/public-search?q={urllib.parse.quote(q)}&events_status=all"
                   f"&limit_per_type=50&page={page}")
            d = get(url)
            ev = d.get("events") or []
            for e in ev:
                slugs[e["slug"]] = e.get("title", "")
            if not (d.get("pagination") or {}).get("hasMore"):
                break
            time.sleep(0.2)
    log.info("найдено событий по запросам: %d", len(slugs))
    markets = []
    for slug in sorted(slugs):
        ev = get(f"{GAMMA}/events?slug={urllib.parse.quote(slug)}")
        time.sleep(0.15)
        for e in ev:
            for m in e.get("markets") or []:
                q = m.get("question") or ""
                if not RELEVANT.search(q + " " + (e.get("title") or "")):
                    continue
                cls = classify(q)
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
                if cls is None or vol < MIN_VOLUME:
                    continue
                try:
                    yes = json.loads(m.get("clobTokenIds") or "[]")[0]
                    outcomes = json.loads(m.get("outcomes") or "[]")
                except (ValueError, IndexError):
                    continue
                if outcomes[:1] != ["Yes"]:
                    continue
                markets.append({"event": slug, "market_id": m.get("id"), "question": q, "cls": cls,
                                "volume": vol, "yes_token": yes,
                                "start": m.get("startDate") or e.get("startDate"),
                                "end": m.get("endDate") or e.get("endDate"),
                                "closed": bool(m.get("closed")),
                                "closed_time": m.get("closedTime")})
    uniq = {m["yes_token"]: m for m in markets}
    return sorted(uniq.values(), key=lambda m: -m["volume"])


def _ts(s: str | None) -> int | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace(" ", "T")
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def history(m: dict) -> list[dict]:
    start = _ts(m["start"]) or int(time.time()) - 400 * 86400
    end = min(int(time.time()), _ts(m.get("closed_time")) or _ts(m["end"]) or int(time.time()))
    pts, t = {}, start
    while t < end:
        t2 = min(end, t + CHUNK_DAYS * 86400)
        d = get(f"{CLOB}/prices-history?market={m['yes_token']}&startTs={t}&endTs={t2}&fidelity=60")
        for p in (d.get("history") or []):
            pts[int(p["t"])] = float(p["p"])
        t = t2
        time.sleep(0.12)
    return [{"t": k, "p": v} for k, v in sorted(pts.items())]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    markets = discover()
    log.info("рынков после отбора: %d (мир %d, война %d)", len(markets),
             sum(m["cls"] == "PEACE" for m in markets), sum(m["cls"] == "WAR" for m in markets))
    rows = []
    for i, m in enumerate(markets, 1):
        h = history(m)
        m["points"] = len(h)
        rows += [{"token": m["yes_token"], "t": x["t"], "p": x["p"]} for x in h]
        log.info("[%d/%d] %s | %s | $%.0f | точек %d", i, len(markets), m["cls"], m["question"][:70],
                 m["volume"], len(h))
    with open(os.path.join(OUT_DIR, "markets.json"), "w", encoding="utf-8") as f:
        json.dump(markets, f, ensure_ascii=False, indent=1)
    import csv
    with open(os.path.join(OUT_DIR, "history_hourly.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["token", "t", "p"])
        w.writeheader()
        w.writerows(rows)
    log.info("готово: рынков %d, часовых точек %d", len(markets), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
