"""
Таблица пересмотров базы расчёта Индекса МосБиржи (IMOEX) за 2022–2026 для модуля 4
(индексные ребалансировки). Только чтение публичного ISS API Мосбиржи.

Метод — без ручной выборки:
1. Состав IMOEX раз в неделю (ISS statistics/.../analytics/IMOEX?date=…; на выходной
   или праздник — шаг назад до 5 дней).
2. Где состав между соседними снимками изменился — двоичный поиск по дням до первой
   даты нового состава (дата вступления в силу, T_eff).
3. Включения и исключения — разность множеств тикеров.
4. Дата анонса T_ann — САМАЯ РАННЯЯ новость ISS sitenews о пересмотре базы индексов
   акций (is_imoex_review: квартальный релиз «Новые базы расчета индексов Московской
   Биржи», «Об изменении / О внеочередном пересмотре баз расчета индексов акций»,
   точечные «… в Индекс МосБиржи») в окне [T_eff − 45 дн; T_eff − 1 дн];
   ссылка https://www.moex.com/n{id}. Не нашлась — T_ann пустой (в тест не идёт).
   Облигации, ESG, РСПП, IPO, недвижимость, «создания стоимости», голубые фишки,
   фьючерсы — не анонсы IMOEX (первая версия ловила их широким фильтром — исправлено
   до прогона модулей).
5. Смена тикера в одном событии (RENAMES: TCSG→T и т. п.) — не включение: пара
   убирается из включений и исключений, пишется в колонку renames.

Выход: research/structural/data/imoex_changes.csv, imoex_snapshots.json, imoex_index_news.json.
Запуск (на сервере, российский IP): python -m research.structural.fetch_imoex_reviews [--from-cache]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import logging
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

log = logging.getLogger("research.structural.fetch_imoex_reviews")

DATA_DIR = os.path.join(ROOT, "research", "structural", "data")
ISS = "https://iss.moex.com/iss"
FROM, TO = dt.date(2022, 6, 1), dt.date(2026, 9, 11)
PAUSE = 0.25
TITLE_RE = re.compile(r"(баз\w*\s+расч[её]та\s+индекс|Индекс\w*\s+МосБиржи|Индекс\w*\s+ММВБ)", re.I)
ANN_WINDOW_DAYS = 45
REVIEW_RE = [
    re.compile(r"^Новые базы расч[её]та индексов Московской [Бб]иржи$"),
    re.compile(r"^Об изменении баз расч[её]та индексов акций$"),
    re.compile(r"^О внеочередном пересмотре баз расч[её]та индексов акций$"),
    re.compile(r"(войдут|включит|включила|исключит|исключила|покинут).* в Индекс МосБиржи$"),
]
# Смена кода бумаги (редомициляция, переименование): старый → новый.
RENAMES = {"TCSG": "T", "YNDX": "YDEX", "FIVE": "X5", "FIXP": "FIXR", "AGRO": "RAGR"}


def is_imoex_review(title: str) -> bool:
    t = html.unescape(title or "").strip()
    return any(r.search(t) for r in REVIEW_RE)


def split_renames(additions: list[str], deletions: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Пары «старый код исключён, новый включён» в одном событии — не включение."""
    adds, dels, ren = list(additions), list(deletions), []
    for old, new in RENAMES.items():
        if old in dels and new in adds:
            dels.remove(old)
            adds.remove(new)
            ren.append(f"{old}->{new}")
    return adds, dels, ren


def _get(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # noqa: BLE001
            if attempt == 3:
                raise
            log.warning("повтор %s: %s", url, e)
            time.sleep(2 * (attempt + 1))
    return {}


def snapshot_raw(d: dt.date) -> set[str] | None:
    out, start = set(), 0
    while True:
        j = _get(f"{ISS}/statistics/engines/stock/markets/index/analytics/IMOEX.json"
                 f"?date={d}&limit=100&start={start}&iss.meta=off")
        a = j.get("analytics", {})
        cols, rows = a.get("columns", []), a.get("data", [])
        if not rows:
            break
        i, di = cols.index("ticker"), cols.index("tradedate")
        if rows[0][di] != d.isoformat():
            return None                            # ISS отдал другой день — этот не торговый
        out |= {r[i] for r in rows}
        if len(rows) < 100:
            break
        start += 100
        time.sleep(PAUSE)
    time.sleep(PAUSE)
    return out or None


def snapshot(d: dt.date, cache: dict) -> tuple[dt.date, set[str]] | None:
    """Состав на d или на ближайший торговый день до него (≤ 5 дней назад)."""
    for k in range(6):
        x = d - dt.timedelta(days=k)
        if x.isoformat() not in cache:
            s = snapshot_raw(x)
            cache[x.isoformat()] = sorted(s) if s else None
        if cache[x.isoformat()]:
            return x, set(cache[x.isoformat()])
    return None


def change_points(cache: dict) -> list[dict]:
    weeks, d = [], FROM
    while d <= TO:
        s = snapshot(d, cache)
        if s:
            weeks.append(s)
        d += dt.timedelta(days=7)
    events = []
    for (d0, s0), (d1, s1) in zip(weeks, weeks[1:]):
        if s0 == s1:
            continue
        lo, hi = d0, d1                            # lo — старый состав, hi — новый
        while (hi - lo).days > 1:
            mid = lo + dt.timedelta(days=(hi - lo).days // 2)
            s = snapshot(mid, cache)
            if s is None or s[0] <= lo:
                lo = mid
                continue
            if s[1] == s0:
                lo = s[0]
            else:
                hi = s[0]
        eff = snapshot(hi, cache)
        new = eff[1] if eff else s1
        events.append({"eff_date": hi.isoformat(), "before_date": lo.isoformat(),
                       "additions": sorted(new - s0), "deletions": sorted(s0 - new)})
        log.info("изменение %s: +%s −%s", hi, sorted(new - s0), sorted(s0 - new))
    return events


def index_news() -> list[dict]:
    out, start = [], 0
    while True:
        j = _get(f"{ISS}/sitenews.json?iss.meta=off&start={start}")
        n = j.get("sitenews", {})
        cols, rows = n.get("columns", []), n.get("data", [])
        if not rows:
            break
        ii, ti, pi = cols.index("id"), cols.index("title"), cols.index("published_at")
        stop = False
        for r in rows:
            pub = r[pi][:10]
            if pub < FROM.isoformat():
                stop = True
                break
            if TITLE_RE.search(r[ti] or ""):
                out.append({"id": r[ii], "title": r[ti], "published": r[pi]})
        if stop:
            break
        start += len(rows)
        time.sleep(PAUSE)
    return out


def attach_announcements(events: list[dict], news: list[dict]) -> list[dict]:
    relevant = [x for x in news if is_imoex_review(x["title"])]
    for e in events:
        eff = dt.date.fromisoformat(e["eff_date"])
        cands = [x for x in relevant
                 if 1 <= (eff - dt.date.fromisoformat(x["published"][:10])).days <= ANN_WINDOW_DAYS]
        best = min(cands, key=lambda x: x["published"]) if cands else None
        e.update(ann_date=best["published"][:10] if best else "", ann_id=best["id"] if best else "",
                 ann_title=html.unescape(best["title"]) if best else "",
                 ann_url=f"https://www.moex.com/n{best['id']}" if best else "")
    return events


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Пересмотры базы IMOEX из ISS")
    ap.add_argument("--from-cache", action="store_true",
                    help="снимки и новости из сохранённых JSON (без прокрутки ленты)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(DATA_DIR, exist_ok=True)
    snap_path = os.path.join(DATA_DIR, "imoex_snapshots.json")
    news_path = os.path.join(DATA_DIR, "imoex_index_news.json")
    cache: dict = {}
    if a.from_cache:
        with open(snap_path, encoding="utf-8") as f:
            cache = json.load(f)
    events = change_points(cache)
    if a.from_cache:
        with open(news_path, encoding="utf-8") as f:
            news = json.load(f)
    else:
        news = index_news()
    log.info("новостей по широкому фильтру: %d, из них анонсов пересмотра IMOEX: %d",
             len(news), sum(is_imoex_review(x["title"]) for x in news))
    for e in events:
        e["additions"], e["deletions"], e["renames"] = split_renames(e["additions"], e["deletions"])
    events = attach_announcements(events, news)
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(cache.items())), f, ensure_ascii=False, indent=0)
    with open(news_path, "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA_DIR, "imoex_changes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eff_date", "before_date", "ann_date", "ann_url", "ann_title",
                                          "additions", "deletions", "renames"])
        w.writeheader()
        for e in events:
            w.writerow({k: (" ".join(v) if isinstance(v, list) else v) for k, v in e.items() if k != "ann_id"})
    log.info("событий изменения состава: %d", len(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
