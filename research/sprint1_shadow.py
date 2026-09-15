"""
Теневой журнал правила Спринта 1 (решение пользователя 15.09.2026: в прод не
включать, копить сделки на свежих данных с 16.09). Заявок нет — только запись.

Правило — замороженное final_rule.json (коммит 31a6a41), без изменений: пост о
компании (связь «объект») категории FINANCIAL с тональностью < 0, лента v2;
шорт внутри дня, вход — первый бар после публикации в основной сессии (ночью и
до открытия — первый бар основной сессии: с 14.09.2026 это 09:10, а не 10:00 —
сдвинулось расписание Мосбиржи), выход — close бара 18:15; нетто = −ход −
издержки круга («Премиум» + спред, cost_model base). Справочно — вход через
10 минут (период сборщика news_tg).

Наблюдение, не испытание: в реестр trials.jsonl не пишется.
Журнал: audit/r4_research/sprint1/shadow/journal.csv (ключ message_id + ticker +
вариант: повторный прогон ничего не дублирует), сводка — summary.md там же.
Каждый прогон добирает последние 5 календарных дней: если 5-минутки дня
загрузились поздно, сделка попадёт в журнал следующим вечером.
Текстов постов в журнале нет (репозиторий публичный).

Запуск (на сервере): python -m research.sprint1_shadow [--day YYYY-MM-DD]
Крон: scripts/r4_shadow_crontab → /etc/cron.d/etl-r4-shadow.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import news_classify as ncl                # noqa: E402
from research import news_event_study as ns              # noqa: E402
from research import short_rule as sr                    # noqa: E402
from research import sprint1_final as sf                 # noqa: E402

log = logging.getLogger("research.sprint1_shadow")

SHADOW_FROM = dt.date(2026, 9, 16)
LOOKBACK_DAYS = 5
SHADOW_DIR = os.path.join(es.SPRINT_DIR, "shadow")
JOURNAL = os.path.join(SHADOW_DIR, "journal.csv")
SUMMARY = os.path.join(SHADOW_DIR, "summary.md")
VARIANTS = (("main", es.LAG_REACTION), ("lag10", es.LAG_TRADE))
FIELDS = ["date", "variant", "message_id", "ticker", "posted", "t0", "p0", "move", "ar",
          "cost", "net_short", "sentiment", "logged_at"]


def day_trades(ev: pd.DataFrame, bars_of: dict, idx: es.Bars, tdays: list[dt.date],
               day: dt.date, cost) -> list[dict]:
    """Сделки правила с входом в день `day` (оба варианта входа)."""
    rows = []
    for e in sf.rule_events(ev).itertuples(index=False):
        b = bars_of.get(e.ticker)
        if b is None:
            continue
        for variant, lag in VARIANTS:
            o = sf.intraday_outcome(e.posted, lag, b, idx, tdays)
            if o is None or o["t0"].date() != day or not sr.quantum_ok(o["p0"]):
                continue
            c = float(cost(e.ticker))
            rows.append({"date": day.isoformat(), "variant": variant, "message_id": int(e.message_id),
                         "ticker": e.ticker, "posted": e.posted.isoformat(timespec="minutes"),
                         "t0": o["t0"].isoformat(timespec="minutes"), "p0": round(o["p0"], 6),
                         "move": round(o["move"], 4), "ar": round(o["ar"], 4), "cost": round(c, 4),
                         "net_short": round(-o["move"] - c, 4), "sentiment": round(float(e.sentiment), 3)})
    return rows


def _key(r: dict) -> tuple:
    return (str(r["message_id"]), r["ticker"], r["variant"])


def merge_journal(rows: list[dict], path: str = JOURNAL) -> int:
    """Дописывает новые строки; уже записанные (по ключу) пропускает."""
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            seen = {_key(r) for r in csv.DictReader(f)}
    new = [r for r in rows if _key(r) not in seen]
    if not new:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fresh = not os.path.exists(path)
    now = dt.datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for r in new:
            w.writerow({**r, "logged_at": now})
    return len(new)


def summary(path: str = JOURNAL) -> str:
    L = ["# Теневой журнал правила Спринта 1", "",
         "FINANCIAL, тональность < 0 → шорт внутри дня до 18:20 (final_rule.json, без изменений). "
         f"Наблюдение с {SHADOW_FROM:%d.%m.%Y}, заявок нет, в реестр испытаний не пишется. "
         "Итог на отложенной выборке 2022–2024: ход −0,379 % (t −2,8), нетто +0,122 % (t 0,7).", ""]
    if not os.path.exists(path):
        return "\n".join(L + ["Сделок пока нет.", ""])
    df = pd.read_csv(path)
    L += ["| вариант | сделок | дат | аномальный ход, % (t) | нетто шорта, % (t) | доля плюсовых |",
          "|---|---|---|---|---|---|"]
    for variant, _ in VARIANTS:
        x = df[df["variant"] == variant]
        if x.empty:
            continue
        ar, net = es.by_date(x["ar"], x["date"]), es.by_date(x["net_short"], x["date"])
        L.append(f"| {variant} | {len(x)} | {x['date'].nunique()} | {sf._f(ar.get('mean'))} "
                 f"({sf._f(ar.get('t'), 1)}) | {sf._f(net.get('mean'))} ({sf._f(net.get('t'), 1)}) | "
                 f"{sf._f(float((x['net_short'] > 0).mean()), 2)} |")
    return "\n".join(L) + "\n"


def run(day: dt.date) -> int:
    with open(sf.RULE_PATH, encoding="utf-8") as f:
        rule = json.load(f)
    clf = ncl.Classifier()
    if clf.version != rule["dicts"]:
        raise SystemExit(f"словари {clf.version} ≠ замороженным {rule['dicts']}")
    days = [day - dt.timedelta(days=k) for k in range(LOOKBACK_DAYS)]
    days = sorted(d for d in days if d >= SHADOW_FROM and d.weekday() < 5)
    if not days:
        log.info("день %s вне наблюдения", day)
        return 0
    import database
    conn = database.get_connection()
    try:
        posts = ns.load_posts(conn, "markettwits", days[0] - dt.timedelta(days=LOOKBACK_DAYS))
        posts = posts[posts["msk"].dt.date <= day]
        events, _ = es.build_events(posts, clf)
        ev = sf.rule_events(events) if len(events) else events
        tickers = sorted(set(ev["ticker"])) if len(ev) else []
        lo = days[0] - dt.timedelta(days=2 * LOOKBACK_DAYS)
        bars_of = es.load_bars(conn, "market_data_5m", tickers, lo, day) if tickers else {}
        idx = es.load_bars(conn, "market_data_5m", [es.INDEX], lo, day).get(es.INDEX)
    finally:
        conn.close()
    if idx is None:
        log.warning("нет 5-минуток %s — прогон пропущен", es.INDEX)
        return 0
    tdays = sorted(d for d in idx.day_close if d.weekday() < 5)
    spreads = cm.load_spreads()
    rows = []
    for d in days:
        if d not in idx.day_close:
            log.warning("%s: 5-минуток %s нет (не торговый день или данные ещё не загружены)", d, es.INDEX)
            continue
        got = day_trades(ev, bars_of, idx, tdays, d, lambda tk: cm.round_trip(tk, "base", spreads))
        log.info("%s: сделок правила %d", d, sum(r["variant"] == "main" for r in got))
        rows += got
    added = merge_journal(rows)
    os.makedirs(SHADOW_DIR, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary())
    log.info("в журнал добавлено %d (событий правила в окне %d)", added, len(ev))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Теневой журнал правила Спринта 1")
    ap.add_argument("--day", default=None, help="день наблюдения, по умолчанию сегодня")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(dt.date.fromisoformat(a.day) if a.day else dt.date.today())


if __name__ == "__main__":
    raise SystemExit(main())
