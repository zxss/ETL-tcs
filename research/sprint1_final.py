"""
Спринт 1, финальная предрегистрированная гипотеза (решение пользователя 15.09.2026).
ОДНО правило, ОДИН прогон на нетронутой выборке 01.01.2022–20.05.2024
(research_bars_5m). Правило заморожено в audit/r4_research/sprint1/final_rule.json
и этим файлом (коммит до прогона); испытание — в реестре trials.jsonl.

Правило:
  событие  — пост о компании (связь «объект») категории FINANCIAL с тональностью
             < 0; классификатор и словари v2; лента как в event study v2 (без спама,
             дайджестов, отчётов о цене, повторов категории в течение 2 часов);
  стратегия — intraday_short: вход по open первого бара после публикации в
             основной сессии; ночью, в премаркет и выходные — первая сделка 10:00
             следующего торгового дня; вход позже 18:10 (до выхода не остаётся бара)
             — тоже 10:00 следующего торгового дня;
  выход    — close бара 18:15 того же дня (фаза 18:20);
  издержки — круг: комиссия «Премиум» 0,08 % + спред бумаги (cost_model base);
             перенос 0 (позиция закрыта внутри дня).
Критерий подтверждения: t по датам аномального хода бумаги (против IMOEX)
≤ −2,0 И среднее нетто шорта > 0.
Справочно (не тест): вход через 10 минут после публикации — период сборщика news_tg.

Запуск (на сервере): python -m research.sprint1_final
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import news_classify as ncl                # noqa: E402
from research import news_event_study as ns              # noqa: E402
from research import short_rule as sr                    # noqa: E402

log = logging.getLogger("research.sprint1_final")

RULE_PATH = os.path.join(ROOT, "audit", "r4_research", "sprint1", "final_rule.json")
PERIOD = (dt.date(2022, 1, 1), dt.date(2024, 5, 20), "research_bars_5m")
EXIT_BAR = dt.time(18, 15)
LAST_ENTRY = dt.time(18, 10)
T_MAX, NET_MIN = -2.0, 0.0


def rule_events(ev: pd.DataFrame) -> pd.DataFrame:
    return ev[(ev["category"] == "FINANCIAL") & (ev["sentiment"] < 0)]


def intraday_outcome(posted: dt.datetime, lag: dt.timedelta, bars: es.Bars, idx: es.Bars,
                     tdays: list[dt.date]) -> dict | None:
    """Шорт внутри дня: вход по правилу, выход close бара 18:15 того же дня."""
    tset = set(tdays)
    m = es.entry_moment(posted + lag, tdays, tset)
    for _ in range(2):
        i0 = bars.entry(m) if m else None
        if i0 is None:
            return None
        t0 = pd.Timestamp(bars.t[i0]).to_pydatetime()
        if t0.date() in tset and t0.time() <= LAST_ENTRY:
            break
        m = es.entry_moment(dt.datetime.combine(t0.date(), dt.time(23, 59)), tdays, tset)
    else:
        return None
    exit_t = dt.datetime.combine(t0.date(), EXIT_BAR)
    p0, p1 = float(bars.o[i0]), bars.close_at(exit_t)
    x0, x1 = idx.price_at(t0), idx.close_at(exit_t)
    if not (p0 > 0 and p1 == p1 and x0 and x0 == x0 and x1 == x1):
        return None
    move = (p1 / p0 - 1.0) * 100.0
    return {"t0": t0, "p0": p0, "move": move, "ar": move - (x1 / x0 - 1.0) * 100.0}


def evaluate(rows: pd.DataFrame, days: list[dt.date], n_trials: int) -> dict:
    """rows: date, ar, net_short. Критерий — из докстринга."""
    ar = es.by_date(rows["ar"], rows["date"])
    net = es.by_date(rows["net_short"], rows["date"])
    daily = rows.groupby("date")["net_short"].mean().reindex(days).fillna(0.0)
    ok = (ar.get("t") is not None and ar["t"] <= T_MAX and (net.get("mean") or -1) > NET_MIN)
    return {"ar": ar, "net": net, "hit": float((rows["net_short"] > 0).mean()) if len(rows) else None,
            "ir": es.information_ratio(daily), "dsr": es.deflated_sharpe(daily, max(2, n_trials)),
            "verdict": "подтверждено" if ok else "не подтверждено"}


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}".replace(".", ",")


def report(res: dict, meta: dict) -> str:
    L = ["# Спринт 1 — финальная гипотеза: шорт внутри дня после негативной отчётности", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правило `final_rule.json`, словари "
         f"`{meta['dicts']}`. Период {meta['from']} … {meta['to']} ({meta['table']}), единственный прогон.", "",
         f"Событий FINANCIAL с тональностью < 0: {meta['events']}; с ценами (основной вариант) {meta['ok']}. "
         f"Испытаний в реестре: {meta['trials_total']}.", "",
         "Аномальный ход — ход бумаги от входа до 18:20 минус ход IMOEX; нетто шорта — −ход − издержки "
         "(«Премиум» + спред), перенос 0. t — по датам.", "",
         "| вариант | сделок | дат | аномальный ход, % (t) | нетто шорта, % (t) | доля плюсовых | IR | DSR | вердикт |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {r['ar'].get('n', 0)} | {r['ar'].get('dates', 0)} | "
                 f"{_f(r['ar'].get('mean'))} ({_f(r['ar'].get('t'), 1)}) | {_f(r['net'].get('mean'))} "
                 f"({_f(r['net'].get('t'), 1)}) | {_f(r['hit'], 2)} | {_f(r['ir'], 2)} | {_f(r['dsr'], 2)} | "
                 f"{r['verdict'] if name.startswith('основной') else 'справочно'} |")
    L += ["", f"Критерий: t аномального хода ≤ {T_MAX:g} и среднее нетто шорта > 0 — только основной вариант.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 1: финальная гипотеза на отложенной выборке")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(RULE_PATH, encoding="utf-8") as f:
        rule = json.load(f)
    import database
    d_from, d_to, table = PERIOD
    clf = ncl.Classifier()
    if clf.version != rule["dicts"]:
        raise SystemExit(f"словари {clf.version} ≠ замороженным {rule['dicts']}")
    conn = database.get_connection()
    try:
        posts = ns.load_posts(conn, "markettwits", d_from)
        posts = posts[posts["msk"].dt.date <= d_to]
        events, _ = es.build_events(posts, clf)
        ev = rule_events(events)
        bars_of = es.load_bars(conn, table, sorted(set(ev["ticker"])), d_from, d_to)
        idx = es.load_bars(conn, table, [es.INDEX], d_from, d_to).get(es.INDEX)
    finally:
        conn.close()
    tdays = sorted(d for d in idx.day_close if d_from <= d <= d_to + dt.timedelta(days=6) and d.weekday() < 5)
    days = [d for d in tdays if d <= d_to]
    spreads = cm.load_spreads()
    rev = es._revision()
    total = es.register_trials("sprint1-final", 1, rev)
    res, ok_main = {}, 0
    for name, lag in (("основной: первый бар после новости", es.LAG_REACTION),
                      ("справочно: вход через 10 минут (сборщик)", es.LAG_TRADE)):
        rows = []
        for e in ev.itertuples(index=False):
            b = bars_of.get(e.ticker)
            o = intraday_outcome(e.posted, lag, b, idx, tdays) if b else None
            if o is None or not sr.quantum_ok(o["p0"]):
                continue
            rows.append({"date": o["t0"].date(), "ticker": e.ticker, "ar": o["ar"],
                         "net_short": -o["move"] - cm.round_trip(e.ticker, "base", spreads)})
        df = pd.DataFrame(rows)
        if name.startswith("основной"):
            ok_main = len(df)
        res[name] = evaluate(df, days, total) if len(df) else {"ar": {}, "net": {}, "hit": None,
                                                              "ir": None, "dsr": None, "verdict": "нет данных"}
    now = dt.datetime.now()
    meta = {"created": now.strftime("%d.%m.%Y %H:%M"), "revision": rev, "dicts": clf.version,
            "from": str(d_from), "to": str(d_to), "table": table, "events": int(len(ev)),
            "ok": ok_main, "trials_total": total}
    text = report(res, meta)
    out = a.out or os.path.join(ROOT, "audit", "r4_research", f"sprint1-final-{now:%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rule": rule, "results": res}, f, ensure_ascii=False, indent=1, default=str)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
