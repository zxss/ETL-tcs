"""
Теневой журнал «золотой ночи» — гипотеза 2B_gold Спринта 2 (решение пользователя
15.09.2026: на отложенной выборке не подтверждена — нетто +0,081 %, t 1,4; копить
чистый форвард и использовать как признак в Спринте 3). Заявок нет.

Правило — research/sprint2_rules.json (2B_gold) без изменений: ближний фьючерс на
золото GD (объём предыдущего дня; выбывает за 2 торговых дня до экспирации) вырос с
10:00 до 18:30 → ночной лонг PLZL и SELG от цены 18:30 до первой сделки следующего
торгового дня; нетто = ход с дивидендом − издержки круга − рост пая фонда за ночь.
Справочно — вход по цене 18:35 (фаза OVERNIGHT r3).

Сделка попадает в журнал вечером дня выхода (утренняя цена уже есть); прогон
добирает последние 5 дней. Фьючерсы, IMOEX (календарь торгов) и пай TMON@ —
GetCandles в память; акции — market_data_5m; дивиденды PLZL/SELG — GetDividends
на каждом прогоне. Наблюдение, не испытание: в реестр trials.jsonl не пишется.
Журнал: audit/r4_research/sprint2/shadow/journal.csv (ключ: дата входа + бумага).

Запуск (на сервере): python -m research.sprint2_shadow [--day YYYY-MM-DD]
Крон: scripts/r4_shadow_crontab → /etc/cron.d/etl-r4-shadow.
"""
from __future__ import annotations

import argparse
import asyncio
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
from research import fetch_dividends as fd               # noqa: E402
from research import futures_loader as fl                # noqa: E402
from research import session_timing as st                # noqa: E402
from research import sprint1_shadow as sh                # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402

log = logging.getLogger("research.sprint2_shadow")

HYP_ID = "2B_gold"
SHADOW_FROM = dt.date(2026, 9, 16)
LOOKBACK_DAYS = 5
SHADOW_DIR = os.path.join(ll.OUT_DIR, "shadow")
JOURNAL = os.path.join(SHADOW_DIR, "journal.csv")
SUMMARY = os.path.join(SHADOW_DIR, "summary.md")
FIELDS = ["date", "exit_date", "ticker", "front", "sig", "move", "div", "cost", "hurdle",
          "net_long", "net_long_1835", "logged_at"]
FUND = "TMON@"
MSK = dt.timezone(dt.timedelta(hours=3))


def _key(r: dict) -> tuple:
    return (str(r["date"]), r["ticker"])


def hypothesis(rules: dict) -> dict:
    return next(h for h in rules["hypotheses"] if h["id"] == HYP_ID)


def entry_days(day: dt.date) -> list[dt.date]:
    """Дни входа, чей выход (первая сделка следующего дня) мог наступить к `day`."""
    out = [day - dt.timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)]
    return sorted(d for d in out if d >= SHADOW_FROM and d.weekday() < 5)


def daily_volumes(candles: dict, root: str) -> pd.DataFrame:
    rows = []
    for tk, xs in candles.items():
        for r in xs:
            d = dt.datetime.fromisoformat(r[0].replace("Z", "+00:00")).astimezone(MSK).date()
            rows.append((tk, d, float(r[5])))
    df = pd.DataFrame(rows, columns=["ticker", "d", "vol"])
    df = df.groupby(["ticker", "d"], as_index=False)["vol"].sum()
    return df.assign(root=root)


def dividends_by_exdate(raw: dict, tdays: list[dt.date]) -> dict:
    """GetDividends → {бумага: [(дата отсечки = первый торговый день после последнего дня покупки, сумма)]}."""
    out = {}
    for tk, rows in raw.items():
        for r in rows:
            ex = es.next_day(tdays, dt.date.fromisoformat(r["last_buy_date"]))
            if ex:
                out.setdefault(tk, []).append((ex, float(r["dividend_net"])))
    return {k: sorted(v) for k, v in out.items()}


def extend_fund(hurdle: es.Hurdle, closes: list[tuple]) -> es.Hurdle:
    """Дописывает к ряду TMON@ свежие закрытия (после последней даты файла)."""
    s = dict(hurdle.series.get(FUND, []))
    for d, c in closes:
        d = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)[:10])
        if c and c > 0:
            s[d] = float(c)
    hurdle.series[FUND] = sorted(s.items())
    return hurdle


def trades(h: dict, fb, sbars: dict, days: list[dt.date], tdays: list[dt.date], divs: dict,
           cost, hurdle, front_of) -> list[dict]:
    """Сделки основного правила (сырьё > 0) с выходом внутри известного календаря."""
    rows, _ = ll.rows_2b(h, fb, sbars, days, tdays, divs, cost, hurdle, lambda n: 0.0)
    out = []
    for r in rows:
        if r["sig"] <= 0:
            continue
        out.append({"date": r["date"].isoformat(), "exit_date": es.next_day(tdays, r["date"]).isoformat(),
                    "ticker": r["ticker"], "front": front_of(r["date"]), "sig": round(r["sig"], 4),
                    "move": round(r["move"], 4), "div": r["div"], "cost": round(r["cost"], 4),
                    "hurdle": round(r["hurdle"], 5), "net_long": round(r["net_long"], 4),
                    "net_long_1835": round(r["net_long_1835"], 4)})
    return out


def summary(path: str = JOURNAL) -> str:
    L = ["# Теневой журнал «золотой ночи» (2B_gold, Спринт 2)", "",
         "Рост ближнего фьючерса GD с 10:00 до 18:30 → ночной лонг PLZL и SELG до первой сделки "
         f"следующего дня (sprint2_rules.json, без изменений). Наблюдение с {SHADOW_FROM:%d.%m.%Y}, заявок нет.",
         "Для сравнения: разработка 2025–2026 — нетто +0,208 % (t 2,8); отложенная 2022–2024 — +0,081 % (t 1,4).", ""]
    if not os.path.exists(path):
        return "\n".join(L + ["Сделок пока нет.", ""])
    df = pd.read_csv(path)
    L += ["| вход | сделок | дат | нетто, % (t) | доля плюсовых |", "|---|---|---|---|---|"]
    for name, col in (("18:30", "net_long"), ("18:35", "net_long_1835")):
        x = es.by_date(df[col], df["date"])
        L.append(f"| {name} | {x.get('n', 0)} | {x.get('dates', 0)} | {es._f(x.get('mean'))} "
                 f"({es._f(x.get('t'), 1)}) | {es._f(float((df[col] > 0).mean()), 2)} |")
    return "\n".join(L) + "\n"


async def _load(root: str, stocks: list[str], d_from: dt.date, d_to: dt.date):
    from loaders import moex_loader
    from services import backfill_5m as bf
    start = dt.datetime.combine(d_from, dt.time(), MSK)
    end = min(dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), MSK), dt.datetime.now(MSK))
    async with bf._session() as s:
        fut = await moex_loader._api_post(s, "InstrumentsService/Futures",
                                          {"instrumentStatus": "INSTRUMENT_STATUS_ALL"})
        cons = [c for c in fl.select_contracts(fut.get("instruments", []), d_from, d_to) if c["root"] == root]
        candles = {c["ticker"]: await moex_loader.fetch_5m_candles(s, c["uid"], start, end) for c in cons}
        idx_rows = await moex_loader.fetch_5m_candles(s, await bf.find_uid(s, es.INDEX, index=True), start, end)
        found = await moex_loader._api_post(s, "InstrumentsService/FindInstrument", {"query": FUND})
        fund_uid = next((i["uid"] for i in found.get("instruments", []) if i.get("ticker") == FUND), None)
        fund = await moex_loader.fetch_daily_candles(s, fund_uid, start - dt.timedelta(days=10), end) if fund_uid else []
        divs = {}
        for tk in stocks:
            payload = await moex_loader._api_post(s, "InstrumentsService/GetDividends", {
                "instrumentId": await bf.find_uid(s, tk),
                "from": f"{d_from - dt.timedelta(days=30)}T00:00:00Z", "to": f"{d_to + dt.timedelta(days=60)}T00:00:00Z"})
            divs[tk] = fd.parse_dividends(tk, payload)
    return cons, candles, idx_rows, [(r[0], r[4]) for r in fund], divs


def run(day: dt.date) -> int:
    with open(ll.RULES_PATH, encoding="utf-8") as f:
        rules = json.load(f)
    h = hypothesis(rules)
    days = entry_days(day)
    if not days:
        log.info("день %s: входов в окне наблюдения нет", day)
        return 0
    d_from = days[0] - dt.timedelta(days=10)
    cons, candles, idx_rows, fund, raw_divs = asyncio.run(_load(h["driver"], h["stocks"], d_from, day))
    idx = sh.bars_from_candles(idx_rows)
    if idx is None:
        log.warning("нет 5-минуток %s — прогон пропущен", es.INDEX)
        return 0
    tdays = sorted(d for d in idx.day_close if d.weekday() < 5)
    exp = {c["ticker"]: c["expiration"] for c in cons}
    front = st.front_contracts(daily_volumes(candles, h["driver"]),
                               ll.eligible_until(exp, rules["futures"]["roll_business_days_before_expiration"]))
    fmap = dict(zip(front["d"], front["front"]))
    fbars = {tk: sh.bars_from_candles(xs) for tk, xs in candles.items() if xs}
    import database
    conn = database.get_connection()
    try:
        sbars = es.load_bars(conn, "market_data_5m", h["stocks"], d_from, day)
    finally:
        conn.close()
    spreads = cm.load_spreads()
    hurdle = extend_fund(es.Hurdle(es.HURDLE_PATH), fund)
    ok = [d for d in days if d in tdays and (es.next_day(tdays, d) or dt.date.max) <= day]
    rows = trades(h, lambda d: fbars.get(fmap.get(d)), sbars, ok, tdays,
                  dividends_by_exdate(raw_divs, tdays),
                  lambda tk: cm.round_trip(tk, rules["costs"]["scenario"], spreads), hurdle.growth,
                  lambda d: fmap.get(d, ""))
    added = sh.merge_journal(rows, JOURNAL, FIELDS, _key)
    os.makedirs(SHADOW_DIR, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary())
    log.info("дни входа %s: сделок %d, в журнал добавлено %d", [str(d) for d in ok], len(rows), added)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Теневой журнал «золотой ночи» (2B_gold)")
    ap.add_argument("--day", default=None, help="день прогона, по умолчанию сегодня")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(dt.date.fromisoformat(a.day) if a.day else dt.date.today())


if __name__ == "__main__":
    raise SystemExit(main())
