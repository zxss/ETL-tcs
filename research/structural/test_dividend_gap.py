"""
Модуль 3: систематическое закрытие дивидендного гэпа (Michaely & Vila, 1995;
Elton & Gruber, 1970). Правила — rules.json, modules.dividend_gap.
"""
from __future__ import annotations

import csv
import datetime as dt

import pandas as pd

from research import event_study_news as es
from research.structural import common as cmn

MODULE = "dividend_gap"


def gap_trade(ohlc: pd.DataFrame, days: list[dt.date], ex_day: dt.date, tp_share: float = 0.8,
              max_sessions: int = 20) -> dict | None:
    """ohlc — дневки бумаги по датам. Возвращает вход/выход или причину пропуска."""
    if ex_day not in days:
        return None
    i = days.index(ex_day)
    if i < 1 or i + 1 + max_sessions >= len(days):
        return None
    need = [days[i - 1], days[i], days[i + 1]]
    if any(d not in ohlc.index for d in need):
        return None
    prev_close, ex_open = float(ohlc.at[days[i - 1], "close"]), float(ohlc.at[days[i], "open"])
    gap = prev_close - ex_open
    if gap <= 0:
        return {"skip": "нет гэпа"}
    tp = ex_open + tp_share * gap
    entry_day, entry = days[i + 1], float(ohlc.at[days[i + 1], "close"])
    if entry >= tp:
        return {"skip": "гэп закрылся до входа"}
    for k in range(1, max_sessions + 1):
        d = days[i + 1 + k]
        if d not in ohlc.index:
            continue
        o, h = float(ohlc.at[d, "open"]), float(ohlc.at[d, "high"])
        if o >= tp:
            return {"entry_day": entry_day, "entry": entry, "exit_day": d, "exit": o, "reason": "тейк на открытии",
                    "gap_pct": gap / prev_close * 100.0}
        if h >= tp:
            return {"entry_day": entry_day, "entry": entry, "exit_day": d, "exit": tp, "reason": "тейк",
                    "gap_pct": gap / prev_close * 100.0}
    d = days[i + 1 + max_sessions]
    if d not in ohlc.index:
        return None
    return {"entry_day": entry_day, "entry": entry, "exit_day": d, "exit": float(ohlc.at[d, "close"]),
            "reason": "тайм-стоп", "gap_pct": gap / prev_close * 100.0}


def run(conn, rules: dict, stage: str, ctx: cmn.Context) -> tuple[pd.DataFrame, dict]:
    m = rules["modules"]["dividend_gap"]
    d_from, d_to = cmn.period(rules, stage)
    with open(es.DIV_PATH, encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
    events = []
    for r in reg:
        ex = cmn.next_day(ctx.tdays, dt.date.fromisoformat(r["last_buy_date"]))
        if ex and d_from <= ex <= d_to:
            events.append((r["ticker"], ex, float(r["dividend_net"])))
    tickers = sorted({e[0] for e in events})
    daily = cmn.union_daily(conn, tickers, d_from - dt.timedelta(days=10), d_to + dt.timedelta(days=45))
    by_tk = {tk: g.set_index("date") for tk, g in daily.groupby("ticker")} if len(daily) else {}
    days = ctx.tdays
    rows, skips = [], {}
    for tk, ex, amount in events:
        basket = "A" if tk in m["basket_a"] else "B"
        oh = by_tk.get(tk)
        res = gap_trade(oh, days, ex) if oh is not None else None
        if res is None:
            skips["нет данных"] = skips.get("нет данных", 0) + 1
            continue
        if "skip" in res:
            skips[res["skip"]] = skips.get(res["skip"], 0) + 1
            continue
        n = cmn.lots_notional(m["position_rub"], res["entry"], ctx.lot(tk))
        if not n:
            skips["лот дороже позиции"] = skips.get("лот дороже позиции", 0) + 1
            continue
        g = (res["exit"] / res["entry"] - 1.0) * 100.0
        r = cmn.trade(MODULE, basket, tk, res["entry_day"], res["exit_day"], n, g, ctx.cost_rt(tk),
                      ctx.fund_pct(res["entry_day"], res["exit_day"]),
                      f"{res['reason']}; гэп {res['gap_pct']:.2f}%; дивиденд {amount}")
        r["reason"] = res["reason"]
        rows.append(r)
    tr = pd.DataFrame(rows)
    extra = {"events": len(events), "skipped": skips}
    for b in ("A", "B"):
        x = tr[tr["variant"] == b] if len(tr) else tr
        extra[f"take_rate_{b}"] = float(x["reason"].isin(["тейк", "тейк на открытии"]).mean()) if len(x) else None
    return (tr[cmn.TRADE_FIELDS] if len(tr) else pd.DataFrame(columns=cmn.TRADE_FIELDS)), extra
