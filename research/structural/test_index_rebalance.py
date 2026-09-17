"""
Модуль 4: индексные ребалансировки IMOEX (Petajisto, 2011; Chen, Noronha, Singal, 2004).
Правила — rules.json, modules.index_rebalance; события —
research/structural/data/imoex_changes.csv (fetch_imoex_reviews.py).
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import os

import pandas as pd

from research import session_calendar as sc
from research import sprint1_shadow as sh
from research import sprint2_leadlag as ll
from research.structural import common as cmn

MODULE = "index_rebalance"
EVENTS_PATH = os.path.join(cmn.ROOT, "research", "structural", "data", "imoex_changes.csv")


def event_days(ann: dt.date, eff: dt.date, days: list[dt.date]) -> tuple[dt.date, dt.date] | None:
    """Вход — торговый день после анонса; выход — торговый день перед вступлением в силу."""
    entry, exit_ = cmn.next_day(days, ann), cmn.prev_day(days, eff)
    if entry is None or exit_ is None or entry > exit_:
        return None
    return entry, exit_


def read_events(path: str = EVENTS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def _api_bars(ticker: str, days: list[dt.date]):
    from loaders import moex_loader
    from services import backfill_5m as bf
    rows = []
    async with bf._session() as s:
        try:
            uid = await bf.find_uid(s, ticker)
        except Exception:                       # noqa: BLE001 — бумага не находится (старый код)
            return None
        for d in days:
            rows += await moex_loader.fetch_5m_candles(s, uid, dt.datetime.combine(d, dt.time(), cmn.MSK),
                                                       dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), cmn.MSK))
    return sh.bars_from_candles(rows)


def run(conn, rules: dict, stage: str, ctx: cmn.Context) -> tuple[pd.DataFrame, dict]:
    m = rules["modules"]["index_rebalance"]
    d_from, d_to = cmn.period(rules, stage)
    days = ctx.tdays
    events = [e for e in read_events() if d_from <= dt.date.fromisoformat(e["eff_date"]) <= d_to]
    idx = cmn.union_bars(conn, [cmn.INDEX], d_from - dt.timedelta(days=60), d_to).get(cmn.INDEX)
    rows, skips, api_used = [], {}, []
    for e in events:
        adds = [x for x in (e["additions"] or "").split() if x]
        if not adds:
            continue
        if not e["ann_date"]:
            skips["нет даты анонса"] = skips.get("нет даты анонса", 0) + len(adds)
            continue
        dd = event_days(dt.date.fromisoformat(e["ann_date"]), dt.date.fromisoformat(e["eff_date"]), days)
        if dd is None:
            skips["вход позже выхода"] = skips.get("вход позже выхода", 0) + len(adds)
            continue
        entry_day, exit_day = dd
        for tk in adds:
            b = cmn.union_bars(conn, [tk], entry_day, exit_day).get(tk)
            if b is None or entry_day not in b.day_close or exit_day not in b.day_close:
                b = asyncio.run(_api_bars(tk, [entry_day, exit_day]))
                api_used.append(tk)
            o = ll.open_at(b, entry_day, sc.short_entry(entry_day)) if b is not None else None
            p1 = b.day_close.get(exit_day) if b is not None else None
            x0 = idx.price_at(dt.datetime.combine(entry_day, o[0])) if (idx is not None and o) else None
            x1 = idx.day_close.get(exit_day) if idx is not None else None
            if not (o and p1 and x0 and x1):
                skips["нет цен"] = skips.get("нет цен", 0) + 1
                continue
            n = cmn.lots_notional(m["position_rub"], o[1], ctx.lot(tk))
            if not n:
                n = m["position_rub"]                    # лот вне справочника вселенной — номинал позиции
            div = ctx.dividends(tk, entry_day, exit_day)
            g = ((p1 + div) / o[1] - 1.0) * 100.0
            car = g - (x1 / x0 - 1.0) * 100.0
            r = cmn.trade(MODULE, "additions", tk, entry_day, exit_day, n, g, ctx.cost_rt(tk),
                          ctx.fund_pct(entry_day, exit_day),
                          f"анонс {e['ann_date']} ({e['ann_url']}); вступление {e['eff_date']}; CAR {car:.2f}%")
            r["car_pct"] = car
            rows.append(r)
    tr = pd.DataFrame(rows)
    extra = {"events": len(events), "additions": sum(len((e["additions"] or "").split()) for e in events),
             "deletions": sum(len((e["deletions"] or "").split()) for e in events), "skipped": skips,
             "prices_from_api": sorted(set(api_used)),
             "car_mean_pct": float(tr["car_pct"].mean()) if len(tr) else None}
    return (tr[cmn.TRADE_FIELDS] if len(tr) else pd.DataFrame(columns=cmn.TRADE_FIELDS)), extra
