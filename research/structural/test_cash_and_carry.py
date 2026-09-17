"""
Модуль 2: синтетический Cash-and-Carry — базисный арбитраж акция против фьючерса
(Fama & French, 1987). Правила — rules.json, modules.cash_and_carry.
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from research import sprint2_leadlag as ll
from research.structural import common as cmn

MODULE = "cash_and_carry"
CONTRACTS_PATH = os.path.join(cmn.ROOT, "audit", "r4_research", "structural", "contracts_stockfut.csv")
T18 = dt.time(17, 55)


def implied_rate_pct(f_share: float, spot: float, divs: float, dte_days: int) -> float:
    """(F − S + дивиденды до экспирации) / S × 365 / DTE, % годовых."""
    return (f_share - spot + divs) / spot * 365.0 / dte_days * 100.0


def read_contracts(path: str = CONTRACTS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{"ticker": r["ticker"], "root": r["root"], "uid": r["uid"],
                 "expiration": dt.date.fromisoformat(r["expiration"])} for r in csv.DictReader(f)]


async def _sizes() -> dict:
    from loaders import moex_loader
    from services import backfill_5m as bf
    async with bf._session() as s:
        r = await moex_loader._api_post(s, "InstrumentsService/Futures", {"instrumentStatus": "INSTRUMENT_STATUS_ALL"})
    out = {}
    for f in r.get("instruments", []):
        size = f.get("basicAssetSize")
        units = size.get("units") if isinstance(size, dict) else size
        try:
            out[f.get("ticker")] = float(units)
        except (TypeError, ValueError):
            continue
    return out


def run(conn, rules: dict, stage: str, ctx: cmn.Context) -> tuple[pd.DataFrame, dict]:
    m = rules["modules"]["cash_and_carry"]
    d_from, d_to = cmn.period(rules, stage)
    pairs = m["pairs"]
    lo_dte, hi_dte = m["dte_range_days"]
    contracts = [c for c in read_contracts() if c["root"] in pairs
                 and d_from <= c["expiration"] <= d_to + dt.timedelta(days=hi_dte + 5)]
    sizes = asyncio.run(_sizes())
    stocks = cmn.union_bars(conn, list(pairs.values()), d_from, d_to + dt.timedelta(days=hi_dte + 5),
                            t0="17:30", t1="18:05")
    futs = cmn.union_bars(conn, [c["ticker"] for c in contracts], d_from, d_to + dt.timedelta(days=hi_dte + 5),
                          t0="17:30", t1="18:05", table_override="research_fut_5m")
    last_day = {}
    for c in contracts:
        b = futs.get(c["ticker"])
        if b is not None and len(b.t):
            ds = sorted({pd.Timestamp(x).date() for x in b.t})
            ds = [d for d in ds if d <= c["expiration"]]
            if ds:
                last_day[c["ticker"]] = ds[-1]
    days_all = [d for d in ctx.tdays if d_from <= d <= d_to + dt.timedelta(days=hi_dte + 5)]
    rows, signal_days, open_pos = [], set(), {}
    fut_cost = rules["costs"]["futures_rt_pct"]
    for d in days_all:
        rate = ctx.fund_rate_annual(d)
        for root, stock in pairs.items():
            sb = stocks.get(stock)
            s = ll.close_on_day(sb, d, T18) if sb is not None else None
            pos = open_pos.get(stock)
            if pos:
                c = pos["contract"]
                fb = futs.get(c["ticker"])
                fp = ll.close_on_day(fb, d, T18) if fb is not None else None
                if s is None or fp is None:
                    continue
                size = pos["size"]
                dte = (c["expiration"] - d).days
                mtm = pos["shares"] * (s - pos["s0"]) - pos["n"] * (fp - pos["f0"])
                pos["min_mtm"] = min(pos["min_mtm"], mtm)
                implied = implied_rate_pct(fp / size, s, ctx.dividends(stock, d, c["expiration"]), max(dte, 1))
                if d >= last_day.get(c["ticker"], c["expiration"]) or implied <= rate:
                    div = ctx.dividends(stock, pos["d0"], d)
                    gross_rub = pos["shares"] * (s - pos["s0"] + div) - pos["n"] * (fp - pos["f0"])
                    stock_notional = pos["shares"] * pos["s0"]
                    fut_notional = pos["n"] * pos["f0"]
                    cost_pct = ctx.cost_rt(stock) + fut_cost * fut_notional / stock_notional
                    reason = "экспирация" if d >= last_day.get(c["ticker"], c["expiration"]) else "базис схлопнулся"
                    r = cmn.trade(MODULE, "C&C", stock, pos["d0"], d, stock_notional,
                                  gross_rub / stock_notional * 100.0, cost_pct, ctx.fund_pct(pos["d0"], d),
                                  f"{c['ticker']}; {reason}; вход {pos['implied0']:.2f}% при фонде {pos['rate0']:.2f}%; "
                                  f"мин. переоценка {pos['min_mtm']:.0f} ₽")
                    r["min_mtm_rub"] = pos["min_mtm"]
                    rows.append(r)
                    del open_pos[stock]
                continue
            if d > d_to or s is None:
                continue
            cands = [c for c in contracts if c["root"] == root and lo_dte <= (c["expiration"] - d).days <= hi_dte]
            if not cands:
                continue
            c = min(cands, key=lambda x: x["expiration"])
            fb = futs.get(c["ticker"])
            fp = ll.close_on_day(fb, d, T18) if fb is not None else None
            size = sizes.get(c["ticker"])
            if fp is None or not size:
                continue
            f_share = fp / size
            if not (0.8 <= f_share / s <= 1.25):
                continue
            dte = (c["expiration"] - d).days
            implied = implied_rate_pct(f_share, s, ctx.dividends(stock, d, c["expiration"]), dte)
            if implied < rate + m["entry_premium_pct"]:
                continue
            signal_days.add(d)
            n = math.floor(m["position_rub"] / (s * size))
            if n <= 0:
                continue
            open_pos[stock] = {"contract": c, "size": size, "n": n, "shares": n * size, "s0": s, "f0": fp,
                               "d0": d, "implied0": implied, "rate0": rate, "min_mtm": 0.0}
    tr = pd.DataFrame(rows)
    years = ((d_to - d_from).days + 1) / 365.25
    extra = {"signal_days": len(signal_days), "signal_days_per_year": len(signal_days) / years,
             "contracts_used": len(contracts), "sizes_found": sum(1 for c in contracts if sizes.get(c["ticker"])),
             "max_drawdown_rub": float(tr["min_mtm_rub"].min()) if len(tr) else None,
             "open_at_end": len(open_pos)}
    return (tr[cmn.TRADE_FIELDS] if len(tr) else pd.DataFrame(columns=cmn.TRADE_FIELDS)), extra
