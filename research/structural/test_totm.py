"""
Модуль 1: календарный эффект конца месяца (Turn-of-the-Month; McConnell & Xu, 2008).
Правила — rules.json, modules.totm. Вариант 1А — ночи T−1→T+3, вариант 1Б — удержание
с вечера T−2 до вечера T+3; контроль — 4-сессионные окна вне TOTM.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from research import session_calendar as sc
from research import sprint2_leadlag as ll
from research.structural import common as cmn

MODULE = "totm"


def totm_windows(days: list[dt.date]) -> list[dict]:
    """Окна по границам подряд идущих месяцев: T−2, T−1 (конец месяца), T+1..T+3."""
    by: dict[tuple, list] = {}
    for d in days:
        by.setdefault((d.year, d.month), []).append(d)
    keys = sorted(by)
    out = []
    for cur, nxt in zip(keys, keys[1:]):
        if (nxt[0] * 12 + nxt[1]) - (cur[0] * 12 + cur[1]) != 1:
            continue
        c, n = by[cur], by[nxt]
        if len(c) < 2 or len(n) < 3:
            continue
        out.append({"month": f"{cur[0]}-{cur[1]:02d}", "t_m2": c[-2], "t_m1": c[-1],
                    "t_p1": n[0], "t_p2": n[1], "t_p3": n[2]})
    return out


def control_windows(days: list[dt.date], windows: list[dict], length: int = 4) -> list[tuple]:
    """Неперекрывающиеся окна (вечер дня k → вечер дня k+length) вне дней T−2…T+3."""
    idx = {d: i for i, d in enumerate(days)}
    blocked = set()
    for w in windows:
        if w["t_m2"] in idx and w["t_p3"] in idx:
            blocked.update(range(idx[w["t_m2"]], idx[w["t_p3"]] + 1))
    out, k = [], 0
    while k + length < len(days):
        if any(j in blocked for j in range(k, k + length + 1)):
            k += 1
            continue
        out.append((days[k], days[k + length]))
        k += length
    return out


def _evening(b, d):
    return ll.close_on_day(b, d, sc.EVENING_ENTRY_BAR) if b is not None else None


def _morning(b, d):
    o = ll.open_at(b, d, sc.main_open(d)) if b is not None else None
    return o[1] if o else None


def _exit_evening(b, d):
    return ll.close_on_day(b, d, sc.SHORT_EXIT_BAR) if b is not None else None


def run(conn, rules: dict, stage: str, ctx: cmn.Context) -> tuple[pd.DataFrame, dict]:
    m = rules["modules"]["totm"]
    d_from, d_to = cmn.period(rules, stage)
    tickers = m["tickers"]
    cap = rules["account"]["capital_rub"]
    per_ticker = cap / len(tickers)
    bars = cmn.union_bars(conn, tickers, d_from - dt.timedelta(days=10), d_to + dt.timedelta(days=15),
                          t0="09:55", t1="18:35")
    days = [d for d in ctx.tdays if d_from - dt.timedelta(days=10) <= d <= d_to + dt.timedelta(days=15)]
    windows = [w for w in totm_windows(days) if d_from <= w["t_m2"] and w["t_p3"] <= d_to + dt.timedelta(days=15)
               and w["t_m1"] <= d_to]
    rows, tot_basket, ctl_basket = [], [], []

    def one(variant, tk, d0, d1, p0, p1):
        if not (p0 and p1):
            return None
        div = ctx.dividends(tk, d0, d1)
        g = ((p1 + div) / p0 - 1.0) * 100.0
        if abs(g) > 30:
            return None
        n = cmn.lots_notional(per_ticker, p0, ctx.lot(tk))
        if not n:
            return None
        return cmn.trade(MODULE, variant, tk, d0, d1, n, g, ctx.cost_rt(tk), ctx.fund_pct(d0, d1))

    for w in windows:
        g1b = []
        for tk in tickers:
            b = bars.get(tk)
            for d0, d1 in ((w["t_m1"], w["t_p1"]), (w["t_p1"], w["t_p2"]), (w["t_p2"], w["t_p3"])):
                r = one("1A", tk, d0, d1, _evening(b, d0), _morning(b, d1))
                if r:
                    rows.append(r)
            r = one("1B", tk, w["t_m2"], w["t_p3"], _evening(b, w["t_m2"]), _exit_evening(b, w["t_p3"]))
            if r:
                rows.append(r)
                g1b.append(r["gross_pct"])
        if g1b:
            tot_basket.append(float(np.mean(g1b)))
    for d0, d1 in control_windows(days, totm_windows(days)):
        if not (d_from <= d0 <= d_to):
            continue
        g = []
        for tk in tickers:
            b = bars.get(tk)
            p0, p1 = _evening(b, d0), _exit_evening(b, d1)
            if p0 and p1:
                x = ((p1 + ctx.dividends(tk, d0, d1)) / p0 - 1.0) * 100.0
                if abs(x) <= 30:
                    g.append(x)
        if g:
            ctl_basket.append(float(np.mean(g)))
    extra = {"windows": len(windows), "totm_basket_gross_mean_pct": float(np.mean(tot_basket)) if tot_basket else None,
             "control_basket_gross_mean_pct": float(np.mean(ctl_basket)) if ctl_basket else None,
             "control_windows": len(ctl_basket), "welch_t_totm_vs_control": cmn.welch_t(tot_basket, ctl_basket)}
    return pd.DataFrame(rows, columns=cmn.TRADE_FIELDS), extra
