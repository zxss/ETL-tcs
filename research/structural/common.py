"""
Общие части ретро-теста структурных моделей (ТЗ пользователя 17.09.2026).
Правила — research/structural/rules.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import sprint4_gatekeeper as sg            # noqa: E402

RULES_PATH = os.path.join(ROOT, "research", "structural", "rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "structural")
REPORT_PATH = os.path.join(ROOT, "RESEARCH-MACRO-STRUCTURAL-REPORT.md")
INDEX = "IMOEX"
MSK = dt.timezone(dt.timedelta(hours=3))
SPLIT_DATE = dt.date(2024, 5, 21)               # граница research_bars_5m / market_data_5m
TABLES = (("research_bars_5m", dt.date(2021, 12, 1), dt.date(2024, 5, 20)),
          ("market_data_5m", SPLIT_DATE, dt.date(2026, 12, 31)))
TRADE_FIELDS = ["module", "variant", "ticker", "entry_day", "exit_day", "hold_days", "notional",
                "gross_pct", "cost_pct", "fund_pct", "net_excess_pct", "pnl_excess_rub", "note"]

BAR_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, low, close
FROM {table}
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN %(t0)s AND %(t1)s
ORDER BY ticker, ts
"""


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def period(rules: dict, stage: str) -> tuple[dt.date, dt.date]:
    s = rules["samples"][stage]
    return dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])


# ── Цены ─────────────────────────────────────────────────────────────────────

def union_daily(conn, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    """Дневки из 5-минуток обеих таблиц (все сессии), сплиты пересчитаны."""
    parts = []
    for table, lo, hi in TABLES:
        a, b = max(lo, d_from), min(hi, d_to)
        if a <= b:
            parts.append(sg.daily_from_5m(conn, table, list(tickers), a, b))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return sg.adjust_daily_splits(df) if len(df) else df


def union_bars(conn, tickers: list[str], d_from: dt.date, d_to: dt.date,
               t0: str = "00:00", t1: str = "23:59", table_override: str | None = None) -> dict:
    """ticker → es.Bars по 5-минуткам обеих таблиц в окне времени (сырые цены)."""
    frames = []
    tables = [(table_override, d_from, d_to)] if table_override else TABLES
    for table, lo, hi in tables:
        a, b = max(lo, d_from), min(hi, d_to)
        if a > b:
            continue
        frames.append(pd.read_sql(BAR_SQL.format(table=table), conn, params={
            "tk": list(tickers), "f": dt.datetime.combine(a, dt.time(), MSK),
            "t": dt.datetime.combine(b + dt.timedelta(days=1), dt.time(), MSK), "t0": t0, "t1": t1}))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for tk, g in df.groupby("ticker"):
        out[tk] = es.Bars(g["tm"], g["open"].astype(float), g["close"].astype(float))
    return out


def trading_days(daily: pd.DataFrame) -> list[dt.date]:
    ix = daily[daily["ticker"] == INDEX]
    return sorted(d for d in ix["date"] if d.weekday() < 5)


def next_day(days: list[dt.date], d: dt.date) -> dt.date | None:
    return es.next_day(days, d)


def prev_day(days: list[dt.date], d: dt.date) -> dt.date | None:
    import bisect
    i = bisect.bisect_left(days, d)
    return days[i - 1] if i > 0 else None


# ── Фонд и издержки ──────────────────────────────────────────────────────────

class Context:
    """Всё, что общее для модулей: фонд, издержки, лоты, дивиденды."""

    def __init__(self, tdays: list[dt.date]):
        from research import short_rule as sr
        self.hurdle = es.Hurdle(es.HURDLE_PATH)
        self.spreads = cm.load_spreads()
        self.lots, _ = sr.load_lots_and_blocked()
        self.tdays = tdays
        self.divs = es.load_dividends(es.DIV_PATH, tdays)

    def fund_pct(self, d0: dt.date, d1: dt.date) -> float:
        return float(self.hurdle.growth(d0, d1))

    def fund_rate_annual(self, d: dt.date, window: int = 30) -> float:
        return self.fund_pct(d - dt.timedelta(days=window), d) * 365.0 / window

    def cost_rt(self, ticker: str) -> float:
        return float(cm.round_trip(ticker, "base", self.spreads))

    def lot(self, ticker: str) -> int:
        return int(self.lots.get(ticker, 1) or 1)

    def dividends(self, ticker: str, d0: dt.date, d1: dt.date) -> float:
        return float(es.dividends_between(self.divs.get(ticker, []), d0, d1))


def lots_notional(position_rub: float, price: float, lot: int) -> float:
    n = math.floor(position_rub / (price * lot)) if price > 0 else 0
    return n * price * lot if n > 0 else 0.0


def trade(module: str, variant: str, ticker: str, entry_day: dt.date, exit_day: dt.date, notional: float,
          gross_pct: float, cost_pct: float, fund_pct: float, note: str = "") -> dict:
    net = gross_pct - cost_pct - fund_pct
    return {"module": module, "variant": variant, "ticker": ticker, "entry_day": entry_day, "exit_day": exit_day,
            "hold_days": max(1, (exit_day - entry_day).days), "notional": notional, "gross_pct": gross_pct,
            "cost_pct": cost_pct, "fund_pct": fund_pct, "net_excess_pct": net,
            "pnl_excess_rub": notional * net / 100.0, "note": note}


# ── Сводка ───────────────────────────────────────────────────────────────────

def summarize(tr: pd.DataFrame, d_from: dt.date, d_to: dt.date, capital: float, crit: dict) -> dict:
    years = ((d_to - d_from).days + 1) / 365.25
    if tr is None or tr.empty:
        return {"trades": 0, "passed": False}
    bd = es.by_date(tr["net_excess_pct"], tr["entry_day"])
    deployed = float((tr["notional"] * tr["hold_days"] / 365.0).sum())          # рубле-годы
    excess_rub = float(tr["pnl_excess_rub"].sum())
    employed = excess_rub / deployed * 100.0 if deployed > 0 else float("nan")
    days = [d.date() for d in pd.bdate_range(d_from, d_to)]
    daily = (tr.groupby("exit_day")["pnl_excess_rub"].sum() / capital).reindex(days).fillna(0.0)
    t = bd.get("t")
    passed = bool(employed == employed and employed >= crit["excess_annual_min_pct"]
                  and t is not None and t >= crit["t_min"])
    return {"trades": int(len(tr)), "trades_per_year": len(tr) / years,
            "avg_hold_days": float(tr["hold_days"].mean()),
            "net_excess_pct_trade": float(tr["net_excess_pct"].mean()),
            "median_net_excess_pct": float(tr["net_excess_pct"].median()),
            "win_rate": float((tr["net_excess_pct"] > 0).mean()),
            "t": t, "dates": bd.get("dates"),
            "excess_employed_annual_pct": employed,
            "contribution_pct_capital_year": excess_rub / capital / years * 100.0,
            "pnl_excess_rub": excess_rub, "ir": es.information_ratio(daily), "passed": passed}


def welch_t(a: list[float], b: list[float]) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None
