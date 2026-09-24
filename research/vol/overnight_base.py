"""
Общая база для H-V2, H-R1, H-U1 — прокси ночной корзины контура r4.

Прокси, а не сам контур: правило BEST_TRADES зависит от прогнозов и новостей,
которых на всей истории нет. Берётся тот же прокси, что в прошлом тесте
программы (research/intraday/overnight_filter.py): топ-K по обороту за 60 дней,
покупка по закрытию основной сессии, продажа по открытию следующего дня.
Это ровно та экспозиция, которой управляет сайзинг и режимный фильтр.

Издержки: круг «Премиум» 0,08 % + спред бумаги (cost_model, сценарий base).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                         # noqa: E402

TOPK = 5
LIQ_WIN = 60
SAMPLES = {
    "dev_2022_2024": ("research_bars_5m", dt.date(2022, 1, 3), dt.date(2024, 5, 20)),
    "holdout_2024_2026": ("market_data_5m", dt.date(2024, 5, 21), dt.date(2026, 9, 24)),
}

SQL = """
WITH b AS (
  SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close::float8 AS c,
         high::float8 AS h, low::float8 AS l, volume::float8 AS v
  FROM {table}
  WHERE close > 0 AND ts >= %(f)s AND ts < %(t)s
    AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN '10:00' AND '18:40'
)
SELECT ticker, tm::date AS d,
       max(h) AS high, min(l) AS low, sum(v * c) AS value,
       (array_agg(c ORDER BY tm))[1] AS open,
       (array_agg(c ORDER BY tm DESC))[1] AS close,
       count(*) AS nbars
FROM b GROUP BY ticker, tm::date ORDER BY ticker, d
"""


def load_daily(conn, table: str, d0: dt.date, d1: dt.date) -> pd.DataFrame:
    df = pd.read_sql(SQL.format(table=table), conn,
                     params={"f": f"{d0} 00:00+03",
                             "t": f"{d1 + dt.timedelta(days=1)} 00:00+03"})
    df = df[df["nbars"] >= 60].copy()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    for c in ("open", "close", "high", "low", "value"):
        df[c] = df[c].astype(float)
    return df.sort_values(["ticker", "d"]).reset_index(drop=True)


def basket(df: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    """Ночные сделки прокси-корзины: close[D] → open[D+1], с издержками."""
    spreads = cm.load_spreads()
    piv_close = df.pivot(index="d", columns="ticker", values="close").sort_index()
    piv_open = df.pivot(index="d", columns="ticker", values="open").sort_index()
    piv_val = df.pivot(index="d", columns="ticker", values="value").sort_index()
    liq = piv_val.rolling(LIQ_WIN, min_periods=40).mean()
    days = list(piv_close.index)
    rows = []
    for i in range(LIQ_WIN, len(days) - 1):
        d, dn = days[i], days[i + 1]
        row = liq.loc[d].dropna()
        if len(row) < topk:
            continue
        picks = list(row.sort_values(ascending=False).index[:topk])
        for tk in picks:
            c0, o1 = piv_close.at[d, tk], piv_open.at[dn, tk]
            if not (np.isfinite(c0) and np.isfinite(o1) and c0 > 0):
                continue
            gross = (o1 / c0 - 1.0) * 100.0
            cost = cm.round_trip(tk, "base", spreads)
            rows.append({"d": d, "d_exit": dn, "ticker": tk,
                         "gross_pct": gross, "cost_pct": cost,
                         "net_pct": gross - cost})
    return pd.DataFrame(rows)


def daily_returns(trades: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Доходность корзины за ночь: равный вес или заданные веса (нормируются)."""
    if trades.empty:
        return pd.Series(dtype=float)
    if weights is None:
        return trades.groupby("d")["net_pct"].mean()
    t = trades.copy()
    t["w"] = weights.reindex(t.index).to_numpy()
    t["w"] = t["w"].fillna(0.0)
    out = {}
    for d, g in t.groupby("d"):
        s = g["w"].sum()
        out[d] = float((g["net_pct"] * g["w"]).sum() / s) if s > 0 else np.nan
    return pd.Series(out).sort_index()
