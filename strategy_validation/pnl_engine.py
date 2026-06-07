#!/usr/bin/env python3
"""
Единый модуль расчёта P&L по дневным стратегиям.
Используется backtest.py, advanced_stats.py и top3_report.py,
чтобы исключить расхождения в учёте издержек (cost charged per trade,
NOT once for the whole holding period).

Каждая функция возвращает (gross_%, returns_series, n_trades) где
n_trades == len(returns_series) — число сделок (round-trip), и
net = gross - cost_rt * n_trades.
"""
from __future__ import annotations
import pandas as pd


def _series_for(d: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy in ("intraday_short",):
        return -d["intraday"].dropna()
    if strategy in ("long_intraday",):
        return d["intraday"].dropna()
    if strategy in ("long_overnight",):
        return d["overnight"].dropna()
    if strategy in ("short_hold",):
        return -d["total"].dropna()
    raise ValueError(strategy)


def pnl(d: pd.DataFrame, strategy: str, cost_rt: float):
    """Возвращает (gross_%, net_%, returns_series, n_trades).

    cost_rt начисляется ОДИН РАЗ ЗА КАЖДУЮ сделку (round-trip),
    т.е. net = gross - cost_rt * n_trades. Для всех стратегий здесь
    одна сделка = один торговый день (одно открытие+закрытие позиции),
    поэтому n_trades == len(returns).
    """
    ret = _series_for(d, strategy)
    n_trades = len(ret)
    gross = ret.sum()
    net = gross - cost_rt * n_trades
    return gross, net, ret, n_trades
