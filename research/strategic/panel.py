"""
Панель данных стратегического исследования: дневки, обороты, факторы.

Дневки собираются из 5-минуток ОДИНАКОВО для обеих выборок (research_bars_5m до
20.05.2024, market_data_5m после) — иначе разница между dev и holdout была бы
разницей источников, а не рынка. Сплиты пересчитываются.

Вселенная фиксируется пересечением бумаг, у которых есть 5-минутки в обеих
выборках: состав не должен меняться между периодами.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import short_rule as sr                      # noqa: E402
from research import sprint4_gatekeeper as sg              # noqa: E402
from research.strategic import costs as sc                 # noqa: E402

INDEX = "IMOEX"
MSK = dt.timezone(dt.timedelta(hours=3))
TABLES = (("research_bars_5m", dt.date(2021, 12, 1), dt.date(2024, 5, 20)),
          ("market_data_5m", dt.date(2024, 5, 21), dt.date(2026, 12, 31)))
POSITION_RUB = 100_000.0


def universe(conn) -> list[str]:
    """Бумаги с 5-минутками в обеих таблицах (без индекса)."""
    q = "SELECT DISTINCT ticker FROM {t}"
    sets = []
    for table, _, _ in TABLES:
        sets.append({r[0] for r in pd.read_sql(q.format(t=table), conn).itertuples(index=False)})
    both = sorted((sets[0] & sets[1]) - {INDEX})
    return both


def load_daily(conn, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    """Дневки из 5-минуток обеих таблиц, сплиты пересчитаны."""
    parts = []
    for table, lo, hi in TABLES:
        a, b = max(lo, d_from), min(hi, d_to)
        if a <= b:
            parts.append(sg.daily_from_5m(conn, table, list(tickers) + [INDEX], a, b))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = sg.adjust_daily_splits(df)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def with_turnover(daily: pd.DataFrame, lots: dict) -> pd.DataFrame:
    """Оборот в рублях: объём в лотах × размер лота × close."""
    df = daily.copy()
    df["lot"] = [int(lots.get(t, 1) or 1) for t in df["ticker"]]
    df["turnover_rub"] = df["volume"].astype(float) * df["lot"] * df["close"].astype(float)
    return df


def _rolling_beta(r: pd.Series, m: pd.Series, window: int = 120) -> pd.Series:
    cov = r.rolling(window, min_periods=window // 2).cov(m)
    var = m.rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0.0, np.nan)).clip(-3.0, 3.0)


def features(daily: pd.DataFrame) -> pd.DataFrame:
    """Факторы ТЗ на каждую пару (дата, бумага). Все окна смотрят строго назад."""
    ix = daily[daily["ticker"] == INDEX].set_index("date")["close"].astype(float).sort_index()
    ix_ret = np.log(ix / ix.shift(1))
    rows = []
    for tk, g in daily[daily["ticker"] != INDEX].groupby("ticker", sort=False):
        g = g.sort_values("date").set_index("date")
        c, h, l, o = (g[x].astype(float) for x in ("close", "high", "low", "open"))
        r = np.log(c / c.shift(1))
        m = ix_ret.reindex(c.index)
        beta = _rolling_beta(r, m, 120)
        f = pd.DataFrame(index=c.index)
        f["ticker"] = tk
        f["close"] = c
        f["turnover_rub"] = g["turnover_rub"].astype(float)
        f["adv_rub"] = g["turnover_rub"].astype(float).rolling(20, min_periods=10).mean().shift(1)
        # 1. остаточный моментум, очищенный от беты IMOEX
        for w in (30, 90):
            stock = np.log(c / c.shift(w))
            index = np.log(ix.reindex(c.index) / ix.reindex(c.index).shift(w))
            f[f"resid_mom_{w}"] = (stock - beta * index) * 100.0
        # 2. волатильность Гармана–Класса (дневная, %)
        gk = 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
        f["garman_klass_vol"] = np.sqrt(gk.rolling(20, min_periods=15).mean().clip(lower=0.0)) * 100.0
        # 3. неликвидность Амихуда (|доходность| на миллион рублей оборота)
        illiq = (r.abs() * 100.0) / (g["turnover_rub"].astype(float) / 1e6).replace(0.0, np.nan)
        f["amihud_illiq"] = illiq.rolling(20, min_periods=15).mean()
        # 4. расстояние до скользящих, в единицах волатильности
        sd20 = r.rolling(20, min_periods=15).std()
        for span in (50, 200):
            ema = c.ewm(span=span, adjust=False, ignore_na=True).mean()
            f[f"dist_ema{span}"] = np.log(c / ema) / sd20.replace(0.0, np.nan)
        f["cs_spread_pct"] = sc.corwin_schultz(h, l).rolling(20, min_periods=10).median()
        f["rel_volume"] = g["turnover_rub"].astype(float) / f["adv_rub"]
        f["ret_1"] = r * 100.0
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["imoex_ret_5"] = out["date"].map((ix / ix.shift(5) - 1.0).mul(100.0))
    return out


FACTORS = ("resid_mom_30", "resid_mom_90", "garman_klass_vol", "amihud_illiq",
           "dist_ema50", "dist_ema200")


def cross_section_z(feat: pd.DataFrame, cols=FACTORS) -> pd.DataFrame:
    """Кросс-секционный z-score по каждой дате (winsor ±3σ)."""
    out = feat.copy()
    for c in cols:
        g = out.groupby("date")[c]
        out[c + "_z"] = ((out[c] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)).clip(-3.0, 3.0)
    return out


def forward_return(feat: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Валовая доходность вперёд на horizon торговых дней (close→close), %."""
    out = feat.sort_values(["ticker", "date"]).copy()
    out["fwd_ret"] = out.groupby("ticker")["close"].transform(lambda s: s.shift(-horizon) / s - 1.0) * 100.0
    out["fwd_date"] = out.groupby("ticker")["date"].shift(-horizon)
    g = out.groupby("date")["fwd_ret"]
    out["fwd_ret_rel"] = out["fwd_ret"] - g.transform("mean")           # цель ранжирования
    return out


def load_lots() -> dict:
    lots, _ = sr.load_lots_and_blocked()
    return lots
