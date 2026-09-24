"""
Панель характеристик для блока C (кросс-секция доходностей), план 04_plan.

Строится из полного среза TQBR (audit/r4_research/longhist), со снятыми с торгов
бумагами и дивидендами — survivorship-free. Признаки зафиксированы ДО прогона,
их ровно 12, перебора признаков нет.

Универс задаётся диапазоном рангов по среднему обороту за 60 дней:
  (1, 50)   — тот же универс, что во всех прошлых тестах программы (S1–S3);
  (51, 150) — свежий срез S4, ни разу не использованный.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "audit", "r4_research", "longhist")
LIQ_WIN, LIQ_MIN = 60, 50
MAX_ABS_R = 0.40
H = 21                                   # горизонт прогноза, торговых дней

FEATURES = ["mom_12_1", "ret_1m", "ret_6m", "atr_pct", "rvol_21", "turnover",
            "amihud", "div_yield", "px_to_high", "hl_range", "zero_days", "beta"]


def load(prefixes=("tqbr", "tqbrh")) -> dict[str, pd.DataFrame]:
    """Широкие таблицы по всей доске за все доступные годы."""
    frames = []
    for pref in prefixes:
        for p in sorted(glob.glob(os.path.join(DATA_DIR, f"{pref}_*.csv.gz"))):
            df = pd.read_csv(p)
            df["__pref"] = pref
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["CLOSE"] > 0) & (df["VALUE"] > 0)].copy()
    df["d"] = pd.to_datetime(df["TRADEDATE"]).dt.date
    df = df.sort_values(["SECID", "d", "__pref"]).drop_duplicates(["SECID", "d"], keep="first")
    wide = {c.lower(): df.pivot(index="d", columns="SECID", values=c).sort_index()
            for c in ("CLOSE", "HIGH", "LOW", "VALUE", "VOLUME")}
    close = wide["close"]

    div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name in ("dividends.csv", "dividends_tqbrh.csv"):
        dp = os.path.join(DATA_DIR, name)
        if not (os.path.exists(dp) and os.path.getsize(dp) > 5):
            continue
        dv = pd.read_csv(dp)
        days = np.array(close.index)
        for _, r in dv.iterrows():
            if r["ticker"] not in div.columns:
                continue
            lb = dt.date.fromisoformat(str(r["last_buy_date"])[:10])
            i = np.searchsorted(days, lb, side="right")
            if i < len(days):
                div.iat[i, div.columns.get_loc(r["ticker"])] += float(r["dividend_net"])

    ret = (close + div) / close.shift(1) - 1.0
    ret = ret.where(ret.abs() <= MAX_ABS_R)
    prev = close.shift(1)
    tr = pd.concat([wide["high"] - wide["low"], (wide["high"] - prev).abs(),
                    (wide["low"] - prev).abs()]).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=12).mean() / close * 100.0
    liq = wide["value"].rolling(LIQ_WIN, min_periods=LIQ_MIN).mean()
    return {"close": close, "ret": ret, "atr": atr, "liq": liq, "div": div,
            "high": wide["high"], "low": wide["low"], "value": wide["value"]}


def universe(liq: pd.DataFrame, d, lo_rank: int, hi_rank: int) -> list[str]:
    row = liq.loc[d].dropna()
    order = row.sort_values(ascending=False).index
    return list(order[lo_rank - 1: hi_rank])


def _cum(ret: pd.DataFrame, i0: int, i1: int) -> pd.Series:
    """Накопленная доходность по строкам [i0, i1) (NaN → 0, но нужно ≥ 60 % дней)."""
    sl = ret.iloc[i0:i1]
    n_ok = sl.notna().sum()
    out = (1.0 + sl.fillna(0.0)).prod() - 1.0
    return out.where(n_ok >= max(5, int(0.6 * len(sl))))


def characteristics(data: dict, d, tickers: list[str]) -> pd.DataFrame:
    """12 признаков на дату d по списку бумаг. Только прошлое."""
    ret, close, atr = data["ret"], data["close"], data["atr"]
    i = ret.index.get_loc(d)
    if i < 260:
        return pd.DataFrame()
    tk = [t for t in tickers if t in ret.columns]
    f = pd.DataFrame(index=tk)

    f["mom_12_1"] = _cum(ret[tk], i - 252, i - 21)
    f["ret_1m"] = _cum(ret[tk], i - 21, i + 1)
    f["ret_6m"] = _cum(ret[tk], i - 126, i + 1)
    f["atr_pct"] = atr[tk].iloc[i]
    f["rvol_21"] = ret[tk].iloc[i - 21:i + 1].std(ddof=1) * 100.0
    val = data["value"][tk]
    f["turnover"] = np.log1p(val.iloc[i - 60:i + 1].mean())
    amih = (ret[tk].abs().iloc[i - 60:i + 1] / val.iloc[i - 60:i + 1].replace(0, np.nan))
    f["amihud"] = np.log1p(amih.mean() * 1e9)
    div_sum = data["div"][tk].iloc[max(0, i - 252):i + 1].sum()
    f["div_yield"] = (div_sum / close[tk].iloc[i]).replace([np.inf, -np.inf], np.nan) * 100.0
    f["px_to_high"] = close[tk].iloc[i] / close[tk].iloc[i - 252:i + 1].max()
    f["hl_range"] = ((data["high"][tk] - data["low"][tk]) / close[tk]).iloc[i - 21:i + 1].mean() * 100.0
    f["zero_days"] = (ret[tk].iloc[i - 60:i + 1].abs() < 1e-9).mean()

    mkt = ret[tk].iloc[i - 252:i + 1].mean(axis=1)
    sub = ret[tk].iloc[i - 252:i + 1]
    mv = mkt.var(ddof=1)
    f["beta"] = (sub.apply(lambda s: s.cov(mkt)) / mv) if mv and mv > 0 else np.nan
    return f


def forward_return(data: dict, d, tickers: list[str], h: int = H) -> pd.Series:
    """Доходность за следующие h торговых дней (с дивидендами)."""
    ret = data["ret"]
    i = ret.index.get_loc(d)
    if i + h >= len(ret):
        return pd.Series(dtype=float)
    tk = [t for t in tickers if t in ret.columns]
    return _cum(ret[tk], i + 1, i + 1 + h)


def rebalance_dates(index, d0: dt.date, d1: dt.date, step: int = H) -> list:
    days = [d for d in index if d0 <= d <= d1]
    return days[::step]


def build(data: dict, d0: dt.date, d1: dt.date, lo_rank: int, hi_rank: int,
          step: int = H) -> pd.DataFrame:
    """Длинная панель: строка = (дата, бумага) с признаками и forward-доходностью."""
    rows = []
    for d in rebalance_dates(data["ret"].index, d0, d1, step):
        tk = universe(data["liq"], d, lo_rank, hi_rank)
        if len(tk) < 10:
            continue
        f = characteristics(data, d, tk)
        if f.empty:
            continue
        y = forward_return(data, d, list(f.index))
        if y.empty:
            continue
        f = f.assign(d=d, ticker=f.index, y=y)
        rows.append(f)
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p = p.dropna(subset=["y"])
    # кросс-секционное демеанирование цели: относительная доходность внутри универса
    p["y_rel"] = p["y"] - p.groupby("d")["y"].transform("mean")
    return p
