"""
Аномалия низкой волатильности (единственный воспроизведённый на holdout сигнал
из ТА-батареи) — торгуема ли при НИЗКОЙ оборачиваемости, и как тилт корзины
(ТЗ пользователя 24.09.2026, «комбинируй ТА с текущими моделями»).

Батарея (research/ta/battery) показала: низкий ATR% → выше завтрашняя
относительная доходность, IC≈−0,05, t≈−3,8 на dev И holdout. Но дневной
long-short это не отбивает (издержки). Здесь проверяем то, как low-vol реально
харвестят: редкая перебалансировка (горизонты 1/5/20 дней), лонг-онли нижний
квинтиль ATR против универса, издержки с учётом РЕАЛЬНОГО оборота квинтиля
(меняется только часть состава). Плюс тилт: улучшает ли low-ATR-фильтр ночную
корзину top-K по обороту (прокси BEST_TRADES).

Выборки: dev 2024-05-21…2025-11-30, holdout 2025-12-01…2026-09-22.
Запуск (на сервере): python -m research.ta.lowvol
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                      # noqa: E402
from research import news_event_study as ns               # noqa: E402

log = logging.getLogger("research.ta.lowvol")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "ta")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2025, 11, 30))
HOLDOUT = (dt.date(2025, 12, 1), dt.date(2026, 9, 22))
HORIZONS = [1, 5, 20]
QUINT = 0.2
TOPK = 5


def build(conn) -> pd.DataFrame:
    tickers = sorted(set(ns.UNIVERSE))
    df = pd.read_sql("SELECT ticker,date,high,low,close,volume FROM market_data "
                     "WHERE ticker = ANY(%(tk)s) AND close>0 ORDER BY ticker,date",
                     conn, params={"tk": tickers})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for x in ("high", "low", "close", "volume"):
        df[x] = df[x].astype(float)
    parts = []
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        c, h, l = g["close"], g["high"], g["low"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        g["atr_pct"] = tr.rolling(14).mean() / c * 100
        g["turn_prev"] = (c * g["volume"]).shift(1)
        parts.append(g[["ticker", "date", "close", "atr_pct", "turn_prev"]])
    return pd.concat(parts, ignore_index=True)


def _t(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 8 or s.std(ddof=1) == 0:
        return {"n": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"n": n, "mean": float(s.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def lowvol_horizon(panel: pd.DataFrame, H: int, spreads: dict) -> dict:
    """Лонг-онли нижний квинтиль ATR против универса, перебалансировка каждые H дней.
    Издержки = круг × оборот состава квинтиля, амортизировано."""
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    atr = panel.pivot(index="date", columns="ticker", values="atr_pct").sort_index()
    days = list(close.index)
    exc, days_used, prev_set = [], [], None
    fee_rt = 2 * cm.FEE_SIDE_PCT + cm.load_spreads()[cm._FALLBACK][0]   # круг: комиссия + спред блю-чипов
    for i in range(0, len(days) - H, H):
        d = days[i]
        a = atr.loc[d].dropna()
        if len(a) < 15:
            continue
        thr = a.quantile(QUINT)
        low = set(a[a <= thr].index)                      # низковолатильные
        c0 = close.loc[d]
        cH = close.loc[days[i + H]]
        r_low = np.nanmean([(cH[t] / c0[t] - 1) * 100 for t in low if c0.get(t) and cH.get(t)])
        r_uni = np.nanmean([(cH[t] / c0[t] - 1) * 100 for t in a.index if c0.get(t) and cH.get(t)])
        turnover = 1.0 if prev_set is None else 1 - len(low & prev_set) / max(1, len(low))
        cost = fee_rt * turnover                            # платим только за сменившийся состав
        exc.append((r_low - r_uni) - cost)
        days_used.append(d)
        prev_set = low
    s = pd.Series(exc, index=days_used)
    r = _t(s)
    # годовые: среднее за ребаланс × (252/H)
    r["excess_annual_pct"] = float(s.mean() * (252 / H)) if len(s) else None
    r["rebalances"] = len(s)
    return r


def overnight_tilt(panel: pd.DataFrame, spreads: dict) -> dict:
    """Прокси BEST_TRADES: top-K по обороту. Ночная (H=1) доходность корзины
    против той же корзины, тилтованной в сторону низкого ATR (замена верхнего
    ATR-имени в корзине на следующий низко-ATR из кандидатов). Без лишнего круга."""
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    atr = panel.pivot(index="date", columns="ticker", values="atr_pct").sort_index()
    turn = panel.pivot(index="date", columns="ticker", values="turn_prev").sort_index()
    days = list(close.index)
    base, tilt = [], []
    d_used = []
    for i in range(len(days) - 1):
        d, dn = days[i], days[i + 1]
        tp = turn.loc[d].dropna()
        a = atr.loc[d].dropna()
        cand = [t for t in tp.index if t in a.index]
        if len(cand) < TOPK + 3:
            continue
        ranked = tp[cand].sort_values(ascending=False)
        basket = list(ranked.index[:TOPK])
        # тилт: из топ-2K кандидатов берём K с наименьшим ATR
        pool = list(ranked.index[:2 * TOPK])
        tilted = sorted(pool, key=lambda t: a[t])[:TOPK]
        def on(names):
            return np.nanmean([(close.loc[dn][t] / close.loc[d][t] - 1) * 100
                               for t in names if close.loc[d].get(t) and close.loc[dn].get(t)])
        base.append(on(basket)); tilt.append(on(tilted)); d_used.append(d)
    b = pd.Series(base, index=d_used); tl = pd.Series(tilt, index=d_used)
    lift = _t(tl - b)
    return {"base_overnight_mean": float(b.mean()), "tilt_overnight_mean": float(tl.mean()),
            "lift": lift}


def run_sample(conn, d_from, d_to, spreads) -> dict:
    panel = build(conn)
    panel = panel[(panel["date"] >= d_from) & (panel["date"] <= d_to)].dropna(subset=["atr_pct"])
    out = {"lowvol_longonly": {f"H{H}": lowvol_horizon(panel, H, spreads) for H in HORIZONS},
           "overnight_tilt": overnight_tilt(panel, spreads)}
    return out


def run(conn) -> dict:
    spreads = cm.load_spreads()
    return {"note": "low-vol лонг-онли нижний квинтиль ATR против универса, издержки с учётом оборота; тилт ночной корзины",
            "dev": run_sample(conn, *DEV, spreads), "holdout": run_sample(conn, *HOLDOUT, spreads)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "lowvol_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 19,
                            "stage": "ta_lowvol_tradability", "trials": 4, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
