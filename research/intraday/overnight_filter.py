"""
Внутридневная реверсия как ФИЛЬТР/вес ночной корзины BEST_TRADES (ТЗ пользователя
23.09.2026 — единственный живой лид из сводки программы).

Идея: реверсия к закрытию (research/intraday/regime_scan) реальна, но как
отдельная сделка убыточна из-за издержек. Здесь её применяем БЕЗ лишнего круга:
корзину long_overnight стратегия и так покупает в 18:35 — вопрос лишь, какие из
уже отобранных бумаг держать. Проверяем: предсказывает ли аномальный ПОЗДНИЙ ход
бумаги за день её НОЧНУЮ доходность (close→открытие). Если растянутые вверх за
день бумаги отдают на ночь — их надо недовешивать; это переотбор, не новая сделка.

Сигнал (известен к входу 18:35): late_move[бумага, D] = ход за окно
(16:00→18:15 и др.) в z-оценках к собственной истории окна (40 дней, назад).
Цель: ночная доходность open[D+1]/close[D], кросс-секционно демеанированная
(вес внутри корзины — относительный). Гипотеза реверсии: corr(late_move_z,
overnight_rel) ОТРИЦАТЕЛЬНА (поздний рост → слабее на ночь).

Замеры:
  1) кросс-секц. корреляция late_move_z ↔ overnight_rel, t по дням, Холм по окнам;
  2) «прибавка фильтра»: ночная доходность нижнего квинтиля late_move минус
     верхнего (растянутые), t по дням — прямая оценка выгоды переотбора;
  3) на прокси-корзине top-K по обороту: ночная доходность корзины БЕЗ фильтра
     против корзины с выкинутыми верхне-децильными late-движками.

Выборки (стоки в БД с 2024-05, период находки реверсии — 2025-12+, поэтому dev
берём РАНЬШЕ, чтобы разработка была на не-смотренном для этого сигнала окне):
  dev     2024-05-21 … 2025-11-30
  holdout 2025-12-01 … 2026-09-22
Запуск (на сервере): python -m research.intraday.overnight_filter
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

from research import news_event_study as ns                # noqa: E402

log = logging.getLogger("research.intraday.overnight_filter")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "intraday_regime")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2025, 11, 30))
HOLDOUT = (dt.date(2025, 12, 1), dt.date(2026, 9, 22))
WINDOWS = {"15-1815": (dt.time(15, 0), dt.time(18, 15)),
           "16-1815": (dt.time(16, 0), dt.time(18, 15)),
           "17-1815": (dt.time(17, 0), dt.time(18, 15))}
HEADLINE = "16-1815"
Z_WINDOW, Z_MIN = 40, 20
TOPK = 5                                   # прокси-корзина ~ BEST_TRADES_TOP_N

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close
FROM market_data_5m
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN '14:55' AND '18:20'
"""
DAILY_SQL = """
SELECT ticker, date, open, close, volume FROM market_data
WHERE ticker = ANY(%(tk)s) AND close > 0 AND date >= %(f)s AND date <= %(t)s
ORDER BY ticker, date
"""


def _close_at(s: pd.Series, t0: dt.time, mode: str) -> float | None:
    """первый close с временем ≥ t0 (mode=ge) или последний ≤ t0 (mode=le)."""
    if mode == "ge":
        m = s[s.index.time >= t0]
        return float(m.iloc[0]) if len(m) else None
    m = s[s.index.time <= t0]
    return float(m.iloc[-1]) if len(m) else None


def late_moves(conn, tickers, d_from, d_to) -> pd.DataFrame:
    bars = pd.read_sql(BARS_SQL, conn, params={"tk": tickers,
                       "f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"})
    bars["tm"] = pd.to_datetime(bars["tm"])
    bars["d"] = bars["tm"].dt.date
    bars["close"] = bars["close"].astype(float)
    rows = []
    for (tk, d), g in bars.groupby(["ticker", "d"]):
        s = g.set_index("tm")["close"].sort_index()
        rec = {"ticker": tk, "d": d}
        for name, (t0, t1) in WINDOWS.items():
            a = _close_at(s, t0, "ge")
            b = _close_at(s, t1, "le")
            rec[name] = (b / a - 1.0) * 100.0 if (a and b) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def zscore(df: pd.DataFrame, col: str) -> pd.Series:
    out = []
    for tk, g in df.sort_values("d").groupby("ticker"):
        g = g.sort_values("d")
        mu = g[col].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).mean()
        sd = g[col].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).std(ddof=1)
        out.append(pd.Series((g[col] - mu) / sd.replace(0.0, np.nan), index=g.index))
    return pd.concat(out).sort_index()


def overnight(conn, tickers, d_from, d_to) -> pd.DataFrame:
    d = pd.read_sql(DAILY_SQL, conn, params={"tk": tickers, "f": d_from, "t": d_to + dt.timedelta(days=5)})
    d["date"] = pd.to_datetime(d["date"]).dt.date
    for c in ("open", "close", "volume"):
        d[c] = d[c].astype(float)
    d = d.sort_values(["ticker", "date"])
    d["next_open"] = d.groupby("ticker")["open"].shift(-1)
    d["overnight"] = (d["next_open"] / d["close"] - 1.0) * 100.0
    d["turn_prev"] = d.groupby("ticker").apply(
        lambda x: (x["close"] * x["volume"]).shift(1), include_groups=False).reset_index(level=0, drop=True)
    d = d.rename(columns={"date": "d"})
    d["on_rel"] = d["overnight"] - d.groupby("d")["overnight"].transform("mean")
    return d[["ticker", "d", "overnight", "on_rel", "turn_prev"]]


def t_by_day(vals: pd.Series, days: pd.Series) -> dict:
    s = pd.Series(vals.to_numpy(float), index=days.to_numpy())
    g = s.groupby(level=0).mean().dropna()
    n = len(g)
    if n < 10 or g.std(ddof=1) == 0:
        return {"days": n}
    t = float(g.mean() / (g.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"days": n, "mean": float(g.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def daily_corr(panel: pd.DataFrame, xcol: str, ycol: str) -> dict:
    vals, days = [], []
    for d, g in panel.groupby("d"):
        g = g.dropna(subset=[xcol, ycol])
        if len(g) >= 6 and g[xcol].std() > 0:
            vals.append(float(g[xcol].corr(g[ycol])))
            days.append(d)
    return t_by_day(pd.Series(vals), pd.Series(days))


def quintile_lift(panel: pd.DataFrame, zcol: str) -> dict:
    """Ночная доходность нижнего квинтиля late_move минус верхнего (растянутые), по дням."""
    diffs, days = [], []
    for d, g in panel.groupby("d"):
        g = g.dropna(subset=[zcol, "overnight"])
        if len(g) < 10:
            continue
        q = g[zcol].quantile([0.2, 0.8])
        low = g[g[zcol] <= q.iloc[0]]["overnight"].mean()
        high = g[g[zcol] >= q.iloc[1]]["overnight"].mean()
        if pd.notna(low) and pd.notna(high):
            diffs.append(low - high)
            days.append(d)
    return t_by_day(pd.Series(diffs), pd.Series(days))


def basket_filter(panel: pd.DataFrame, zcol: str) -> dict:
    """Прокси BEST_TRADES: топ-K по обороту вчера. Ночная доходность корзины
    БЕЗ фильтра против корзины с выкинутыми верхне-децильными late-движками (замена — кэш=0)."""
    base, filt, days = [], [], []
    for d, g in panel.groupby("d"):
        g = g.dropna(subset=["turn_prev", "overnight"])
        if len(g) < TOPK + 3:
            continue
        basket = g.nlargest(TOPK, "turn_prev")
        base.append(basket["overnight"].mean())
        thr = g[zcol].quantile(0.9)
        kept = basket[~(basket[zcol] >= thr)]
        # выкинутая позиция → кэш (ночная 0), поэтому среднее по K с нулями
        filt.append(kept["overnight"].sum() / TOPK)
        days.append(d)
    base = pd.Series(base, index=days)
    filt = pd.Series(filt, index=days)
    lift = filt - base
    n = len(lift.dropna())
    from scipy import stats
    t = float(lift.mean() / (lift.std(ddof=1) / math.sqrt(n))) if n > 2 and lift.std(ddof=1) else None
    return {"days": n, "base_mean_overnight": float(base.mean()), "filt_mean_overnight": float(filt.mean()),
            "lift_mean": float(lift.mean()) if n else None, "lift_t": t,
            "lift_p": float(2 * stats.t.sf(abs(t), n - 1)) if t is not None else None}


def run_sample(conn, tickers, d_from, d_to) -> dict:
    lm = late_moves(conn, tickers, d_from - dt.timedelta(days=90), d_to)
    on = overnight(conn, tickers, d_from - dt.timedelta(days=5), d_to)
    out = {}
    pvals = {}
    for w in WINDOWS:
        lm[f"z_{w}"] = zscore(lm, w)
    panel_full = lm.merge(on, on=["ticker", "d"], how="inner")
    panel_full = panel_full[(panel_full["d"] >= d_from) & (panel_full["d"] <= d_to)]
    for w in WINDOWS:
        c = daily_corr(panel_full, f"z_{w}", "on_rel")
        out[w] = {"corr_zlate_overnight": c}
        if c.get("p") is not None:
            pvals[w] = c["p"]
    items = sorted((v, k) for k, v in pvals.items())
    run_p, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        out[k]["corr_zlate_overnight"]["p_holm"] = run_p
    zc = f"z_{HEADLINE}"
    out["headline_window"] = HEADLINE
    out["quintile_lift"] = quintile_lift(panel_full, zc)
    out["basket_filter"] = basket_filter(panel_full, zc)
    out["days"] = int(panel_full["d"].nunique())
    return out


def run(conn) -> dict:
    tickers = sorted(set(ns.UNIVERSE))
    return {"hypothesis": "поздний внутридневной ход бумаги → её ночная доходность; реверсия ⇒ corr<0, фильтр даёт + к корзине",
            "topk": TOPK, "dev": run_sample(conn, tickers, *DEV),
            "holdout": run_sample(conn, tickers, *HOLDOUT)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "overnight_filter_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 17,
                            "stage": "overnight_reversion_filter", "trials": 5, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
