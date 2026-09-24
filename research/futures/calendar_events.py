"""
Календарные потоки на фьючерсах (ТЗ пользователя 24.09.2026, гипотезы B1, B2).
Фьючерсы выбраны за издержки: круг ~0,02–0,05 % против ~0,14 % по акциям.

B1. Налоговый период. Экспортёры продают валюту к сроку налогов (28-е число,
    с 2023 — единый налоговый платёж). Гипотеза: USD/RUB (Si) и CNY/RUB (CR)
    ниже нормы в 5 торговых дней до T (T = последний торговый день ≤ 28-го) и
    выше нормы 5 дней после. Окна: pre = close[T−5]→close[T], post =
    close[T]→close[T+5], внутри одного контракта (ближний на начало окна).
    Норма = 5 × средняя дневная доходность инструмента по той же выборке.
    Сетка: 2 инструмента × 2 окна = 4 испытания. Правило: шорт pre + лонг post,
    издержки 0,05 % за круг на ногу.

B2. Решения ЦБ по ключевой ставке (плановые заседания, 13:30 МСК; внеплановые
    исключены — их нельзя знать заранее). (a) дрейф перед решением (аналог
    pre-FOMC): close[D−1, 18:45] → 13:25 D; (b) разворот реакции: ход
    13:25→14:00 против хода 14:00→18:45, правило фейдит первую реакцию.
    Инструменты MX (индекс) и Si. Сетка 2 × 2 = 4 испытания.

Все 8 испытаний под одной поправкой Холма. Выборки как в прочих тестах
фьючерсов: dev 2024-05-21…2026-09-11, holdout 2022-01-03…2024-05-20 (событий
мало — B2 даёт ~17–19 заседаний на выборку, мощность слабая, это сказано
заранее). Запуск (сервер): python -m research.futures.calendar_events
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

log = logging.getLogger("research.futures.calendar_events")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "futures")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2026, 9, 11))
HOLDOUT = (dt.date(2022, 1, 3), dt.date(2024, 5, 20))
COST = 0.05                                  # % за круг, консервативно
TAX_DAY, WIN = 28, 5
CLOSE_T = dt.time(18, 45)

# Плановые заседания по ключевой ставке (cbr.ru/dkp/cal_mp), решение в 13:30 МСК.
CBR_MEETINGS = [
    "2022-02-11", "2022-03-18", "2022-04-29", "2022-06-10", "2022-07-22", "2022-09-16",
    "2022-10-28", "2022-12-16",
    "2023-02-10", "2023-03-17", "2023-04-28", "2023-06-09", "2023-07-21", "2023-09-15",
    "2023-10-27", "2023-12-15",
    "2024-02-16", "2024-03-22", "2024-04-26", "2024-06-07", "2024-07-26", "2024-09-13",
    "2024-10-25", "2024-12-20",
    "2025-02-14", "2025-03-21", "2025-04-25", "2025-06-06", "2025-07-25", "2025-09-12",
    "2025-10-24", "2025-12-19",
    "2026-02-13", "2026-03-20", "2026-04-24", "2026-06-19", "2026-07-24", "2026-09-11",
]

SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close, volume
FROM research_fut_5m
WHERE ticker LIKE %(root)s AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN '10:00' AND '18:45'
"""


def load(conn, root: str) -> pd.DataFrame:
    df = pd.read_sql(SQL, conn, params={"root": root + "%", "f": "2021-12-01 00:00+03",
                                        "t": "2026-09-20 00:00+03"})
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("tm")


def near_map(df: pd.DataFrame) -> dict:
    vol = df.groupby(["d", "ticker"])["volume"].sum().unstack(fill_value=0.0).sort_index()
    prev = vol.shift(1)
    out = {}
    for d in vol.index:
        pv = prev.loc[d].dropna()
        pv = pv[pv > 0]
        if len(pv):
            out[d] = pv.idxmax()
    return out


_CACHE: dict = {}


def px_at(df: pd.DataFrame, tk: str, d: dt.date, t: dt.time) -> float | None:
    """последний close контракта tk в день d не позже t."""
    key = id(df)
    if key not in _CACHE:
        _CACHE[key] = {k: g for k, g in df.groupby(["ticker", "d"])}
    g = _CACHE[key].get((tk, d))
    if g is None:
        return None
    s = g[g["tm"].dt.time <= t]
    return float(s["close"].iloc[-1]) if len(s) else None


def daily_closes(df: pd.DataFrame) -> pd.DataFrame:
    """(d, ticker) → close на 18:45 (последний бар дня)."""
    return df.groupby(["d", "ticker"])["close"].last().unstack()


def _stats(x: list[float]) -> dict:
    s = pd.Series(x, dtype=float).dropna()
    n = len(s)
    if n < 6 or s.std(ddof=1) == 0:
        return {"n": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"n": n, "mean_pct": float(s.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "hit": float((s > 0).mean())}


# ── B1 налоговый период ──────────────────────────────────────────────────────

def tax_windows(df: pd.DataFrame, d0: dt.date, d1: dt.date) -> dict:
    near = near_map(df)
    closes = daily_closes(df)
    days = [d for d in closes.index if d0 <= d <= d1 and d in near]
    # норма: средняя дневная доходность ближнего контракта (внутри контракта)
    daily = []
    for a, b in zip(days[:-1], days[1:]):
        tk = near[a]
        pa, pb = closes.at[a, tk], closes.at[b, tk]
        if pd.notna(pa) and pd.notna(pb):
            daily.append((pb / pa - 1.0) * 100.0)
    drift = float(np.mean(daily)) if daily else 0.0
    pre, post, rule = [], [], []
    months = sorted({(d.year, d.month) for d in days})
    for y, m in months:
        cand = [d for d in days if d.year == y and d.month == m and d.day <= TAX_DAY]
        if not cand:
            continue
        T = cand[-1]
        i = days.index(T)
        if i - WIN < 0 or i + WIN >= len(days):
            continue
        a, b = days[i - WIN], days[i + WIN]
        tk_pre, tk_post = near[a], near[T]
        def ret(tk, x, y_):
            px, py = closes.at[x, tk], closes.at[y_, tk]
            return (py / px - 1.0) * 100.0 if pd.notna(px) and pd.notna(py) else np.nan
        r_pre = ret(tk_pre, a, T) - WIN * drift
        r_post = ret(tk_post, T, b) - WIN * drift
        pre.append(r_pre)
        post.append(r_post)
        rule.append((-(r_pre + WIN * drift) + (r_post + WIN * drift)) - 2 * COST)   # сырые, без нормы
    return {"drift_daily_pct": drift, "pre_excess": _stats(pre), "post_excess": _stats(post),
            "rule_short_pre_long_post_net": _stats(rule)}


# ── B2 решения ЦБ ────────────────────────────────────────────────────────────

def cbr_events(df: pd.DataFrame, d0: dt.date, d1: dt.date) -> dict:
    near = near_map(df)
    all_days = sorted(df["d"].unique())
    pre, r1s, r2s, fade = [], [], [], []
    base = []                                            # тот же отрезок в обычные дни
    meet = {dt.date.fromisoformat(x) for x in CBR_MEETINGS}
    for i, d in enumerate(all_days):
        if not (d0 <= d <= d1) or i == 0 or d not in near:
            continue
        tk, prev = near[d], all_days[i - 1]
        p_prev = px_at(df, tk, prev, CLOSE_T)
        p_1325 = px_at(df, tk, d, dt.time(13, 25))
        if not (p_prev and p_1325):
            continue
        drift = (p_1325 / p_prev - 1.0) * 100.0
        if d not in meet:
            base.append(drift)
            continue
        pre.append(drift)
        p_1400 = px_at(df, tk, d, dt.time(13, 55))
        p_cl = px_at(df, tk, d, CLOSE_T)
        if p_1400 and p_cl:
            r1 = (p_1400 / p_1325 - 1.0) * 100.0
            r2 = (p_cl / p_1400 - 1.0) * 100.0
            r1s.append(r1); r2s.append(r2)
            fade.append(-np.sign(r1) * r2 - COST)
    base_mean = float(np.mean(base)) if base else 0.0
    corr = None
    if len(r1s) >= 6:
        r = float(np.corrcoef(r1s, r2s)[0, 1])
        n = len(r1s)
        t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
        from scipy import stats
        corr = {"n": n, "r": r, "t": t, "p": float(2 * stats.t.sf(abs(t), n - 2))}
    return {"pre_drift_excess": _stats([x - base_mean for x in pre]),
            "pre_drift_raw_mean_pct": float(np.mean(pre)) if pre else None,
            "base_same_window_mean_pct": base_mean,
            "reaction_corr_r1_r2": corr, "fade_rule_net": _stats(fade)}


def holm(pvals: dict) -> dict:
    items = sorted((v, k) for k, v in pvals.items() if v is not None)
    out, run_p, m = {}, 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        out[k] = run_p
    return out


def run(conn) -> dict:
    res = {"cost_round_trip_pct": COST, "dev": {}, "holdout": {}}
    data = {r: load(conn, r) for r in ("Si", "CR", "MX")}
    for name, (d0, d1) in (("dev", DEV), ("holdout", HOLDOUT)):
        res[name]["B1_tax"] = {r: tax_windows(data[r], d0, d1) for r in ("Si", "CR")}
        res[name]["B2_cbr"] = {r: cbr_events(data[r], d0, d1) for r in ("MX", "Si")}
        log.info("%s готов", name)
    d = res["dev"]
    p = {}
    for r in ("Si", "CR"):
        p[f"B1/{r}/pre"] = d["B1_tax"][r]["pre_excess"].get("p")
        p[f"B1/{r}/post"] = d["B1_tax"][r]["post_excess"].get("p")
    for r in ("MX", "Si"):
        p[f"B2/{r}/pre_drift"] = d["B2_cbr"][r]["pre_drift_excess"].get("p")
        p[f"B2/{r}/reaction"] = (d["B2_cbr"][r]["reaction_corr_r1_r2"] or {}).get("p")
    res["holm_dev"] = holm(p)
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "calendar_events_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
