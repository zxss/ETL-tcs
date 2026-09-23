"""
Внутридневная реверсия на фьючерсах (ТЗ пользователя 23.09.2026, приоритет №2).

Гипотеза (предзарегистрирована ДО прогона): аномально большой ход фьючерса в
часовом окне основной сессии откатывается к закрытию сессии — тот же эффект,
что найден на акциях (research/intraday/regime_scan: реальный, но не торгуемый
после издержек на одной бумаге). У фьючерсов издержки относительно хода НИЖЕ,
поэтому проверяем, не становится ли он торгуемым здесь. Правило контрариан:
фейдим ход (|z|≥2), вход в конце часа, выход на закрытии сессии.

Инструменты: Brent (BR), газ NG, золото $ (GD), USD/RUB (Si), индекс MOEX (MX).
Ближний контракт по объёму предыдущего дня; z-оценка часового хода — к
собственной истории того же инструмента и того же часа (40 дней, только назад).
Основная сессия 10:00–18:45 (у фьючерсов есть и вечерняя, но берём аналог
акционного окна). Издержки круга — два сценария: 0,02 % (реалистично для
ликвидного ближнего) и 0,05 % (консервативно).

Выборки: dev 2024-05-21…2026-09-11, holdout 2022-01-03…2024-05-20.

Запуск (на сервере): python -m research.futures.intraday_reversion
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

log = logging.getLogger("research.futures.intraday_reversion")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "futures")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
ROOTS = {"BR": "Brent", "NG": "газ NG", "GD": "золото $", "Si": "USD/RUB", "MX": "индекс MOEX"}
DEV = (dt.date(2024, 5, 21), dt.date(2026, 9, 11))
HOLDOUT = (dt.date(2022, 1, 3), dt.date(2024, 5, 20))
SES0, SES1 = dt.time(10, 0), dt.time(18, 45)
BUCKETS = [(dt.time(h, 0), dt.time(h + 1, 0)) for h in range(10, 18)]     # 10..18, 8 окон
Z_WINDOW, Z_MIN, Z_THRESH = 40, 20, 2.0
COSTS = {"real": 0.02, "cons": 0.05}

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close, volume
FROM research_fut_5m
WHERE ticker LIKE %(root)s AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time >= %(s0)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time <= %(s1)s
"""


def load_bars(conn, root: str, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    df = pd.read_sql(BARS_SQL, conn, params={"root": root + "%", "s0": SES0, "s1": SES1,
                     "f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"})
    if df.empty:
        return df
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def near_map(df: pd.DataFrame) -> dict:
    """day → ближний контракт (макс. объём предыдущего дня)."""
    vol = df.groupby(["d", "ticker"])["volume"].sum().unstack(fill_value=0.0).sort_index()
    prev = vol.shift(1)
    out = {}
    for d in vol.index:
        pv = prev.loc[d].dropna()
        pv = pv[pv > 0]
        if len(pv):
            out[d] = pv.idxmax()
    return out


def panel(df: pd.DataFrame) -> pd.DataFrame:
    """(day, bucket) ближнего контракта → bucket_ret и aftermath (до закрытия сессии)."""
    near = near_map(df)
    rows = []
    for d, g in df.groupby("d"):
        tk = near.get(d)
        if not tk:
            continue
        s = g[g["ticker"] == tk].sort_values("tm")
        if len(s) < 20:
            continue
        ses_close = float(s["close"].iloc[-1])
        for bi, (t0, t1) in enumerate(BUCKETS):
            b = s[(s["tm"].dt.time >= t0) & (s["tm"].dt.time < t1)]
            if len(b) < 6:
                continue
            o, c = float(b["close"].iloc[0]), float(b["close"].iloc[-1])
            rows.append({"d": d, "bucket": bi, "bret": (c / o - 1.0) * 100.0,
                         "aftermath": (ses_close / c - 1.0) * 100.0})
    return pd.DataFrame(rows)


def add_z(pl: pd.DataFrame) -> pd.DataFrame:
    pl = pl.sort_values("d").copy()
    out = []
    for bi, g in pl.groupby("bucket"):
        g = g.sort_values("d")
        mu = g["bret"].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).mean()
        sd = g["bret"].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).std(ddof=1)
        g = g.assign(z=(g["bret"] - mu) / sd.replace(0.0, np.nan))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def corr_t(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(df)
    if n < 30 or df["x"].std() == 0:
        return {"n": n}
    r = float(df["x"].corr(df["y"]))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
    from scipy import stats
    return {"n": n, "r": r, "t": t, "p": float(2 * stats.t.sf(abs(t), n - 2))}


def rule(pl: pd.DataFrame, cost: float) -> dict:
    """Фейд: |z|≥порог → позиция против хода, выход на закрытии сессии. Нетто по дням."""
    sig = pl.dropna(subset=["z", "aftermath"])
    sig = sig[sig["z"].abs() >= Z_THRESH]
    if len(sig) < 20:
        return {"trades": int(len(sig))}
    sig = sig.assign(net=-np.sign(sig["bret"]) * sig["aftermath"] - cost)
    per_day = sig.groupby("d")["net"].mean()
    n = len(per_day)
    t = float(per_day.mean() / (per_day.std(ddof=1) / math.sqrt(n))) if per_day.std(ddof=1) else None
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), n - 1)) if t is not None else None
    return {"trades": int(len(sig)), "days": n, "mean_day_pct": float(per_day.mean()),
            "t": t, "p": p, "win_rate": float((sig["net"] > 0).mean()),
            "sum_pct": float(per_day.sum())}


def run_root(conn, root: str, d_from: dt.date, d_to: dt.date) -> dict:
    df = load_bars(conn, root, d_from - dt.timedelta(days=90), d_to)
    if df.empty:
        return {"note": "нет данных"}
    pl = add_z(panel(df))
    pl = pl[(pl["d"] >= d_from) & (pl["d"] <= d_to)]
    if pl.empty:
        return {"note": "пусто после фильтра"}
    overall = corr_t(pl["z"], pl["aftermath"])            # пулинг по всем окнам
    by_bucket = {}
    for bi, g in pl.groupby("bucket"):
        by_bucket[f"{BUCKETS[bi][0].strftime('%H')}-{BUCKETS[bi][1].strftime('%H')}"] = corr_t(g["z"], g["aftermath"])
    return {"days": int(pl["d"].nunique()), "pooled_corr": overall,
            "by_bucket": by_bucket, "rule": {k: rule(pl, c) for k, c in COSTS.items()}}


def run(conn) -> dict:
    res = {"hypothesis": "аномальный часовой ход фьючерса → откат до закрытия (реверсия); правило фейдит ход",
           "z_thresh": Z_THRESH, "costs": COSTS, "dev": {}, "holdout": {}}
    pvals = {}
    for root, name in ROOTS.items():
        res["dev"][root] = {"name": name, **run_root(conn, root, *DEV)}
        res["holdout"][root] = {"name": name, **run_root(conn, root, *HOLDOUT)}
        pc = res["dev"][root].get("pooled_corr", {})
        if pc.get("p") is not None:
            pvals[root] = pc["p"]
        log.info("%s готов", root)
    # Холм по пулинг-корреляциям dev
    items = sorted((v, k) for k, v in pvals.items())
    run_p, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        res["dev"][k].setdefault("pooled_corr", {})["p_holm"] = run_p
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
    with open(os.path.join(OUT_DIR, "intraday_reversion_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 15,
                            "stage": "futures_intraday_reversion", "trials": 10, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
