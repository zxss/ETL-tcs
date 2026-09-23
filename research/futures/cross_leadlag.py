"""
Кросс-активный внутридневной lead-lag между фьючерсами (ТЗ пользователя
23.09.2026). Предсказывает ли 5-минутный ход одного фьючерса ход другого в
СЛЕДУЮЩЕМ баре — направление, отличное от «сырьё→акция» (Спринт 2) и от
внутридневной динамики одного инструмента (intraday_reversion).

Пары и ожидаемые знаки (предзарегистрированы ДО прогона):
  BR ↔ NG  — энергокомплекс (нефть/газ), ожидаем ПОЛОЖИТЕЛЬНУЮ связь;
  BR ↔ Si  — нефть → рубль: нефть вверх → USD/RUB вниз, ожидаем ОТРИЦАТЕЛЬНУЮ;
  GD ↔ MX  — золото ↔ индекс MOEX, знак НЕ задан (двусторонняя проверка).
Каждую пару тестируем в обе стороны (A ведёт B и B ведёт A).

Метод: ближний контракт по объёму вчера, 5-минутные лог-доходности внутри дня
(без переноса через день). Для каждого дня корреляция ret_A[t] с ret_B[t+1]
(лид на 1 бар) и одновременная ret_A[t]↔ret_B[t] (базовая ко-движения); t по
дням (среднее дневных корреляций), поправка Холма по всем лид-замерам.
Торговое правило на значимом лиде — вторично: держать 1 бар (5 мин) при круговых
издержках 0,02–0,05 % почти наверняка съедает эффект, но считаем.

Выборки: dev 2024-05-21…2026-09-11, holdout 2022-01-03…2024-05-20.
Сессия 10:00–18:45. Запуск (на сервере): python -m research.futures.cross_leadlag
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

log = logging.getLogger("research.futures.cross_leadlag")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "futures")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2026, 9, 11))
HOLDOUT = (dt.date(2022, 1, 3), dt.date(2024, 5, 20))
SES0, SES1 = dt.time(10, 0), dt.time(18, 45)
COSTS = {"real": 0.02, "cons": 0.05}
LEAD_QTILE = 0.90                          # «сильный» ход ведущего = верхний дециль |ret|
PAIRS = [("BR", "NG", +1, "нефть↔газ"), ("BR", "Si", -1, "нефть→рубль"),
         ("GD", "MX", 0, "золото↔индекс")]

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close, volume
FROM research_fut_5m
WHERE ticker LIKE %(root)s AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time >= %(s0)s
  AND (ts AT TIME ZONE 'Europe/Moscow')::time <= %(s1)s
"""


def near_returns(conn, root: str, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    """5-минутные лог-доходности ближнего контракта внутри дня: [tm, d, ret]."""
    df = pd.read_sql(BARS_SQL, conn, params={"root": root + "%", "s0": SES0, "s1": SES1,
                     "f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"})
    if df.empty:
        return df
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    vol = df.groupby(["d", "ticker"])["volume"].sum().unstack(fill_value=0.0).sort_index()
    prev = vol.shift(1)
    near = {d: prev.loc[d].dropna()[prev.loc[d].dropna() > 0].idxmax()
            for d in vol.index if prev.loc[d].dropna().gt(0).any()}
    out = []
    for d, g in df.groupby("d"):
        tk = near.get(d)
        if not tk:
            continue
        s = g[g["ticker"] == tk].sort_values("tm")
        if len(s) < 10:
            continue
        r = np.log(s["close"].to_numpy()[1:] / s["close"].to_numpy()[:-1]) * 100.0
        out.append(pd.DataFrame({"tm": s["tm"].to_numpy()[1:], "d": d, "ret": r}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def daily_corr(merged: pd.DataFrame, xcol: str, ycol: str) -> tuple[pd.Series, pd.Series]:
    """Корреляция xcol↔ycol по каждому дню; возвращает (series коэфф., series дат)."""
    vals, days = [], []
    for d, g in merged.groupby("d"):
        g = g.dropna(subset=[xcol, ycol])
        if len(g) >= 10 and g[xcol].std() > 0 and g[ycol].std() > 0:
            vals.append(float(g[xcol].corr(g[ycol])))
            days.append(d)
    return pd.Series(vals), pd.Series(days)


def t_of(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 10 or s.std(ddof=1) == 0:
        return {"days": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"days": n, "mean_corr": float(s.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def lead_rule(merged: pd.DataFrame, lead: str, tgt: str, sign: int, cost: float) -> dict:
    """Сильный ход ведущего (|ret|≥дециль дня) → сделка по цели в следующем баре в сторону sign*знак(ход)."""
    m = merged.dropna(subset=[lead, tgt]).copy()
    if len(m) < 200:
        return {"trades": 0}
    thr = m[lead].abs().quantile(LEAD_QTILE)
    sig = m[m[lead].abs() >= thr]
    if len(sig) < 50:
        return {"trades": int(len(sig))}
    direction = sign if sign != 0 else 1     # для двустороннего берём знак из корреляции отдельно; тут sign задаёт гипотезу
    net = direction * np.sign(sig[lead]) * sig[tgt] - cost
    per_day = net.groupby(sig["d"]).mean()
    n = len(per_day)
    t = float(per_day.mean() / (per_day.std(ddof=1) / math.sqrt(n))) if per_day.std(ddof=1) else None
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), n - 1)) if t is not None else None
    return {"trades": int(len(sig)), "days": n, "mean_day_pct": float(per_day.mean()),
            "t": t, "p": p, "win_rate": float((net > 0).mean())}


def run_sample(conn, d_from, d_to) -> dict:
    cache = {}
    def rets(root):
        if root not in cache:
            cache[root] = near_returns(conn, root, d_from - dt.timedelta(days=5), d_to)
        return cache[root]
    out = {}
    pvals = {}
    for a, b, sign, name in PAIRS:
        ra, rb = rets(a), rets(b)
        if ra.empty or rb.empty:
            out[f"{a}-{b}"] = {"note": "нет данных"}
            continue
        m = ra[["tm", "d", "ret"]].merge(rb[["tm", "ret"]], on="tm", suffixes=("_a", "_b"))
        m = m[(m["d"] >= d_from) & (m["d"] <= d_to)].sort_values("tm")
        # следующий бар цели внутри дня
        m["b_next"] = m.groupby("d")["ret_b"].shift(-1)
        m["a_next"] = m.groupby("d")["ret_a"].shift(-1)
        contemp, _ = daily_corr(m, "ret_a", "ret_b")
        lead_ab, dab = daily_corr(m, "ret_a", "b_next")     # A ведёт B
        lead_ba, dba = daily_corr(m.rename(columns={}), "ret_b", "a_next")  # B ведёт A
        res = {"name": name, "expect_sign": sign,
               "contemp": t_of(contemp),
               "lead_A_to_B": t_of(lead_ab), "lead_B_to_A": t_of(lead_ba)}
        for tag, blk in (("AB", res["lead_A_to_B"]), ("BA", res["lead_B_to_A"])):
            if blk.get("p") is not None:
                pvals[f"{a}-{b}:{tag}"] = blk["p"]
        res["rule_A_to_B"] = {k: lead_rule(m, "ret_a", "b_next", sign, c) for k, c in COSTS.items()}
        res["rule_B_to_A"] = {k: lead_rule(m, "ret_b", "a_next", sign, c) for k, c in COSTS.items()}
        out[f"{a}-{b}"] = res
    items = sorted((v, k) for k, v in pvals.items())
    run_p, mm = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (mm - i) * v))
        pair, tag = k.split(":")
        out[pair].setdefault("holm", {})[tag] = run_p
    return out


def run(conn) -> dict:
    return {"pairs": [f"{a}-{b} ({n}, знак {s})" for a, b, s, n in PAIRS],
            "lead_qtile": LEAD_QTILE, "costs": COSTS,
            "dev": run_sample(conn, *DEV), "holdout": run_sample(conn, *HOLDOUT)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cross_leadlag_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 16,
                            "stage": "futures_cross_leadlag", "trials": 6, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
