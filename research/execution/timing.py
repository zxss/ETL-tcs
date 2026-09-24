"""
Исполнение ночных лонгов без лишнего круга (ТЗ пользователя 24.09.2026,
«сделай последовательно все проверки», гипотезы A1 и A2).

Урок программы (237 испытаний): реальные эффекты есть (внутридневная реверсия,
low-vol), но отдельной сделкой их съедают издержки. Здесь сигнал применяется к
сделкам, которые r4 делает в любом случае, — издержки не растут.

A1. Момент выхода по гэпу. Сейчас ночная позиция продаётся на первой цене
    основной сессии (sc.main_open). Вопрос: даёт ли отсрочка выхода на
    +30/+60/+120 минут прибавку при сильном гэпе. Гэп = open первого бара /
    цена входа (close бара 18:30 накануне) − 1, в z-оценках к собственной
    истории бумаги (40 сессий, только назад). Условия: гэп вниз z≤−1, гэп
    вверх z≥+1. Метрика: exit_alt/exit_base − 1, среднее по бумагам за день,
    t по дням. Предрегистрация: 3 горизонта × 2 условия = 6 испытаний, Холм;
    двусторонний тест, правило «откладывать выход» принимается только если на
    dev знак > 0, Холм < 0,05 И на holdout тот же знак с t ≥ 2.

A2. Лимитный вход вместо рыночного. Сейчас покупка в 18:35 по цене +0,1 %.
    Альтернатива: в 17:30 выставить лимит L = P(17:25)·(1−X), исполнение если
    low любого бара 17:30…18:30 строго ниже L (очередь), иначе покупка в 18:35
    как сейчас. Выигрыш = P_base/entry − 1 (выход тот же). Неблагоприятный
    отбор учтён автоматически: заливка на падении, после которого цена может
    уйти ещё ниже к 18:35. Сетка: X ∈ {0,2; 0,5} % × условие {все дни;
    растянутые вверх: ход 15:00→17:25 z ≥ 2} = 4 испытания, Холм.

Выборки: dev 2024-05-21…2025-11-30, holdout 2025-12-01…2026-09-22.
Универс ns.UNIVERSE (прокси корзины BEST_TRADES). Запуск (сервер):
python -m research.execution.timing
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

from research import news_event_study as ns               # noqa: E402
from research import session_calendar as sc                # noqa: E402

log = logging.getLogger("research.execution.timing")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "execution")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2025, 11, 30))
HOLDOUT = (dt.date(2025, 12, 1), dt.date(2026, 9, 22))
Z_WINDOW, Z_MIN = 40, 20
EXIT_DELAYS = [30, 60, 120]                 # минут после main_open
GAP_Z = 1.0
LIMIT_X = [0.2, 0.5]                        # % ниже цены 17:25
LIMIT_T0 = dt.time(17, 30)
STRETCH_Z = 2.0

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, low, close
FROM market_data_5m
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN '09:00' AND '18:30'
"""


def load(conn, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    df = pd.read_sql(BARS_SQL, conn, params={"tk": list(ns.UNIVERSE),
                     "f": f"{d_from - dt.timedelta(days=80)} 00:00+03",
                     "t": f"{d_to + dt.timedelta(days=6)} 00:00+03"})
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    for c in ("open", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values(["ticker", "tm"]).reset_index(drop=True)


def _plus(t: dt.time, m: int) -> dt.time:
    return (dt.datetime.combine(dt.date(2000, 1, 1), t) + dt.timedelta(minutes=m)).time()


def day_features(bars: pd.DataFrame) -> pd.DataFrame:
    """(ticker, d) → цены, нужные обоим тестам."""
    rows = []
    for (tk, d), g in bars.groupby(["ticker", "d"]):
        g = g.set_index("tm")
        t = g.index.time
        mo = sc.main_open(d)
        rec = {"ticker": tk, "d": d}
        first = g[t == mo]
        rec["open_main"] = float(first["open"].iloc[0]) if len(first) else np.nan
        for m in EXIT_DELAYS:
            tb = _plus(mo, m - 5)                                # close бара, заканчивающегося в mo+m
            b = g[t == tb]
            rec[f"px_{m}"] = float(b["close"].iloc[0]) if len(b) else np.nan
        e = g[t == sc.EVENING_ENTRY_BAR]
        rec["entry"] = float(e["close"].iloc[0]) if len(e) else np.nan
        p0 = g[t == dt.time(17, 25)]
        rec["p1725"] = float(p0["close"].iloc[0]) if len(p0) else np.nan
        s0 = g[t == dt.time(15, 0)]
        rec["p1500"] = float(s0["open"].iloc[0]) if len(s0) else np.nan
        win = g[(t >= LIMIT_T0) & (t <= sc.EVENING_ENTRY_BAR)]
        rec["low_win"] = float(win["low"].min()) if len(win) >= 8 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    out = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("d")
        mu = g[col].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).mean()
        sd = g[col].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN).std(ddof=1)
        out.append((g[col] - mu) / sd.replace(0.0, np.nan))
    return pd.concat(out).reindex(df.index)


def t_by_day(df: pd.DataFrame, col: str) -> dict:
    g = df.dropna(subset=[col]).groupby("d")[col].mean()
    n = len(g)
    if n < 10 or g.std(ddof=1) == 0:
        return {"days": n}
    t = float(g.mean() / (g.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"days": n, "obs": int(df[col].notna().sum()), "mean_pct": float(g.mean()),
            "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def panel(bars: pd.DataFrame) -> pd.DataFrame:
    f = day_features(bars).sort_values(["ticker", "d"]).reset_index(drop=True)
    # следующий торговый день бумаги: выход ночной позиции, открытой вечером d
    nxt = f.groupby("ticker")[["d", "open_main"] + [f"px_{m}" for m in EXIT_DELAYS]].shift(-1)
    f["exit_d"] = nxt["d"]
    f["gap"] = (nxt["open_main"] / f["entry"] - 1.0) * 100.0
    for m in EXIT_DELAYS:
        f[f"hold_{m}"] = (nxt[f"px_{m}"] / nxt["open_main"] - 1.0) * 100.0
    # отсечь разрывы данных (выход позже чем через 5 календарных дней)
    gapdays = (pd.to_datetime(f["exit_d"]) - pd.to_datetime(f["d"])).dt.days
    f.loc[gapdays > 5, ["gap"] + [f"hold_{m}" for m in EXIT_DELAYS]] = np.nan
    f["gap_z"] = _z(f, "gap")
    f["stretch"] = (f["p1725"] / f["p1500"] - 1.0) * 100.0
    f["stretch_z"] = _z(f, "stretch")
    return f


def a1(f: pd.DataFrame) -> dict:
    out = {"unconditional": {f"+{m}м": t_by_day(f, f"hold_{m}") for m in EXIT_DELAYS}}
    for cond, mask in (("gap_down", f["gap_z"] <= -GAP_Z), ("gap_up", f["gap_z"] >= GAP_Z)):
        out[cond] = {f"+{m}м": t_by_day(f[mask], f"hold_{m}") for m in EXIT_DELAYS}
    return out


def a2(f: pd.DataFrame) -> dict:
    out = {}
    base = f.dropna(subset=["entry", "p1725", "low_win"])
    for cond, sub in (("all", base), ("stretched_up", base[base["stretch_z"] >= STRETCH_Z])):
        for x in LIMIT_X:
            lim = sub["p1725"] * (1 - x / 100.0)
            filled = sub["low_win"] < lim
            entry = np.where(filled, lim, sub["entry"])
            s = sub.assign(gain=(sub["entry"] / entry - 1.0) * 100.0)
            r = t_by_day(s, "gain")
            r["fill_rate"] = float(filled.mean()) if len(sub) else None
            # среди заливок: цена в 18:35 против лимита (неблагоприятный отбор)
            r["gain_when_filled_pct"] = float(s.loc[filled, "gain"].mean()) if filled.any() else None
            out[f"{cond}_X{x}"] = r
    return out


def holm(res_dev: dict, keys: list[tuple[str, str]]) -> dict:
    ps = [(res_dev[a][b].get("p", 1.0), f"{a}/{b}") for a, b in keys]
    ps.sort()
    out, run_p, m = {}, 0.0, len(ps)
    for i, (p, k) in enumerate(ps):
        run_p = max(run_p, min(1.0, (m - i) * p))
        out[k] = run_p
    return out


def run(conn) -> dict:
    res = {}
    for name, (d0, d1) in (("dev", DEV), ("holdout", HOLDOUT)):
        f = panel(load(conn, d0, d1))
        f = f[(f["d"] >= d0) & (f["d"] <= d1)]
        res[name] = {"A1_exit_delay": a1(f), "A2_limit_entry": a2(f),
                     "days": int(f["d"].nunique())}
        log.info("%s готов: %d дней", name, res[name]["days"])
    d = res["dev"]
    res["holm_dev"] = {
        "A1": holm(d["A1_exit_delay"], [(c, f"+{m}м") for c in ("gap_down", "gap_up") for m in EXIT_DELAYS]),
        "A2": holm({"x": d["A2_limit_entry"]}, [("x", k) for k in d["A2_limit_entry"]]),
    }
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
    with open(os.path.join(OUT_DIR, "timing_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 20,
                            "stage": "execution_timing_A1_A2", "trials": 10, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
