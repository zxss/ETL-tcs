"""
H-V1 — HAR-RV как базовая модель волатильности (план 04_plan, волна 1).

ПРЕДРЕГИСТРАЦИЯ (записана до прогона, коммит be59a8c+):
  HAR-RV, оценённая на 5-минутках, даёт МЕНЬШИЙ QLIKE, чем GARCH(1,1) и чем
  ATR-прокси, на горизонтах 1 и 5 дней. Знак задан: улучшение, а не «отличие».
  Спецификаций ровно три, перебора нет:
    HAR      RV_{t+1} = c + b_d·RV_d + b_w·RV_w + b_m·RV_m
    HAR-J    + b_j·J_d,  J = max(RV − BPV, 0)        (скачки)
    HAR-lev  + b_l·min(r_t, 0)                        (леверидж)
  Бенчмарки: GARCH(1,1) на дневных доходностях; ATR-прокси (OLS RV на ATR²).
  Критерий: значимое улучшение QLIKE (Диболд–Мариано, Холм) на dev И на holdout.

Выборки:
  dev      research_bars_5m  2022-01-03 … 2024-05-20  (для волатильности НЕ трогали)
  holdout  market_data_5m    2024-05-21 … 2026-09-24  (по доходностям трогали,
           по RV — нет; целевая переменная другая)

Единицы: RV в %² за день по основной сессии 10:00–18:40 (вечерняя сессия и
ночной гэп исключены — это мир торгового контура r4). Прогноз GARCH приводится
к тем же единицам множителем mean(RV)/mean(σ̂²), оценённым НА ОБУЧЕНИИ, иначе
сравнивались бы разные величины (GARCH меряет close-to-close с гэпом).

Запуск (на сервере): python -m research.vol.har
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import database                                              # noqa: E402
from research.validation import core as V                    # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave1")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")

SAMPLES = {
    "dev_2022_2024": ("research_bars_5m", dt.date(2022, 1, 3), dt.date(2024, 5, 20)),
    "holdout_2024_2026": ("market_data_5m", dt.date(2024, 5, 21), dt.date(2026, 9, 24)),
}
SESSION = ("10:00", "18:40")
MIN_BARS = 60
HORIZONS = (1, 5)
TRAIN_MIN, REFIT, EMBARGO = 250, 21, 1
SPECS = ("HAR", "HAR-J", "HAR-lev")
BENCH = ("GARCH", "ATR")

RV_SQL = """
WITH b AS (
  SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close::float8 AS c
  FROM {table}
  WHERE close > 0 AND ts >= %(f)s AND ts < %(t)s
    AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN %(s0)s AND %(s1)s
), r AS (
  SELECT ticker, tm::date AS d, tm, c,
         ln(c / NULLIF(lag(c) OVER (PARTITION BY ticker, tm::date ORDER BY tm), 0)) * 100.0 AS ret
  FROM b
), r2 AS (
  SELECT ticker, d, tm, ret,
         lag(ret) OVER (PARTITION BY ticker, d ORDER BY tm) AS ret1
  FROM r
)
SELECT ticker, d, count(ret) AS nbars, sum(ret * ret) AS rv,
       sum(abs(ret) * abs(ret1)) AS bp
FROM r2 WHERE ret IS NOT NULL GROUP BY ticker, d ORDER BY ticker, d
"""

DAILY_SQL = """
SELECT DISTINCT ON (ticker, (tm)::date) ticker, (tm)::date AS d, c AS close,
       hi AS high, lo AS low
FROM (
  SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, close::float8 AS c,
         max(high::float8) OVER (PARTITION BY ticker, (ts AT TIME ZONE 'Europe/Moscow')::date) AS hi,
         min(low::float8)  OVER (PARTITION BY ticker, (ts AT TIME ZONE 'Europe/Moscow')::date) AS lo
  FROM {table}
  WHERE close > 0 AND ts >= %(f)s AND ts < %(t)s
    AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN %(s0)s AND %(s1)s
) x
ORDER BY ticker, (tm)::date, tm DESC
"""


# ── данные ───────────────────────────────────────────────────────────────────

def load_sample(conn, table: str, d0: dt.date, d1: dt.date) -> pd.DataFrame:
    p = {"f": f"{d0} 00:00+03", "t": f"{d1 + dt.timedelta(days=1)} 00:00+03",
         "s0": SESSION[0], "s1": SESSION[1]}
    rv = pd.read_sql(RV_SQL.format(table=table), conn, params=p)
    dl = pd.read_sql(DAILY_SQL.format(table=table), conn, params=p)
    df = rv.merge(dl, on=["ticker", "d"], how="inner")
    df = df[df["nbars"] >= MIN_BARS].copy()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    df["rv"] = df["rv"].astype(float)
    # BPV с поправкой на число наблюдений; скачковая компонента
    n = df["nbars"].astype(float)
    df["bpv"] = (math.pi / 2.0) * df["bp"].astype(float) * n / (n - 1.0).clip(lower=1.0)
    df["jump"] = (df["rv"] - df["bpv"]).clip(lower=0.0)
    df = df.sort_values(["ticker", "d"]).reset_index(drop=True)
    df["ret"] = df.groupby("ticker")["close"].pct_change() * 100.0
    # ATR (14) в % от цены
    prev_c = df.groupby("ticker")["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr_pct"] = (tr / df["close"] * 100.0)
    df["atr14"] = df.groupby("ticker")["atr_pct"].transform(
        lambda s: s.rolling(14, min_periods=10).mean())
    return df


def features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("d").copy()
    rv = g["rv"]
    g["rv_d"] = rv
    g["rv_w"] = rv.rolling(5, min_periods=5).mean()
    g["rv_m"] = rv.rolling(22, min_periods=22).mean()
    g["j_d"] = g["jump"]
    g["neg_ret"] = g["ret"].clip(upper=0.0)
    for h in HORIZONS:
        g[f"y{h}"] = (rv.shift(-1).rolling(h, min_periods=h).mean().shift(-(h - 1))
                      if h > 1 else rv.shift(-1))
    return g


# ── модели ───────────────────────────────────────────────────────────────────

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _design(g: pd.DataFrame, spec: str) -> np.ndarray:
    cols = ["rv_d", "rv_w", "rv_m"]
    if spec == "HAR-J":
        cols += ["j_d"]
    elif spec == "HAR-lev":
        cols += ["neg_ret"]
    X = g[cols].to_numpy(float)
    return np.column_stack([np.ones(len(X)), X])


def _sigmoid(x: float) -> float:
    """Устойчивая логистическая функция: оптимизатор уходит в ±700 и ломает exp."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


def garch11_fit(r: np.ndarray) -> tuple[float, float, float]:
    """MLE GARCH(1,1) на дневных доходностях (%). Возвращает (ω, α, β)."""
    from scipy.optimize import minimize
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return (float(np.var(r)) if len(r) else 1.0, 0.0, 0.0)
    var0 = float(np.var(r))

    def nll(p):
        w, a, b = math.exp(min(p[0], 50.0)), _sigmoid(p[1]), _sigmoid(p[2])
        if a + b >= 0.999:
            b = 0.999 - a
        s = var0
        out = 0.0
        for x in r:
            out += math.log(s) + x * x / s
            s = w + a * x * x + b * s
            if not np.isfinite(s) or s <= 1e-12:
                return 1e12
        return out

    best, bp = 1e18, None
    for a0, b0 in ((0.1, 0.85), (0.05, 0.9)):
        p0 = [math.log(max(var0 * (1 - a0 - b0), 1e-6)),
              math.log(a0 / (1 - a0)), math.log(b0 / (1 - b0))]
        try:
            res = minimize(nll, p0, method="Nelder-Mead",
                           options={"maxiter": 600, "fatol": 1e-4, "xatol": 1e-4})
            if res.fun < best:
                best, bp = float(res.fun), res.x
        except Exception:                                      # noqa: BLE001
            continue
    if bp is None:
        return (var0, 0.0, 0.0)
    w = math.exp(min(float(bp[0]), 50.0))
    a = _sigmoid(float(bp[1]))
    b = _sigmoid(float(bp[2]))
    if a + b >= 0.999:
        b = 0.999 - a
    return (w, a, b)


def garch_path(r: np.ndarray, w: float, a: float, b: float) -> np.ndarray:
    """Условная дисперсия σ²_t для каждого t (прогноз на t по данным до t−1)."""
    r = np.asarray(r, float)
    s = np.full(len(r), np.nan)
    cur = float(np.nanvar(r[:60])) if len(r) >= 60 else float(np.nanvar(r) or 1.0)
    for i, x in enumerate(r):
        s[i] = cur
        xx = x * x if np.isfinite(x) else cur
        cur = w + a * xx + b * cur
        if not np.isfinite(cur) or cur <= 1e-12:
            cur = max(float(np.nanmean(s[: i + 1])), 1e-6)
    return s


# ── один тикер: walk-forward прогнозы всех моделей ───────────────────────────

def forecasts_for_ticker(g: pd.DataFrame, h: int) -> pd.DataFrame:
    g = g.dropna(subset=["rv_d", "rv_w", "rv_m", f"y{h}"]).reset_index(drop=True)
    n = len(g)
    if n < TRAIN_MIN + EMBARGO + REFIT:
        return pd.DataFrame()
    y = g[f"y{h}"].to_numpy(float)
    out = {k: np.full(n, np.nan) for k in (*SPECS, *BENCH)}
    rets = g["ret"].to_numpy(float)
    atr2 = (g["atr14"].to_numpy(float)) ** 2

    start = TRAIN_MIN
    while start + EMBARGO < n:
        tr = np.arange(0, start - h)                     # хвост обучения обрезан на горизонт
        te = np.arange(start + EMBARGO, min(start + EMBARGO + REFIT, n))
        if len(tr) < TRAIN_MIN // 2 or len(te) == 0:
            break
        for spec in SPECS:
            X = _design(g, spec)
            ok = np.isfinite(X[tr]).all(axis=1) & np.isfinite(y[tr])
            if ok.sum() < 60:
                continue
            beta = _ols(X[tr][ok], y[tr][ok])
            pred = X[te] @ beta
            out[spec][te] = np.where(np.isfinite(pred), np.maximum(pred, 1e-6), np.nan)
        # GARCH: подгон на обучении, путь по всему ряду, масштаб к RV на обучении
        w, a, b = garch11_fit(rets[tr])
        s2 = garch_path(rets, w, a, b)
        with np.errstate(invalid="ignore"):
            scale = np.nanmean(g["rv"].to_numpy(float)[tr]) / np.nanmean(s2[tr])
        if np.isfinite(scale) and scale > 0:
            out["GARCH"][te] = np.maximum(s2[te] * scale, 1e-6)
        # ATR-прокси: OLS RV_{t+h} ~ a + b·ATR²  (честная калибровка масштаба)
        Xa = np.column_stack([np.ones(n), atr2])
        ok = np.isfinite(Xa[tr]).all(axis=1) & np.isfinite(y[tr])
        if ok.sum() >= 60:
            ba = _ols(Xa[tr][ok], y[tr][ok])
            pa = Xa[te] @ ba
            out["ATR"][te] = np.where(np.isfinite(pa), np.maximum(pa, 1e-6), np.nan)
        start += REFIT

    res = pd.DataFrame({"ticker": g["ticker"], "d": g["d"], "y": y})
    for k, v in out.items():
        res[k] = v
    return res.dropna(subset=["y"])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    conn = database.get_connection()
    results: dict = {"preregistered": "HAR < QLIKE(GARCH) и QLIKE(ATR), h=1 и h=5",
                     "samples": {}}
    for sname, (table, d0, d1) in SAMPLES.items():
        raw = load_sample(conn, table, d0, d1)
        print(f"[{sname}] дней-тикеров: {len(raw)}, бумаг: {raw['ticker'].nunique()}, "
              f"{raw['d'].min()}…{raw['d'].max()}")
        feats = raw.groupby("ticker", group_keys=False).apply(features)
        block: dict = {"tickers": int(raw["ticker"].nunique()),
                       "obs": int(len(raw)), "horizons": {}}
        for h in HORIZONS:
            parts = []
            for tk, g in feats.groupby("ticker"):
                f = forecasts_for_ticker(g, h)
                if len(f):
                    parts.append(f)
            if not parts:
                continue
            fc = pd.concat(parts, ignore_index=True)
            cols = [*SPECS, *BENCH]
            fc = fc.dropna(subset=cols)
            hb = {"n_obs": int(len(fc)), "n_tickers": int(fc["ticker"].nunique()),
                  "qlike_mean": {}, "mse_mean": {}, "dm": {}}
            losses = {}
            for m in cols:
                q = V.qlike(fc["y"].to_numpy(), fc[m].to_numpy())
                losses[m] = q
                hb["qlike_mean"][m] = float(np.nanmean(q))
                hb["mse_mean"][m] = float(np.nanmean(V.mse(fc["y"].to_numpy(),
                                                           fc[m].to_numpy())))
            # DM по дневным средним разностям потерь (кластеризация по датам)
            pvals = {}
            for spec in SPECS:
                for bm in BENCH:
                    d = pd.DataFrame({"d": fc["d"], "diff": losses[spec] - losses[bm]})
                    daily = d.groupby("d")["diff"].mean().dropna()
                    t = V.dm_test(daily.to_numpy(), np.zeros(len(daily)))
                    # односторонняя: улучшение = отрицательная разность потерь
                    p1 = t["p"] / 2.0 if t["t"] < 0 else 1.0 - t["p"] / 2.0
                    hb["dm"][f"{spec}_vs_{bm}"] = {"t": t["t"], "p_one_sided": p1,
                                                   "n_days": int(len(daily)),
                                                   "mean_qlike_diff": t["mean_diff"]}
                    pvals[f"{spec}_vs_{bm}"] = p1
            hb["holm"] = V.holm(pvals)
            block["horizons"][f"h{h}"] = hb
            print(f"  h={h}: " + "  ".join(f"{m} QLIKE={hb['qlike_mean'][m]:.4f}"
                                           for m in cols))
            for k, v in hb["dm"].items():
                print(f"     {k:18s} t={v['t']:+.2f}  p1={v['p_one_sided']:.4f}  "
                      f"Холм={hb['holm'][k]['p_adj']:.4f} "
                      f"{'ПРОШЁЛ' if hb['holm'][k]['reject'] and v['t'] < 0 else ''}")
        results["samples"][sname] = block

    path = os.path.join(OUT, "har_results.json")
    json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave1_har_rv_HV1",
                             "trials": 3, "revision": open(
                                 os.path.join(ROOT, "REVISION")).read().strip()
                             if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"},
                            ensure_ascii=False) + "\n")
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
