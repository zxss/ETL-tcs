"""
H-U1 — conformal-интервал как ворота сделки (волна 3).

ПРЕДРЕГИСТРАЦИЯ (до прогона, знак задан):
  Сделки, у которых conformal-интервал прогноза ночной доходности УЖЕ медианы
  дня, дают БОЛЕЕ ВЫСОКУЮ среднюю чистую доходность, чем сделки с широким
  интервалом. Ровно два испытания: (1) разность «узкие − широкие»,
  (2) корзина только из узких против полной корзины.

Почему это новое: все прежние фильтры программы работали по ТОЧЕЧНОМУ прогнозу
(порог сигнала, новостные ворота, реверсия). Фильтр по ШИРИНЕ интервала — то
есть по уверенности модели — не проверялся ни разу.

ГЕЙТ КАЧЕСТВА (выполняется первым): эмпирическое покрытие против номинала 80 %.
Если |покрытие − 0,80| > 0,05, интервалы не калиброваны, фильтр по ним
бессмысленен, и тест останавливается на этом шаге — так записано в плане.

Метод: скользящий split-conformal с адаптацией уровня (ACI, Gibbs & Candès):
точечная модель — GBM на расширяющемся окне, калибровка по последним CAL
остаткам вне обучения, α подстраивается по факту покрытия.

Запуск (на сервере): python -m research.uncertainty.conformal
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

import database                                               # noqa: E402
from research.validation import core as V                     # noqa: E402
from research.vol import overnight_base as OB                 # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave3")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
ALPHA = 0.20                       # номинальное покрытие 80 %
CAL, TRAIN_MIN, REFIT = 250, 500, 42
GAMMA = 0.01                       # шаг адаптации ACI
COVER_TOL = 0.05
PER_YEAR = 252.0
FEATS = ["rv21", "atr14", "ret_1d", "ret_5d", "ret_21d", "turn_z", "gap_prev", "dow"]


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    close = daily.pivot(index="d", columns="ticker", values="close").sort_index()
    open_ = daily.pivot(index="d", columns="ticker", values="open").sort_index()
    high = daily.pivot(index="d", columns="ticker", values="high").sort_index()
    low = daily.pivot(index="d", columns="ticker", values="low").sort_index()
    val = daily.pivot(index="d", columns="ticker", values="value").sort_index()
    r = close.pct_change()
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    feats = {
        "rv21": r.rolling(21, min_periods=15).std(ddof=1) * 100.0,
        "atr14": (tr.rolling(14, min_periods=10).mean() / close) * 100.0,
        "ret_1d": r * 100.0,
        "ret_5d": (close / close.shift(5) - 1.0) * 100.0,
        "ret_21d": (close / close.shift(21) - 1.0) * 100.0,
        "turn_z": (val - val.rolling(60, min_periods=40).mean())
                  / val.rolling(60, min_periods=40).std(ddof=1),
        "gap_prev": (open_ / close.shift(1) - 1.0) * 100.0,
    }
    long = None
    for name, fr in feats.items():
        s = fr.stack().rename(name)
        long = s.to_frame() if long is None else long.join(s, how="outer")
    long = long.reset_index().rename(columns={"level_0": "d", "level_1": "ticker"})
    long["dow"] = pd.to_datetime(long["d"]).dt.dayofweek
    return long


def run_sample(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = build_features(daily)
    df = trades.merge(f, on=["d", "ticker"], how="left").dropna(subset=FEATS)
    df = df.sort_values(["d", "ticker"]).reset_index(drop=True)
    days = sorted(df["d"].unique())
    if len(days) < TRAIN_MIN // 4:
        return {}

    df["pred"] = np.nan
    df["lo"] = np.nan
    df["hi"] = np.nan
    day_pos = {d: i for i, d in enumerate(days)}
    df["di"] = df["d"].map(day_pos)

    model = None
    alpha_t = ALPHA
    resid_hist: list[float] = []
    n_days = len(days)
    i = 0
    # обучение на первых TRAIN_MIN наблюдениях по дням
    start_day = max(60, int(n_days * 0.35))
    for k in range(start_day, n_days):
        te = df[df["di"] == k]
        if te.empty:
            continue
        if model is None or (k - start_day) % REFIT == 0:
            tr = df[df["di"] < k - 1]
            if len(tr) < TRAIN_MIN:
                continue
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                                  learning_rate=0.05,
                                                  min_samples_leaf=30, random_state=0)
            model.fit(tr[FEATS].to_numpy(float), tr["net_pct"].to_numpy(float))
        if model is None:
            continue
        p = model.predict(te[FEATS].to_numpy(float))
        q = (np.quantile(np.abs(resid_hist[-CAL:]), min(max(1.0 - alpha_t, 0.01), 0.999))
             if len(resid_hist) >= 50 else np.nan)
        df.loc[te.index, "pred"] = p
        if np.isfinite(q):
            df.loc[te.index, "lo"] = p - q
            df.loc[te.index, "hi"] = p + q
            cov = float(((te["net_pct"] >= p - q) & (te["net_pct"] <= p + q)).mean())
            alpha_t = float(np.clip(alpha_t + GAMMA * (ALPHA - (1.0 - cov)), 0.01, 0.5))
        resid_hist.extend(list(te["net_pct"].to_numpy(float) - p))
        i += 1

    ev = df.dropna(subset=["lo", "hi"]).copy()
    if len(ev) < 200:
        return {"n": int(len(ev)), "error": "мало наблюдений после разгона"}
    res: dict = {"n_obs": int(len(ev)), "n_days": int(ev["d"].nunique()),
                 "nominal_coverage": 1.0 - ALPHA,
                 "empirical_coverage": V.coverage(ev["net_pct"].to_numpy(),
                                                  ev["lo"].to_numpy(), ev["hi"].to_numpy()),
                 "mean_width_pct": float((ev["hi"] - ev["lo"]).mean())}
    res["gate_calibrated"] = bool(abs(res["empirical_coverage"] - (1.0 - ALPHA)) <= COVER_TOL)
    if not res["gate_calibrated"]:
        res["verdict"] = ("интервалы не калиброваны — фильтр по ширине не проверяется, "
                          "так записано в предрегистрации")
        return res

    ev["width"] = ev["hi"] - ev["lo"]
    ev["narrow"] = ev.groupby("d")["width"].transform(lambda s: s <= s.median())
    narrow = ev[ev["narrow"]].groupby("d")["net_pct"].mean()
    wide = ev[~ev["narrow"]].groupby("d")["net_pct"].mean()
    full = ev.groupby("d")["net_pct"].mean()
    common = narrow.index.intersection(wide.index)
    d = (narrow.reindex(common) - wide.reindex(common)).dropna()
    t1 = V.dm_test(-d.to_numpy(), np.zeros(len(d)))
    d2 = (narrow.reindex(full.index) - full).dropna()
    t2 = V.dm_test(-d2.to_numpy(), np.zeros(len(d2)))
    res["narrow_minus_wide"] = {"annual_pp": float(d.mean() * PER_YEAR), "n": int(len(d)),
                                "t": -t1["t"],
                                "p_one_sided": t1["p"] / 2.0 if t1["t"] < 0 else 1.0 - t1["p"] / 2.0}
    res["narrow_minus_full"] = {"annual_pp": float(d2.mean() * PER_YEAR), "n": int(len(d2)),
                                "t": -t2["t"],
                                "p_one_sided": t2["p"] / 2.0 if t2["t"] < 0 else 1.0 - t2["p"] / 2.0}
    for name, s in (("narrow", narrow), ("wide", wide), ("full", full)):
        res[name] = {"annual_pct": float(s.mean() * PER_YEAR),
                     "sharpe": float(s.mean() / s.std(ddof=1) * math.sqrt(PER_YEAR))
                     if s.std(ddof=1) > 0 else None}
    return res


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    conn = database.get_connection()
    out: dict = {"design": "ACI split-conformal, номинал 80 %, гейт покрытия ±5 пп",
                 "samples": {}}
    for sname, (table, d0, d1) in OB.SAMPLES.items():
        daily = OB.load_daily(conn, table, d0, d1)
        trades = OB.basket(daily)
        if trades.empty:
            continue
        r = run_sample(daily, trades)
        out["samples"][sname] = r
        print(f"\n=== {sname} ===")
        print(f"  покрытие {r.get('empirical_coverage', float('nan')):.3f} "
              f"(номинал {1 - ALPHA:.2f}), ширина {r.get('mean_width_pct', float('nan')):.2f} пп, "
              f"гейт {'ПРОЙДЕН' if r.get('gate_calibrated') else 'ПРОВАЛЕН'}")
        if r.get("gate_calibrated"):
            for k in ("narrow", "wide", "full"):
                print(f"  {k:7s} {r[k]['annual_pct']:+7.2f} %/год  Шарп {r[k]['sharpe']:+.2f}")
            for k in ("narrow_minus_wide", "narrow_minus_full"):
                v = r[k]
                print(f"  {k}: {v['annual_pp']:+.2f} пп/год, t {v['t']:+.2f}, p1 {v['p_one_sided']:.3f}")

    json.dump(out, open(os.path.join(OUT, "conformal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave3_conformal_HU1", "trials": 2,
                             "revision": "wave3"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'conformal.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
