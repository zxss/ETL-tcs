"""
H-V2 — сайзинг по прогнозу волатильности (волна 3).

ПРЕДРЕГИСТРАЦИЯ (до прогона, знак задан):
  Позиция ∝ 1/σ̂ (прогноз HAR) даёт БОЛЕЕ ВЫСОКИЙ Шарп ночной корзины, чем
  фиксированная сумма на бумагу. Ровно два варианта:
    XS  — кросс-секционный перевес внутри корзины при неизменной сумме
          экспозиции (издержки те же, сравнение чистое);
    TS  — временной таргет волатильности: общая экспозиция ∝ 1/σ̂_портфеля,
          нормировка на среднее плечо 1,0 (издержки меняются вместе с оборотом).

ОГОВОРКА, записанная заранее: риск-сайзинг по ATR уже проверялся 17.09.2026
(блок boost-тестов) и дал ≤ +1,8 п.п. незначимо. Здесь меняется только оценка
волатильности — на лучшую. Априорная вероятность успеха низкая; тест берётся
потому, что он почти бесплатен поверх H-V1.

Критерий: Deflated Sharpe выше базового И прирост Шарпа значим (по разности
доходностей, Ньюи–Уэст), на dev И на holdout.

Запуск (на сервере): python -m research.vol.sizing
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
from research.vol import har as H                             # noqa: E402
from research.vol import overnight_base as OB                 # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave3")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
PER_YEAR = 252.0
MAX_LEV, MIN_LEV = 2.0, 0.25


def har_sigma(conn, table: str, d0: dt.date, d1: dt.date) -> pd.DataFrame:
    """Прогноз HAR на 1 день вперёд по каждой бумаге и дате (walk-forward)."""
    raw = H.load_sample(conn, table, d0, d1)
    feats = raw.groupby("ticker", group_keys=False).apply(H.features)
    parts = []
    for tk, g in feats.groupby("ticker"):
        f = H.forecasts_for_ticker(g, 1)
        if len(f):
            parts.append(f[["ticker", "d", "HAR"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"HAR": "rv_hat"})
    out["sigma"] = np.sqrt(out["rv_hat"].clip(lower=1e-6))
    return out


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * math.sqrt(PER_YEAR)) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")


def stats(x: pd.Series, n_trials: int = 281) -> dict:
    x = x.dropna()
    d = V.deflated_sharpe(x.to_numpy() / 100.0, n_trials)
    return {"n": int(len(x)), "mean_pct": float(x.mean()),
            "annual_pct": float(x.mean() * PER_YEAR), "sharpe": sharpe(x),
            "dsr": d["dsr"], "sr_star": d["sr_star"]}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    conn = database.get_connection()
    res: dict = {"design": "XS (перевес при равной экспозиции), TS (таргет вола)",
                 "samples": {}}
    for sname, (table, d0, d1) in OB.SAMPLES.items():
        daily = OB.load_daily(conn, table, d0, d1)
        trades = OB.basket(daily)
        sig = har_sigma(conn, table, d0, d1)
        if trades.empty or sig.empty:
            print(f"[{sname}] нет данных")
            continue
        t = trades.merge(sig[["ticker", "d", "sigma"]], on=["ticker", "d"], how="left")
        t = t.dropna(subset=["sigma"])
        if t.empty:
            print(f"[{sname}] нет пересечения с прогнозом HAR")
            continue

        base = t.groupby("d")["net_pct"].mean().sort_index()
        # XS: веса ∝ 1/σ̂, сумма весов = 1 (та же экспозиция, те же издержки)
        t["w"] = 1.0 / t["sigma"]
        xs = t.groupby("d").apply(
            lambda g: float((g["net_pct"] * g["w"]).sum() / g["w"].sum())).sort_index()
        # TS: плечо ∝ 1/σ̂_портфеля, нормировка на среднее плечо 1 по ПРОШЛОМУ
        sig_p = t.groupby("d")["sigma"].mean().sort_index()
        inv = (1.0 / sig_p)
        norm = inv.expanding(min_periods=60).mean()
        lev = (inv / norm).clip(MIN_LEV, MAX_LEV)
        extra_turn = (lev - lev.shift(1)).abs().fillna(0.0)
        # доп. издержки только на изменение экспозиции (круг 0,128 % на оборот)
        ts = base.reindex(lev.index) * lev - 0.128 * extra_turn
        ts = ts.dropna()

        block = {"base": stats(base), "XS": stats(xs.reindex(base.index).dropna()),
                 "TS": stats(ts)}
        for k in ("XS", "TS"):
            s = (xs if k == "XS" else ts).reindex(base.index)
            d = (s - base).dropna()
            tt = V.dm_test(-d.to_numpy(), np.zeros(len(d)))
            block[f"{k}_minus_base"] = {
                "mean_pp": float(d.mean()), "annual_pp": float(d.mean() * PER_YEAR),
                "t": -tt["t"], "p_one_sided": (tt["p"] / 2.0 if tt["t"] < 0
                                               else 1.0 - tt["p"] / 2.0)}
        res["samples"][sname] = block
        print(f"\n=== {sname} ===")
        for k in ("base", "XS", "TS"):
            b = block[k]
            print(f"  {k:5s} {b['annual_pct']:+7.2f} %/год  Шарп {b['sharpe']:+.2f}  "
                  f"DSR {b['dsr']:.3f}  (n={b['n']})")
        for k in ("XS", "TS"):
            v = block[f"{k}_minus_base"]
            print(f"    {k} − база: {v['annual_pp']:+.2f} пп/год, t {v['t']:+.2f}, "
                  f"p1 {v['p_one_sided']:.3f}")

    json.dump(res, open(os.path.join(OUT, "sizing.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave3_vol_sizing_HV2", "trials": 2,
                             "revision": "wave3"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'sizing.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
