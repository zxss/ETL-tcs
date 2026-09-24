"""
H-R1 — режимная надстройка над ночной корзиной (волна 3).

ПРЕДРЕГИСТРАЦИЯ (до прогона, знаки заданы):
  Ночная корзина даёт ПОЛОЖИТЕЛЬНУЮ среднюю доходность в режиме низкой
  волатильности и НЕПОЛОЖИТЕЛЬНУЮ в режиме высокой; фильтр «не торговать в
  high-vol» УЛУЧШАЕТ Шарп после издержек.
  Режимов ровно два, признак один (реализованная волатильность равновесного
  индекса), модель одна (гауссова HMM, hmmlearn). Перебора нет — иначе выбор
  стратегии по режиму превращается в подгонку.

Протокол против подглядывания: HMM обучается на расширяющемся окне и
переобучается раз в 63 дня; режим на день D определяется фильтрацией по данным
ДО D включительно, решение о сделке принимается на следующий день.

Запуск (на сервере): python -m research.regime.switch
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
N_STATES, TRAIN_MIN, REFIT = 2, 250, 63
PER_YEAR = 252.0


def index_rv(daily: pd.DataFrame) -> pd.Series:
    """Реализованная волатильность равновесного индекса (21 день, %)."""
    close = daily.pivot(index="d", columns="ticker", values="close").sort_index()
    r = close.pct_change()
    idx = r.mean(axis=1)
    return (idx.rolling(21, min_periods=15).std(ddof=1) * 100.0).dropna()


def regimes(rv: pd.Series) -> pd.Series:
    """Онлайн-классификация: обучение на прошлом, фильтрация до текущего дня."""
    from hmmlearn.hmm import GaussianHMM
    x = np.log(rv.to_numpy(float)).reshape(-1, 1)
    days = list(rv.index)
    out = pd.Series(index=rv.index, dtype=float)
    start = TRAIN_MIN
    while start < len(days):
        tr = x[:start]
        try:
            m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=200, random_state=0)
            m.fit(tr)
        except Exception:                                      # noqa: BLE001
            start += REFIT
            continue
        hi = int(np.argmax(m.means_.ravel()))                  # состояние высокой воли
        end = min(start + REFIT, len(days))
        for i in range(start, end):
            st = m.predict(x[: i + 1])[-1]                     # фильтрация по прошлому
            out.iloc[i] = 1.0 if st == hi else 0.0
        start += REFIT
    return out.dropna()


def tstats(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 8 or s.std(ddof=1) == 0:
        return {"n": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    return {"n": n, "mean_pct": float(s.mean()), "annual_pct": float(s.mean() * PER_YEAR),
            "sharpe": float(s.mean() / s.std(ddof=1) * math.sqrt(PER_YEAR)), "t": t}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    conn = database.get_connection()
    res: dict = {"design": "2 режима, признак — RV равновесного индекса, HMM",
                 "samples": {}}
    for sname, (table, d0, d1) in OB.SAMPLES.items():
        daily = OB.load_daily(conn, table, d0, d1)
        trades = OB.basket(daily)
        if trades.empty:
            continue
        base = trades.groupby("d")["net_pct"].mean().sort_index()
        rv = index_rv(daily)
        reg = regimes(rv)
        # решение на D+1 по режиму дня D
        reg_lag = reg.shift(1).dropna()
        common = base.index.intersection(reg_lag.index)
        b = base.reindex(common)
        r = reg_lag.reindex(common)
        low, high = b[r == 0.0], b[r == 1.0]
        filt = b.where(r == 0.0, 0.0)                          # в high-vol не торгуем

        block = {"share_high": float((r == 1.0).mean()),
                 "low_vol": tstats(low), "high_vol": tstats(high),
                 "base": tstats(b), "filtered": tstats(filt)}
        d = (filt - b).dropna()
        tt = V.dm_test(-d.to_numpy(), np.zeros(len(d)))
        block["filtered_minus_base"] = {
            "annual_pp": float(d.mean() * PER_YEAR), "t": -tt["t"],
            "p_one_sided": tt["p"] / 2.0 if tt["t"] < 0 else 1.0 - tt["p"] / 2.0}
        gap = V.dm_test(low.to_numpy(), np.full(len(low), float(high.mean())))
        block["low_minus_high"] = {"pp": float(low.mean() - high.mean()),
                                   "t": gap["t"], "p": gap["p"]}
        res["samples"][sname] = block

        print(f"\n=== {sname}: доля high-vol дней {block['share_high']:.2f} ===")
        for k in ("base", "low_vol", "high_vol", "filtered"):
            v = block[k]
            print(f"  {k:9s} {v.get('annual_pct', float('nan')):+7.2f} %/год  "
                  f"Шарп {v.get('sharpe', float('nan')):+.2f}  t {v.get('t', float('nan')):+.2f}  n={v.get('n')}")
        fb = block["filtered_minus_base"]
        print(f"  фильтр − база: {fb['annual_pp']:+.2f} пп/год, t {fb['t']:+.2f}, "
              f"p1 {fb['p_one_sided']:.3f}")

    json.dump(res, open(os.path.join(OUT, "regime.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave3_regime_HR1", "trials": 2,
                             "revision": "wave3"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'regime.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
