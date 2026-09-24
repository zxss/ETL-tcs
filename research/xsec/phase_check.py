"""
Диагностика (НЕ новое испытание): устойчивость holdout-результата моментума к
фазе сетки ребалансов.

Повод: H-C3 воспроизвёл на том же топ-50 и том же окне 2024-05…2026-09
месячный эффект как +2,9 %/год (t +0,26), тогда как прогон 24.09 дал
−18,6 %/год (t −2,42). Оба считают одно и то же, но начинают сетку из 21 дня с
разного дня. Если вывод зависит от того, с какого дня начать считать месяцы,
то это свойство выборки, а не рынка.

Считаются все 21 возможных сдвига сетки. Ничего не выбирается и никуда не
подставляется — это измерение разброса.

Запуск (на сервере): python -m research.xsec.phase_check
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

from research.xsec import panel as P                          # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave2")
PERIODS = {"2014_2024": (dt.date(2014, 1, 1), dt.date(2024, 5, 20)),
           "2024_2026": (dt.date(2024, 5, 21), dt.date(2026, 9, 22))}
Q, COST = 0.2, 0.2


def legs_for_dates(data: dict, dates: list, lo: int, hi: int) -> pd.DataFrame:
    rows = []
    prev = {"mom_top": set(), "rev_bot": set()}
    for d in dates:
        tk = P.universe(data["liq"], d, lo, hi)
        if len(tk) < 20:
            continue
        f = P.characteristics(data, d, tk)
        if f.empty:
            continue
        y = P.forward_return(data, d, list(f.index))
        if y.empty:
            continue
        g = f.assign(ticker=f.index, y=y).dropna(subset=["y"])
        if len(g) < 20:
            continue
        g["y_rel"] = g["y"] - g["y"].mean()
        k = max(1, int(round(len(g) * Q)))
        rec = {"d": d}
        sel = {"mom_top": set(g.dropna(subset=["mom_12_1"])
                              .sort_values("mom_12_1", ascending=False)["ticker"].iloc[:k]),
               "rev_bot": set(g.dropna(subset=["ret_1m"])
                              .sort_values("ret_1m")["ticker"].iloc[:k])}
        for name, names in sel.items():
            turn = 1.0 if not prev[name] else 1.0 - len(names & prev[name]) / max(1, len(names))
            rec[name] = float(g[g["ticker"].isin(names)]["y_rel"].mean()) * 100.0 - COST * turn
            prev[name] = names
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    data = P.load()
    per_year = 252.0 / P.H
    res = {"note": "диагностика разброса по фазе сетки, не испытание", "periods": {}}
    for pname, (d0, d1) in PERIODS.items():
        days = [d for d in data["ret"].index if d0 <= d <= d1]
        block = {"mom_top": [], "rev_bot": [], "n_phases": 0}
        for phase in range(P.H):
            dates = days[phase::P.H]
            L = legs_for_dates(data, dates, 1, 50)
            if len(L) < 8:
                continue
            block["n_phases"] += 1
            for leg in ("mom_top", "rev_bot"):
                s = L[leg].dropna()
                t = float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))) if len(s) > 3 else float("nan")
                block[leg].append({"phase": phase, "n": int(len(s)),
                                   "annual_pct": float(s.mean() * per_year), "t": t})
        for leg in ("mom_top", "rev_bot"):
            a = np.array([x["annual_pct"] for x in block[leg]], float)
            tt = np.array([x["t"] for x in block[leg]], float)
            block[f"{leg}_summary"] = {
                "annual_min": float(a.min()), "annual_median": float(np.median(a)),
                "annual_max": float(a.max()), "t_min": float(tt.min()),
                "t_median": float(np.median(tt)), "t_max": float(tt.max()),
                "share_t_below_-2": float((tt <= -2.0).mean()),
                "share_t_above_2": float((tt >= 2.0).mean()),
                "sign_flips": bool(a.min() < 0 < a.max())}
        res["periods"][pname] = block
        print(f"\n=== {pname}: {block['n_phases']} фаз сетки ===")
        for leg in ("mom_top", "rev_bot"):
            s = block[f"{leg}_summary"]
            print(f"  {leg}: {s['annual_min']:+.1f} … {s['annual_median']:+.1f} … "
                  f"{s['annual_max']:+.1f} %/год；  t {s['t_min']:+.2f} … "
                  f"{s['t_median']:+.2f} … {s['t_max']:+.2f}；  "
                  f"смена знака: {'ДА' if s['sign_flips'] else 'нет'}； "
                  f"доля |t|>2: {(s['share_t_below_-2'] + s['share_t_above_2']):.2f}")

    json.dump(res, open(os.path.join(OUT, "phase_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n→ {os.path.join(OUT, 'phase_check.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
