"""
H-C3 — моментум на свежем срезе: бумаги 51–150 по обороту (волна 2).

ПРЕДРЕГИСТРАЦИЯ (до прогона, знаки заданы):
  H1: верхний квинтиль моментума 12-1 ЛУЧШЕ равновесного универса среза;
  H2: нижний квинтиль месячной доходности ХУЖЕ универса среза
      (месячный моментум, как на holdout топ-50).
Срез 51–150 ни разу не участвовал ни в одном из 263 испытаний программы —
это независимая кросс-секция того же периода, и она НЕ расходует S3.

ИЗДЕРЖКИ. T0.3 (оценка спреда всей доски по дневным H/L) провалил контроль
качества: Корвин–Шульц монотонен по ликвидности, но корреляция с замером по
5-минуткам 0,11, а уровень завышен вчетверо. Точечной оценке спреда для этих
бумаг доверять нельзя, поэтому вместо неё считается ЛЕСТНИЦА ЧУВСТВИТЕЛЬНОСТИ:
результат при издержках 1×, 2×, 3×, 5× от базовых 0,2 % за круг, и отдельно —
БЕЗУБЫТОЧНЫЙ МНОЖИТЕЛЬ издержек. Это честнее точечного числа: вывод не зависит
от того, чего мы не знаем.

Запуск (на сервере): python -m research.xsec.fresh_slice
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
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
SLICES = {"top50_control": (1, 50), "fresh_51_150": (51, 150)}
PERIODS = {"2014_2024": (dt.date(2014, 1, 1), dt.date(2024, 5, 20)),
           "2024_2026": (dt.date(2024, 5, 21), dt.date(2026, 9, 22))}
Q = 0.2
COST_BASE = 0.2
COST_MULT = (1.0, 2.0, 3.0, 5.0)


def tstat(s: pd.Series, per_year: float) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 8 or s.std(ddof=1) == 0:
        return {"n": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"n": n, "mean_pct": float(s.mean()), "annual_pct": float(s.mean() * per_year),
            "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "hit": float((s > 0).mean())}


def legs(pan: pd.DataFrame) -> pd.DataFrame:
    """Доходности ног (относительно универса среза) БЕЗ издержек + оборот."""
    rows = []
    prev = {"mom_top": set(), "rev_bot": set()}
    for d, g in pan.groupby("d"):
        g = g.dropna(subset=["y_rel"])
        if len(g) < 20:
            continue
        k = max(1, int(round(len(g) * Q)))
        rec = {"d": d, "uni_pct": float(g["y"].mean()) * 100.0, "n": len(g)}
        sel = {
            "mom_top": set(g.dropna(subset=["mom_12_1"])
                           .sort_values("mom_12_1", ascending=False)["ticker"].iloc[:k]),
            "rev_bot": set(g.dropna(subset=["ret_1m"])
                           .sort_values("ret_1m")["ticker"].iloc[:k]),
        }
        for name, names in sel.items():
            if not names:
                continue
            turn = 1.0 if not prev[name] else 1.0 - len(names & prev[name]) / max(1, len(names))
            rec[f"{name}_gross_pct"] = float(g[g["ticker"].isin(names)]["y_rel"].mean()) * 100.0
            rec[f"{name}_turn"] = turn
            prev[name] = names
        rows.append(rec)
    return pd.DataFrame(rows)


def breakeven(gross: pd.Series, turn: pd.Series) -> float:
    """Множитель базовых издержек, при котором среднее уходит в ноль."""
    g, t = gross.dropna(), turn.reindex(gross.dropna().index)
    denom = float((COST_BASE * t).mean())
    return float(g.mean() / denom) if denom > 0 else float("nan")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    data = P.load()
    per_year = 252.0 / P.H
    res: dict = {"design": "знаки предрегистрированы: mom_top > 0, rev_bot < 0",
                 "cost_base_pct": COST_BASE, "slices": {}}

    for sname, (lo, hi) in SLICES.items():
        res["slices"][sname] = {}
        for pname, (d0, d1) in PERIODS.items():
            pan = P.build(data, d0, d1, lo, hi)
            if pan.empty:
                continue
            L = legs(pan)
            if L.empty:
                continue
            block = {"rebalances": int(len(L)),
                     "universe_annual_pct": float(L["uni_pct"].mean() * per_year),
                     "median_names": float(L["n"].median())}
            for leg in ("mom_top", "rev_bot"):
                gc, tc = f"{leg}_gross_pct", f"{leg}_turn"
                if gc not in L:
                    continue
                item = {"gross": tstat(L[gc], per_year),
                        "mean_turnover": float(L[tc].mean()),
                        "breakeven_cost_mult": breakeven(L[gc], L[tc]), "net": {}}
                for m in COST_MULT:
                    net = L[gc] - COST_BASE * m * L[tc]
                    item["net"][f"x{m:g}"] = tstat(net, per_year)
                block[leg] = item
            res["slices"][sname][pname] = block

            print(f"\n=== {sname} / {pname}: {block['rebalances']} ребалансов, "
                  f"универс {block['universe_annual_pct']:+.1f} %/год ===")
            for leg in ("mom_top", "rev_bot"):
                if leg not in block:
                    continue
                it = block[leg]
                g = it["gross"]
                print(f"  {leg}: брутто {g.get('annual_pct', float('nan')):+.2f} %/год "
                      f"(t {g.get('t', float('nan')):+.2f}), оборот {it['mean_turnover']:.2f}, "
                      f"безубыточный множ. издержек {it['breakeven_cost_mult']:.1f}×")
                for m in COST_MULT:
                    n = it["net"][f"x{m:g}"]
                    print(f"      издержки ×{m:g}: {n.get('annual_pct', float('nan')):+.2f} %/год "
                          f"(t {n.get('t', float('nan')):+.2f})")

    json.dump(res, open(os.path.join(OUT, "fresh_slice.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave2_fresh_slice_HC3", "trials": 2,
                             "revision": "wave2"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'fresh_slice.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
