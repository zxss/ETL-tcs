"""
T0.1 — Deflated Sharpe Ratio по всей программе (план 04_plan, волна 0).

Вопрос: моментум выбран из 263 испытаний. Какой Шарп нужен был, чтобы он не был
артефактом перебора? Считаем ожидаемый максимум Шарпа под нулём при N=263
независимых испытаний (Bailey & López de Prado) и DSR наблюдённых вариантов
моментум-портфеля из research/longhist/portfolio.py.

Это НЕ новое испытание реестра — переоценка уже полученного результата.
Запуск (на сервере): python -m research.wave0.dsr_program
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.validation import core as V                       # noqa: E402

LONGHIST = os.path.join(ROOT, "audit", "r4_research", "longhist")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
OUT = os.path.join(ROOT, "audit", "r4_research", "wave0")
VARIANTS = ("LO", "LO_X", "HEDGE", "HEDGE_UNI", "LS_STOCKS")


def program_trials() -> int:
    n = 0
    with open(TRIALS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                n += int(json.loads(line).get("trials", 0))
    return n


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    n_trials = program_trials()
    pf = json.load(open(os.path.join(LONGHIST, "portfolio_results.json"), encoding="utf-8"))

    res = {"n_trials_program": n_trials, "samples": {}}
    for sample in ("long_2014_2024", "holdout_2024_2026"):
        s = pf[sample]
        rows = s["rows"]
        mm = np.array([r.get("mm", np.nan) for r in rows], float)   # денежный рынок
        block = {"rebalances": len(rows)}
        for v in VARIANTS:
            r = np.array([row.get(v, np.nan) for row in rows], float)
            excess = r - mm                      # избыток над денежным рынком
            d_raw = V.deflated_sharpe(r, n_trials)
            d_exc = V.deflated_sharpe(excess, n_trials)
            block[v] = {"raw": d_raw, "excess_over_money": d_exc}
        # сколько Шарпа требует перебор: берём var_sr от главного варианта
        block["sr_star_hedge_excess"] = block["HEDGE"]["excess_over_money"]["sr_star"]
        res["samples"][sample] = block

    # справочно: порог Бонферрони по программе
    from scipy.stats import norm
    res["bonferroni"] = {"n": n_trials, "alpha": 0.05,
                         "two_sided_t": float(norm.ppf(1.0 - 0.05 / (2 * n_trials)))}

    path = os.path.join(OUT, "dsr_program.json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"испытаний в программе: {n_trials}")
    print(f"порог Бонферрони |t| ≥ {res['bonferroni']['two_sided_t']:.2f}")
    for sample, block in res["samples"].items():
        print(f"\n=== {sample} ({block['rebalances']} ребалансов) ===")
        for v in VARIANTS:
            e = block[v]["excess_over_money"]
            print(f"  {v:10s} SR={e['sr']:+.3f} (год {e['sr_annual']:+.2f})  "
                  f"SR*={e['sr_star']:.3f}  DSR={e['dsr']:.3f}")
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
