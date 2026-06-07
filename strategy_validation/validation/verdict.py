#!/usr/bin/env python3
"""
validation/verdict.py — Block 9 final verdict gate.

Runs the full statistical contour per (ticker, strategy) using ONLY the
validation/ layer (significance, white_rc, serial_dependence, entropy,
risk_metrics, probability_of_ruin, monte_carlo) plus a self-contained CSCV PBO.
P&L always from pnl_engine.pnl (single source).

Decision rule
  REJECTED if ANY of:
     White RC p > 0.05  OR  SPA p > 0.05  OR  PBO > 0.5  OR  fails BH-FDR
     OR ProbRuin(-30%) > 0.30  OR  Ljung-Box shows NO structure
     OR entropy near-random (permutation entropy > 0.99)
  CANDIDATE EDGE if ALL:
     White RC < 0.05 AND SPA < 0.05 AND BH-FDR passed AND PBO < 0.2
     AND Hurst stable (|H-0.5| >= 0.05, i.e. not a pure random walk)
     AND ProbRuin(-30%) < 0.10
  else WEAK / INCONCLUSIVE

White RC / SPA / PBO are SELECTION tests over the strategy menu on a ticker, so
they are computed once per ticker (shared by all its strategies). BH-FDR is applied
across the full (ticker x strategy) grid of per-strategy HAC-t p-values.

Usage:
  python3 -m validation.verdict                       # default 6 tickers, 2 strats
  python3 -m validation.verdict --tickers etln selg   --strats long_overnight intraday_short
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pnl_engine
import advanced_stats as adv

from . import significance as sig
from . import white_rc as wrc
from . import serial_dependence as sd
from . import entropy as ent
from . import risk_metrics as rm
from . import probability_of_ruin as por

MENU = ("intraday_short", "long_overnight", "short_hold", "long_intraday")
COST_RT = 0.08


# ── PBO via CSCV across the strategy menu (selection overfitting) ──────────────
def pbo_cscv(ret_matrix: np.ndarray, S: int = 10, seed: int = 42) -> float:
    """ret_matrix: (N_strat, T). Returns PBO in [0,1]. Sharpe as performance."""
    N, T = ret_matrix.shape
    chunk = T // S
    if N < 2 or chunk < 2:
        return float("nan")
    T_use = chunk * S
    R = ret_matrix[:, :T_use]
    chunks = [R[:, i * chunk:(i + 1) * chunk] for i in range(S)]
    S2 = S // 2
    combos = list(combinations(range(S), S2))
    rng = np.random.default_rng(seed)
    if len(combos) > 1000:
        combos = [combos[i] for i in rng.choice(len(combos), 1000, replace=False)]

    def sharpe(block):
        m = block.mean(axis=1)
        s = block.std(axis=1)
        s = np.where(s == 0, 1e-30, s)
        return m / s

    n_overfit = 0
    for c in combos:
        is_idx = set(c)
        oos_idx = [i for i in range(S) if i not in is_idx]
        is_block = np.concatenate([chunks[i] for i in c], axis=1)
        oos_block = np.concatenate([chunks[i] for i in oos_idx], axis=1)
        is_perf = sharpe(is_block)
        oos_perf = sharpe(oos_block)
        best = int(np.argmax(is_perf))
        # rank of best-IS strategy in OOS (fraction beaten); overfit if below median
        oos_rank = np.mean(oos_perf < oos_perf[best])   # fraction it beats
        if oos_rank < 0.5:
            n_overfit += 1
    return n_overfit / len(combos)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns boolean array: True = reject null (significant)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        out = np.zeros(n, bool)
    else:
        kmax = np.max(np.where(passed)[0])
        cutoff = ranked[kmax]
        out = p <= cutoff
    return out


def _returns(d, strat):
    _, _, ret, _ = pnl_engine.pnl(d, strat, COST_RT)
    # net per-period returns (cost charged per trade)
    return (ret - COST_RT).dropna().values


def ticker_selection_tests(d, strats, B=2000, seed=42):
    """White RC / SPA / PBO over the strategy menu present on this ticker."""
    rets = {}
    for s in MENU:
        try:
            r = _returns(d, s)
            if len(r) >= 30:
                rets[s] = r
        except Exception:
            pass
    names = list(rets.keys())
    min_T = min(len(v) for v in rets.values())
    R = np.vstack([rets[s][:min_T] for s in names])     # (N, T)
    wr = wrc.white_reality_check(R.T, B=B, seed=seed)   # expects (T, K)
    pbo = pbo_cscv(R)
    return {"white_rc_p": wr["white_rc_p"], "spa_p": wr["spa_p"], "pbo": pbo}


def run(tickers, strats, base_dir, B=2000):
    rows = []
    # pass 1: per-strategy stats + collect HAC p-values + per-ticker selection tests
    sel_cache = {}
    for tk in tickers:
        path = os.path.join(base_dir, f"{tk.lower()}_candles.csv")
        if not os.path.exists(path):
            print(f"  [skip] {tk}: no file", file=sys.stderr)
            continue
        d = adv.load_daily(path)
        sel = ticker_selection_tests(d, strats, B=B)
        sel_cache[tk] = sel
        for st in strats:
            r = _returns(d, st)
            if len(r) < 30:
                continue
            hac = sig.hac_t_test(r)
            sb = sig.stationary_bootstrap_test(r, B=1000)
            sgn = sig.sign_test(r)
            lb = sd.ljung_box(r)
            runs = sd.runs_test(r)
            hu = sd.hurst_exponent(r)
            er = ent.entropy_report(r)
            risk = rm.risk_report(r)
            ruin = por.prob_of_ruin(r, "block", n_sim=2000)
            rows.append({
                "ticker": tk, "strategy": st, "n": len(r), "net": r.sum(),
                "hac_p": hac["p_value"], "sb_p": sb["p_value"], "sign_p": sgn["p_value"],
                "white_rc_p": sel["white_rc_p"], "spa_p": sel["spa_p"], "pbo": sel["pbo"],
                "lb_struct": lb["any_structure"], "lb_p10": lb[10]["p_value"],
                "runs_p": runs["p_value"], "hurst": hu["hurst"],
                "perm_entropy": er["permutation"],
                "sortino": risk["sortino"], "calmar": risk["calmar"],
                "recovery": risk["recovery_factor"], "ruin30": ruin["ruin_30"],
            })

    # BH-FDR across the whole grid on HAC-t p-values
    pv = np.array([row["hac_p"] for row in rows])
    fdr_pass = bh_fdr(pv) if len(pv) else np.array([], bool)
    for row, fp in zip(rows, fdr_pass):
        row["fdr_pass"] = bool(fp)

    # verdicts
    for row in rows:
        row["verdict"] = classify(row)
    return rows


def classify(r):
    near_random_entropy = (r["perm_entropy"] == r["perm_entropy"]) and r["perm_entropy"] > 0.99
    no_structure = not r["lb_struct"]
    reject = (
        (r["white_rc_p"] > 0.05) or (r["spa_p"] > 0.05) or
        (r["pbo"] > 0.5) or (not r["fdr_pass"]) or
        (r["ruin30"] == r["ruin30"] and r["ruin30"] > 0.30) or
        no_structure or near_random_entropy
    )
    if reject:
        return "REJECTED"
    hurst_stable = (r["hurst"] == r["hurst"]) and abs(r["hurst"] - 0.5) >= 0.05
    candidate = (
        (r["white_rc_p"] < 0.05) and (r["spa_p"] < 0.05) and r["fdr_pass"] and
        (r["pbo"] < 0.2) and hurst_stable and
        (r["ruin30"] == r["ruin30"] and r["ruin30"] < 0.10)
    )
    return "CANDIDATE EDGE" if candidate else "WEAK / INCONCLUSIVE"


def print_report(rows):
    for r in rows:
        print("=" * 72)
        print(f"  {r['ticker']}  /  {r['strategy']}   (n={r['n']}, net={r['net']:+.1f}%)")
        print("-" * 72)
        print(f"  Significance: White RC p={r['white_rc_p']:.3f} / SPA p={r['spa_p']:.3f} / "
              f"BH-FDR {'PASS' if r['fdr_pass'] else 'FAIL'} (HAC-t p={r['hac_p']:.3f})")
        print(f"  Robustness:   PBO={r['pbo']:.3f} / MC(stat-bootstrap p)={r['sb_p']:.3f} / "
              f"Walk-Forward sign-p={r['sign_p']:.3f}")
        print(f"  Structure:    Hurst={r['hurst']:.3f} / Ljung-Box {'struct' if r['lb_struct'] else 'NONE'} "
              f"(p10={r['lb_p10']:.3f}) / Runs p={r['runs_p']:.3f} / PermEntropy={r['perm_entropy']:.3f}")
        print(f"  Risk:         Sortino={r['sortino']:.3f} / Calmar={r['calmar']:.4f} / "
              f"Recovery={r['recovery']:.3f} / ProbRuin(-30%)={r['ruin30']:.3f}")
        print(f"  Verdict:      {r['verdict']}")
    # summary table
    print("=" * 72)
    print("  SUMMARY")
    print(f"  {'ticker':<7}{'strategy':<17}{'WhiteRC':>8}{'SPA':>7}{'PBO':>7}"
          f"{'FDR':>5}{'Ruin30':>8}{'LB':>4}  verdict")
    counts = {"REJECTED": 0, "CANDIDATE EDGE": 0, "WEAK / INCONCLUSIVE": 0}
    for r in rows:
        counts[r["verdict"]] += 1
        print(f"  {r['ticker']:<7}{r['strategy']:<17}{r['white_rc_p']:>8.3f}{r['spa_p']:>7.3f}"
              f"{r['pbo']:>7.3f}{('Y' if r['fdr_pass'] else 'n'):>5}{r['ruin30']:>8.3f}"
              f"{('Y' if r['lb_struct'] else 'n'):>4}  {r['verdict']}")
    print("-" * 72)
    print(f"  REJECTED={counts['REJECTED']}  "
          f"CANDIDATE EDGE={counts['CANDIDATE EDGE']}  "
          f"WEAK={counts['WEAK / INCONCLUSIVE']}")
    return counts


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+",
                    default=["etln", "selg", "smlt", "mgnt", "alrs", "upro"])
    ap.add_argument("--strats", nargs="+",
                    default=["long_overnight", "intraday_short"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    # TGKA stays excluded by construction (not in default list).
    rows = run(a.tickers, a.strats, base, B=a.boot)
    print_report(rows)


if __name__ == "__main__":
    main()
