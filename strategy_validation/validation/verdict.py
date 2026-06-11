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
from . import walk_forward as wf

MENU = ("intraday_short", "long_overnight", "short_hold", "long_intraday")
COST_RT = 0.08

# PR-2: каждая стратегия = (примитив, знак). intraday_short и long_intraday —
# ТОЧНЫЕ зеркала (corr=-1) одного примитива 'intraday'; short_hold — зеркало
# 'total'. BH-FDR должен считать КАЖДЫЙ примитив один раз, иначе семейство
# раздувается гарантированно-нулевыми зеркалами и поправка становится неоправданно
# строгой. Поэтому решение FDR принимается на уровне (ticker, primitive), а не
# (ticker, strategy).
PRIMITIVE = {
    "intraday_short": ("intraday", -1.0),
    "long_intraday":  ("intraday", +1.0),
    "intraday_long":  ("intraday", +1.0),
    "long_overnight": ("overnight", +1.0),
    "short_hold":     ("total", -1.0),
}
_PRIM_COL = {"intraday": "intraday", "overnight": "overnight", "total": "total"}


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


def _gross_returns(d, strat):
    """Per-trade GROSS returns of a strategy (no cost)."""
    _, _, ret, _ = pnl_engine.pnl(d, strat, 0.0)
    return ret.dropna().values


def _net_returns(d, strat, cost_rt):
    """Per-trade NET returns (cost charged once per round-trip trade)."""
    return _gross_returns(d, strat) - cost_rt


def primitive_two_sided_p(d, primitive):
    """PR-2: two-sided HAC-t p that the primitive's GROSS daily mean != 0.

    One p per (ticker, primitive) — charges for searching BOTH directions
    (long/short) without padding the FDR family with mirror nulls. Tests
    statistical structure (direction-agnostic); cost/tradeability is handled by
    the break-even and OOS gates, not here.
    """
    col = _PRIM_COL.get(primitive)
    if col is None or col not in d:
        return float("nan")
    base = d[col].dropna().values
    if len(base) < 30:
        return float("nan")
    p_plus = sig.hac_t_test(base)["p_value"]          # one-sided mean > 0
    return float(2.0 * min(p_plus, 1.0 - p_plus))


def ticker_selection_tests(d, strats, cost_rt, B=2000, seed=42):
    """White RC / SPA / PBO over the strategy menu present on this ticker.

    Both directions of a primitive ARE part of the genuine snooping universe
    (the researcher tried long AND short), so keeping mirror strategies here is
    correct for White RC's max-statistic — it is the FDR family (not White RC)
    that must dedupe mirrors. Net-of-cost returns are used."""
    rets = {}
    for s in MENU:
        try:
            r = _net_returns(d, s, cost_rt)
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


def _cost_for(tk, cost_rt, cost_map):
    return float((cost_map or {}).get(tk.upper(), cost_rt))


def run(tickers, strats, base_dir, B=2000, cost_rt=COST_RT, cost_map=None,
        winsorize=False, universe_tickers=None, oos_fraction=0.30):
    """Statistical contour with realistic costs (PR-1), honest multiple-testing
    family (PR-2), redesigned economic gate (PR-3) and walk-forward OOS (PR-5).

    universe_tickers: superset of tickers over which the BH-FDR family is built
        (data-snooping correction for ticker SELECTION). Defaults to `tickers`.
    """
    rows = []
    # ── PR-2: FDR family is the FULL search universe, one entry per
    #    (ticker, primitive) using two-sided gross p (mirror-deduped). ──────────
    prims = sorted({PRIMITIVE[s][0] for s in strats if s in PRIMITIVE})
    fam_keys, fam_p = [], []
    universe = universe_tickers or tickers
    for utk in universe:
        path = os.path.join(base_dir, f"{utk.lower()}_candles.csv")
        if not os.path.exists(path):
            continue
        du = adv.load_daily(path, winsorize=winsorize)
        for pr in prims:
            p2 = primitive_two_sided_p(du, pr)
            if p2 == p2:                                  # not NaN
                fam_keys.append((utk.upper(), pr))
                fam_p.append(p2)
    fam_pass = bh_fdr(fam_p) if fam_p else np.array([], bool)
    fdr_decision = {k: bool(v) for k, v in zip(fam_keys, fam_pass)}

    # pass 1: per-strategy stats + per-ticker selection tests (report tickers only)
    for tk in tickers:
        path = os.path.join(base_dir, f"{tk.lower()}_candles.csv")
        if not os.path.exists(path):
            print(f"  [skip] {tk}: no file", file=sys.stderr)
            continue
        d = adv.load_daily(path, winsorize=winsorize)
        ck = _cost_for(tk, cost_rt, cost_map)
        sel = ticker_selection_tests(d, strats, ck, B=B)
        for st in strats:
            gross = _gross_returns(d, st)
            r = gross - ck                                # net per trade
            if len(r) < 30:
                continue
            be_cost = float(gross.mean())                 # PR-1: break-even RT cost
            prim = PRIMITIVE.get(st, (None, 0))[0]
            hac = sig.hac_t_test(r)
            sb = sig.stationary_bootstrap_test(r, B=1000)
            sgn = sig.sign_test(r)
            lb = sd.ljung_box(r)
            runs = sd.runs_test(r)
            hu = sd.hurst_exponent(r)
            er = ent.entropy_report(r)
            risk = rm.risk_report(r)
            ruin = por.prob_of_ruin(r, "block", n_sim=2000)
            oos = wf.anchored_oos(r, oos_fraction=oos_fraction)
            rows.append({
                "ticker": tk, "strategy": st, "n": len(r), "net": r.sum(),
                "cost_rt": ck, "be_cost": be_cost,
                "edge_above_cost": bool(be_cost > ck),
                "hac_p": hac["p_value"], "sb_p": sb["p_value"], "sign_p": sgn["p_value"],
                "white_rc_p": sel["white_rc_p"], "spa_p": sel["spa_p"], "pbo": sel["pbo"],
                "fdr_pass": fdr_decision.get((tk.upper(), prim), False),
                "lb_struct": lb["any_structure"], "lb_p10": lb[10]["p_value"],
                "runs_p": runs["p_value"], "hurst": hu["hurst"],
                "perm_entropy": er["permutation"],
                "sortino": risk["sortino"], "calmar": risk["calmar"],
                "recovery": risk["recovery_factor"], "ruin30": ruin["ruin_30"],
                "oos_net": oos["oos_net"], "oos_mean": oos["oos_mean"],
                "oos_p": oos["oos_p"], "oos_decay": oos["decay"],
            })

    # verdicts
    for row in rows:
        row["verdict"] = classify(row)
    return rows


def classify(r):
    # AUDIT REDESIGN (PR-3):
    #   * The permutation-entropy gate was already removed (degenerate: ~1.0 for
    #     every real daily series).
    #   * The Ljung-Box "no_structure" REJECT and the Hurst "stable" REQUIREMENT
    #     are removed from the decision rule for the SAME reason: an overnight
    #     risk-PREMIUM / drift edge has a positive MEAN but iid-looking returns —
    #     no serial autocorrelation (LB insignificant) and Hurst ~= 0.5. Requiring
    #     serial structure therefore rejects exactly the class of edge these
    #     strategies target. Both are still computed and REPORTED for transparency,
    #     but they no longer gate the verdict.
    #   * Replaced by ECONOMIC gates that actually matter (PR-1, PR-5):
    #       - edge must clear realistic costs (break-even RT cost > assumed cost);
    #       - edge must survive a chronological out-of-sample tail (OOS mean > 0).
    ruin = r["ruin30"]
    oos_mean = r.get("oos_mean", float("nan"))
    reject = (
        (r["white_rc_p"] > 0.05) or (r["spa_p"] > 0.05) or
        (r["pbo"] > 0.5) or (not r["fdr_pass"]) or
        (ruin == ruin and ruin > 0.30) or
        (not r["edge_above_cost"]) or                       # PR-1: net edge <= 0
        (oos_mean == oos_mean and oos_mean <= 0)            # PR-5: dies out-of-sample
    )
    if reject:
        return "REJECTED"
    cost_margin = r["be_cost"] > 1.5 * r["cost_rt"]         # comfortable buffer over costs
    candidate = (
        (r["white_rc_p"] < 0.05) and (r["spa_p"] < 0.05) and r["fdr_pass"] and
        (r["pbo"] < 0.2) and cost_margin and
        (ruin == ruin and ruin < 0.10) and
        (oos_mean == oos_mean and oos_mean > 0)
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
        print(f"  Cost (PR-1):  break-even RT={r['be_cost']:+.3f}% vs assumed {r['cost_rt']:.3f}% → "
              f"{'CLEARS' if r['edge_above_cost'] else 'BELOW COSTS'}")
        print(f"  OOS (PR-5):   oos mean={r['oos_mean']:+.4f}%/trade (p={r['oos_p']:.3f}, "
              f"decay={r['oos_decay']:+.2f}, oos net={r['oos_net']:+.1f}%)")
        print(f"  Structure:    Hurst={r['hurst']:.3f} / Ljung-Box {'struct' if r['lb_struct'] else 'NONE'} "
              f"(p10={r['lb_p10']:.3f}) / Runs p={r['runs_p']:.3f} / PermEntropy={r['perm_entropy']:.3f}  "
              f"[informational only — not gated]")
        print(f"  Risk:         Sortino={r['sortino']:.3f} / Calmar={r['calmar']:.4f} / "
              f"Recovery={r['recovery']:.3f} / ProbRuin(-30%)={r['ruin30']:.3f}")
        print(f"  Verdict:      {r['verdict']}")
    # summary table
    print("=" * 72)
    print("  SUMMARY")
    print(f"  {'ticker':<7}{'strategy':<17}{'WhiteRC':>8}{'SPA':>7}{'PBO':>7}"
          f"{'FDR':>5}{'Ruin30':>8}{'BEcost':>8}{'Cost?':>6}{'OOSmean':>9}  verdict")
    counts = {"REJECTED": 0, "CANDIDATE EDGE": 0, "WEAK / INCONCLUSIVE": 0}
    for r in rows:
        counts[r["verdict"]] += 1
        print(f"  {r['ticker']:<7}{r['strategy']:<17}{r['white_rc_p']:>8.3f}{r['spa_p']:>7.3f}"
              f"{r['pbo']:>7.3f}{('Y' if r['fdr_pass'] else 'n'):>5}{r['ruin30']:>8.3f}"
              f"{r['be_cost']:>+8.3f}{('Y' if r['edge_above_cost'] else 'n'):>6}"
              f"{r['oos_mean']:>+9.4f}  {r['verdict']}")
    print("-" * 72)
    print(f"  REJECTED={counts['REJECTED']}  "
          f"CANDIDATE EDGE={counts['CANDIDATE EDGE']}  "
          f"WEAK={counts['WEAK / INCONCLUSIVE']}")
    return counts


def main():
    # CSVs live in ../validation_data (sibling of strategy_validation), NOT in the
    # package dir. Fall back to the package dir for backwards compatibility.
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(pkg_parent)
    default_data = os.path.join(repo_root, "validation_data")
    if not os.path.isdir(default_data):
        default_data = pkg_parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+",
                    default=["etln", "selg", "smlt", "mgnt", "alrs", "upro"])
    ap.add_argument("--strats", nargs="+",
                    default=["long_overnight", "intraday_short"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--data-dir", default=default_data, help="dir with *_candles.csv")
    ap.add_argument("--cost-rt", type=float, default=COST_RT, help="round-trip cost %% (PR-1)")
    ap.add_argument("--winsorize", action="store_true", help="winsorize ex-div gaps (PR-4)")
    ap.add_argument("--oos-fraction", type=float, default=0.30, help="OOS tail fraction (PR-5)")
    a = ap.parse_args()
    # TGKA stays excluded by construction (not in default list).
    rows = run(a.tickers, a.strats, a.data_dir, B=a.boot, cost_rt=a.cost_rt,
               winsorize=a.winsorize, oos_fraction=a.oos_fraction)
    print_report(rows)


if __name__ == "__main__":
    main()
