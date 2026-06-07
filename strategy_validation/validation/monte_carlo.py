#!/usr/bin/env python3
"""
validation/monte_carlo.py — Monte-Carlo resampling of the equity path.

SINGLE SOURCE OF TRUTH: every strategy is fed ONE 1-D array of per-period returns
(strategy.daily_returns / pnl_engine returns series). NO per-strategy column hacks
-- the old top3_report.monte_carlo used the intraday column for sign-flipping even
when P&L was on total/overnight; this module never does that. You pass the actual
strategy return series and it resamples THAT.

Methods:
  bootstrap_mc       i.i.d. resample of the returns (destroys serial structure --
                     baseline; tells you the path-ordering-free distribution)
  block_bootstrap_mc stationary block bootstrap (preserves autocorr / vol clusters)
  regime_mc          resample WITHIN vol-regime buckets (low/med/high |ret| terciles),
                     preserving the regime mix -- guards against a single calm or
                     wild regime driving results

For each, simulates the equity path and reports percentile summaries of:
  CAGR, Sharpe, MaxDD, RecoveryTime (periods to recover the deepest drawdown).
"""
from __future__ import annotations
import numpy as np

from .bootstrap import stationary_bootstrap_indices


def _path_stats(r: np.ndarray, periods_per_year: int) -> tuple:
    eq = np.cumsum(r)
    n = len(r)
    total = eq[-1]
    years = n / periods_per_year
    cagr = total / years if years > 0 else np.nan          # additive % per year
    sd = r.std(ddof=1) if n > 1 else 0.0
    sharpe = (r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    mdd = dd.min()
    # recovery time: longest stretch under water (periods)
    underwater = dd < 0
    rec, cur = 0, 0
    for u in underwater:
        cur = cur + 1 if u else 0
        rec = max(rec, cur)
    return cagr, sharpe, mdd, rec


def _summarize(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"p5": np.nan, "p50": np.nan, "p95": np.nan, "mean": np.nan}
    return {"p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean())}


def _run(resampler, returns, n_sim, periods_per_year, seed):
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    rng = np.random.default_rng(seed)
    cagr, sharpe, mdd, rec = (np.empty(n_sim) for _ in range(4))
    for i in range(n_sim):
        sample = resampler(r, n, rng)
        cagr[i], sharpe[i], mdd[i], rec[i] = _path_stats(sample, periods_per_year)
    return {"CAGR": _summarize(cagr), "Sharpe": _summarize(sharpe),
            "MaxDD": _summarize(mdd), "RecoveryTime": _summarize(rec),
            "n_sim": n_sim, "n": n}


def bootstrap_mc(returns, n_sim=2000, periods_per_year=252, seed=42) -> dict:
    def rs(r, n, rng):
        return r[rng.integers(0, n, size=n)]
    return _run(rs, returns, n_sim, periods_per_year, seed)


def block_bootstrap_mc(returns, n_sim=2000, periods_per_year=252, seed=42) -> dict:
    def rs(r, n, rng):
        eb = max(2.0, np.sqrt(n))
        return r[stationary_bootstrap_indices(n, eb, rng)]
    return _run(rs, returns, n_sim, periods_per_year, seed)


def regime_mc(returns, n_sim=2000, periods_per_year=252, seed=42) -> dict:
    r_all = np.asarray(returns, float)
    r_all = r_all[np.isfinite(r_all)]
    n = len(r_all)
    # bucket by |return| terciles (vol regime proxy)
    absr = np.abs(r_all)
    if n >= 9:
        q1, q2 = np.percentile(absr, [33.33, 66.67])
        buckets = [r_all[absr <= q1], r_all[(absr > q1) & (absr <= q2)], r_all[absr > q2]]
        sizes = [len(b) for b in buckets]
    else:
        buckets, sizes = [r_all], [n]

    def rs(r, nn, rng):
        out = []
        for b, sz in zip(buckets, sizes):
            if len(b) > 0 and sz > 0:
                out.append(b[rng.integers(0, len(b), size=sz)])
        sample = np.concatenate(out)
        rng.shuffle(sample)                    # randomize ordering across regimes
        return sample
    return _run(rs, returns, n_sim, periods_per_year, seed)


def _selftest() -> bool:
    print("=" * 60)
    print("  monte_carlo.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(4)
    pos = rng.normal(0.08, 1.0, 400)       # positive-drift strategy
    neg = rng.normal(-0.08, 1.0, 400)

    for name, fn in [("bootstrap", bootstrap_mc),
                     ("block", block_bootstrap_mc),
                     ("regime", regime_mc)]:
        rp = fn(pos, n_sim=500)
        rn = fn(neg, n_sim=500)
        print(f"  [{name}] +drift CAGR p50={rp['CAGR']['p50']:+.1f} "
              f"Sharpe p50={rp['Sharpe']['p50']:+.2f} MaxDD p50={rp['MaxDD']['p50']:.1f}")
        print(f"  [{name}] -drift CAGR p50={rn['CAGR']['p50']:+.1f} "
              f"Sharpe p50={rn['Sharpe']['p50']:+.2f} MaxDD p50={rn['MaxDD']['p50']:.1f}")

    # sanity: positive drift -> p5 CAGR usually > p5 of negative drift; recovery finite
    rp = bootstrap_mc(pos, n_sim=800)
    rn = bootstrap_mc(neg, n_sim=800)
    ok = (rp["CAGR"]["p50"] > rn["CAGR"]["p50"]) and \
         (rp["Sharpe"]["p50"] > 0 > rn["Sharpe"]["p50"]) and \
         np.isfinite(rp["RecoveryTime"]["p50"])
    print(f"  {'PASS' if ok else 'FAIL'}: +drift dominates -drift across MC variants")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
