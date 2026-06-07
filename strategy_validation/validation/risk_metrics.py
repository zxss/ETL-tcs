#!/usr/bin/env python3
"""
validation/risk_metrics.py — risk-adjusted performance metrics.

All operate on a 1-D returns array (in % per period, additive — matching the
pnl_engine convention of summed percentage returns). Every metric handles
zero-MaxDD / empty / all-negative gracefully (no div-by-zero; returns inf/nan
with a clear convention rather than crashing).

  sortino_ratio(returns, mar=0)            mean / downside deviation
  calmar_ratio(returns, periods_per_year)  CAGR / |MaxDD|
  recovery_factor(returns)                 net profit / |MaxDD|
  tail_ratio(returns)                      |95th pct| / |5th pct|
  omega_ratio(returns, threshold=0)        sum gains above / sum losses below
  max_drawdown(returns)                    helper, on cumulative-sum equity
"""
from __future__ import annotations
import numpy as np


def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def max_drawdown(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())                            # <= 0


def sortino_ratio(returns, mar: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - mar
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / dd)


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    total = r.sum()
    years = len(r) / periods_per_year
    cagr = total / years if years > 0 else float("nan")   # additive annualized %
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / mdd)


def recovery_factor(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    net = r.sum()
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return float("inf") if net > 0 else 0.0
    return float(net / mdd)


def tail_ratio(returns) -> float:
    r = _clean(returns)
    if len(r) < 20:
        return float("nan")
    p95 = np.percentile(r, 95)
    p5 = np.percentile(r, 5)
    if p5 == 0:
        return float("inf") if p95 > 0 else float("nan")
    return float(abs(p95) / abs(p5))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    gains = np.sum(np.maximum(r - threshold, 0))
    losses = np.sum(np.maximum(threshold - r, 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def risk_report(returns, periods_per_year: int = 252) -> dict:
    return {
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "recovery_factor": recovery_factor(returns),
        "tail_ratio": tail_ratio(returns),
        "omega": omega_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }


def _selftest() -> bool:
    print("=" * 60)
    print("  risk_metrics.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(5)
    good = rng.normal(0.10, 1.0, 400)
    bad = rng.normal(-0.10, 1.0, 400)

    rg = risk_report(good)
    rb = risk_report(bad)
    print(f"  +drift: sortino={rg['sortino']:.3f} calmar={rg['calmar']:.4f} "
          f"recovery={rg['recovery_factor']:.3f} omega={rg['omega']:.3f}")
    print(f"  -drift: sortino={rb['sortino']:.3f} calmar={rb['calmar']:.4f} "
          f"recovery={rb['recovery_factor']:.3f} omega={rb['omega']:.3f}")
    ok_dir = (rg["sortino"] > 0 > rb["sortino"]) and (rg["omega"] > 1 > rb["omega"])

    # edge cases
    print("  edge cases (no div-by-zero):")
    e_empty = risk_report(np.array([]))
    e_flat = risk_report(np.ones(50))                 # all positive, zero DD
    e_neg = risk_report(-np.abs(rng.normal(0, 1, 50)))
    print(f"    empty   : {e_empty['sortino']}, calmar={e_empty['calmar']}")
    print(f"    all-pos : sortino={e_flat['sortino']}, recovery={e_flat['recovery_factor']} (MaxDD=0 -> inf)")
    print(f"    all-neg : sortino={e_neg['sortino']:.3f}, omega={e_neg['omega']:.3f}")
    ok_edge = (np.isnan(e_empty["sortino"]) and e_flat["recovery_factor"] == float("inf")
               and e_neg["omega"] == 0.0)

    ok = ok_dir and ok_edge
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
