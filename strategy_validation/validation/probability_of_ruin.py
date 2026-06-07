#!/usr/bin/env python3
"""
validation/probability_of_ruin.py — probability the equity curve breaches a
drawdown threshold (-20% / -30% / -50% of starting capital).

Returns are in additive % per period (pnl_engine convention). We treat cumulative
sum of returns as the equity-in-%-of-capital path: a -20% breach means the running
P&L (from its running peak, or from start) drops by 20 percentage points.

Three estimators:
  (a) bootstrap MC      i.i.d. resample of the return path, count breach fraction
  (b) block bootstrap   stationary block resample (keeps losing streaks intact ->
                         usually HIGHER, more honest ruin estimates)
  (c) classic risk-of-ruin  closed-form from win-rate p and payoff ratio b
                         (avg win / avg loss), units = fraction of capital at risk.

prob_of_ruin(returns, ...) -> {ruin_20, ruin_30, ruin_50} using the chosen method;
prob_of_ruin_all returns all three methods for transparency.
"""
from __future__ import annotations
import numpy as np

from .bootstrap import stationary_bootstrap_indices

THRESHOLDS = (20.0, 30.0, 50.0)


def _max_drawdown_from_peak(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())          # most negative drawdown (in % points)


def _mc_ruin(returns, n_sim, thresholds, resampler, seed):
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return {t: float("nan") for t in thresholds}
    rng = np.random.default_rng(seed)
    breaches = {t: 0 for t in thresholds}
    for _ in range(n_sim):
        sample = resampler(r, n, rng)
        mdd = _max_drawdown_from_peak(np.cumsum(sample))
        for t in thresholds:
            if mdd <= -t:
                breaches[t] += 1
    return {t: breaches[t] / n_sim for t in thresholds}


def bootstrap_ruin(returns, n_sim=3000, thresholds=THRESHOLDS, seed=42) -> dict:
    def rs(r, n, rng):
        return r[rng.integers(0, n, size=n)]
    return _mc_ruin(returns, n_sim, thresholds, rs, seed)


def block_ruin(returns, n_sim=3000, thresholds=THRESHOLDS, seed=42) -> dict:
    def rs(r, n, rng):
        eb = max(2.0, np.sqrt(n))
        return r[stationary_bootstrap_indices(n, eb, rng)]
    return _mc_ruin(returns, n_sim, thresholds, rs, seed)


def classic_risk_of_ruin(returns, thresholds=THRESHOLDS, risk_per_trade=None) -> dict:
    """Classic gambler's risk-of-ruin: RoR = ((1-edge)/(1+edge))^units, where
    edge = p*b - (1-p), p = win rate, b = avg_win/avg_loss, and units = capital
    fraction to lose / fraction risked per trade. We map the threshold (e.g. 20%)
    to 'units' = threshold% / (avg loss as % of capital)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    wins = r[r > 0]; losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return {t: float("nan") for t in thresholds}
    p = len(wins) / len(r)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    b = avg_win / avg_loss if avg_loss > 0 else float("inf")
    edge = p * b - (1 - p)                    # expected units won per trade
    out = {}
    for t in thresholds:
        units = t / avg_loss if avg_loss > 0 else float("inf")
        if edge <= 0:
            out[t] = 1.0                      # negative/zero edge -> ruin certain eventually
        else:
            a = (1 - edge) / (1 + edge)
            a = min(max(a, 0.0), 1.0)
            out[t] = float(a ** units)
    return out


def prob_of_ruin(returns, method="block", **kw) -> dict:
    f = {"bootstrap": bootstrap_ruin, "block": block_ruin,
         "classic": classic_risk_of_ruin}[method]
    res = f(returns, **{k: v for k, v in kw.items()
                        if k in ("n_sim", "thresholds", "seed")})
    return {"ruin_20": res[20.0], "ruin_30": res[30.0], "ruin_50": res[50.0]}


def prob_of_ruin_all(returns, n_sim=3000, seed=42) -> dict:
    return {"bootstrap": prob_of_ruin(returns, "bootstrap", n_sim=n_sim, seed=seed),
            "block": prob_of_ruin(returns, "block", n_sim=n_sim, seed=seed),
            "classic": prob_of_ruin(returns, "classic")}


def _selftest() -> bool:
    print("=" * 60)
    print("  probability_of_ruin.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(8)
    # robust positive strategy: low vol, positive drift -> low ruin
    good = rng.normal(0.10, 0.8, 500)
    # fragile: high vol, slight negative drift -> high ruin
    bad = rng.normal(-0.05, 2.5, 500)

    rg = prob_of_ruin_all(good, n_sim=1500)
    rb = prob_of_ruin_all(bad, n_sim=1500)
    print(f"  GOOD (low-vol +drift):")
    for m in ("bootstrap", "block", "classic"):
        print(f"    {m:<10} ruin20={rg[m]['ruin_20']:.3f} ruin30={rg[m]['ruin_30']:.3f} ruin50={rg[m]['ruin_50']:.3f}")
    print(f"  BAD  (high-vol -drift):")
    for m in ("bootstrap", "block", "classic"):
        print(f"    {m:<10} ruin20={rb[m]['ruin_20']:.3f} ruin30={rb[m]['ruin_30']:.3f} ruin50={rb[m]['ruin_50']:.3f}")

    ok = (rb["block"]["ruin_20"] > rg["block"]["ruin_20"]) and \
         (rg["block"]["ruin_50"] < 0.2) and \
         (0.0 <= rg["bootstrap"]["ruin_20"] <= 1.0)
    print(f"  {'PASS' if ok else 'FAIL'}: bad strategy higher ruin, good strategy low ruin")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
