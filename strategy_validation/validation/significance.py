#!/usr/bin/env python3
"""
validation/significance.py — real significance tests on a 1-D array of per-trade
or daily strategy returns. Replaces the DEGENERATE sum-permutation shuffle test:
a permutation of returns has an identical sum, so the null is a single point and
the test is useless. Each test here has a genuine, non-degenerate null.

  stationary_bootstrap_test(returns, B=2000, p=None)
      Politis & Romano (1994) stationary bootstrap of the MEAN. Preserves
      autocorrelation / vol clustering. Centered bootstrap, one-sided H1: mean>0.
      p = (#{ |boot_mean - obs_mean| >= |obs_mean| } + 1) / (B + 1)   [centered]
      (equivalently boot_centered >= |obs|). Returns observed/bootstrap mean + p.

  sign_test(returns)
      Binomial test of P(return > 0) = 0.5 (one-sided, more positives than half).

  hac_t_test(returns)
      Newey-West / HAC t-test that mean return > 0. Bandwidth (Newey-West 1994
      automatic-lag rule of thumb) L = floor(4*(n/100)^(2/9)). Bartlett weights.
"""
from __future__ import annotations
import math
import numpy as np

try:
    from scipy import stats as _sp
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

from .bootstrap import stationary_bootstrap_indices


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def stationary_bootstrap_test(returns, B: int = 2000, p: float | None = None,
                              seed: int = 42) -> dict:
    r = _clean(returns)
    n = len(r)
    if n < 5:
        return {"observed_mean": float("nan"), "bootstrap_mean": float("nan"),
                "p_value": float("nan"), "n": n}
    expected_block = (1.0 / p) if (p and p > 0) else max(2.0, np.sqrt(n))
    obs = float(r.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(n, expected_block, rng)
        boot[b] = r[idx].mean()
    boot_centered = boot - obs                       # null: mean = 0
    # one-sided for mean > 0: how often centered bootstrap reaches the observed effect
    pval = (np.sum(boot_centered >= abs(obs)) + 1) / (B + 1)
    return {"observed_mean": obs, "bootstrap_mean": float(boot.mean()),
            "p_value": float(pval), "expected_block": float(expected_block), "n": n}


def sign_test(returns) -> dict:
    r = _clean(returns)
    r = r[r != 0.0]
    n = len(r)
    pos = int((r > 0).sum())
    if n < 1:
        return {"stat": float("nan"), "p_value": float("nan"), "n_pos": pos, "n": n}
    if _HAVE_SCIPY:
        res = _sp.binomtest(pos, n, 0.5, alternative="greater")
        pval = float(res.pvalue)
    else:                                            # normal approx fallback
        z = (pos - n / 2) / np.sqrt(n / 4)
        pval = 0.5 * math.erfc(z / np.sqrt(2))
    return {"stat": pos / n, "p_value": pval, "n_pos": pos, "n": n}


def hac_t_test(returns) -> dict:
    r = _clean(returns)
    n = len(r)
    if n < 5:
        return {"tstat": float("nan"), "p_value": float("nan"), "se": float("nan"), "n": n}
    mean = r.mean()
    e = r - mean
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    L = max(0, min(L, n - 1))
    gamma0 = np.dot(e, e) / n
    lrv = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)                      # Bartlett kernel
        gk = np.dot(e[k:], e[:-k]) / n
        lrv += 2.0 * w * gk
    lrv = max(lrv, 1e-30)
    se = np.sqrt(lrv / n)                            # HAC se of the mean
    tstat = mean / se
    if _HAVE_SCIPY:
        pval = float(_sp.t.sf(tstat, df=n - 1))      # one-sided mean > 0
    else:
        pval = 0.5 * math.erfc(tstat / np.sqrt(2))
    return {"tstat": float(tstat), "p_value": float(pval), "se": float(se),
            "lag": L, "n": n}


def _selftest() -> bool:
    print("=" * 60)
    print("  significance.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(1)
    n = 300

    # (1) pure noise -> p should NOT be systematically small
    sb_ps, hac_ps, sg_ps = [], [], []
    for t in range(20):
        noise = np.random.default_rng(100 + t).normal(0, 1, n)
        sb_ps.append(stationary_bootstrap_test(noise, B=500, seed=t)["p_value"])
        hac_ps.append(hac_t_test(noise)["p_value"])
        sg_ps.append(sign_test(noise)["p_value"])
    sb_ps, hac_ps, sg_ps = map(np.array, (sb_ps, hac_ps, sg_ps))
    print("  (1) pure noise (20 runs) -- p should be high / spread, not ~0:")
    print(f"      stationary-bootstrap mean p = {sb_ps.mean():.3f}")
    print(f"      HAC-t                mean p = {hac_ps.mean():.3f}")
    print(f"      sign-test            mean p = {sg_ps.mean():.3f}")
    ok_noise = (sb_ps.mean() > 0.2) and (hac_ps.mean() > 0.2) and (sg_ps.mean() > 0.2)

    # (2) strong positive drift -> all p low
    drift = rng.normal(0.20, 1.0, 500)               # mean 0.20, SNR ~0.20*sqrt(500)~4.5
    sb = stationary_bootstrap_test(drift, B=1000)
    hc = hac_t_test(drift)
    sg = sign_test(drift)
    print("  (2) strong +drift (mean=0.15, sd=1) -- p should be low:")
    print(f"      stationary-bootstrap p = {sb['p_value']:.4f}  (obs mean {sb['observed_mean']:+.3f})")
    print(f"      HAC-t                p = {hc['p_value']:.4f}  (t={hc['tstat']:+.2f}, lag={hc['lag']})")
    print(f"      sign-test            p = {sg['p_value']:.4f}  (P[r>0]={sg['stat']:.2f})")
    ok_drift = (sb["p_value"] < 0.05) and (hc["p_value"] < 0.05) and (sg["p_value"] < 0.05)

    ok = ok_noise and ok_drift
    print(f"  {'PASS' if ok else 'FAIL'}: noise->high p, drift->low p")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
