#!/usr/bin/env python3
"""
validation/serial_dependence.py — tests for temporal structure.

  ljung_box(returns, lags=(5,10,20))
      Q(m) = n(n+2) * sum_{k=1..m} rho_k^2 / (n-k),  chi2 with df=m.
      (Implemented directly; statsmodels is NOT installed in this env.)
      Low p => autocorrelation structure present.

  runs_test(returns)
      Wald-Wolfowitz runs test on the sign sequence. z, two-sided p.
      Significant => non-random ordering of signs.

  hurst_exponent(series)
      R/S analysis. <0.5 antipersistent (mean-reverting), ~0.5 random walk,
      >0.5 persistent / trending.
"""
from __future__ import annotations
import math
import numpy as np

try:
    from scipy import stats as _sp
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False


def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _acf(x, k):
    n = len(x)
    xm = x - x.mean()
    denom = np.dot(xm, xm)
    if denom == 0:
        return 0.0
    return float(np.dot(xm[k:], xm[:-k]) / denom)


def ljung_box(returns, lags=(5, 10, 20)) -> dict:
    x = _clean(returns)
    n = len(x)
    out = {}
    for m in lags:
        if n <= m + 1:
            out[m] = {"Q": float("nan"), "p_value": float("nan"), "df": m}
            continue
        s = sum(_acf(x, k) ** 2 / (n - k) for k in range(1, m + 1))
        Q = n * (n + 2) * s
        if _HAVE_SCIPY:
            p = float(_sp.chi2.sf(Q, df=m))
        else:
            p = float("nan")
        out[m] = {"Q": float(Q), "p_value": p, "df": m}
    out["any_structure"] = any(
        (v["p_value"] == v["p_value"]) and v["p_value"] < 0.05
        for k, v in out.items() if isinstance(v, dict))
    return out


def runs_test(returns) -> dict:
    x = _clean(returns)
    s = np.sign(x)
    s = s[s != 0]
    n = len(s)
    if n < 3:
        return {"z": float("nan"), "p_value": float("nan"), "runs": 0, "n": n}
    n_pos = int((s > 0).sum())
    n_neg = int((s < 0).sum())
    runs = 1 + int((s[1:] != s[:-1]).sum())
    if n_pos == 0 or n_neg == 0:
        return {"z": float("nan"), "p_value": float("nan"), "runs": runs, "n": n}
    mu = 1 + 2.0 * n_pos * n_neg / n
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
    if var <= 0:
        return {"z": float("nan"), "p_value": float("nan"), "runs": runs, "n": n}
    z = (runs - mu) / np.sqrt(var)
    if _HAVE_SCIPY:
        p = float(2 * _sp.norm.sf(abs(z)))
    else:
        p = float(math.erfc(abs(z) / np.sqrt(2)))
    return {"z": float(z), "p_value": p, "runs": runs, "expected_runs": float(mu), "n": n}


def hurst_exponent(series) -> dict:
    """R/S analysis on the cumulative series of the (mean-removed) input.
    For a returns series, we treat the returns themselves as the increments."""
    x = _clean(series)
    n = len(x)
    if n < 32:
        return {"hurst": float("nan"), "interpretation": "n<32, unreliable", "n": n}
    min_chunk = 8
    max_chunk = n // 2
    sizes = np.unique(np.floor(np.logspace(
        np.log10(min_chunk), np.log10(max_chunk), 12)).astype(int))
    sizes = sizes[sizes >= min_chunk]
    rs_vals, log_sizes = [], []
    for s in sizes:
        n_chunks = n // s
        if n_chunks < 1:
            continue
        rs_chunk = []
        for c in range(n_chunks):
            seg = x[c * s:(c + 1) * s]
            m = seg.mean()
            y = np.cumsum(seg - m)
            R = y.max() - y.min()
            S = seg.std(ddof=0)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            log_sizes.append(np.log(s))
    if len(rs_vals) < 3:
        return {"hurst": float("nan"), "interpretation": "insufficient scales", "n": n}
    H = float(np.polyfit(log_sizes, np.log(rs_vals), 1)[0])
    if H < 0.45:
        interp = "antipersistent (mean-reverting)"
    elif H <= 0.55:
        interp = "random walk (no structure)"
    else:
        interp = "persistent (trending)"
    return {"hurst": H, "interpretation": interp, "n": n}


def _selftest() -> bool:
    print("=" * 60)
    print("  serial_dependence.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(3)
    n = 1000

    iid = rng.normal(0, 1, n)
    ar = np.zeros(n); eps = rng.normal(0, 1, n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + eps[t]

    lb_iid = ljung_box(iid)
    lb_ar = ljung_box(ar)
    print("  (1) Ljung-Box: iid should be insignificant, AR(1) significant")
    print(f"      iid   lag10 Q={lb_iid[10]['Q']:.1f} p={lb_iid[10]['p_value']:.3f} any_struct={lb_iid['any_structure']}")
    print(f"      AR(1) lag10 Q={lb_ar[10]['Q']:.1f} p={lb_ar[10]['p_value']:.3g} any_struct={lb_ar['any_structure']}")
    ok_lb = (not lb_iid["any_structure"]) and lb_ar["any_structure"]

    # (2) Hurst: trending (random walk of drift) > 0.5, mean-reverting < 0.5
    trend = np.cumsum(rng.normal(0.05, 0.3, n))      # the *level* trends; use increments
    trend_incr = np.diff(np.cumsum(np.abs(rng.normal(0, 1, n)) + 0.1))  # persistent positives
    # construct clearly persistent fBm-like: integrate positively correlated noise
    persist = np.zeros(n); e = rng.normal(0, 1, n)
    for t in range(1, n):
        persist[t] = 0.8 * persist[t - 1] + e[t]     # strong positive AC -> H>0.5
    revert = -0.7 * np.roll(iid, 1) + iid            # anti-correlated -> H<0.5
    h_p = hurst_exponent(persist)
    h_r = hurst_exponent(revert)
    h_i = hurst_exponent(iid)
    print("  (2) Hurst: persistent>0.5, iid~0.5, mean-reverting<0.5")
    print(f"      persistent  H={h_p['hurst']:.3f} ({h_p['interpretation']})")
    print(f"      iid         H={h_i['hurst']:.3f} ({h_i['interpretation']})")
    print(f"      reverting   H={h_r['hurst']:.3f} ({h_r['interpretation']})")
    ok_h = (h_p["hurst"] > 0.55) and (h_r["hurst"] < 0.5)

    # (3) runs test: alternating sign series should be significant
    alt = np.array([1.0, -1.0] * (n // 2))
    rt_alt = runs_test(alt)
    rt_iid = runs_test(iid)
    print("  (3) Runs test: alternating signs significant, iid not")
    print(f"      alternating z={rt_alt['z']:.1f} p={rt_alt['p_value']:.3g}")
    print(f"      iid         z={rt_iid['z']:.2f} p={rt_iid['p_value']:.3f}")
    ok_runs = (rt_alt["p_value"] < 0.05) and (rt_iid["p_value"] > 0.05)

    ok = ok_lb and ok_h and ok_runs
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
