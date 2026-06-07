#!/usr/bin/env python3
"""
validation/white_rc.py — White (2000) Reality Check + Hansen (2005) SPA.

Input: matrix R of strategy return series, shape (T, K) (rows=time, cols=strategy),
and a benchmark series (default zero). Differential f_k,t = R[t,k] - bench[t].

White RC statistic:  V = sqrt(T) * max_k mean(f_k)
Null distribution via STATIONARY block bootstrap of the f columns (preserves serial
dependence), each column recentered on its own mean (H0: no strategy beats bench).
    white_rc_p = (#{ V* >= V } + 1) / (B + 1)     <-- FIX: (count+1)/(B+1), not np.mean

Hansen (2005) SPA: studentizes each strategy by its own bootstrap std and uses the
consistent recentering threshold (drop strategies whose mean is below
-sqrt(2 log log T / T) * std, i.e. clearly-bad strategies do not inflate the null).
    spa_p = (#{ T*_spa >= T_spa } + 1) / (B + 1)
The "lower" / consistent variant is the standard, less conservative SPA p-value.

Both p high on noise (no strategy genuinely beats the benchmark).
"""
from __future__ import annotations
import numpy as np

from .bootstrap import stationary_bootstrap_indices


def white_reality_check(R: np.ndarray, benchmark=None, B: int = 2000,
                        seed: int = 42) -> dict:
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    T, K = R.shape
    bench = np.zeros(T) if benchmark is None else np.asarray(benchmark, float)
    F = R - bench[:, None]                      # (T,K) differential

    means = F.mean(axis=0)                       # per-strategy mean excess
    sqrtT = np.sqrt(T)
    V_obs = sqrtT * means.max()

    # studentization for SPA: per-strategy std of sqrt(T)*mean via bootstrap-free
    # plug-in (sample std of the column is the simple consistent choice)
    std = F.std(axis=0, ddof=1)
    std = np.where(std < 1e-30, 1e-30, std)

    expected_block = max(2.0, np.sqrt(T))
    rng = np.random.default_rng(seed)

    Fc = F - means[None, :]                      # recenter each column on H0 mean=0

    # SPA consistent threshold: keep strategy in null only if not clearly bad
    thresh = -np.sqrt(2.0 * np.log(np.log(max(T, 3))) / T) * std
    spa_keep = means >= thresh                   # boolean per strategy

    T_spa_obs = np.max(sqrtT * means / std)      # studentized max

    V_star = np.empty(B)
    T_spa_star = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(T, expected_block, rng)
        bm = Fc[idx, :].mean(axis=0)             # bootstrap means of recentered diffs
        V_star[b] = sqrtT * bm.max()
        stud = sqrtT * bm / std
        if spa_keep.any():
            T_spa_star[b] = np.max(stud[spa_keep])
        else:
            T_spa_star[b] = stud.max()

    white_rc_p = (np.sum(V_star >= V_obs) + 1) / (B + 1)
    spa_p = (np.sum(T_spa_star >= T_spa_obs) + 1) / (B + 1)
    return {"white_rc_p": float(white_rc_p), "spa_p": float(spa_p),
            "V_obs": float(V_obs), "best_strategy": int(np.argmax(means)),
            "n_strategies": K, "T": T}


def _selftest() -> bool:
    print("=" * 60)
    print("  white_rc.py self-test")
    print("=" * 60)
    # (1) noise: 6 strategies pure noise -> both p high, p-value bounded by 1/(B+1)
    wp, sp = [], []
    for t in range(12):
        rng = np.random.default_rng(t)
        R = rng.normal(0, 1, size=(250, 6))
        res = white_reality_check(R, B=600, seed=t)
        wp.append(res["white_rc_p"]); sp.append(res["spa_p"])
    wp, sp = np.array(wp), np.array(sp)
    print(f"  (1) noise, 6 strategies (12 runs):")
    print(f"      White RC mean p = {wp.mean():.3f}   SPA mean p = {sp.mean():.3f}")
    ok_noise = (wp.mean() > 0.2) and (sp.mean() > 0.2)
    # p-value floor is 1/(B+1) -> never exactly 0
    ok_floor = (wp.min() >= 1.0 / 601) and (sp.min() >= 1.0 / 601)
    print(f"      min White p = {wp.min():.4f} >= 1/(B+1)={1/601:.4f}: {ok_floor}")

    # (2) one strategy has real edge -> p should drop
    rng = np.random.default_rng(99)
    R = rng.normal(0, 1, size=(400, 6))
    R[:, 2] += 0.18                              # strategy 2 has genuine drift
    res = white_reality_check(R, B=1500, seed=5)
    print(f"  (2) strategy #2 has +0.18 drift among 6:")
    print(f"      White RC p = {res['white_rc_p']:.4f}  SPA p = {res['spa_p']:.4f}  best={res['best_strategy']}")
    ok_edge = (res["white_rc_p"] < 0.05) and (res["spa_p"] < 0.05)

    ok = ok_noise and ok_floor and ok_edge
    print(f"  {'PASS' if ok else 'FAIL'}: noise->high p, real edge->low p, p-floor respected")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
