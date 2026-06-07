#!/usr/bin/env python3
"""
validation/bootstrap.py — shared bootstrap primitives.

These return INDEX arrays of length n so callers can resample any aligned
quantity (returns, signs, etc.). Both preserve serial structure, unlike the
i.i.d. permutation that the old shuffle test relied on.

  stationary_bootstrap_indices(n, expected_block, rng)
      Politis & Romano (1994). Geometric block lengths with mean = expected_block,
      i.e. at each step continue the current block with prob (1 - 1/expected_block),
      else jump to a fresh uniform start. Wraps around (circular).

  moving_block_bootstrap_indices(n, block_len, rng)
      Fixed-length non-overlapping concatenation of random block starts, trimmed
      to n. Same primitive advanced_stats used, exposed here as the single source.
"""
from __future__ import annotations
import numpy as np


def stationary_bootstrap_indices(n: int, expected_block: float, rng) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=int)
    expected_block = max(1.0, float(expected_block))
    p = 1.0 / expected_block           # restart probability
    idx = np.empty(n, dtype=int)
    cur = int(rng.integers(0, n))
    for t in range(n):
        idx[t] = cur
        if rng.random() < p:
            cur = int(rng.integers(0, n))   # start a new block
        else:
            cur = (cur + 1) % n             # continue block (circular)
    return idx


def moving_block_bootstrap_indices(n: int, block_len: int, rng) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=int)
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def _lag1_ac(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _selftest() -> bool:
    print("=" * 60)
    print("  bootstrap.py self-test: serial structure preserved")
    print("=" * 60)
    rng = np.random.default_rng(0)
    T, phi = 600, 0.6
    x = np.zeros(T)
    eps = rng.normal(0, 1, T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    true_ac = _lag1_ac(x)
    eb = int(round(np.sqrt(T)))

    sb_ac, mb_ac, iid_ac = [], [], []
    for _ in range(300):
        sb_ac.append(_lag1_ac(x[stationary_bootstrap_indices(T, eb, rng)]))
        mb_ac.append(_lag1_ac(x[moving_block_bootstrap_indices(T, eb, rng)]))
        iid_ac.append(_lag1_ac(rng.permutation(x)))
    sb_m, mb_m, iid_m = np.mean(sb_ac), np.mean(mb_ac), np.mean(iid_ac)
    print(f"  true lag-1 AC of AR(1,phi=0.6)     : {true_ac:+.3f}")
    print(f"  stationary bootstrap mean lag-1 AC : {sb_m:+.3f}")
    print(f"  moving-block    bootstrap lag-1 AC : {mb_m:+.3f}")
    print(f"  i.i.d. permutation       lag-1 AC : {iid_m:+.3f} (should be ~0)")
    ok = (sb_m > 0.25) and (mb_m > 0.25) and (abs(iid_m) < 0.1)
    print(f"  {'PASS' if ok else 'FAIL'}: block bootstraps keep AC, permutation destroys it")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
