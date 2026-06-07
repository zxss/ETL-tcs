#!/usr/bin/env python3
"""
validation/entropy.py — entropy / complexity measures of a returns series.

Random/unpredictable series -> high entropy. Structured (constant/periodic) -> low.

  shannon_entropy(x, bins=20)        histogram-binned Shannon entropy, normalized to [0,1]
  permutation_entropy(x, order=3)    Bandt-Pompe ordinal-pattern entropy, normalized [0,1]
  sample_entropy(x, m=2, r=0.2*std)  SampEn (Richman & Moorman)
  approximate_entropy(x, m=2, r=...) ApEn (Pincus)

entropy_report(x) returns all of them in one dict.
"""
from __future__ import annotations
import math
import numpy as np
from itertools import permutations


def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def shannon_entropy(x, bins: int = 20) -> float:
    x = _clean(x)
    if len(x) < 2 or x.std() == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / np.log(bins))                  # normalized to [0,1]


def permutation_entropy(x, order: int = 3, delay: int = 1) -> float:
    x = _clean(x)
    n = len(x)
    if n < order + 1:
        return float("nan")
    patterns = {p: 0 for p in permutations(range(order))}
    count = 0
    for i in range(n - (order - 1) * delay):
        window = x[i:i + order * delay:delay]
        patt = tuple(np.argsort(window))
        patterns[patt] += 1
        count += 1
    probs = np.array([c for c in patterns.values() if c > 0], float) / count
    H = -np.sum(probs * np.log(probs))
    return float(H / np.log(math.factorial(order)))   # normalized [0,1]


def _phi(x, m, r):
    n = len(x)
    templates = np.array([x[i:i + m] for i in range(n - m + 1)])
    count = 0
    total = 0
    for i in range(len(templates)):
        d = np.max(np.abs(templates - templates[i]), axis=1)
        count += np.sum(d <= r) - 1                  # exclude self
        total += len(templates) - 1
    return count / total if total > 0 else 0.0


def sample_entropy(x, m: int = 2, r: float | None = None) -> float:
    x = _clean(x)
    n = len(x)
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * x.std()
    if r <= 0:
        return 0.0
    B = _phi(x, m, r)
    A = _phi(x, m + 1, r)
    if A == 0 or B == 0:
        return float("inf")
    return float(-np.log(A / B))


def approximate_entropy(x, m: int = 2, r: float | None = None) -> float:
    x = _clean(x)
    n = len(x)
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * x.std()
    if r <= 0:
        return 0.0

    def phi(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = np.zeros(len(templates))
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            c[i] = np.sum(d <= r) / len(templates)
        return np.mean(np.log(c[c > 0]))

    return float(phi(m) - phi(m + 1))


def entropy_report(x) -> dict:
    return {
        "shannon": shannon_entropy(x),
        "permutation": permutation_entropy(x),
        "sample": sample_entropy(x),
        "approximate": approximate_entropy(x),
    }


def _selftest() -> bool:
    print("=" * 60)
    print("  entropy.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(7)
    rnd = rng.normal(0, 1, 500)
    periodic = np.tile([1.0, 2.0, 3.0, 2.0], 125)
    const = np.ones(500)

    er = entropy_report(rnd)
    ep = entropy_report(periodic)
    ec = entropy_report(const)
    print(f"  random   : shannon={er['shannon']:.3f} perm={er['permutation']:.3f} "
          f"samp={er['sample']:.3f} apen={er['approximate']:.3f}")
    print(f"  periodic : shannon={ep['shannon']:.3f} perm={ep['permutation']:.3f} "
          f"samp={ep['sample']:.3f} apen={ep['approximate']:.3f}")
    print(f"  constant : shannon={ec['shannon']:.3f} perm={ec['permutation']}")
    ok = (er["shannon"] > ep["shannon"]) and \
         (er["permutation"] > ep["permutation"]) and \
         (ec["shannon"] < 0.01) and \
         (er["permutation"] > 0.9)
    print(f"  {'PASS' if ok else 'FAIL'}: random high entropy, periodic/constant low")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
