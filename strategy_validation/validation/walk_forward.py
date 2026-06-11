#!/usr/bin/env python3
"""
validation/walk_forward.py — honest chronological out-of-sample test (PR-5).

The CSCV/PBO machinery shuffles time chunks, which answers "is the selection
overfit" but NOT "does the edge survive forward in time". A real strategy must
keep working on data it was never measured on. Here we do the simplest honest
version: an ANCHORED holdout — fit nothing (the strategies are parameter-free),
but measure the edge on the first (1 - oos_fraction) of the series (in-sample)
and re-measure it on the final oos_fraction (out-of-sample), in calendar order.

  anchored_oos(returns, oos_fraction=0.30) -> dict
      is_net / oos_net      summed net % in each segment
      is_mean / oos_mean    mean per-trade net %
      oos_p                 one-sided HAC-t p-value that OOS mean > 0
      decay                 oos_mean / is_mean (1.0 = no decay, <0 = sign flip)

An edge that only exists in-sample (decay <= 0 or oos_mean <= 0) is a red flag
the in-sample-only tests (White RC / PBO on the whole series) cannot catch.
"""
from __future__ import annotations
import numpy as np

from .significance import hac_t_test


def anchored_oos(returns, oos_fraction: float = 0.30) -> dict:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    nan = float("nan")
    if n < 40:
        return {"is_net": nan, "oos_net": nan, "is_mean": nan, "oos_mean": nan,
                "oos_p": nan, "decay": nan, "n_is": 0, "n_oos": 0}
    n_oos = max(10, int(round(n * oos_fraction)))
    n_oos = min(n_oos, n - 10)              # keep at least 10 in-sample
    is_seg = r[: n - n_oos]
    oos_seg = r[n - n_oos:]
    is_mean = float(is_seg.mean())
    oos_mean = float(oos_seg.mean())
    oos_p = float(hac_t_test(oos_seg)["p_value"])
    decay = float(oos_mean / is_mean) if is_mean != 0 else nan
    return {
        "is_net": float(is_seg.sum()), "oos_net": float(oos_seg.sum()),
        "is_mean": is_mean, "oos_mean": oos_mean,
        "oos_p": oos_p, "decay": decay,
        "n_is": len(is_seg), "n_oos": len(oos_seg),
    }


def _selftest() -> bool:
    print("=" * 60)
    print("  walk_forward.py self-test")
    print("=" * 60)
    rng = np.random.default_rng(2)
    # (1) stable edge: strong, low-noise positive drift -> OOS survives unambiguously
    stable = rng.normal(0.30, 0.8, 600)          # t_oos ~ 0.30*sqrt(180)/0.8 ~ 5
    rs = anchored_oos(stable)
    print(f"  (1) stable +drift: is_mean={rs['is_mean']:+.3f} oos_mean={rs['oos_mean']:+.3f} "
          f"oos_p={rs['oos_p']:.3f} decay={rs['decay']:+.2f}")
    ok1 = (rs["oos_mean"] > 0) and (rs["oos_p"] < 0.05) and (rs["decay"] > 0.3)

    # (2) in-sample-only edge: strong drift in first 70%, pure noise in OOS tail
    fake = np.concatenate([rng.normal(0.40, 0.8, 420), rng.normal(0.0, 0.8, 180)])
    rf = anchored_oos(fake)
    print(f"  (2) IS-only edge:  is_mean={rf['is_mean']:+.3f} oos_mean={rf['oos_mean']:+.3f} "
          f"oos_p={rf['oos_p']:.3f} decay={rf['decay']:+.2f}")
    ok2 = (rf["oos_mean"] < rf["is_mean"]) and (rf["oos_p"] > 0.05)

    ok = ok1 and ok2
    print(f"  {'PASS' if ok else 'FAIL'}: stable edge survives OOS, IS-only edge decays")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
