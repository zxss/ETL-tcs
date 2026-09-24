"""
T0.3 — оценка спреда для ВСЕЙ доски TQBR по дневным данным.

Зачем: тесты блока C идут по бумагам вне топ-50 по обороту, а cost_model знает
спреды только для 49 торгуемых бумаг и подставляет остальным медиану голубых
фишек — это занижение издержек в разы.

Два оценщика по дневным данным:
  CS  — Корвин–Шульц (2012) по H/L соседних дней, С поправкой на ночной гэп
        (без неё гэп попадает в двухдневный диапазон и завышает спред) и с
        помесячным усреднением α (в статье отрицательные средние → 0);
  AR  — Abdi–Ranaldo (2017) CHL: S² = 4·E[(c_t − η_t)(c_t − η_{t+1})],
        где η = (ln H + ln L)/2. Устойчивее CS на неликвидных бумагах.

КОНТРОЛЬ КАЧЕСТВА (решает, можно ли вообще пользоваться оценкой):
  1) корреляция с матрицей по 5-минуткам на 46 общих бумагах;
  2) монотонность по ликвидности — спред неликвидных бумаг ОБЯЗАН быть выше,
     чем у голубых фишек. Если нет — оценщик меряет волатильность, а не спред.

Запуск (на сервере): python -m research.wave0.board_spreads
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                           # noqa: E402

LONGHIST = os.path.join(ROOT, "audit", "r4_research", "longhist")
OUT = os.path.join(ROOT, "audit", "r4_research", "wave0")
K = 3.0 - 2.0 * math.sqrt(2.0)
MIN_OBS = 120
MAX_SPREAD_PCT = 15.0


def load_board() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(LONGHIST, "tqbr*_*.csv.gz"))):
        df = pd.read_csv(path)
        frames.append(df)
    b = pd.concat(frames, ignore_index=True)
    b.columns = [c.lower() for c in b.columns]
    b["tradedate"] = pd.to_datetime(b["tradedate"])
    for c in ("open", "high", "low", "close", "volume", "value"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[(b["high"] > 0) & (b["low"] > 0) & (b["close"] > 0) & (b["high"] >= b["low"])]
    b = b.drop_duplicates(["secid", "tradedate"]).sort_values(["secid", "tradedate"])
    return b


def cs_alpha(g: pd.DataFrame) -> pd.Series:
    """α Корвина–Шульца по парам дней с поправкой на ночной гэп."""
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    c = g["close"].to_numpy(float)
    if len(h) < 3:
        return pd.Series(dtype=float)
    h0, l0 = h[:-1], l[:-1]
    h1, l1 = h[1:].copy(), l[1:].copy()
    c0 = c[:-1]
    # поправка на гэп: если вчерашний close вне сегодняшнего диапазона,
    # сдвигаем сегодняшний диапазон на величину гэпа (Corwin & Schultz, §I.B)
    gap = np.where(c0 < l1, l1 - c0, np.where(c0 > h1, h1 - c0, 0.0))
    h1 = h1 - gap
    l1 = l1 - gap
    ok = (h0 > 0) & (l0 > 0) & (h1 > 0) & (l1 > 0)
    beta = np.where(ok, np.log(np.divide(h0, l0, where=ok, out=np.ones_like(h0))) ** 2
                    + np.log(np.divide(h1, l1, where=ok, out=np.ones_like(h1))) ** 2, np.nan)
    hmax = np.maximum(h0, h1)
    lmin = np.minimum(l0, l1)
    gamma = np.where(ok, np.log(np.divide(hmax, lmin, where=ok,
                                          out=np.ones_like(hmax))) ** 2, np.nan)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / K - np.sqrt(np.maximum(gamma, 0.0) / K)
    return pd.Series(alpha, index=g.index[1:])


def ar_spread(g: pd.DataFrame) -> float:
    """Abdi–Ranaldo CHL: S = 2√max(E[(c−η_t)(c−η_{t+1})], 0)."""
    c = np.log(g["close"].to_numpy(float))
    eta = (np.log(g["high"].to_numpy(float)) + np.log(g["low"].to_numpy(float))) / 2.0
    if len(c) < 3:
        return float("nan")
    x = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    x = x[np.isfinite(x)]
    if len(x) < MIN_OBS:
        return float("nan")
    return float(2.0 * math.sqrt(max(x.mean(), 0.0)) * 100.0)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    b = load_board()
    rows = []
    for tk, g in b.groupby("secid"):
        g = g.sort_values("tradedate").reset_index(drop=True)
        if len(g) < MIN_OBS:
            continue
        a = cs_alpha(g)
        if len(a) < MIN_OBS:
            continue
        # помесячное усреднение α, отрицательные средние → 0 (как в статье)
        month = g["tradedate"].dt.to_period("M").reindex(a.index)
        am = a.groupby(month).mean()
        am = am[np.isfinite(am)]
        if len(am) < 6:
            continue
        s_cs = 2.0 * (np.exp(am) - 1.0) / (1.0 + np.exp(am))
        s_cs = np.clip(s_cs, 0.0, MAX_SPREAD_PCT / 100.0) * 100.0
        rows.append({"ticker": tk, "n_months": int(len(am)), "days": int(len(g)),
                     "spread_cs_pct": float(np.median(s_cs)),
                     "spread_cs_p95_pct": float(np.quantile(s_cs, 0.95)),
                     "spread_ar_pct": ar_spread(g),
                     "adv_rub": float(g["value"].median()),
                     "first": str(g["tradedate"].iloc[0].date()),
                     "last": str(g["tradedate"].iloc[-1].date())})
    out = pd.DataFrame(rows).sort_values("adv_rub", ascending=False).reset_index(drop=True)
    out["liq_rank"] = np.arange(1, len(out) + 1)

    # ── контроль качества ───────────────────────────────────────────────────
    qc: dict = {}
    sp = cm.load_spreads()
    known = {k: v[0] for k, v in sp.items() if k != cm._FALLBACK}
    m = out[out["ticker"].isin(known)].copy()
    m["spread_5m"] = m["ticker"].map(known)
    for est in ("spread_cs_pct", "spread_ar_pct"):
        mm = m[np.isfinite(m[est])]
        tiers = {}
        for name, lo, hi in (("top50", 1, 50), ("mid", 51, 150), ("tail", 151, 10_000)):
            sl = out[(out["liq_rank"] >= lo) & (out["liq_rank"] <= hi)][est].dropna()
            tiers[name] = float(sl.median()) if len(sl) else None
        mono = (tiers["top50"] is not None and tiers["mid"] is not None
                and tiers["tail"] is not None
                and tiers["top50"] < tiers["mid"] < tiers["tail"])
        qc[est] = {"n_common": int(len(mm)),
                   "corr_with_5m": float(np.corrcoef(mm[est], mm["spread_5m"])[0, 1])
                   if len(mm) >= 10 else None,
                   "median_est_top50": tiers["top50"], "median_5m_top50":
                       float(mm["spread_5m"].median()) if len(mm) else None,
                   "tiers": tiers, "monotone_in_liquidity": bool(mono)}

    best = max(qc, key=lambda k: (qc[k]["monotone_in_liquidity"],
                                  qc[k]["corr_with_5m"] or -1))
    usable = qc[best]["monotone_in_liquidity"] and (qc[best]["corr_with_5m"] or 0) >= 0.5
    qc["verdict"] = {"best_estimator": best, "usable": bool(usable),
                     "rule": "годен, если монотонен по ликвидности И корр. с 5м ≥ 0.5"}

    out.to_csv(os.path.join(OUT, "board_spreads.csv"), index=False)
    json.dump(qc, open(os.path.join(OUT, "board_spreads_check.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"бумаг с оценкой: {len(out)}")
    for est in ("spread_cs_pct", "spread_ar_pct"):
        q = qc[est]
        print(f"\n{est}: корр. с 5-минутками {q['corr_with_5m']}, "
              f"монотонность {q['monotone_in_liquidity']}")
        print(f"   тиры (медиана, %): {q['tiers']}, по 5м топ-50: {q['median_5m_top50']}")
    print(f"\nВЕРДИКТ: {qc['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
