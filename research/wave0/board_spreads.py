"""
T0.3 — оценка спреда для ВСЕЙ доски TQBR по дневным H/L (Корвин–Шульц).

Зачем: тесты блока C идут по бумагам вне топ-50 по обороту, а cost_model знает
спреды только для 49 торгуемых бумаг и подставляет остальным медиану голубых
фишек — это занижение издержек в разы. Корвин–Шульц (2012) оценивает спред из
дневных максимума/минимума двух соседних дней, а их у нас есть на всю доску
с 2014 года.

Контроль качества: для 49 бумаг, где спред уже посчитан по 5-минуткам
(audit/out/cost_matrix.csv), сравниваем оценки — если корреляция низкая,
оценке по доске доверять нельзя.

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
MIN_OBS = 60


def load_board() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(LONGHIST, "tqbr*_*.csv.gz"))):
        df = pd.read_csv(path)
        df["src"] = "holdout" if os.path.basename(path).startswith("tqbrh") else "dev"
        frames.append(df)
    b = pd.concat(frames, ignore_index=True)
    b.columns = [c.lower() for c in b.columns]
    b["tradedate"] = pd.to_datetime(b["tradedate"]).dt.date
    for c in ("open", "high", "low", "close", "volume", "value"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[(b["high"] > 0) & (b["low"] > 0) & (b["close"] > 0) & (b["high"] >= b["low"])]
    b = b.drop_duplicates(["secid", "tradedate"]).sort_values(["secid", "tradedate"])
    return b


def corwin_schultz(g: pd.DataFrame) -> pd.Series:
    """Спред в % по парам соседних дней. Отрицательные оценки → 0 (как в статье)."""
    h, l = g["high"].to_numpy(float), g["low"].to_numpy(float)
    if len(h) < 2:
        return pd.Series(dtype=float)
    hl = np.log(h / l) ** 2
    beta = hl[:-1] + hl[1:]
    hmax = np.maximum(h[:-1], h[1:])
    lmin = np.minimum(l[:-1], l[1:])
    gamma = np.log(hmax / lmin) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / K - np.sqrt(gamma / K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), s, np.nan)
    s = np.clip(s, 0.0, 0.25)                       # >25 % — заведомо мусор
    return pd.Series(s * 100.0)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    b = load_board()
    rows = []
    for tk, g in b.groupby("secid"):
        g = g.sort_values("tradedate")
        s = corwin_schultz(g)
        s = s.dropna()
        if len(s) < MIN_OBS:
            continue
        rows.append({"ticker": tk, "n_obs": int(len(s)),
                     "spread_pct": float(s.median()),
                     "spread_p95_pct": float(s.quantile(0.95)),
                     "adv_rub": float(g["value"].median()),
                     "days": int(len(g)),
                     "first": str(g["tradedate"].iloc[0]), "last": str(g["tradedate"].iloc[-1])})
    out = pd.DataFrame(rows).sort_values("adv_rub", ascending=False).reset_index(drop=True)
    out["liq_rank"] = np.arange(1, len(out) + 1)

    # ── контроль: сверка с матрицей по 5-минуткам ───────────────────────────
    check = {"note": "cost_matrix недоступна"}
    try:
        sp = cm.load_spreads()
        known = {k: v[0] for k, v in sp.items() if k != cm._FALLBACK}
        m = out[out["ticker"].isin(known)].copy()
        m["spread_5m"] = m["ticker"].map(known)
        if len(m) >= 10:
            r = float(np.corrcoef(m["spread_pct"], m["spread_5m"])[0, 1])
            check = {"n": int(len(m)), "corr": r,
                     "median_cs": float(m["spread_pct"].median()),
                     "median_5m": float(m["spread_5m"].median()),
                     "ratio_cs_to_5m": float(m["spread_pct"].median() / m["spread_5m"].median())}
    except Exception as exc:                                   # noqa: BLE001
        check = {"error": str(exc)}

    path = os.path.join(OUT, "board_spreads.csv")
    out.to_csv(path, index=False)
    json.dump(check, open(os.path.join(OUT, "board_spreads_check.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"бумаг с оценкой спреда: {len(out)}")
    for lo, hi in ((1, 50), (51, 150), (151, 10_000)):
        sl = out[(out["liq_rank"] >= lo) & (out["liq_rank"] <= hi)]
        if len(sl):
            print(f"  ранг {lo:>3}–{min(hi, len(out)):<4} n={len(sl):>3}  "
                  f"медианный спред {sl['spread_pct'].median():.3f} %  "
                  f"медианный оборот {sl['adv_rub'].median()/1e6:.1f} млн ₽/день")
    print(f"сверка с 5-минутками: {check}")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
