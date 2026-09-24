"""
Факторы на длинной истории MOEX 2014–2024 (ТЗ пользователя 24.09.2026, C1).

Зачем: low-vol — единственный индикатор ТА, воспроизведённый на holdout
(IC ≈ −0,05), но как портфель на H=20 у нас было всего 13–20 ребалансов — нет
мощности. 2014-01…2024-05 — выборка, которую не видел ни один наш тест по
акциям (market_data начинается 2024-05), т. е. это и мощность, и свежий holdout.

Данные: research/longhist/iss_loader — полный срез TQBR по датам (со снятыми с
торгов бумагами), дивиденды T-Invest (полная доходность; у снятых бумаг
дивидендов может не быть — смещение против дивидендных, т. е. против low-vol).
Сплиты/ошибки: дневные |r| > 40 % выбрасываются.
Универс на дату: топ-50 по среднему обороту за 60 дней, торговалась ≥ 50 из 60.

Предрегистрация (4 испытания, Холм, ожидаемые знаки):
  T1 IC: ATR%(14) ↔ завтрашняя относительная доходность, ожидание < 0 —
     прямая репликация находки батареи ТА;
  T2 low-vol: лонг нижний квинтиль ATR% против равновесного универса,
     ребаланс каждые 21 день, издержки 0,2 % × доля сменившегося состава, > 0;
  T3 моментум 12-1: лонг верхний квинтиль доходности d−252…d−21, > 0;
  T4 месячный разворот: лонг нижний квинтиль доходности d−21…d, > 0.
Периоды: P1 2014-01…2022-02-18 (до закрытия рынка), P2 2022-03-24…2024-05-20;
главный вывод — по объединению, периоды отдельно как проверка устойчивости.
Запуск (сервер): python -m research.longhist.factors
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

log = logging.getLogger("research.longhist.factors")
DATA_DIR = os.path.join(ROOT, "audit", "r4_research", "longhist")
TOP_N, LIQ_WIN, LIQ_MIN = 50, 60, 50
H, Q = 21, 0.2
COST = 0.2
MAX_ABS_R = 0.40
P1 = (dt.date(2014, 1, 1), dt.date(2022, 2, 18))
P2 = (dt.date(2022, 3, 24), dt.date(2024, 5, 20))


def load() -> dict[str, pd.DataFrame]:
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "tqbr_*.csv.gz")))])
    df = df[(df["CLOSE"] > 0) & (df["VALUE"] > 0)].copy()
    df["d"] = pd.to_datetime(df["TRADEDATE"]).dt.date
    df = df.drop_duplicates(["SECID", "d"])
    wide = {c: df.pivot(index="d", columns="SECID", values=c).sort_index()
            for c in ("CLOSE", "HIGH", "LOW", "VALUE")}
    close = wide["CLOSE"]
    # дивиденды: ex-дата = первый торговый день после последнего дня покупки
    div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    dp = os.path.join(DATA_DIR, "dividends.csv")
    if os.path.exists(dp) and os.path.getsize(dp) > 5:
        dv = pd.read_csv(dp)
        days = np.array(close.index)
        for _, r in dv.iterrows():
            if r["ticker"] not in div.columns:
                continue
            lb = dt.date.fromisoformat(str(r["last_buy_date"])[:10])
            i = np.searchsorted(days, lb, side="right")
            if i < len(days):
                div.iat[i, div.columns.get_loc(r["ticker"])] += float(r["dividend_net"])
    ret = (close + div) / close.shift(1) - 1.0
    ret = ret.where(ret.abs() <= MAX_ABS_R)
    prev = close.shift(1)
    tr = pd.concat([wide["HIGH"] - wide["LOW"], (wide["HIGH"] - prev).abs(), (wide["LOW"] - prev).abs()]
                   ).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=12).mean() / close * 100.0
    liq_mean = wide["VALUE"].rolling(LIQ_WIN, min_periods=LIQ_MIN).mean()
    return {"close": close, "ret": ret, "atr": atr, "liq": liq_mean}


def universe(liq: pd.DataFrame, d) -> list[str]:
    row = liq.loc[d].dropna()
    return list(row.sort_values(ascending=False).index[:TOP_N])


def in_period(d, per) -> bool:
    return per[0] <= d <= per[1]


def ic_daily(D: dict) -> pd.Series:
    """T1: ранговая корреляция ATR%(d) с относительной доходностью d+1."""
    from scipy import stats
    ret, atr, liq = D["ret"], D["atr"], D["liq"]
    days = list(ret.index)
    out = {}
    for i in range(len(days) - 1):
        d, dn = days[i], days[i + 1]
        u = universe(liq, d)
        if len(u) < 30:
            continue
        a = atr.loc[d, u]
        r = ret.loc[dn, u]
        m = a.notna() & r.notna()
        if m.sum() < 25:
            continue
        rr = r[m] - r[m].mean()
        out[d] = stats.spearmanr(a[m], rr)[0]
    return pd.Series(out)


def hold_ret(ret: pd.DataFrame, i: int, names: list[str]) -> pd.Series:
    """доходность за H дней после индекса i (пропуски = 0 — бумага не торговалась)."""
    blk = ret.iloc[i + 1:i + 1 + H][names].fillna(0.0)
    return (1.0 + blk).prod() - 1.0


def portfolios(D: dict) -> pd.DataFrame:
    ret, atr, liq, close = D["ret"], D["atr"], D["liq"], D["close"]
    cum = (1.0 + ret.fillna(0.0)).cumprod()
    days = list(ret.index)
    rows, prev = [], {"lowvol": set(), "mom": set(), "rev": set()}
    for i in range(252, len(days) - H, H):
        d = days[i]
        u = [t for t in universe(liq, d) if pd.notna(close.at[d, t])]
        if len(u) < 30:
            continue
        k = max(1, int(round(len(u) * Q)))
        sig = {
            "lowvol": atr.loc[d, u].dropna().sort_values().index[:k],
            "mom": (cum.iloc[i - 21][u] / cum.iloc[i - 252][u] - 1).dropna().sort_values(ascending=False).index[:k],
            "rev": (cum.iloc[i][u] / cum.iloc[i - 21][u] - 1).dropna().sort_values().index[:k],
        }
        hr = hold_ret(ret, i, u)
        r_uni = float(hr.mean())
        rec = {"d": d, "uni_pct": r_uni * 100}
        for name, names in sig.items():
            names = set(names)
            turn = 1.0 if not prev[name] else 1 - len(names & prev[name]) / max(1, len(names))
            rec[f"{name}_pct"] = (float(hr[list(names)].mean()) - r_uni) * 100 - COST * turn
            prev[name] = names
        rows.append(rec)
    return pd.DataFrame(rows)


def tstat(s: pd.Series, periods_per_year: float) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 8 or s.std(ddof=1) == 0:
        return {"n": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"n": n, "mean": float(s.mean()), "annual_pct": float(s.mean() * periods_per_year),
            "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1)), "hit": float((s > 0).mean())}


def run() -> dict:
    D = load()
    log.info("данные: %d дней × %d бумаг", *D["close"].shape)
    ic = ic_daily(D)
    pf = portfolios(D)
    res = {"params": {"top_n": TOP_N, "H": H, "quintile": Q, "cost_pct": COST},
           "days": int(len(D["close"])), "securities": int(D["close"].shape[1])}
    for per_name, per in (("pooled", (P1[0], P2[1])), ("P1_2014_2022", P1), ("P2_2022_2024", P2)):
        ics = ic[[in_period(d, per) for d in ic.index]]
        p = pf[[in_period(d, per) for d in pf["d"]]]
        res[per_name] = {
            "T1_ic_atr": {**tstat(ics, 1), "mean_ic": float(ics.mean()) if len(ics) else None},
            "T2_lowvol": tstat(p["lowvol_pct"], 252 / H),
            "T3_mom_12_1": tstat(p["mom_pct"], 252 / H),
            "T4_rev_1m": tstat(p["rev_pct"], 252 / H),
            "universe_ew_annual_pct": float(p["uni_pct"].mean() * 252 / H) if len(p) else None,
        }
    items = sorted((res["pooled"][k].get("p", 1.0), k) for k in ("T1_ic_atr", "T2_lowvol", "T3_mom_12_1", "T4_rev_1m"))
    run_p, m, holm = 0.0, len(items), {}
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        holm[k] = run_p
    res["holm_pooled"] = holm
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run()
    with open(os.path.join(DATA_DIR, "factors_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
