"""
Моментум как ПОРТФЕЛЬ в рублях: абсолютная доходность, хедж, просадки, TMON
(вопрос пользователя 24.09.2026 «можем накатить стратегию на демо?»).

Это НЕ новая гипотеза и не новые испытания реестра: сигнал тот же, что прошёл
предрегистрированный holdout (research/longhist/factors). Здесь считается лишь
исполнимость — относительный эффект (+11,5 %/год против равновесного универса)
переводится в деньги, потому что сам универс в 2024–26 падал ~20 %/год и
лонг-онли в абсолюте мог проигрывать TMON.

Варианты (выбор ноги хеджа сделан ПО МЕХАНИЗМУ, не по бэктесту):
  LO   лонг-онли верхний квинтиль 12-1;
  LO_X то же, минус аутсайдеры прошлого месяца (оба утверждения holdout);
  HEDGE LO_X минус фьючерс MX (шорт индекса, нотионал 1:1) — у фьючерса нет
        платы за перенос, в отличие от шорта акций (тариф «Премиум»: ночь от
        45 ₽/день на позицию ≈ 1,1 %/мес при позиции 100 тыс. ₽, это съедает
        весь эффект — поэтому шорт АКЦИЙ как ногу хеджа не рассматриваем);
  для справки LS_STOCKS — лонг-шорт на акциях с этой платой, чтобы показать её.
Бенчмарк: TMON (до листинга RUSFAR − 0,4 %/год), он же альтернатива деньгам r4.

Издержки: лонг-нога cost_model round_trip по факту оборота состава; фьючерс
0,02 % круг на ребаланс + 4 ролла в год; плата за перенос шорта акций —
cost_model.carry_pct при позиции POSITION_RUB.
Запуск (сервер): python -m research.longhist.portfolio
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                        # noqa: E402
from research.longhist import factors as fa                  # noqa: E402

log = logging.getLogger("research.longhist.portfolio")
DATA_DIR = fa.DATA_DIR
POSITION_RUB = 100_000.0
FUT_RT = 0.02                                   # круг фьючерса MX, %
FUT_ROLLS_PER_YEAR = 4
SAMPLES = {"long_2014_2024": ("tqbr", dt.date(2014, 1, 1), dt.date(2024, 5, 20)),
           "holdout_2024_2026": ("tqbrh", dt.date(2024, 5, 21), dt.date(2026, 9, 22))}


def mx_daily(conn) -> pd.Series:
    """Дневная доходность ближнего фьючерса MX (внутри контракта, ролл не даёт скачка)."""
    q = """SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
                  (array_agg(close ORDER BY ts DESC))[1] AS close, sum(volume) AS vol
           FROM research_fut_5m WHERE ticker LIKE 'MX%%' AND close > 0
           GROUP BY 1, 2"""
    df = pd.read_sql(q, conn)
    df["close"] = df["close"].astype(float)
    close = df.pivot(index="d", columns="ticker", values="close").sort_index()
    vol = df.pivot(index="d", columns="ticker", values="vol").astype(float).sort_index()
    near = vol.shift(1).idxmax(axis=1)
    out = {}
    days = list(close.index)
    for a, b in zip(days[:-1], days[1:]):
        tk = near.get(b)
        if tk and pd.notna(close.at[a, tk]) and pd.notna(close.at[b, tk]):
            out[b] = close.at[b, tk] / close.at[a, tk] - 1.0
    return pd.Series(out)


def money_daily() -> pd.Series:
    """Дневная доходность денежного рынка: TMON, до него RUSFAR − комиссия."""
    from research.treasury.carry import money_index
    idx, tmon = money_index("2013-06-01")
    s = tmon.combine_first(idx / idx.iloc[0] * float(tmon.dropna().iloc[0])
                           if len(tmon.dropna()) else idx)
    r = s.ffill().pct_change()
    return r[r.abs() < 0.01]                     # отсев склейки индексов


def legs(D: dict, spreads: dict) -> pd.DataFrame:
    """По каждому ребалансу: доходность ног за период удержания и издержки."""
    ret, atr, liq, close = D["ret"], D["atr"], D["liq"], D["close"]
    cum = (1.0 + ret.fillna(0.0)).cumprod()
    days = list(ret.index)
    rows, prev = [], {"LO": set(), "LO_X": set(), "SHORT": set()}
    for i in range(252, len(days) - fa.H, fa.H):
        d = days[i]
        u = [t for t in fa.universe(liq, d) if pd.notna(close.at[d, t])]
        if len(u) < 30:
            continue
        k = max(1, int(round(len(u) * fa.Q)))
        mom = (cum.iloc[i - 21][u] / cum.iloc[i - 252][u] - 1).dropna().sort_values(ascending=False)
        m1 = (cum.iloc[i][u] / cum.iloc[i - 21][u] - 1).dropna().sort_values()
        losers = set(m1.index[:k])
        sel = {"LO": list(mom.index[:k]),
               "LO_X": [t for t in mom.index if t not in losers][:k],
               "SHORT": list(losers)}
        hr = fa.hold_ret(ret, i, u)
        rec = {"d": d, "end": days[min(i + fa.H, len(days) - 1)], "uni": float(hr.mean())}
        for name, names in sel.items():
            s = set(names)
            turn = 1.0 if not prev[name] else 1 - len(s & prev[name]) / max(1, len(s))
            cost = float(np.mean([cm.round_trip(t, "base", spreads) for t in names])) * turn / 100.0
            rec[f"{name}_gross"] = float(hr[names].mean())
            rec[f"{name}_cost"] = cost
            prev[name] = s
        rows.append(rec)
    return pd.DataFrame(rows)


def window_ret(s: pd.Series, a, b) -> float:
    """NaN, если ряда на этом окне нет (MX с 2022, RUSFAR/TMON позже) — чтобы
    отсутствующий бенчмарк не превращался в ноль и не завышал сравнение."""
    if not len(s) or a < s.index.min() or b > s.index.max():
        return float("nan")
    w = s[(s.index > a) & (s.index <= b)]
    return float((1.0 + w).prod() - 1.0) if len(w) else float("nan")


def build(pf: pd.DataFrame, mx: pd.Series, mm: pd.Series) -> pd.DataFrame:
    fut_cost = (FUT_RT + FUT_RT * FUT_ROLLS_PER_YEAR * fa.H / 252) / 100.0
    carry = cm.carry_pct(POSITION_RUB, fa.H) / 100.0        # перенос шорта акций
    rows = []
    for _, r in pf.iterrows():
        mxr = window_ret(mx, r["d"], r["end"])
        mmr = window_ret(mm, r["d"], r["end"])
        lo = r["LO_gross"] - r["LO_cost"]
        lox = r["LO_X_gross"] - r["LO_X_cost"]
        sh = r["SHORT_gross"] + r["SHORT_cost"]             # шорт: издержки против нас
        rows.append({"d": r["d"], "uni": r["uni"], "mm": mmr, "mx": mxr,
                     "LO": lo, "LO_X": lox,
                     "HEDGE": lox - mxr - fut_cost,
                     # контроль: тот же хедж на ВСЁМ универсе — сколько даёт сам
                     # базис (дивиденды акций против ценового индекса) без моментума
                     "HEDGE_UNI": r["uni"] - mxr - fut_cost - 0.0014,
                     "LS_STOCKS": lox - sh - carry})
    return pd.DataFrame(rows)


def stats(s: pd.Series, per_year: float, bench: pd.Series | None = None) -> dict:
    if bench is not None:
        bench = bench[s.notna() & bench.notna()]
    s = s.dropna()
    n = len(s)
    if n < 6:
        return {"n": n}
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min() * 100)
    out = {"n": n, "annual_pct": float(((1 + s.mean()) ** per_year - 1) * 100),
           "total_pct": float((eq.iloc[-1] - 1) * 100), "max_dd_pct": dd,
           "hit": float((s > 0).mean())}
    if bench is not None and len(bench) >= 6:
        x = (s - bench).dropna()
        t = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if x.std(ddof=1) else None
        from scipy import stats as st
        out["vs_money_annual_pp"] = float((s.mean() - bench.mean()) * per_year * 100)
        out["vs_money_t"] = t
        out["vs_money_p"] = float(2 * st.t.sf(abs(t), len(x) - 1)) if t else None
    return out


def run(conn) -> dict:
    spreads = cm.load_spreads()
    mx, mm = mx_daily(conn), money_daily()
    per_year = 252 / fa.H
    res = {"params": {"position_rub": POSITION_RUB, "H": fa.H, "top_n": fa.TOP_N,
                      "fut_round_trip_pct": FUT_RT,
                      "stock_short_carry_pct_per_period": cm.carry_pct(POSITION_RUB, fa.H)}}
    for name, (prefix, d0, d1) in SAMPLES.items():
        D = fa.load(prefix)
        pf = legs(D, spreads)
        pf = pf[[d0 <= d <= d1 for d in pf["d"]]]
        b = build(pf, mx, mm)
        res[name] = {"rebalances": len(b),
                     "rebalances_with_money": int(b["mm"].notna().sum()),
                     "rebalances_with_mx": int(b["mx"].notna().sum()),
                     "money_annual_pct": float(((1 + b["mm"].mean()) ** per_year - 1) * 100),
                     "universe_annual_pct": float(((1 + b["uni"].mean()) ** per_year - 1) * 100),
                     "mx_annual_pct": float(((1 + b["mx"].mean()) ** per_year - 1) * 100)}
        for col in ("LO", "LO_X", "HEDGE", "HEDGE_UNI", "LS_STOCKS"):
            res[name][col] = stats(b[col], per_year, b["mm"])
        res[name]["rows"] = b.round(4).assign(d=b["d"].astype(str)).to_dict("records")
        log.info("%s готов", name)
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    with open(os.path.join(DATA_DIR, "portfolio_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "params"},
                     ensure_ascii=False, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
