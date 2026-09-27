"""
Моментум 12-1 на ФЬЮЧЕРСАХ с плечом: лонг фьючерсов на 10 лучших бумаг, шорт
фьючерса на индекс, свободные деньги — в фонде денежного рынка.

Зачем фьючерсы (запрос пользователя 27.09.2026 «максимальная доходность с плечом»):
маржинальный кредит брокера — плата за непокрытую позицию ~27–35 % годовых
(research/cost_model.carry_pct), дороже, чем приносит любая стратегия программы, —
такое плечо уменьшает доходность. У фьючерса платы за перенос нет, нужен только
ГО (акции 18–27 %, индекс ~12 %), остаток лежит в фонде под ключевую ставку.
Лонг фьючерсов на акции против шорта фьючерса на индекс гасит базис, который
съедал ~10 п./год у варианта «акции + шорт MX» (hypotheses-abc, 24.09):
    доходность ≈ ставка × (1 − доля ГО) + L × (TR_топ − TR_индекса) − издержки.

Сигнал не новый: моментум 12-1, топ-10 равными весами, ребаланс раз в месяц —
тот, что прошёл предрегистрированный holdout (research/longhist/factors).
Новое — вселенная (только бумаги с фьючерсами, по ТЕКУЩЕМУ списку FORTS →
смещение выживания) и плечо. Это форвард-кандидат для песочницы, НЕ
подтверждённая стратегия: DSR моментума 0,019, значимость зависит от фазы сетки.

Плечо выбирается по ограничениям, а не по максимуму доходности в прошлом:
  * ГО: L × (ГО лонга + ГО индекса) ≤ 1 − запас;
  * маржин-колл: после худшего исторического месяца капитал ≥ требуемого ГО;
  * справочно — Келли L* = μ / σ² по избыточной доходности.

Данные: longhist (TQBR + TQBRH с дивидендами, 2014–2026), индекс полной доходности
MCFTR и ключевая ставка ЦБ (ISS / cbr.ru). Только чтение.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import urllib.request

import numpy as np
import pandas as pd

from research.vrp.backtest import key_rate_history, rate_on

# бумаги вселенной робота, у которых есть фьючерс на FORTS (ISS, 27.09.2026)
FUT_UNIVERSE = ("AFLT ALRS ASTR BSPB CBOM ENPG FEES FIXR FLOT GMKN HYDR IRAO LENT LKOH "
                "MAGN MGNT MOEX MVID PHOR POSI ROSN RTKM RUAL SMLT TATN UPRO VKCO VTBR "
                "X5 YDEX SBER GAZP SNGS SNGSP MTSS").split()
TOP_N = 10
GO_STOCK = 0.215           # медиана ГО фьючерсов на акции (ISS 27.09)
GO_INDEX = 0.119           # ГО MXZ6
MARGIN_BUFFER = 0.25       # доля капитала, которую ГО не должно занимать
STOCK_FUT_RT = 0.10        # круг фьючерса на акцию, % (консервативно)
INDEX_FUT_RT = 0.02        # круг MX, %
ROLLS_PER_YEAR = 4
MAX_ABS_R = 0.40


def load_longhist(path: str):
    frames = []
    for pref in ("tqbr", "tqbrh"):
        for p in sorted(glob.glob(os.path.join(path, f"{pref}_*.csv.gz"))):
            df = pd.read_csv(p)
            df["__p"] = pref
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["CLOSE"] > 0)].copy()
    df["d"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values(["SECID", "d", "__p"]).drop_duplicates(["SECID", "d"])
    close = df.pivot(index="d", columns="SECID", values="CLOSE").sort_index()
    div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name in ("dividends.csv", "dividends_tqbrh.csv"):
        dp = os.path.join(path, name)
        if not os.path.exists(dp):
            continue
        for r in pd.read_csv(dp).itertuples():
            if r.ticker not in div.columns:
                continue
            i = np.searchsorted(close.index.values, np.datetime64(r.last_buy_date), side="right")
            if i < len(close.index):
                div.iat[i, div.columns.get_loc(r.ticker)] += float(r.dividend_net)
    ret = (close + div) / close.shift(1) - 1.0
    return ret.where(ret.abs() <= MAX_ABS_R)


def iss_index(sec: str) -> pd.Series:
    out, s = {}, 0
    while True:
        u = (f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/"
             f"{sec}.json?iss.meta=off&from=2013-01-01&start={s}"
             f"&history.columns=TRADEDATE,CLOSE")
        h = json.loads(urllib.request.urlopen(u, timeout=60).read())["history"]["data"]
        for d, c in h:
            if c:
                out[pd.Timestamp(d)] = float(c)
        if len(h) < 100:
            return pd.Series(out).sort_index()
        s += len(h)


def run(longhist: str) -> dict:
    ret = load_longhist(longhist)
    names = [t for t in FUT_UNIVERSE if t in ret.columns]
    ret = ret[names]
    idx = iss_index("MCFTR").pct_change()
    rates = key_rate_history()
    month_ends = ret.resample("ME").last().index
    rows = []
    prev_top: set = set()
    for k in range(13, len(month_ends) - 1):
        t0 = month_ends[k]
        lb_a, lb_b = month_ends[k - 12], month_ends[k - 1]
        win = ret.loc[(ret.index > lb_a) & (ret.index <= lb_b)]
        ok = win.notna().sum() >= 0.8 * len(win)
        mom = (1 + win.fillna(0)).prod() - 1
        mom = mom[ok]
        if len(mom) < TOP_N + 5:
            continue
        top = list(mom.sort_values(ascending=False).index[:TOP_N])
        nxt = ret.loc[(ret.index > t0) & (ret.index <= month_ends[k + 1])]
        leg = (1 + nxt[top].fillna(0)).prod() - 1
        ix = idx.loc[(idx.index > t0) & (idx.index <= month_ends[k + 1])]
        if len(ix) < 10:
            continue
        spread = float(leg.mean() - ((1 + ix).prod() - 1))
        turnover = len(set(top) ^ prev_top) / (2 * TOP_N) if prev_top else 1.0
        prev_top = set(top)
        cost = (turnover * STOCK_FUT_RT + ROLLS_PER_YEAR / 12 * (STOCK_FUT_RT + INDEX_FUT_RT)) / 100
        rf = rate_on(rates, t0.date()) / 100 / 12
        rows.append({"month": month_ends[k + 1].strftime("%Y-%m"), "spread": spread,
                     "cost": cost, "rf": rf, "top": top})
    m = pd.DataFrame(rows).set_index("month")
    net = m.spread - m.cost                                   # на 1× нотионала
    mu, sd = float(net.mean()), float(net.std(ddof=1))
    worst = float(net.min())
    go = GO_STOCK + GO_INDEX
    l_margin = (1 - MARGIN_BUFFER) / go
    # после худшего месяца капитал E(1 + L·worst) должен покрывать ГО L·go·E
    l_call = 1 / (go - worst) if worst < 0 else l_margin
    l_kelly = mu / sd ** 2 if sd > 0 else float("nan")

    def total(L):
        cash = max(0.0, 1 - L * go)
        r = m.rf * cash + L * net
        eq = (1 + r).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        ann = float((1 + r).prod() ** (12 / len(r)) - 1)
        ex = r - m.rf
        t = float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex))))
        hold = r[r.index >= "2024-06"]
        return {"L": L, "annual": ann, "max_dd": dd, "worst_month": float(r.min()),
                "t_excess_vs_rf": t, "annual_rf": float((1 + m.rf).prod() ** (12 / len(r)) - 1),
                "holdout_annual": float((1 + hold).prod() ** (12 / len(hold)) - 1) if len(hold) else None}

    # максимум, допустимый ОБОИМИ ограничениями (шаг 0,05) — запрос «доходность
    # должна быть максимальной для теста»
    chosen = math.floor(min(l_margin, l_call) * 20) / 20
    grid = [total(L) for L in sorted({0.0, 1.0, 1.5, 2.0, chosen}) if L <= l_margin + 1e-9]
    return {"months": len(m), "period": f"{m.index[0]} … {m.index[-1]}",
            "names": len(names), "spread_mean_m": mu, "spread_sd_m": sd,
            "spread_t": mu / (sd / math.sqrt(len(net))), "worst_month_1x": worst,
            "l_margin": l_margin, "l_margin_call": l_call, "l_kelly": l_kelly,
            "L_chosen": chosen, "grid": grid, "chosen": total(chosen),
            "by_year": {y: float(g.sum()) for y, g in net.groupby(net.index.str[:4])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--longhist", default="/opt/etl-r4/audit/r4_research/longhist")
    ap.add_argument("--out-dir", default="/opt/research-data/levmom")
    a = ap.parse_args(argv)
    r = run(a.longhist)
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.flowkit import Checks, save_state, versioned
    chk = Checks("levmom-backtest")
    chk.ok("выбранное плечо не выше обоих ограничений",
           r["L_chosen"] <= min(r["l_margin"], r["l_margin_call"]) + 1e-9)
    g0 = next((g for g in r["grid"] if g["L"] == 0.0), None)
    chk.ok("тождество: без плеча доходность = ставка (все деньги в фонде)",
           g0 is not None and abs(g0["annual"] - g0["annual_rf"]) < 1e-12)
    chk.close("t спреда пересчитан из среднего и разброса",
              r["spread_t"], r["spread_mean_m"] / (r["spread_sd_m"] / math.sqrt(r["months"])), tol=1e-9)
    out = versioned(a.out_dir, "levmom_backtest", "json")
    save_state(out, params={"TOP_N": TOP_N, "GO_STOCK": GO_STOCK, "GO_INDEX": GO_INDEX,
                            "MARGIN_BUFFER": MARGIN_BUFFER, "STOCK_FUT_RT": STOCK_FUT_RT,
                            "INDEX_FUT_RT": INDEX_FUT_RT, "UNIVERSE": FUT_UNIVERSE},
               results=r, checks=chk, inputs={"longhist": a.longhist})
    print(chk.report())
    print("результат:", out)
    print(f"месяцев {r['months']} ({r['period']}), бумаг с фьючерсом {r['names']}")
    print(f"спред топ-10 − индекс (1×, нетто): {r['spread_mean_m']*100:+.2f} %/мес, "
          f"t {r['spread_t']:+.2f}, худший месяц {r['worst_month_1x']*100:+.1f} %")
    print("по годам, % спреда:", {y: round(v * 100, 1) for y, v in r["by_year"].items()})
    print(f"плечо: по ГО ≤ {r['l_margin']:.2f}, по маржин-коллу ≤ {r['l_margin_call']:.2f}, "
          f"Келли {r['l_kelly']:.2f} → выбрано {r['L_chosen']}")
    print(f"{'L':>5}{'годовых':>10}{'ставка':>9}{'просадка':>10}{'худш.мес':>10}{'t сверх ставки':>16}{'2024-06+':>10}")
    for g in r["grid"]:
        print(f"{g['L']:>5}{g['annual']*100:>9.1f}%{g['annual_rf']*100:>8.1f}%"
              f"{g['max_dd']*100:>9.1f}%{g['worst_month']*100:>9.1f}%{g['t_excess_vs_rf']:>16.2f}"
              f"{(g['holdout_annual'] or 0)*100:>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
