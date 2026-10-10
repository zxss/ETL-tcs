"""
Испытания № 290–294 строго по research/crypto/PREREG.md (коммит 2fc94c5).

Что подаём:  <data>/daily/<SYMBOL>.csv из research.crypto.download.
Что запускаем: python -m research.crypto.trials --data <каталог> [--out <каталог>]
Что получаем:  <out>/crypto_trials_wNN.json — параметры, самопроверки, недельные и дневные
               ряды, итоги и вердикты; сводка в stdout.

Тайминг (без заглядывания вперёд): сигнал по закрытию дня s, вход по закрытию следующего
дня. № 290 — решение по закрытию t, позиция с закрытия t+1. № 291–293 — сигнал по закрытию
воскресенья, вход по закрытию понедельника, выход по закрытию следующего понедельника.
Монета, пропавшая из архива (делистинг), оценивается по последнему доступному закрытию.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.flowkit import Checks, save_state, versioned  # noqa: E402

START, DEV_END, END = dt.date(2019, 1, 1), dt.date(2023, 12, 31), dt.date(2026, 9, 30)
LIST_FROM = dt.date(2021, 1, 1)
N_TRIALS = 294
COST_MAJOR, COST_ALT = 0.001, 0.002
MAJORS = ("BTCUSDT", "ETHUSDT")
MIN_HIST, LIQ_DAYS, MIN_LIQ = 60, 30, 5e6
TSMOM_LB = 90
MOM_LB, REV_LB, VOL_LB = 28, 7, 30
TELEGRAM = ("TON", "NOT", "DOGS", "HMSTR", "CATI", "MAJOR", "BLUM", "X", "PX", "MEMEFI", "TAPS")


# ── данные ───────────────────────────────────────────────────────────────────

def load(data: str):
    files = sorted(glob.glob(os.path.join(data, "daily", "*.csv")))
    series = {}
    for f in files:
        sym = os.path.basename(f)[:-4]
        rows = {}
        with open(f) as fh:
            for r in csv.DictReader(fh):
                c = float(r["close"])
                if c > 0:
                    rows[dt.date.fromisoformat(r["date"])] = (c, float(r["quote_volume"]))
        if rows:
            series[sym] = rows
    first = min(min(s) for s in series.values())
    days = [first + dt.timedelta(days=i) for i in range((END - first).days + 1)]
    idx = {d: i for i, d in enumerate(days)}
    syms = sorted(series)
    C = np.full((len(days), len(syms)), np.nan)
    V = np.full((len(days), len(syms)), np.nan)
    for j, s in enumerate(syms):
        for d, (c, v) in series[s].items():
            if d in idx:
                C[idx[d], j], V[idx[d], j] = c, v
    return days, idx, syms, C, V


def last_close(C, j, i0, i1):
    """Последнее доступное закрытие пары j на отрезке [i0, i1] (делистинг — последняя цена)."""
    col = C[i0:i1 + 1, j]
    ok = np.where(~np.isnan(col))[0]
    return col[ok[-1]] if len(ok) else np.nan


# ── статистика ───────────────────────────────────────────────────────────────

def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def pval(t):
    return float(math.erfc(abs(t) / math.sqrt(2))) if math.isfinite(t) else float("nan")


def dsr(x, n):
    from scipy.stats import norm
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    v = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1)
    em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / n) + em * norm.ppf(1 - 1 / (n * math.e)))
    return float(norm.cdf((sr - sr0) / math.sqrt(v)))


def summary(x, dates):
    x = np.asarray(x, float)
    dev = np.array([v for v, d in zip(x, dates) if d <= DEV_END])
    hold = np.array([v for v, d in zip(x, dates) if d > DEV_END])
    return {"n": int(np.isfinite(x).sum()), "mean": float(np.nanmean(x)), "t": tstat(x),
            "dev_n": len(dev), "dev_mean": float(np.nanmean(dev)) if len(dev) else None, "dev_t": tstat(dev),
            "hold_n": len(hold), "hold_mean": float(np.nanmean(hold)) if len(hold) else None, "hold_t": tstat(hold)}


# ── № 290 тренд ──────────────────────────────────────────────────────────────

def tsmom(days, syms, C, cost_mult=1.0):
    out = {}
    i_start = days.index(START)
    for s in MAJORS:
        j = syms.index(s)
        c = C[:, j]
        r = np.full(len(days), np.nan)
        r[1:] = c[1:] / c[:-1] - 1
        sig = np.full(len(days), np.nan)
        sig[TSMOM_LB:] = (c[TSMOM_LB:] / c[:-TSMOM_LB] - 1 > 0).astype(float)
        sig[np.isnan(c)] = np.nan
        pos = np.full(len(days), np.nan)
        pos[2:] = sig[:-2]                     # решение по закрытию t → доходность дня t+2
        strat, bh, ds = [], [], []
        prev = 0.0
        for i in range(i_start, len(days)):
            if np.isnan(pos[i]) or np.isnan(r[i]):
                continue
            fee = abs(pos[i] - prev) * COST_MAJOR * cost_mult
            strat.append(pos[i] * r[i] - fee)
            bh.append(r[i])
            ds.append(days[i])
            prev = pos[i]
        strat, bh = np.array(strat), np.array(bh)
        sh = lambda x: float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if x.std(ddof=1) > 0 else float("nan")
        part = lambda m: (strat[m], bh[m])
        dev_m = np.array([d <= DEV_END for d in ds]); hold_m = ~dev_m
        res = {"summary": summary(strat, ds), "sharpe": sh(strat), "sharpe_bh": sh(bh),
               "dev_sharpe": sh(part(dev_m)[0]), "dev_sharpe_bh": sh(part(dev_m)[1]),
               "hold_sharpe": sh(part(hold_m)[0]), "hold_sharpe_bh": sh(part(hold_m)[1]),
               "annual": float((1 + strat).prod() ** (365 / len(strat)) - 1),
               "annual_bh": float((1 + bh).prod() ** (365 / len(bh)) - 1),
               "time_in_market": float(np.mean([p for p in pos[i_start:] if np.isfinite(p)]))}
        sm = res["summary"]
        res["passed"] = bool(sm["t"] >= 2 and sm["mean"] > 0
                             and res["dev_sharpe"] > res["dev_sharpe_bh"] and res["hold_sharpe"] > res["hold_sharpe_bh"]
                             and (sm["dev_mean"] or 0) > 0 and (sm["hold_mean"] or 0) > 0)
        out[s] = res
    return out


# ── № 291–293 недельные факторы ──────────────────────────────────────────────

def weekly(days, syms, C, V, cost_mult=1.0):
    idx = {d: i for i, d in enumerate(days)}
    mondays = [d for d in days if d.weekday() == 0 and START <= d <= END]
    rets = np.full_like(C, np.nan)
    rets[1:] = C[1:] / C[:-1] - 1
    cost = np.array([COST_MAJOR if s in MAJORS else COST_ALT for s in syms]) * cost_mult
    rows = {k: [] for k in ("mom", "rev", "lowvol")}
    prev_w = {k: {} for k in rows}
    meta = []
    for e, e2 in zip(mondays, mondays[1:]):
        ie, ie2 = idx[e], idx[e2]
        s = ie - 1                                         # сигнал — закрытие воскресенья
        if s - max(MIN_HIST, MOM_LB, VOL_LB) < 0:
            continue
        hist_ok = ~np.isnan(C[s - MIN_HIST + 1, :]) & ~np.isnan(C[s, :]) & ~np.isnan(C[ie, :])
        liq = np.nanmedian(V[s - LIQ_DAYS + 1:s + 1, :], axis=0)
        univ = np.where(hist_ok & (liq >= MIN_LIQ))[0]
        if len(univ) < 10:
            continue
        exit_px = np.array([last_close(C, j, ie, ie2) for j in univ])
        wr = exit_px / C[ie, univ] - 1
        ur = float(np.mean(wr))
        mom = C[s - 1, univ] / C[s - MOM_LB, univ] - 1
        rev = C[s, univ] / C[s - REV_LB, univ] - 1
        vol = np.nanstd(rets[s - VOL_LB + 1:s + 1, univ], axis=0, ddof=1)
        k = max(1, len(univ) // 5)
        picks = {"mom": np.argsort(-np.nan_to_num(mom, nan=-np.inf))[:k],
                 "rev": np.argsort(np.nan_to_num(rev, nan=np.inf))[:k],
                 "lowvol": np.argsort(np.nan_to_num(vol, nan=np.inf))[:k]}
        for name, p in picks.items():
            js = univ[p]
            w = {int(j): 1.0 / len(js) for j in js}
            turn = sum(abs(w.get(j, 0) - prev_w[name].get(j, 0)) for j in set(w) | set(prev_w[name]))
            fee = sum(abs(w.get(j, 0) - prev_w[name].get(j, 0)) * cost[j] for j in set(w) | set(prev_w[name]))
            pr = float(np.mean(wr[p]))
            rows[name].append({"week": str(e), "port": pr, "univ": ur, "fee": fee, "turnover": turn,
                               "excess": pr - fee - ur, "n": int(len(js))})
            prev_w[name] = w
        meta.append({"week": str(e), "universe": int(len(univ))})
    return rows, meta


# ── № 294 листинги ───────────────────────────────────────────────────────────

def listings(days, syms, C):
    jb = syms.index("BTCUSDT")
    out = []
    for j, s in enumerate(syms):
        ok = np.where(~np.isnan(C[:, j]))[0]
        if not len(ok):
            continue
        d0 = days[ok[0]]
        if d0 < LIST_FROM or d0 + dt.timedelta(days=97) > END or s == "BTCUSDT":
            continue
        i7, i97 = ok[0] + 7, ok[0] + 97
        p7 = last_close(C, j, ok[0], i7)
        p97 = last_close(C, j, i7, i97)
        if not (p7 > 0 and p97 > 0):
            continue
        r = p97 / p7 - 1
        rb = C[i97, jb] / C[i7, jb] - 1
        out.append({"symbol": s, "listed": str(d0), "ret": float(r), "btc": float(rb), "excess": float(r - rb),
                    "delisted_in_window": bool(np.isnan(C[i97, j])),
                    "telegram": s[:-4] in TELEGRAM})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    days, idx, syms, C, V = load(a.data)
    chk = Checks("crypto-trials")
    chk.ok("BTC и ETH есть в данных", all(s in syms for s in MAJORS))
    chk.ok("история доходит до конца периода", np.isfinite(C[idx[END], syms.index("BTCUSDT")]))

    t290 = {"x1": tsmom(days, syms, C, 1.0), "x2": tsmom(days, syms, C, 2.0)}
    w1, meta = weekly(days, syms, C, V, 1.0)
    w2, _ = weekly(days, syms, C, V, 2.0)
    fac = {}
    for name, trial in (("mom", 291), ("rev", 292), ("lowvol", 293)):
        ex = [r["excess"] for r in w1[name]]
        ws = [dt.date.fromisoformat(r["week"]) for r in w1[name]]
        sm = summary(ex, ws)
        sm2 = summary([r["excess"] for r in w2[name]], ws)
        fac[name] = {"trial": trial, "summary": sm, "summary_x2": sm2, "p": pval(sm["t"]), "dsr": dsr(ex, N_TRIALS),
                     "annual_port": float(np.prod([1 + r["port"] - r["fee"] for r in w1[name]]) ** (52 / len(ex)) - 1),
                     "annual_univ": float(np.prod([1 + r["univ"] for r in w1[name]]) ** (52 / len(ex)) - 1),
                     "turnover": float(np.mean([r["turnover"] for r in w1[name]]))}
    # Холм внутри трёх
    order = sorted(fac, key=lambda k: fac[k]["p"])
    for rank, k in enumerate(order):
        fac[k]["holm_p"] = min(1.0, fac[k]["p"] * (3 - rank))
    for k, f in fac.items():
        sm, sm2 = f["summary"], f["summary_x2"]
        f["passed"] = bool(sm["mean"] > 0 and sm["t"] >= 2 and (sm["dev_mean"] or 0) > 0 and (sm["hold_mean"] or 0) > 0
                           and f["holm_p"] < 0.05 and f["dsr"] >= 0.95 and sm2["mean"] > 0)

    lst = listings(days, syms, C)
    ex = [r["excess"] for r in lst]
    dates = [dt.date.fromisoformat(r["listed"]) for r in lst]
    t294 = {"summary": summary(ex, dates), "median": float(np.median(ex)) if ex else None,
            "share_under": float(np.mean(np.array(ex) < 0)) if ex else None,
            "telegram": [r for r in lst if r["telegram"]]}
    s294 = t294["summary"]
    t294["passed"] = bool(s294["mean"] < 0 and s294["t"] <= -2 and (s294["dev_mean"] or 0) < 0 and (s294["hold_mean"] or 0) < 0)

    # самопроверки
    for name in w1:
        chk.ok(f"{name}: недель одинаково при ×1 и ×2", len(w1[name]) == len(w2[name]))
        chk.ok(f"{name}: портфель ≤ квинтиля вселенной", all(r["n"] <= max(1, m["universe"] // 5)
                                                           for r, m in zip(w1[name], meta)))
        chk.close(f"{name}: средний избыток пересчитан", fac[name]["summary"]["mean"],
                  float(np.mean([r["port"] - r["fee"] - r["univ"] for r in w1[name]])), tol=1e-12)
        chk.ok(f"{name}: издержки ×2 не лучше ×1", fac[name]["summary_x2"]["mean"] <= fac[name]["summary"]["mean"] + 1e-15)
    chk.ok("недель в выборке ≥ 350", len(meta) >= 350, str(len(meta)))
    chk.ok("вселенная ≥ 10 каждую неделю", all(m["universe"] >= 10 for m in meta))
    for s in MAJORS:
        chk.ok(f"290 {s}: ×2 не лучше ×1", t290["x2"][s]["summary"]["mean"] <= t290["x1"][s]["summary"]["mean"] + 1e-15)
    chk.ok("294: листингов ≥ 100", len(lst) >= 100, str(len(lst)))

    out = versioned(a.out or os.path.join(a.data, "results"), "crypto_trials", "json")
    save_state(out, params={"prereg": "research/crypto/PREREG.md", "START": START, "DEV_END": DEV_END, "END": END,
                            "N_TRIALS": N_TRIALS, "costs": [COST_MAJOR, COST_ALT], "MIN_LIQ": MIN_LIQ},
               results={"t290": t290, "factors": fac, "t294": t294, "weekly": w1, "universe": meta,
                        "listings": lst, "symbols": len(syms)},
               checks=chk, inputs={"data": a.data})
    print(chk.report())
    pct = lambda v: "—" if v is None or not math.isfinite(v) else f"{v * 100:+.2f}"
    for s, r in t290["x1"].items():
        sm = r["summary"]
        print(f"№290 {s}: {r['annual'] * 100:6.1f} %/год против «держать» {r['annual_bh'] * 100:6.1f}; Шарп {r['sharpe']:.2f} против {r['sharpe_bh']:.2f}; "
              f"разработка {r['dev_sharpe']:.2f}/{r['dev_sharpe_bh']:.2f}, отложенная {r['hold_sharpe']:.2f}/{r['hold_sharpe_bh']:.2f}; "
              f"в рынке {r['time_in_market'] * 100:.0f} %; ×2 Шарп {t290['x2'][s]['sharpe']:.2f} → {'ПРОЙДЕН' if r['passed'] else 'не пройден'}")
    for k, f in fac.items():
        sm = f["summary"]
        print(f"№{f['trial']} {k}: избыток {pct(sm['mean'])} %/нед t {sm['t']:+.2f} (разр. {pct(sm['dev_mean'])} t {sm['dev_t']:+.2f}; "
              f"отл. {pct(sm['hold_mean'])} t {sm['hold_t']:+.2f}); ×2 {pct(f['summary_x2']['mean'])}; Холм {f['holm_p']:.3f}; DSR {f['dsr']:.3f}; "
              f"портфель {f['annual_port'] * 100:.0f} %/год, вселенная {f['annual_univ'] * 100:.0f} %; оборот {f['turnover']:.2f} → "
              f"{'ПРОЙДЕН' if f['passed'] else 'не пройден'}")
    print(f"№294 листинги: n {s294['n']}, избыток к BTC за 90 дней {pct(s294['mean'])} % (медиана {pct(t294['median'])}), t {s294['t']:+.2f}; "
          f"2021–23 {pct(s294['dev_mean'])} (n {s294['dev_n']}), 2024–26 {pct(s294['hold_mean'])} (n {s294['hold_n']}); "
          f"отстали {t294['share_under'] * 100:.0f} % → {'ПРОЙДЕН' if t294['passed'] else 'не пройден'}")
    for r in t294["telegram"]:
        print(f"   телеграм {r['symbol']}: листинг {r['listed']}, 90 дней {pct(r['ret'])} %, к BTC {pct(r['excess'])} %")
    print("результат:", out)
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
