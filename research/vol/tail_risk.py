"""
H-V3 — GARCH-EVT-Copula: хвостовой риск ночного портфеля (волна 4).

Зачем: у контура нет НИКАКОЙ модели хвостового риска, при том что на ночь
выносятся позиции по 100 000 ₽, а на боевом счёте «Робот» — реальные деньги.
Это не про доходность, а про то, чтобы знать размер возможного убытка.

ПРЕДРЕГИСТРАЦИЯ (до прогона):
  Двухшаговая схема (GARCH-фильтр → EVT/GPD на хвостах стандартизованных
  остатков → гауссова копула для зависимости) даёт VaR/ES ночного портфеля,
  ПРОХОДЯЩИЙ формальные тесты покрытия (Kupiec LR_uc и Christoffersen LR_cc)
  на уровнях 95 % и 99 %. Бенчмарки: историческое моделирование (250 дней) и
  нормальное приближение с EWMA-волатильностью.
  Ровно 2 испытания: уровень 95 % и уровень 99 %.

Портфель: та же прокси-корзина, что в H-V2/H-R1 — топ-5 по обороту, равный вес,
экспозиция close[D] → open[D+1]. Риск, который меряем, — ночной гэп.

Запуск (на сервере): python -m research.vol.tail_risk
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import database                                               # noqa: E402
from research.vol import har as H                             # noqa: E402
from research.vol import overnight_base as OB                 # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave4")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
LEVELS = (0.95, 0.99)
TRAIN_MIN, REFIT = 250, 63
N_SIM = 20000
TAIL_Q = 0.10
POSITION_RUB = 100_000.0


# ── маргиналы: эмпирика в центре, GPD на хвостах ─────────────────────────────

def fit_gpd(x: np.ndarray):
    """MLE обобщённого Парето для превышений (x > 0)."""
    from scipy.stats import genpareto
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 25:
        return None
    try:
        c, loc, scale = genpareto.fit(x, floc=0.0)
        return (float(c), float(scale)) if np.isfinite(c) and scale > 0 else None
    except Exception:                                          # noqa: BLE001
        return None


class SemiParametric:
    """F(z): эмпирика между квантилями TAIL_Q и 1−TAIL_Q, GPD за ними."""

    def __init__(self, z: np.ndarray):
        z = np.sort(z[np.isfinite(z)])
        self.z = z
        self.n = len(z)
        self.lo = float(np.quantile(z, TAIL_Q))
        self.hi = float(np.quantile(z, 1.0 - TAIL_Q))
        self.g_hi = fit_gpd(z[z > self.hi] - self.hi)
        self.g_lo = fit_gpd(self.lo - z[z < self.lo])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        from scipy.stats import genpareto
        u = np.clip(np.asarray(u, float), 1e-6, 1 - 1e-6)
        out = np.quantile(self.z, u)
        if self.g_hi is not None:
            m = u > 1.0 - TAIL_Q
            if m.any():
                uu = (u[m] - (1.0 - TAIL_Q)) / TAIL_Q
                out[m] = self.hi + genpareto.ppf(np.clip(uu, 0, 1 - 1e-9),
                                                 self.g_hi[0], 0.0, self.g_hi[1])
        if self.g_lo is not None:
            m = u < TAIL_Q
            if m.any():
                uu = (TAIL_Q - u[m]) / TAIL_Q
                out[m] = self.lo - genpareto.ppf(np.clip(uu, 0, 1 - 1e-9),
                                                 self.g_lo[0], 0.0, self.g_lo[1])
        return out


# ── бэктесты покрытия ────────────────────────────────────────────────────────

def kupiec(hits: np.ndarray, p: float) -> dict:
    n, x = len(hits), int(hits.sum())
    if n < 50:
        return {"n": n, "x": x}
    pi = x / n
    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return x * math.log(q) + (n - x) * math.log(1 - q)
    lr = -2.0 * (ll(p) - ll(pi))
    from scipy.stats import chi2
    return {"n": n, "x": x, "rate": pi, "expected": p, "LR_uc": float(lr),
            "p": float(chi2.sf(lr, 1))}


def christoffersen(hits: np.ndarray, p: float) -> dict:
    from scipy.stats import chi2
    h = hits.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    def safe(a, b):
        return a / b if b > 0 else 0.0
    p01, p11 = safe(n01, n00 + n01), safe(n11, n10 + n11)
    pi = safe(n01 + n11, n00 + n01 + n10 + n11)
    def ll(q, k, m):
        if q <= 0 or q >= 1:
            return 0.0
        return k * math.log(q) + m * math.log(1 - q)
    l0 = ll(pi, n01 + n11, n00 + n10)
    l1 = ll(p01, n01, n00) + ll(p11, n11, n10)
    lr_ind = -2.0 * (l0 - l1)
    uc = kupiec(hits, p)
    lr_cc = lr_ind + uc.get("LR_uc", 0.0)
    return {"LR_ind": float(lr_ind), "p_ind": float(chi2.sf(lr_ind, 1)),
            "LR_cc": float(lr_cc), "p_cc": float(chi2.sf(lr_cc, 2))}


def es_backtest(loss: np.ndarray, var: np.ndarray, es: np.ndarray) -> dict:
    """Acerbi–Szekely Z2: среднее (loss/ES − 1) по превышениям должно быть ≈ 0."""
    m = loss > var
    if m.sum() < 5:
        return {"exceedances": int(m.sum())}
    z = float(np.mean(loss[m] / np.maximum(es[m], 1e-9) - 1.0))
    return {"exceedances": int(m.sum()), "Z2": z,
            "verdict": "ES занижен" if z > 0.10 else ("ES завышен" if z < -0.10 else "ES адекватен")}


# ── основной прогон ──────────────────────────────────────────────────────────

def overnight_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    close = daily.pivot(index="d", columns="ticker", values="close").sort_index()
    open_ = daily.pivot(index="d", columns="ticker", values="open").sort_index()
    on = (open_.shift(-1) / close - 1.0) * 100.0
    return on.iloc[:-1]


def run_sample(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    on = overnight_matrix(daily)
    picks = trades.groupby("d")["ticker"].apply(list)
    names = sorted({t for lst in picks for t in lst})
    on = on[[c for c in names if c in on.columns]].copy()
    days = [d for d in on.index if d in picks.index]
    if len(days) < TRAIN_MIN + 60:
        return {"error": "мало дней"}

    rng = np.random.default_rng(7)
    rows = []
    garch_par: dict[str, tuple] = {}
    sig_path: dict[str, np.ndarray] = {}
    marg: dict[str, SemiParametric] = {}
    corr = None
    last_fit = -10**9

    all_days = list(on.index)
    pos = {d: i for i, d in enumerate(all_days)}
    for d in days:
        i = pos[d]
        if i < TRAIN_MIN:
            continue
        if i - last_fit >= REFIT:
            last_fit = i
            resid = {}
            for tk in on.columns:
                r = on[tk].to_numpy(float)[:i]
                r = np.where(np.isfinite(r), r, 0.0)
                if np.count_nonzero(r) < 150:
                    continue
                w, a, b = H.garch11_fit(r)
                garch_par[tk] = (w, a, b)
                s2 = H.garch_path(on[tk].to_numpy(float), w, a, b)
                sig_path[tk] = np.sqrt(np.maximum(s2, 1e-9))
                z = r / np.maximum(sig_path[tk][:i], 1e-9)
                z = z[np.isfinite(z)]
                if len(z) > 150:
                    marg[tk] = SemiParametric(z)
                    resid[tk] = pd.Series(z[-500:])
            if len(resid) >= 2:
                Zr = pd.DataFrame(resid).dropna(axis=1, how="all")
                rho_s = Zr.corr(method="spearman").to_numpy()
                corr_m = 2.0 * np.sin(np.pi * rho_s / 6.0)
                corr_m = np.nan_to_num(corr_m, nan=0.0)
                np.fill_diagonal(corr_m, 1.0)
                ev, V_ = np.linalg.eigh(corr_m)
                ev = np.maximum(ev, 1e-6)
                corr_m = V_ @ np.diag(ev) @ V_.T
                dsc = np.sqrt(np.diag(corr_m))
                corr = (corr_m / np.outer(dsc, dsc), list(Zr.columns))

        sel = [t for t in picks.loc[d] if t in marg and t in sig_path]
        if len(sel) < 3 or corr is None:
            continue
        cols = corr[1]
        idx = [cols.index(t) for t in sel if t in cols]
        sel = [t for t in sel if t in cols]
        if len(sel) < 3:
            continue
        C = corr[0][np.ix_(idx, idx)]
        try:
            L = np.linalg.cholesky(C + 1e-8 * np.eye(len(C)))
        except np.linalg.LinAlgError:
            continue
        from scipy.stats import norm
        g = rng.standard_normal((N_SIM, len(sel))) @ L.T
        u = norm.cdf(g)
        sim = np.column_stack([marg[t].ppf(u[:, j]) * sig_path[t][i]
                               for j, t in enumerate(sel)])
        port = sim.mean(axis=1)                       # равный вес
        actual = float(np.nanmean([on.at[d, t] for t in sel]))
        rec = {"d": d, "actual_pct": actual, "n_names": len(sel)}
        # исторический и нормальный бенчмарки на уровне портфеля
        hist_port = np.nanmean(on[sel].to_numpy(float)[max(0, i - 250):i], axis=1)
        lam = 0.94
        w_ = lam ** np.arange(len(hist_port))[::-1]
        mu_h = float(np.nansum(w_ * hist_port) / np.nansum(w_))
        sd_h = math.sqrt(float(np.nansum(w_ * (hist_port - mu_h) ** 2) / np.nansum(w_)))
        for lv in LEVELS:
            q = 1.0 - lv
            rec[f"evt_var_{lv}"] = -float(np.quantile(port, q))
            tail = port[port <= np.quantile(port, q)]
            rec[f"evt_es_{lv}"] = -float(tail.mean()) if len(tail) else np.nan
            hp = hist_port[np.isfinite(hist_port)]
            rec[f"hist_var_{lv}"] = -float(np.quantile(hp, q)) if len(hp) > 50 else np.nan
            ht = hp[hp <= np.quantile(hp, q)] if len(hp) > 50 else np.array([])
            rec[f"hist_es_{lv}"] = -float(ht.mean()) if len(ht) else np.nan
            rec[f"norm_var_{lv}"] = -float(mu_h + sd_h * norm.ppf(q))
            rec[f"norm_es_{lv}"] = -float(mu_h - sd_h * norm.pdf(norm.ppf(q)) / q)
        rows.append(rec)

    bt = pd.DataFrame(rows)
    if len(bt) < 100:
        return {"error": f"мало оценок: {len(bt)}"}
    res: dict = {"n_days": int(len(bt)), "names": len(names),
                 "period": [str(bt['d'].min()), str(bt['d'].max())], "levels": {}}
    loss = -bt["actual_pct"].to_numpy(float)
    for lv in LEVELS:
        block = {}
        for model in ("evt", "hist", "norm"):
            var = bt[f"{model}_var_{lv}"].to_numpy(float)
            es = bt[f"{model}_es_{lv}"].to_numpy(float)
            ok = np.isfinite(var) & np.isfinite(loss)
            hits = (loss[ok] > var[ok]).astype(int)
            k = kupiec(hits, 1.0 - lv)
            c = christoffersen(hits, 1.0 - lv)
            block[model] = {"kupiec": k, "christoffersen": c,
                            "es": es_backtest(loss[ok], var[ok], es[ok]),
                            "mean_var_pct": float(np.nanmean(var)),
                            "pass_uc": bool(k.get("p", 0) > 0.05),
                            "pass_cc": bool(c.get("p_cc", 0) > 0.05)}
        res["levels"][str(lv)] = block
        # лимит экспозиции: при VaR 99 % на позицию 100 000 ₽
        if lv == 0.99:
            res["limit"] = {
                "var99_mean_pct": block["evt"]["mean_var_pct"],
                "loss_per_position_rub": block["evt"]["mean_var_pct"] / 100.0 * POSITION_RUB,
                "note": "средний ночной VaR 99 % на одну позицию 100 000 ₽"}
    return res


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    conn = database.get_connection()
    out: dict = {"design": "GARCH(1,1) → GPD-хвосты → гауссова копула; "
                           "бенчмарки: историческое моделирование, нормальное",
                 "samples": {}}
    for sname, (table, d0, d1) in OB.SAMPLES.items():
        daily = OB.load_daily(conn, table, d0, d1)
        trades = OB.basket(daily)
        if trades.empty:
            continue
        r = run_sample(daily, trades)
        out["samples"][sname] = r
        print(f"\n=== {sname} ===")
        if "error" in r:
            print("  " + r["error"])
            continue
        print(f"  дней в бэктесте: {r['n_days']}, бумаг: {r['names']}")
        for lv, block in r["levels"].items():
            print(f"  уровень {lv}:")
            for model, b in block.items():
                k = b["kupiec"]
                print(f"    {model:5s} VaR {b['mean_var_pct']:5.2f} %  "
                      f"превышений {k.get('rate', float('nan')):.4f} "
                      f"(ожидание {k.get('expected', float('nan')):.4f})  "
                      f"Kupiec p={k.get('p', float('nan')):.3f} "
                      f"{'OK' if b['pass_uc'] else 'НЕ ПРОШЁЛ'}  "
                      f"CC p={b['christoffersen'].get('p_cc', float('nan')):.3f} "
                      f"{'OK' if b['pass_cc'] else 'НЕ ПРОШЁЛ'}  "
                      f"ES: {b['es'].get('verdict', '—')}")
        if "limit" in r:
            print(f"  → ночной VaR 99 % ≈ {r['limit']['var99_mean_pct']:.2f} % = "
                  f"{r['limit']['loss_per_position_rub']:,.0f} ₽ на позицию 100 000 ₽")

    json.dump(out, open(os.path.join(OUT, "tail_risk.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave4_tail_risk_HV3", "trials": 2,
                             "revision": "wave4"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'tail_risk.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
