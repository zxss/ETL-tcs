"""
H-C2 — IPCA против статической факторной модели (волна 2).

ПРЕДРЕГИСТРАЦИЯ (до прогона):
  IPCA с нагрузками, инструментированными характеристиками, объясняет
  кросс-секцию ЛУЧШЕ статической факторной модели на тех же данных — по
  суммарному R² и по прогнозному R². Знак задан: выше.
  K = 3 фактора, зафиксировано заранее; перебора K нет.

Смысл для программы: понять, меняются ли нагрузки во времени. Если да — это
объясняет, почему одиночные факторы то работают, то нет (low-vol реален внутри
дня, но теряется на месяце).

Модель:   r_{i,t+1} = Z_{i,t} Γ f_{t+1} + ε,  Z — 12 характеристик (ранги,
          кросс-секционно стандартизованы) + константа.
Оценка:   ALS с нормировкой Γ'Γ = I.
R²:
  total      — с реализованными f_{t+1} (насколько структура вообще описывает);
  predictive — с безусловным средним f̄, оценённым ТОЛЬКО на обучении
               (настоящий прогноз: r̂ = Z_{i,t} Γ f̄).
Бенчмарк: статические факторы (PCA по доходностям), нагрузки на обучении,
          прогноз тем же способом через f̄.

Запуск (на сервере): python -m research.xsec.ipca
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.xsec import panel as P                          # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave2")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
S1 = (dt.date(2014, 1, 1), dt.date(2018, 12, 31))
S2 = (dt.date(2019, 1, 1), dt.date(2024, 5, 20))
K = 3
MAX_ITER, TOL = 200, 1e-6


def instrument(pan: pd.DataFrame) -> dict:
    """Z по датам: кросс-секционные ранги признаков в [-0.5, 0.5] + константа."""
    Z, R, names = {}, {}, None
    for d, g in pan.groupby("d"):
        g = g.dropna(subset=["y_rel"])
        if len(g) < 20:
            continue
        z = g[P.FEATURES].rank(pct=True) - 0.5
        z = z.fillna(0.0)
        z["const"] = 1.0
        Z[d] = z.to_numpy(float)
        R[d] = g["y_rel"].to_numpy(float)
        names = list(z.columns)
    return {"Z": Z, "R": R, "names": names}


def ipca_fit(Z: dict, R: dict, k: int = K) -> tuple[np.ndarray, dict]:
    dates = sorted(Z)
    L = Z[dates[0]].shape[1]
    rng = np.random.default_rng(0)
    G = np.linalg.qr(rng.normal(size=(L, k)))[0]
    f = {d: np.zeros(k) for d in dates}
    prev = np.inf
    for _ in range(MAX_ITER):
        for d in dates:
            A = Z[d] @ G
            f[d] = np.linalg.lstsq(A, R[d], rcond=None)[0]
        num = np.zeros((L * k,))
        den = np.zeros((L * k, L * k))
        for d in dates:
            zz = Z[d].T @ Z[d]
            ff = np.outer(f[d], f[d])
            den += np.kron(zz, ff)
            num += (np.outer(f[d], Z[d].T @ R[d])).T.reshape(-1)
        try:
            g = np.linalg.solve(den + 1e-8 * np.eye(len(den)), num)
        except np.linalg.LinAlgError:
            break
        G = g.reshape(L, k)
        G = np.linalg.qr(G)[0]
        sse = sum(float(((R[d] - Z[d] @ G @ f[d]) ** 2).sum()) for d in dates)
        if abs(prev - sse) < TOL * max(1.0, abs(prev)):
            break
        prev = sse
    return G, f


def static_fit(R: dict, tickers: dict, k: int = K) -> tuple[dict, np.ndarray]:
    """PCA по панели доходностей (бумаги × даты) на обучении → нагрузки β."""
    dates = sorted(R)
    all_tk = sorted({t for d in dates for t in tickers[d]})
    M = pd.DataFrame(index=all_tk, columns=dates, dtype=float)
    for d in dates:
        M.loc[tickers[d], d] = R[d]
    M = M.dropna(thresh=max(8, int(0.5 * len(dates))))
    X = M.fillna(0.0).to_numpy(float)
    if X.shape[0] < k + 2:
        return {}, np.zeros(k)
    U, S, Vt = np.linalg.svd(X - X.mean(axis=1, keepdims=True), full_matrices=False)
    beta = {tk: U[i, :k] * S[:k] for i, tk in enumerate(M.index)}
    fbar = Vt[:k].mean(axis=1)
    return beta, fbar


def r2(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = np.isfinite(a) & np.isfinite(p)
    if ok.sum() < 10:
        return float("nan")
    return float(1.0 - ((a[ok] - p[ok]) ** 2).sum() / (a[ok] ** 2).sum())


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    data = P.load()
    pan = P.build(data, S1[0], S2[1], 1, 50)
    inst = instrument(pan)
    Z, R = inst["Z"], inst["R"]
    tickers = {d: list(g["ticker"]) for d, g in pan.groupby("d") if d in Z}

    tr_dates = [d for d in sorted(Z) if S1[0] <= d <= S1[1]]
    te_dates = [d for d in sorted(Z) if S2[0] <= d <= S2[1]]
    print(f"IPCA: обучение {len(tr_dates)} дат, тест {len(te_dates)} дат, "
          f"признаков {len(inst['names'])}, K={K}")

    G, f_tr = ipca_fit({d: Z[d] for d in tr_dates}, {d: R[d] for d in tr_dates})
    fbar = np.mean([f_tr[d] for d in tr_dates], axis=0)
    beta, fbar_s = static_fit({d: R[d] for d in tr_dates},
                              {d: tickers[d] for d in tr_dates})

    a_all, p_ipca_pred, p_ipca_tot, p_stat = [], [], [], []
    for d in te_dates:
        r = R[d]
        a_all.append(r)
        p_ipca_pred.append(Z[d] @ G @ fbar)
        f_d = np.linalg.lstsq(Z[d] @ G, r, rcond=None)[0]           # реализованный f
        p_ipca_tot.append(Z[d] @ G @ f_d)
        b = np.array([beta.get(t, np.zeros(K)) for t in tickers[d]])
        p_stat.append(b @ fbar_s)
    a = np.concatenate(a_all)

    res = {"K": K, "n_train_dates": len(tr_dates), "n_test_dates": len(te_dates),
           "features": inst["names"],
           "R2_predictive": {"IPCA": r2(a, np.concatenate(p_ipca_pred)),
                             "STATIC": r2(a, np.concatenate(p_stat))},
           "R2_total": {"IPCA": r2(a, np.concatenate(p_ipca_tot))},
           "gamma_loadings": {n: [float(x) for x in G[i]]
                              for i, n in enumerate(inst["names"])}}
    res["verdict"] = {
        "predictive_ipca_better": bool(res["R2_predictive"]["IPCA"]
                                       > res["R2_predictive"]["STATIC"]),
        "note": "знак предрегистрирован: IPCA выше статики"}

    json.dump(res, open(os.path.join(OUT, "ipca.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"  R² прогнозный: IPCA {res['R2_predictive']['IPCA']:+.5f}  "
          f"статика {res['R2_predictive']['STATIC']:+.5f}")
    print(f"  R² суммарный (с реализованными факторами): IPCA {res['R2_total']['IPCA']:+.5f}")
    print("  веса Γ (первый фактор):")
    for n, v in sorted(res["gamma_loadings"].items(), key=lambda kv: -abs(kv[1][0]))[:6]:
        print(f"    {n:12s} {v[0]:+.3f}")
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave2_ipca_HC2", "trials": 2,
                             "revision": "wave2"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'ipca.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
