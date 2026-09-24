"""
H-C1 — нелинейный комбинатор против лучшего одиночного фактора (волна 2).

ПРЕДРЕГИСТРАЦИЯ (до прогона):
  Градиентный бустинг по панели из 12 характеристик даёт БОЛЕЕ ВЫСОКИЙ IC и
  БОЛЕЕ ВЫСОКУЮ доходность квинтильного портфеля ПОСЛЕ ИЗДЕРЖЕК, чем сортировка
  по одному моментуму 12-1. Знак задан: выше, а не «отличается».
  Моделей ровно три (перебора нет, гиперпараметры зафиксированы в коде):
    EN   — ElasticNet (α по внутренней CV на обучении, это не испытание);
    GBM  — HistGradientBoostingRegressor(depth 3, 200 итераций, lr 0.05);
    AVG  — среднее рангов EN и GBM.
  Бенчмарк: сортировка по mom_12_1 (единственный выживший сигнал программы).

Выборки (план 04_plan §2):
  S1 2014-01-01…2018-12-31 — разработка, свободно;
  S2 2019-01-01…2024-05-20 — ОДИН взгляд замороженной конфигурацией;
  S3 2024-05-21…2026-09-22 — НЕ трогаем (остался один выстрел на программу).

Протокол: расширяющееся окно, переобучение на каждый ребаланс, эмбарго 21
торговый день + горизонт между обучением и целью. Издержки 0,2 % × доля
сменившегося состава — как в research/longhist/factors.py, чтобы числа были
сравнимы с моментумом.

Запуск (на сервере): python -m research.xsec.ml_xsec
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

from research.validation import core as V                     # noqa: E402
from research.xsec import panel as P                          # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "wave2")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
S1 = (dt.date(2014, 1, 1), dt.date(2018, 12, 31))
S2 = (dt.date(2019, 1, 1), dt.date(2024, 5, 20))
Q, COST = 0.2, 0.2
EMBARGO_DAYS = 42                       # 21 эмбарго + 21 горизонт цели
MIN_TRAIN = 600
MODELS = ("EN", "GBM", "AVG")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    feats = P.FEATURES
    Xtr = train[feats].to_numpy(float)
    ytr = train["y_rel"].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[ok], ytr[ok]
    if len(ytr) < MIN_TRAIN:
        return {}
    med = np.nanmedian(Xtr, axis=0)
    Xte = np.where(np.isfinite(Xte), Xte, med)

    sc = StandardScaler().fit(Xtr)
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000, n_jobs=1)
    en.fit(sc.transform(Xtr), ytr)
    p_en = en.predict(sc.transform(Xte))

    gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                        min_samples_leaf=20, l2_regularization=1.0,
                                        random_state=0)
    gbm.fit(Xtr, ytr)
    p_gbm = gbm.predict(Xte)

    r_en = pd.Series(p_en).rank(pct=True).to_numpy()
    r_gb = pd.Series(p_gbm).rank(pct=True).to_numpy()
    return {"EN": p_en, "GBM": p_gbm, "AVG": (r_en + r_gb) / 2.0}


def walk_forward(pan: pd.DataFrame, eval_from: dt.date, eval_to: dt.date) -> pd.DataFrame:
    dates = sorted(d for d in pan["d"].unique() if eval_from <= d <= eval_to)
    out = []
    for d in dates:
        cut = d - dt.timedelta(days=int(EMBARGO_DAYS * 1.45))    # календарный запас
        train = pan[pan["d"] <= cut]
        test = pan[pan["d"] == d].copy()
        if len(train) < MIN_TRAIN or len(test) < 20:
            continue
        preds = fit_predict(train, test)
        if not preds:
            continue
        for k, v in preds.items():
            test[f"p_{k}"] = v
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(fc: pd.DataFrame) -> dict:
    from scipy import stats
    res: dict = {"rebalances": int(fc["d"].nunique())}
    signals = {m: f"p_{m}" for m in MODELS}
    signals["MOM"] = "mom_12_1"

    ic_rows, pf_rows = [], []
    prev: dict[str, set] = {k: set() for k in signals}
    for d, g in fc.groupby("d"):
        g = g.dropna(subset=["y_rel"])
        if len(g) < 20:
            continue
        rec_ic, rec_pf = {"d": d}, {"d": d, "uni_pct": float(g["y"].mean()) * 100.0}
        k = max(1, int(round(len(g) * Q)))
        for name, col in signals.items():
            s = g[col]
            m = s.notna()
            if m.sum() < 20:
                continue
            rec_ic[name] = float(stats.spearmanr(s[m], g["y_rel"][m])[0])
            top = set(g.loc[m].sort_values(col, ascending=False)["ticker"].iloc[:k])
            turn = 1.0 if not prev[name] else 1.0 - len(top & prev[name]) / max(1, len(top))
            sel = g[g["ticker"].isin(top)]
            rec_pf[name] = float(sel["y_rel"].mean()) * 100.0 - COST * turn
            prev[name] = top
        ic_rows.append(rec_ic)
        pf_rows.append(rec_pf)

    ic = pd.DataFrame(ic_rows)
    pf = pd.DataFrame(pf_rows)
    per_year = 252.0 / P.H
    res["ic"] = {}
    res["portfolio"] = {}
    for name in signals:
        if name in ic:
            s = ic[name].dropna()
            res["ic"][name] = {"mean": float(s.mean()), "n": int(len(s)),
                               "t": float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s))))
                               if len(s) > 3 and s.std(ddof=1) > 0 else None}
        if name in pf:
            s = pf[name].dropna()
            t = (float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s))))
                 if len(s) > 3 and s.std(ddof=1) > 0 else None)
            res["portfolio"][name] = {"mean_pct": float(s.mean()), "n": int(len(s)),
                                      "annual_pct": float(s.mean() * per_year),
                                      "t": t, "hit": float((s > 0).mean())}
    # прямое сравнение с бенчмарком: парная разность по ребалансам
    res["vs_MOM"] = {}
    pvals = {}
    for name in MODELS:
        if name not in pf or "MOM" not in pf:
            continue
        d_ic = (ic[name] - ic["MOM"]).dropna()
        d_pf = (pf[name] - pf["MOM"]).dropna()
        t_ic = V.dm_test(-d_ic.to_numpy(), np.zeros(len(d_ic)))    # знак: выше = лучше
        t_pf = V.dm_test(-d_pf.to_numpy(), np.zeros(len(d_pf)))
        p_ic = t_ic["p"] / 2.0 if t_ic["t"] < 0 else 1.0 - t_ic["p"] / 2.0
        p_pf = t_pf["p"] / 2.0 if t_pf["t"] < 0 else 1.0 - t_pf["p"] / 2.0
        res["vs_MOM"][name] = {
            "d_ic_mean": float(d_ic.mean()), "d_ic_t": -t_ic["t"], "d_ic_p1": p_ic,
            "d_pf_mean_pct": float(d_pf.mean()), "d_pf_t": -t_pf["t"], "d_pf_p1": p_pf,
            "d_pf_annual_pp": float(d_pf.mean() * per_year)}
        pvals[f"{name}_pf"] = p_pf
    res["holm_portfolio"] = V.holm(pvals)
    return res


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    data = P.load()
    pan = P.build(data, S1[0], S2[1], 1, 50)
    print(f"панель: {len(pan)} строк, {pan['d'].nunique()} дат, "
          f"{pan['ticker'].nunique()} бумаг, {pan['d'].min()}…{pan['d'].max()}")
    pan.to_csv(os.path.join(OUT, "panel_top50.csv.gz"), index=False, compression="gzip")

    res = {"features": P.FEATURES, "samples": {}}
    for name, (a, b) in (("S1_dev_2014_2018", S1), ("S2_val_2019_2024", S2)):
        fc = walk_forward(pan, a, b)
        if fc.empty:
            print(f"[{name}] нет прогнозов")
            continue
        r = evaluate(fc)
        res["samples"][name] = r
        print(f"\n=== {name}: {r['rebalances']} ребалансов ===")
        for k in (*MODELS, "MOM"):
            ic = r["ic"].get(k, {})
            pf = r["portfolio"].get(k, {})
            print(f"  {k:4s} IC={ic.get('mean', float('nan')):+.4f} (t {ic.get('t') or float('nan'):+.2f})"
                  f"   портфель {pf.get('annual_pct', float('nan')):+.2f} %/год "
                  f"(t {pf.get('t') or float('nan'):+.2f}, hit {pf.get('hit', float('nan')):.2f})")
        for k, v in r["vs_MOM"].items():
            print(f"    {k} − MOM: ΔIC={v['d_ic_mean']:+.4f} (t {v['d_ic_t']:+.2f}), "
                  f"Δпортфель={v['d_pf_annual_pp']:+.2f} пп/год (t {v['d_pf_t']:+.2f}, "
                  f"p1={v['d_pf_p1']:.3f}, Холм={r['holm_portfolio'][k + '_pf']['p_adj']:.3f})")

    json.dump(res, open(os.path.join(OUT, "ml_xsec.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="minutes"),
                             "sprint": 24, "stage": "wave2_ml_xsec_HC1", "trials": 3,
                             "revision": "wave2"}, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.join(OUT, 'ml_xsec.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
