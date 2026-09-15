"""
Спринт 3 (ТЗ 50D): кластерные модели превышения барьера, walk-forward.
Параметры — research/sprint3_config.json; панель — research/sprint3_panel.py.

По каждому кластеру: переобучение в первый торговый день месяца на 12 месяцах
(минимум 6), в обучение — только строки, чей выход раньше начала тестового месяца
(перекрытие удержаний не просачивается). Сделка — вероятность превышения барьера
≥ порога. Результат сделки — excess (доходность − фонд − издержки). Удержания 2 дня
перекрываются, поэтому t — Ньюи–Уэст (лаг 1) по датам входа.
Ворота — как в Спринте 2; число испытаний = число кластеров, в реестр trials.jsonl.

Запуск (на сервере): python -m research.sprint3_wf --stage dev
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import event_study_news as es              # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402
from research import sprint3_panel as sp                 # noqa: E402

log = logging.getLogger("research.sprint3_wf")
ALL = "ВСЕ КЛАСТЕРЫ"


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    d = pd.to_datetime(pd.Series(dates)).sort_values()
    return list(d.groupby(d.dt.to_period("M")).min())


def walk_forward(panel: pd.DataFrame, features: list[str], model_cfg: dict, wf: dict) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = panel.copy()
    p["d"] = pd.to_datetime(p["d"])
    p["exit_d"] = pd.to_datetime(p["exit_d"])
    params = {k: model_cfg[k] for k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                                        "l2_regularization", "random_state")}
    out = []
    for cl, g in p.groupby("cluster"):
        starts = month_starts(g["d"])
        for i, start in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else g["d"].max() + pd.Timedelta(days=1)
            tr = g[(g["exit_d"] < start) & (g["d"] >= start - pd.DateOffset(months=wf["train_months"]))]
            tr = tr.dropna(subset=["y"])
            if tr.empty or (tr["d"].max() - tr["d"].min()).days < wf["min_train_months"] * 30 \
                    or len(tr) < wf["min_train_rows"] or tr["y"].nunique() < 2:
                continue
            te = g[(g["d"] >= start) & (g["d"] < nxt)]
            if te.empty:
                continue
            m = HistGradientBoostingClassifier(**params).fit(tr[features], tr["y"].astype(int))
            out.append(te[["ticker", "cluster", "d", "exit_d", "y", "R", "excess", "cost", "hurdle"]]
                       .assign(p=m.predict_proba(te[features])[:, 1], fit=start, n_train=len(tr),
                               base_train=float(tr["y"].mean())))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def newey_west_t(x: pd.Series, lag: int = 1) -> float | None:
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 10:
        return None
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * float(e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else None


def evaluate(pred: pd.DataFrame, threshold: float, n_trials: int, gate: dict) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    groups = [(cl, g) for cl, g in pred.groupby("cluster")] + [(ALL, pred)]
    for cl, g in groups:
        lab = g.dropna(subset=["y"])
        days = sorted(g["d"].unique())
        tr = lab[lab["p"] >= threshold]
        per = tr.groupby("d")["excess"].mean()
        daily = per.reindex(days).fillna(0.0)
        rec = {"rows": int(len(lab)), "dates": int(len(days)), "base_rate": float(lab["y"].mean()) if len(lab) else None,
               "auc": float(roc_auc_score(lab["y"], lab["p"])) if lab["y"].nunique() == 2 else None,
               "trades": int(len(tr)), "trade_dates": int(len(per)),
               "precision": float(tr["y"].mean()) if len(tr) else None,
               "excess": es.by_date(tr["excess"], tr["d"]) if len(tr) else {"n": 0, "dates": 0},
               "t_nw": newey_west_t(per, 1), "hit": float((tr["excess"] > 0).mean()) if len(tr) else None,
               "ir": es.information_ratio(daily), "dsr": es.deflated_sharpe(daily, max(2, n_trials)),
               "all_rows_excess": es.by_date(lab["excess"], lab["d"])}
        rec["gate"] = bool(rec["t_nw"] is not None and rec["t_nw"] >= gate["t_min"]
                           and rec["trade_dates"] >= gate["dates_min"]
                           and (rec["excess"].get("mean") or -1) > gate["net_min"])
        out[cl] = rec
    return out


def deciles(pred: pd.DataFrame) -> pd.DataFrame:
    lab = pred.dropna(subset=["y"]).copy()
    lab["bin"] = pd.qcut(lab["p"], 10, labels=False, duplicates="drop")
    return lab.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), excess=("excess", "mean"),
                                  rows=("y", "size")).reset_index()


def report(stage: str, res: dict, dec: pd.DataFrame, meta: dict) -> str:
    f = es._f
    L = [f"# Спринт 3 — кластерные модели превышения барьера, {'разработка' if stage == 'dev' else 'отложенная выборка'}",
         "", f"Сформировано {meta['created']}. Код `{meta['revision']}`, конфигурация `{meta['config']}`. "
         f"Период {meta['from']} … {meta['to']}; прогнозы вне обучения {meta['oos_from']} … {meta['oos_to']}. "
         f"Испытаний в реестре: {meta['trials_total']}.", "",
         f"Цель: доходность за {meta['horizon']} дня > фонд + издержки + {meta['margin']:g} п.п. Сделка — "
         f"вероятность ≥ {meta['threshold']:g}; excess — доходность − фонд − издержки, %. t — Ньюи–Уэст по датам входа.", "",
         "| кластер | строк | доля y=1 | AUC | сделок | дат | точность | excess, % (t НУ) | доля плюсовых | IR | DSR | все строки, % | ворота |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cl, r in res.items():
        L.append(f"| {cl} | {r['rows']} | {f(r['base_rate'], 3)} | {f(r['auc'], 3)} | {r['trades']} | {r['trade_dates']} | "
                 f"{f(r['precision'], 3)} | {f(r['excess'].get('mean'))} ({f(r['t_nw'], 1)}) | {f(r['hit'], 2)} | "
                 f"{f(r['ir'], 2)} | {f(r['dsr'], 2)} | {f(r['all_rows_excess'].get('mean'))} | "
                 f"{'да' if r['gate'] else 'нет'} |")
    L += ["", "## Калибровка: децили вероятности (все кластеры)", "",
          "| дециль | средняя вероятность | доля y=1 | excess, % | строк |", "|---|---|---|---|---|"]
    for r in dec.itertuples():
        L.append(f"| {int(r.bin) + 1} | {r.p_mean:.3f} | {r.y_rate:.3f} | {f(r.excess)} | {r.rows} |")
    L += ["", "«Все строки» — excess, если входить в каждую бумагу каждый день (база для сравнения).",
          "Ворота: t НУ ≥ 2, дат со сделками ≥ 30, средний excess > 0.", ""]
    return "\n".join(L)


def run(stage: str, rebuild: bool = False) -> int:
    cfg = sp.load_config()
    if stage == "holdout" and not cfg["samples"]["holdout"].get("approved"):
        raise SystemExit("протокол отложенной выборки не утверждён пользователем")
    path = sp.panel_path(stage)
    if rebuild or not os.path.exists(path):
        import database
        conn = database.get_connection()
        try:
            panel = sp.build_panel(conn, cfg, stage)
        finally:
            conn.close()
        os.makedirs(sp.OUT_DIR, exist_ok=True)
        panel.to_csv(path, index=False, float_format="%.6g")
    panel = pd.read_csv(path, parse_dates=["d", "exit_d"])
    feats = sp.feature_list(cfg)
    rev = es._revision()
    total = ll.register(f"sprint3-{stage}", len(cfg["clusters"]) if stage == "dev" else 0, rev, sprint=3)
    pred = walk_forward(panel, feats, cfg["model"], cfg["walk_forward"])
    res = evaluate(pred, cfg["decision"]["threshold"], total, cfg["gate"])
    out = os.path.join(sp.OUT_DIR, stage)
    os.makedirs(out, exist_ok=True)
    meta = {"created": dt.datetime.now().strftime("%d.%m.%Y %H:%M"), "revision": rev, "config": cfg["version"],
            "from": cfg["samples"][stage]["from"], "to": cfg["samples"][stage]["to"],
            "oos_from": str(pred["d"].min().date()), "oos_to": str(pred["d"].max().date()),
            "trials_total": total, "horizon": cfg["target"]["horizon_days"], "margin": cfg["target"]["margin_pct"],
            "threshold": cfg["decision"]["threshold"], "panel_rows": int(len(panel))}
    text = report(stage, res, deciles(pred), meta)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "passed": [k for k, r in res.items() if r["gate"] and k != ALL], "results": res},
                  f, ensure_ascii=False, indent=1, default=str)
    pred.to_csv(os.path.join(out, "predictions.csv"), index=False, float_format="%.5f")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 3: кластерные модели, walk-forward")
    ap.add_argument("--stage", choices=("dev", "holdout"), required=True)
    ap.add_argument("--rebuild", action="store_true", help="пересобрать панель")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(a.stage, a.rebuild)


if __name__ == "__main__":
    raise SystemExit(main())
