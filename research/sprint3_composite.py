"""
Спринт 3, контрольная точка (решение пользователя 15.09.2026): одно испытание без
машинного обучения. Правило — research/sprint3_composite.json, фиксируется коммитом
до прогона.

Скор = z(RS30) + z(EMA50) + z(тональность 48 ч) − z(спред Корвина–Шульца), z —
поперечный внутри дня по пулу из 9 ликвидных бумаг. Вход — только когда скор выше
97,5-го перцентиля скора пула за предыдущие 250 торговых дней (порог из прошлого).
Сделка — как в v1: лонг бумаги 18:35 + шорт фьючерса MX, выход через 2 торговых дня;
результат net_mn против «всё в TMON». t — Ньюи–Уэст (лаг 1) по датам входа.

Если ворота пройдены — один прогон на отложенной выборке (тот же код внутри окна).

Запуск (на сервере): python -m research.sprint3_composite --stage dev
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import event_study_news as es              # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402
from research import sprint3_panel as sp                 # noqa: E402
from research import sprint3_wf as wf                    # noqa: E402

log = logging.getLogger("research.sprint3_composite")

SPEC_PATH = os.path.join(ROOT, "research", "sprint3_composite.json")
NET, EXTRAS = "net_mn", ("net_alpha", "net_mn_fee")


def load_spec(path: str = SPEC_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cross_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Поперечный z-скор внутри дня; нулевой разброс или пропуск → 0."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        g = df.groupby("d")[c]
        mu, sd = g.transform("mean"), g.transform(lambda s: s.std(ddof=0))
        z = (df[c] - mu) / sd.where(sd > 0)
        out[c] = z.fillna(0.0)
    return out


def score(df: pd.DataFrame, terms: list) -> pd.Series:
    z = cross_z(df, [c for c, _ in terms])
    return sum(float(sign) * z[c] for c, sign in terms)


def rolling_threshold(df: pd.DataFrame, q: float, lookback: int, min_hist: int) -> pd.Series:
    """Порог на дату: перцентиль q скоров за предыдущие lookback торговых дней (строго до даты)."""
    days = sorted(df["d"].unique())
    by_day = {d: g["score"].to_numpy(float) for d, g in df.groupby("d")}
    thr = {}
    for i, d in enumerate(days):
        win = days[max(0, i - lookback):i]
        thr[d] = float(np.quantile(np.concatenate([by_day[x] for x in win]), q)) if len(win) >= min_hist else np.nan
    return df["d"].map(thr)


def select(panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    p = panel[panel["ticker"].isin(spec["pool"])].copy()
    p["score"] = score(p, spec["score"]["terms"])
    t = spec["tail"]
    p["thr"] = rolling_threshold(p, t["quantile"], t["lookback_days"], t["min_history_days"])
    p["entry"] = p["thr"].notna() & (p["score"] > p["thr"])
    return p


def evaluate(p: pd.DataFrame, n_trials: int, gate: dict) -> dict:
    eligible = p[p["thr"].notna()]
    days = sorted(eligible["d"].unique())
    tr = eligible[eligible["entry"]].dropna(subset=[NET])
    per = tr.groupby("d")[NET].mean()
    daily = per.reindex(days).fillna(0.0)
    res = {"eligible_days": len(days), "eligible_rows": int(len(eligible)), "trades": int(len(tr)),
           "trade_dates": int(len(per)), "net": es.by_date(tr[NET], tr["d"]) if len(tr) else {"n": 0, "dates": 0},
           "t_nw": wf.newey_west_t(per, 1), "hit": float((tr[NET] > 0).mean()) if len(tr) else None,
           "precision": float(tr["y_rel"].mean()) if len(tr) else None,
           "base_rate": float(eligible["y_rel"].mean()),
           "ir": es.information_ratio(daily), "dsr": es.deflated_sharpe(daily, max(2, n_trials)),
           "all_rows": es.by_date(eligible[NET], eligible["d"]),
           "extras": {c: {"mean": float(tr[c].mean()) if len(tr) else None,
                          "t_nw": wf.newey_west_t(tr.groupby("d")[c].mean(), 1)} for c in EXTRAS},
           "by_ticker": tr.groupby("ticker")[NET].agg(["count", "mean"]).round(4).to_dict("index")}
    res["gate"] = bool(res["t_nw"] is not None and res["t_nw"] >= gate["t_min"]
                       and res["trade_dates"] >= gate["dates_min"]
                       and (res["net"].get("mean") or -1) > gate["net_min"])
    return res


def report(stage: str, r: dict, meta: dict) -> str:
    f = es._f
    L = [f"# Спринт 3 — контрольная точка без ML (линейный скор, хвост 2,5 %), "
         f"{'разработка' if stage == 'dev' else 'отложенная выборка (единственный прогон)'}", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правило `{meta['spec']}`. "
         f"Дни с порогом {meta['from']} … {meta['to']}. Испытаний в реестре: {meta['trials_total']}.", "",
         "Скор: z(RS30) + z(EMA50) + z(тональность 48 ч) − z(спред Корвина–Шульца), пул: " + ", ".join(meta["pool"]) + ". "
         "Вход — скор выше 97,5-го перцентиля за предыдущие 250 дней. Связка: лонг бумаги + шорт фьючерса MX, 2 дня.", "",
         "| дней с порогом | сделок | дат | точность (доля y=1 в пуле) | нетто связки, % (t НУ) | альфа − издержки, % (t НУ) "
         "| Б: связка при комиссии 0,08 %, % (t НУ) | доля плюсовых | IR | DSR | все строки пула, % | ворота |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| {r['eligible_days']} | {r['trades']} | {r['trade_dates']} | {f(r['precision'], 3)} ({f(r['base_rate'], 3)}) | "
         f"{f(r['net'].get('mean'))} ({f(r['t_nw'], 1)}) | {f(r['extras']['net_alpha']['mean'])} "
         f"({f(r['extras']['net_alpha']['t_nw'], 1)}) | {f(r['extras']['net_mn_fee']['mean'])} "
         f"({f(r['extras']['net_mn_fee']['t_nw'], 1)}) | {f(r['hit'], 2)} | {f(r['ir'], 2)} | {f(r['dsr'], 2)} | "
         f"{f(r['all_rows'].get('mean'))} | {'да' if r['gate'] else 'нет'} |", "",
         "| бумага | сделок | нетто связки, % |", "|---|---|---|"]
    for tk, x in sorted(r["by_ticker"].items()):
        L.append(f"| {tk} | {int(x['count'])} | {f(x['mean'])} |")
    L += ["", "Ворота: t НУ ≥ 2, дат со сделками ≥ 30, средний результат > 0.", ""]
    return "\n".join(L)


def holdout_allowed(out_dir: str = sp.OUT_DIR) -> None:
    if os.path.exists(os.path.join(out_dir, "composite_holdout", "results.json")):
        raise SystemExit("отложенная выборка контрольной точки уже прогонялась — повтор запрещён")
    path = os.path.join(out_dir, "composite_dev", "results.json")
    if not os.path.exists(path):
        raise SystemExit("нет результата разработки")
    with open(path, encoding="utf-8") as f:
        if not json.load(f)["result"]["gate"]:
            raise SystemExit("ворота разработки не пройдены — отложенная выборка не нужна")


def run(stage: str) -> int:
    spec, cfg = load_spec(), sp.load_config()
    path = sp.panel_path(stage, cfg["version"])
    if stage == "holdout":
        holdout_allowed()
        if not os.path.exists(path):
            import database
            conn = database.get_connection()
            try:
                sp.build_panel(conn, cfg, "holdout").to_csv(path, index=False, float_format="%.6g")
            finally:
                conn.close()
    panel = pd.read_csv(path, parse_dates=["d", "exit_d"])
    p = select(panel, spec)
    rev = es._revision()
    total = ll.register(f"sprint3-composite-{stage}", 1 if stage == "dev" else 0, rev, sprint=3)
    r = evaluate(p, total, spec["gate"])
    elig = p[p["thr"].notna()]
    meta = {"created": dt.datetime.now().strftime("%d.%m.%Y %H:%M"), "revision": rev, "spec": spec["version"],
            "from": str(elig["d"].min().date()), "to": str(elig["d"].max().date()), "trials_total": total,
            "pool": spec["pool"]}
    out = os.path.join(sp.OUT_DIR, f"composite_{stage}")
    os.makedirs(out, exist_ok=True)
    text = report(stage, r, meta)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "result": r}, f, ensure_ascii=False, indent=1, default=str)
    p[p["entry"]][["ticker", "d", "exit_d", "score", "thr", "y_rel", NET, *EXTRAS]].to_csv(
        os.path.join(out, "trades.csv"), index=False, float_format="%.5f")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 3: контрольная точка без ML")
    ap.add_argument("--stage", choices=("dev", "holdout"), required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(a.stage)


if __name__ == "__main__":
    raise SystemExit(main())
