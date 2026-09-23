"""
H6. Дневное движение индекса Polymarket «мир»/«война» как фильтр для
long_overnight (ТЗ пользователя 23.09.2026 — «собери данные, прогони через
торговые модели, найти закономерности с доходностью»).

Отличие от прежнего замера (research/polymarket/analyze.py, 22.09.2026):
там все 4 теста (A_same/B_gap/C_intra/D_next) были либо одновременными
(сигнал и IMOEX известны в один момент — не сыграть), либо использовали
Δ ПОСЛЕ решения о входе. Здесь сигнал вырезан РОВНО до времени решения
long_overnight (18:35 МСК = 15:35 UTC, config.STAGE2_OVERNIGHT_TIME) и
проверяется на предсказание гэпа, который наступает ПОСЛЕ этого момента —
единственная тайминг-корректная (тем самым в принципе торгуемая) проверка.

  sig[D]    = Δ индекса класса за окно (cutoff[D-1] .. cutoff[D]), п.п.
  target[D] = гэп IMOEX (open[D+1] / close[D] − 1) × 100, %  — то, что
              реально ловит long_overnight, войдя в D по сигналу известному
              к cutoff[D].

Дев-выборка — та же, что в H1–H5 (research/strategic/rules.json):
2024-05-21…2026-09-11, отложенная НЕ ТРОГАЕТСЯ.

Порог фильтра выбирается на CPCV-разбиениях (5 блоков, embargo 5 дней, как в
H1) — чтобы не подгонять точку входа по всей истории. PBO считается по сетке
порогов методом CSCV. Издержки — research/strategic/costs.CostModel (комиссия
+ спред блю-чипов по умолчанию + воздействие с консервативным ADV), ставка
фонда — приближение (нет локальной истории TMON@/LQDT), это явно помечено в
отчёте, а не выдаётся за точную цифру.

Запуск: python -m research.strategic.h6_polymarket_signal
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

from research.polymarket.analyze import load as pm_load, class_index          # noqa: E402
from research.strategic import costs as sc                                    # noqa: E402
from research.strategic import validation as va                               # noqa: E402

log = logging.getLogger("research.strategic.h6")

MODULE = "h6_polymarket_signal"
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "polymarket")
DEV_FROM, DEV_TO = dt.date(2024, 5, 21), dt.date(2026, 9, 11)
CUTOFF_UTC = dt.time(15, 35)                 # config.STAGE2_OVERNIGHT_TIME 18:35 МСК
POSITION_RUB = 100_000.0                     # BEST_TRADES_POSITION_RUB боевого r4
CLASSES = ("PEACE", "WAR")
THRESH_GRID = {"any_up": 0.0, "median_pos": None, "p75": None}      # median_pos/p75 считаются по трейну
FUND_ANNUAL_PCT_APPROX = 18.0                 # приближение ставки денежного рынка 2024–2026
                                               # (локальной истории TMON@/LQDT нет — см. отчёт)


def imoex_daily(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT date, open, close FROM market_data WHERE ticker = 'IMOEX' "
                     "AND close > 0 ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[[d.weekday() < 5 for d in df["date"]]].drop_duplicates("date", keep="last")
    df["open"], df["close"] = df["open"].astype(float), df["close"].astype(float)
    df["prev_close"] = df["close"].shift(1)
    df["prev_date"] = df["date"].shift(1)
    df["next_open"] = df["open"].shift(-1)
    df["next_date"] = df["date"].shift(-1)
    df["gap_next"] = (df["next_open"] / df["close"] - 1) * 100.0
    return df.dropna(subset=["prev_date", "next_date"])


def signal_series(ix: pd.Series, days: pd.DataFrame) -> pd.Series:
    """Δ индекса за окно (cutoff[D-1] .. cutoff[D]) — известно к моменту входа D."""
    out = {}
    for r in days.itertuples(index=False):
        a = pd.Timestamp.combine(r.prev_date, CUTOFF_UTC).tz_localize("UTC")
        b = pd.Timestamp.combine(r.date, CUTOFF_UTC).tz_localize("UTC")
        s = ix[(ix.index > a) & (ix.index <= b)]
        out[r.date] = float(s.sum()) if len(s) else np.nan
    return pd.Series(out)


def fund_pct(d0: dt.date, d1: dt.date) -> float:
    """Приближение ставки фонда: FUND_ANNUAL_PCT_APPROX годовых, по календарным дням."""
    return FUND_ANNUAL_PCT_APPROX / 365.0 * max(0, (d1 - d0).days)


def build_panel(conn) -> pd.DataFrame:
    markets, h = pm_load()
    days = imoex_daily(conn)
    days = days[(days["date"] >= DEV_FROM) & (days["date"] <= DEV_TO)]
    panel = days.set_index("date")[["prev_date", "next_date", "gap_next", "close"]].copy()
    for cls in CLASSES:
        ix = class_index(markets, h, cls)
        sig = signal_series(ix, days)
        panel[f"sig_{cls}"] = sig
    return panel.dropna(subset=["gap_next"])


def trades_for_rule(panel: pd.DataFrame, cls: str, thresh: float, model: sc.CostModel,
                    position_rub: float = POSITION_RUB) -> pd.DataFrame:
    """Сделки long_overnight, отфильтрованные условием sig_cls > thresh."""
    sel = panel[panel[f"sig_{cls}"] > thresh].dropna(subset=["gap_next"])
    if sel.empty:
        return pd.DataFrame()
    cost = model.round_trip_pct("__IMOEX_PROXY__", sigma_pct=1.0, notional_rub=position_rub,
                                adv_rub=5_000_000_000.0)
    rows = []
    for d, r in sel.iterrows():
        fund = fund_pct(d, r["next_date"])
        net = r["gap_next"] - cost - fund
        rows.append({"entry_day": d, "exit_day": r["next_date"], "notional": position_rub,
                     "gross_pct": r["gap_next"], "cost_pct": cost, "fund_pct": fund,
                     "net_excess_pct": net, "pnl_excess_rub": position_rub * net / 100.0})
    return pd.DataFrame(rows)


def baseline_trades(panel: pd.DataFrame, model: sc.CostModel,
                    position_rub: float = POSITION_RUB) -> pd.DataFrame:
    """Безусловный long_overnight каждый торговый день — точка отсчёта."""
    cost = model.round_trip_pct("__IMOEX_PROXY__", sigma_pct=1.0, notional_rub=position_rub,
                                adv_rub=5_000_000_000.0)
    rows = []
    for d, r in panel.iterrows():
        fund = fund_pct(d, r["next_date"])
        net = r["gap_next"] - cost - fund
        rows.append({"entry_day": d, "exit_day": r["next_date"], "notional": position_rub,
                     "gross_pct": r["gap_next"], "cost_pct": cost, "fund_pct": fund,
                     "net_excess_pct": net, "pnl_excess_rub": position_rub * net / 100.0})
    return pd.DataFrame(rows)


def _grid_thresholds(train_sig: pd.Series) -> dict:
    pos = train_sig[train_sig > 0]
    return {"any_up": 0.0,
            "median_pos": float(pos.median()) if len(pos) else None,
            "p75": float(train_sig.quantile(0.75))}


def oos_filtered_trades(panel: pd.DataFrame, cls: str, model: sc.CostModel, cpcv: dict) -> dict:
    """CPCV: порог фильтра выбирается на train (макс. net_excess_pct_trade),
    сделки собираются на test. Возвращает итоговую OOS-таблицу сделок и
    per-config матрицу для PBO."""
    dates = sorted(panel.index)
    splits = va.cpcv_splits(dates, cpcv["blocks"], cpcv["k_test"], cpcv["embargo_days"], label_days=2)
    oos_rows, perf_rows = [], []
    for sp in splits:
        train = panel.loc[sorted(sp["train"] & set(panel.index))]
        test = panel.loc[sorted(sp["test"] & set(panel.index))]
        if train.empty or test.empty:
            continue
        grid = _grid_thresholds(train[f"sig_{cls}"].dropna())
        best_name, best_score, per_cfg_test = None, -np.inf, {}
        for name, th in grid.items():
            if th is None:
                continue
            tr_tr = trades_for_rule(train, cls, th, model)
            te_tr = trades_for_rule(test, cls, th, model)
            per_cfg_test[name] = te_tr
            score = tr_tr["net_excess_pct"].mean() if not tr_tr.empty else -np.inf
            if score > best_score:
                best_name, best_score = name, score
        if best_name and not per_cfg_test.get(best_name, pd.DataFrame()).empty:
            oos_rows.append(per_cfg_test[best_name])
        perf_rows.append({k: (v["net_excess_pct"].mean() if not v.empty else np.nan)
                          for k, v in per_cfg_test.items()})
    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    perf = pd.DataFrame(perf_rows)
    return {"oos_trades": oos, "perf_matrix": perf}


def signal_correlation(panel: pd.DataFrame) -> dict:
    """Прямая проверка предсказательности (без порога/фильтра): corr(sig[D], gap[D+1])."""
    out = {}
    pvals = {}
    for cls in CLASSES:
        s = panel[[f"sig_{cls}", "gap_next"]].dropna()
        s = s[s[f"sig_{cls}"] != 0]
        n = len(s)
        if n < 20:
            out[cls] = {"n": n}
            continue
        r = float(s[f"sig_{cls}"].corr(s["gap_next"]))
        t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 2))
        out[cls] = {"n": n, "r": r, "t": t, "p": p}
        pvals[cls] = p
    items = sorted((v, k) for k, v in pvals.items())
    run, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k]["p_holm"] = run
    return out


def run(conn, rules: dict) -> dict:
    model = sc.CostModel(impact_y=rules["costs"]["impact_Y"])
    panel = build_panel(conn)
    log.info("[H6] панель: %d торговых дней (%s..%s)", len(panel), panel.index.min(), panel.index.max())

    corr = signal_correlation(panel)
    base = baseline_trades(panel, model)
    base_summary = va.summarize(base, DEV_FROM, DEV_TO, 5_000_000.0)

    per_class = {}
    for cls in CLASSES:
        oos = oos_filtered_trades(panel, cls, model, rules["cpcv"])
        summary = va.summarize(oos["oos_trades"], DEV_FROM, DEV_TO, 5_000_000.0)
        pbo = va.cscv_pbo(oos["perf_matrix"], s=min(8, max(2, len(oos["perf_matrix"]) // 2 * 2)))
        gate = va.dev_gate(summary, pbo.get("pbo"), 50_000_000.0, rules["gates"]["dev"]) \
            if summary.get("trades") else (False, ["сделок нет"])
        per_class[cls] = {"correlation": corr.get(cls, {}), "oos_summary": summary,
                          "pbo": pbo, "passed": gate[0], "failed": gate[1],
                          "coverage_pct": float(len(oos["oos_trades"]) / max(1, len(base)) * 100.0)
                          if not oos["oos_trades"].empty else 0.0}

    result = {"module": MODULE, "dev_sample": {"from": str(DEV_FROM), "to": str(DEV_TO)},
             "days": int(len(panel)), "baseline": base_summary, "classes": per_class,
             "fund_approx_annual_pct": FUND_ANNUAL_PCT_APPROX,
             "fund_approx_note": "приближение — локальной истории TMON@/LQDT в этой БД нет"}
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    with open(os.path.join(ROOT, "research", "strategic", "rules.json"), encoding="utf-8") as f:
        rules = json.load(f)
    conn = database.get_connection()
    try:
        res = run(conn, rules)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "h6_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(os.path.join(ROOT, "audit", "r4_research", "trials.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 10,
                            "stage": "h6_polymarket_signal", "trials": 6, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
