"""
Сводный анализ по результатам walk-forward реплея.

Блок 2.2 — Rank IC формулы FinalScore и её компонент, матрица избыточности.
Этап 1  — сигнальный бэктест портфеля и критерии Go/No-Go.
Этап 2  — моделирование исполнения лимиток на 5-минутных барах.

Запуск (после audit/run_walkforward.py):
    python3 -m audit.run_analysis
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import backtest, costs, data, ic  # noqa: E402

log = logging.getLogger("audit.analysis")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Критерии перехода Этап 1 → Этап 2 (ТЗ, раздел 4).
GATE_STAGE1 = {
    "sharpe": 1.3,
    "profit_factor": 1.45,
    "max_drawdown_pct": -12.0,      # не хуже
    "win_rate": 0.52,
    "alpha_annual_pct": 15.0,
}

COMPONENTS = ["exp_score", "prob_score", "liq_score_n", "rs_score",
              "regime_score", "vol_score", "final_score"]


def gate_check(perf: dict, alpha: dict) -> dict:
    """Проверка критериев Go/No-Go Этапа 1."""
    got = {
        "sharpe": perf.get("sharpe"),
        "profit_factor": perf.get("profit_factor"),
        "max_drawdown_pct": perf.get("max_drawdown_pct"),
        "win_rate": perf.get("win_rate"),
        "alpha_annual_pct": alpha.get("alpha_annual_pct"),
    }
    checks = {}
    for k, need in GATE_STAGE1.items():
        v = got.get(k)
        if v is None or not np.isfinite(v):
            checks[k] = {"got": None, "need": need, "pass": False}
            continue
        ok = (v >= need) if k != "max_drawdown_pct" else (v >= need)
        checks[k] = {"got": float(v), "need": need, "pass": bool(ok)}
    checks["_all_pass"] = all(c["pass"] for c in checks.values() if isinstance(c, dict))
    return checks


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    res: dict = {}

    wf = pd.read_csv(os.path.join(OUT, "walkforward.csv"), parse_dates=["asof_date", "fit_date"])
    log.info("Реплей: %s строк, %d дат, %d тикеров",
             f"{len(wf):,}", wf["asof_date"].nunique(), wf["ticker"].nunique())

    cm = pd.read_csv(os.path.join(OUT, "cost_matrix.csv"))
    cost_map = {r.ticker: float(r.cost_rt_base) for r in cm.itertuples()
                if np.isfinite(r.cost_rt_base)}
    instr = costs.load_instruments()
    short_blocked = set(instr.loc[instr["short_enabled"] != True, "ticker"])  # noqa: E712

    # Только первый месяц каждого окна — непересекающийся ряд.
    wfp = wf[wf["fold_primary"]].copy()
    log.info("Непересекающийся OOS: %s строк, %d дат",
             f"{len(wfp):,}", wfp["asof_date"].nunique())

    # ── Блок 2.2: компоненты и Rank IC ──────────────────────────────────────
    comp = ic.build_components(wfp)
    comp.to_csv(os.path.join(OUT, "components.csv"), index=False)

    tbl = ic.ic_table(comp, COMPONENTS, "realized_net", by_strategy=True)
    tbl.to_csv(os.path.join(OUT, "rank_ic.csv"), index=False)
    res["rank_ic_all"] = tbl[tbl["scope"] == "ВСЕ"].to_dict("records")
    res["rank_ic_by_strategy"] = tbl[tbl["scope"] != "ВСЕ"].to_dict("records")

    # Валовая доходность (без издержек) — отделяет «нет сигнала» от «съели издержки».
    tbl_gross = ic.ic_table(comp, ["final_score", "exp_score"], "realized",
                            by_strategy=False)
    res["rank_ic_gross"] = tbl_gross.to_dict("records")

    corr = ic.component_correlations(comp, COMPONENTS[:-1])
    corr.to_csv(os.path.join(OUT, "component_corr.csv"))
    res["component_corr"] = corr.round(3).to_dict()

    decay = ic.decay_by_model_age(ic.build_components(wf), "final_score")
    decay.to_csv(os.path.join(OUT, "ic_decay.csv"), index=False)
    res["ic_decay"] = decay.to_dict("records")

    # ── Этап 1: сигнальный бэктест ──────────────────────────────────────────
    daily = data.load_daily()
    bench = backtest.benchmark_returns(daily, "IMOEX")

    # Меню стратегий production (config.VALIDATION_STRATS): short_hold в бою
    # не торгуется, поэтому базовые варианты его не включают.
    PROD = {"long_overnight", "intraday_short", "intraday_long"}
    variants = {
        "top10_equal": dict(top_n=10, weighting="equal", allowed_strats=PROD),
        "top10_riskparity": dict(top_n=10, weighting="risk_parity", allowed_strats=PROD),
        "top5_equal": dict(top_n=5, weighting="equal", allowed_strats=PROD),
        "top20_equal": dict(top_n=20, weighting="equal", allowed_strats=PROD),
        "top10_long_only": dict(top_n=10, weighting="equal", long_only=True),
        "top10_all_strats": dict(top_n=10, weighting="equal"),
    }
    bt = {}
    for name, kw in variants.items():
        daily_ret, picks = backtest.signal_backtest(
            comp, "final_score", ret_col="realized_net",
            short_blocked=short_blocked, **kw)
        if daily_ret.empty:
            continue
        perf = backtest.performance(daily_ret)
        alpha = backtest.alpha_vs_benchmark(daily_ret, bench)
        perf.pop("equity", None)
        perf.pop("drawdown", None)
        bt[name] = {"performance": perf, "alpha": alpha,
                    "gate": gate_check(perf, alpha)}
        daily_ret.to_csv(os.path.join(OUT, f"equity_{name}.csv"))
        if name == "top10_equal":
            picks.to_csv(os.path.join(OUT, "picks_top10.csv"), index=False)
        log.info("  %-20s Sharpe=%6.2f PF=%5.2f MDD=%7.2f%% WR=%.3f альфа=%+.1f%%",
                 name, perf.get("sharpe", np.nan), perf.get("profit_factor", np.nan),
                 perf.get("max_drawdown_pct", np.nan), perf.get("win_rate", np.nan),
                 alpha.get("alpha_annual_pct", np.nan))

    # Контроль: случайный отбор той же мощности — сколько даёт «ничего».
    rng = np.random.default_rng(7)
    comp_rand = comp.copy()
    comp_rand["rand_score"] = rng.random(len(comp_rand))
    rnd_ret, _ = backtest.signal_backtest(comp_rand, "rand_score", top_n=10,
                                          ret_col="realized_net",
                                          short_blocked=short_blocked)
    rnd_perf = backtest.performance(rnd_ret)
    rnd_perf.pop("equity", None)
    rnd_perf.pop("drawdown", None)
    bt["control_random"] = {"performance": rnd_perf,
                            "alpha": backtest.alpha_vs_benchmark(rnd_ret, bench)}
    res["backtest"] = bt

    bench_perf = backtest.performance(bench)
    bench_perf.pop("equity", None)
    bench_perf.pop("drawdown", None)
    res["benchmark_imoex"] = bench_perf

    # ── Этап 2: исполнение лимиток на 5-минутках ────────────────────────────
    log.info("Моделирование исполнения на 5-минутных барах...")
    d5 = data.load_5m(main_session_only=True)
    d5_start = d5["date"].min()
    sig = comp[comp["asof_date"] >= d5_start].copy()
    sig = (sig.sort_values("final_score", ascending=False)
              .groupby(["asof_date", "ticker"], as_index=False).head(1))
    sig = (sig.sort_values(["asof_date", "final_score"], ascending=[True, False])
              .groupby("asof_date", as_index=False).head(10))
    sig = sig[~((~sig["strategy"].isin(ic.LONG_STRATS)) &
                sig["ticker"].isin(short_blocked))]
    log.info("  сигналов к исполнению: %s (с %s)", f"{len(sig):,}", d5_start.date())

    ex = {}
    for ef, tpf, tag in [(0.2, 1.0, "entry0.2_tp1.0"),
                         (0.2, 0.5, "entry0.2_tp0.5"),
                         (0.8, 1.0, "entry0.8_tp1.0")]:
        fills = backtest.simulate_limit_fills(sig, d5, entry_frac=ef, tp_frac=tpf)
        if fills.empty:
            continue
        fills.to_csv(os.path.join(OUT, f"fills_{tag}.csv"), index=False)
        ex[tag] = backtest.execution_report(fills, cost_map)
        log.info("  %-18s fill_rate=%.3f adverse=%.3f net_mean=%+.4f%%", tag,
                 ex[tag].get("fill_rate", np.nan), ex[tag].get("adverse_share", np.nan),
                 ex[tag].get("net_mean_pct", np.nan))
    res["execution"] = ex

    with open(os.path.join(OUT, "analysis_summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log.info("Готово → audit/out/analysis_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
