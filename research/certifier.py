"""
Шлюз сертификации гипотез (ТЗ 17.09.2026, раздел 2.1 и шаг 3).

Стратегия получает статус CERTIFIED только если на ОТЛОЖЕННОЙ выборке
(2022-09-12…2024-05-20) выполнены все условия:
  t по датам входа ≥ 2,0; избыточная доходность сверх фонда ≥ +2,0 % годовых на
  задействованный капитал; IR к фонду ≥ 0,8; DSR ≥ 0,95 с учётом ВСЕХ испытаний
  реестра (audit/r4_research/trials.jsonl); для интрадей — ноль позиций,
  оставшихся на ночь.

Вход — таблица сделок бэктеста (CSV) с колонками:
  entry_day, exit_day, notional, net_excess_pct, pnl_excess_rub [, horizon]
Такую таблицу отдают модули research/structural/*.

Запуск:
  python -m research.certifier --strategy cash_and_carry \
      --trades audit/r4_research/structural/holdout/trades.csv --variant "C&C-2" [--apply]
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

log = logging.getLogger("research.certifier")

TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "certification")
TRADING_DAYS = 252


def thresholds(registry) -> dict:
    return dict(registry.data.get("certification") or {})


def count_trials(path: str = TRIALS_PATH) -> int:
    if not os.path.exists(path):
        return 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                total += int(json.loads(line).get("trials", 0))
            except ValueError:
                continue
    return total


def t_by_date(values: pd.Series, dates: pd.Series) -> float | None:
    s = pd.Series(np.asarray(values, float), index=np.asarray(dates)).dropna()
    if s.empty:
        return None
    g = s.groupby(level=0).mean()
    if len(g) < 3 or not g.std(ddof=1):
        return None
    return float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g))))


def deflated_sharpe(daily: pd.Series, n_trials: int) -> float | None:
    """DSR (Lopez de Prado): вероятность, что истинный Sharpe > 0 с учётом числа
    испытаний и негауссовости."""
    from scipy import stats
    r = pd.Series(daily).dropna()
    if len(r) < 30 or n_trials < 2 or not r.std(ddof=1):
        return None
    sr = float(r.mean() / r.std(ddof=1))
    var = (1 + 0.5 * sr ** 2) / len(r)
    g = 0.5772156649015329
    sr0 = math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                            + g * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / den))


def metrics(trades: pd.DataFrame, capital_rub: float, n_trials: int) -> dict:
    tr = trades.copy()
    for c in ("entry_day", "exit_day"):
        tr[c] = pd.to_datetime(tr[c]).dt.date
    tr["hold_days"] = [max(1, (b - a).days) for a, b in zip(tr["entry_day"], tr["exit_day"])]
    deployed = float((tr["notional"] * tr["hold_days"] / 365.0).sum())
    excess_rub = float(tr["pnl_excess_rub"].sum())
    days = [d.date() for d in pd.bdate_range(min(tr["entry_day"]), max(tr["exit_day"]))]
    daily = (tr.groupby("exit_day")["pnl_excess_rub"].sum() / capital_rub).reindex(days).fillna(0.0)
    sd = daily.std(ddof=1)
    intraday = tr[tr.get("horizon", pd.Series(["", ] * len(tr))) == "intraday"] if "horizon" in tr else tr.iloc[0:0]
    return {"trades": int(len(tr)), "dates": int(tr["entry_day"].nunique()),
            "net_excess_pct_trade": float(tr["net_excess_pct"].mean()),
            "t": t_by_date(tr["net_excess_pct"], tr["entry_day"]),
            "excess_annual_pct": excess_rub / deployed * 100.0 if deployed > 0 else None,
            "ir": float(daily.mean() / sd * math.sqrt(TRADING_DAYS)) if sd else None,
            "dsr": deflated_sharpe(daily, max(2, n_trials)), "trials": n_trials,
            "square_off_violations": int((intraday["exit_day"] > intraday["entry_day"]).sum()) if len(intraday) else 0}


def verdict(m: dict, th: dict) -> tuple[bool, list[str]]:
    fails = []
    checks = [("t", m["t"], th["t_min"], "t по датам"),
              ("excess_annual_pct", m["excess_annual_pct"], th["excess_annual_min_pct"],
               "избыточная доходность сверх фонда, % годовых"),
              ("ir", m["ir"], th["ir_min"], "IR к фонду"),
              ("dsr", m["dsr"], th["dsr_min"], "DSR")]
    for key, got, need, name in checks:
        if got is None or got < need:
            fails.append(f"{name}: {'—' if got is None else f'{got:.2f}'} < {need}")
    if m["square_off_violations"] > int(th.get("square_off_violations_max", 0)):
        fails.append(f"интрадей-позиции, оставшиеся на ночь: {m['square_off_violations']}")
    return (not fails), fails


def certify(strategy_id: str, trades: pd.DataFrame, *, registry, capital_rub: float | None = None,
            n_trials: int | None = None, apply: bool = False) -> dict:
    th = thresholds(registry)
    cap = capital_rub or float((registry.data.get("limits") or {}).get("capital_rub", 5_000_000))
    trials = count_trials() if n_trials is None else n_trials
    m = metrics(trades, cap, trials)
    ok, fails = verdict(m, th)
    res = {"strategy_id": strategy_id, "verdict": "CERTIFICATION_PASS" if ok else "CERTIFICATION_REJECTED",
           "metrics": m, "thresholds": th, "failed": fails,
           "created": dt.datetime.now().isoformat(timespec="seconds")}
    if apply:
        registry.set_status(strategy_id, "CERTIFIED" if ok else "RESEARCH", metrics=m,
                            note="; ".join(fails) if fails else "критерии ТЗ 2.1 выполнены")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{strategy_id}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сертификация гипотезы по критериям ТЗ 2.1")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--trades", required=True, help="CSV сделок отложенной выборки")
    ap.add_argument("--variant", default="", help="фильтр по колонке variant")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="записать статус в реестр стратегий")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from services.strategies.registry import StrategyRegistry
    tr = pd.read_csv(a.trades)
    if a.variant:
        tr = tr[tr["variant"] == a.variant]
    if tr.empty:
        raise SystemExit("нет сделок для сертификации")
    res = certify(a.strategy, tr, registry=StrategyRegistry(), n_trials=a.trials, apply=a.apply)
    m = res["metrics"]
    print(f"{res['verdict']}: {a.strategy} ({m['trades']} сделок, {m['dates']} дат)")
    print(f"  t {m['t']:.2f} | сверх фонда {m['excess_annual_pct']:.2f} % год | IR {m['ir']:.2f} | "
          f"DSR {m['dsr'] if m['dsr'] is None else round(m['dsr'], 3)} (испытаний {m['trials']})")
    for f in res["failed"]:
        print("  не выполнено —", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
