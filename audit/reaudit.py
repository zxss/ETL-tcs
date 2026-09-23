"""
Повторный квант-аудит — против ТЕКУЩЕЙ конфигурации.

Первый аудит (AUDIT-PROFITABILITY-REPORT.md) измерял код на коммите eabcb08.
С тех пор закрыты 8 из 12 его приоритетов, и вердикт «0 из 5 критериев»
описывает уже несуществующую систему.

Реплей (audit/out/walkforward.csv) переиспользуется намеренно: модельный слой
(features.py, dataset.py, model.py) с момента его генерации НЕ МЕНЯЛСЯ, в
forecast.py добавлены только выходные поля контекста. Изменился слой скоринга
и фильтрации — именно он и переизмеряется.

Ключевое добавление к первому аудиту: Deflated Sharpe Ratio. Конфигурация
подбиралась примерно по пятнадцати вариантам на одной выборке, и без поправки
на число испытаний любой Sharpe здесь завышен по построению.

Запуск:
    python3 -m audit.reaudit
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import backtest, costs, data, ic  # noqa: E402
from tft_forecast.combined import (  # noqa: E402
    _score_row, apply_strategy_specialisation,
)

log = logging.getLogger("audit.reaudit")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Сколько независимых конфигураций было испытано на этой выборке за весь
# цикл ремедиации: 3 режима скоринга x (штрафы вкл/выкл) = 6, плюс 3 варианта
# знаменателя TradeScore, плюс 7 вариантов фильтра импульса, плюс 5 порогов
# овернайта, плюс 2 значения top_n. Округлено вниз — оценка консервативна
# в сторону ЗАВЫШЕНИЯ значимости.
N_TRIALS = 15

DIR = {"long_overnight": "LONG", "intraday_long": "LONG",
       "intraday_short": "SHORT", "short_hold": "SHORT"}


# ── Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) ────────────────────

def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Ожидаемый МАКСИМАЛЬНЫЙ Sharpe из n_trials испытаний при НУЛЕВОЙ альфе.

    Если перебрать 15 конфигураций на одних данных, лучшая покажет
    положительный Sharpe даже когда ни одна не имеет преимущества. Эта
    величина — та планка, которую надо превзойти, чтобы результат что-то значил.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329          # постоянная Эйлера — Маскерони
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sr_variance) * ((1 - gamma) * z1 + gamma * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    sr_variance: float | None = None,
                    freq: int = 252) -> dict:
    """DSR — вероятность того, что истинный Sharpe больше нуля С УЧЁТОМ отбора.

    Поправка на две вещи сразу: на число испытаний (через порог sr0) и на
    негауссовость доходностей (через асимметрию и эксцесс). Ряд с редкими
    крупными выигрышами и частыми мелкими потерями даёт завышенный обычный
    Sharpe — DSR это штрафует.
    """
    r = returns.dropna()
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return {"n": T}
    sr_period = float(r.mean() / r.std(ddof=1))          # Sharpe за период (день)
    sr_annual = sr_period * np.sqrt(freq)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))        # обычный эксцесс

    if sr_variance is None:
        # Оценка разброса Sharpe между испытаниями: при отсутствии реальной
        # выборки испытаний берём дисперсию оценки самого Sharpe.
        sr_variance = (1 + 0.5 * sr_period ** 2) / T
    sr0 = expected_max_sharpe(n_trials, sr_variance)

    denom = np.sqrt(1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return {"n": T, "sharpe_annual": sr_annual}
    z = (sr_period - sr0) * np.sqrt(T - 1) / denom
    return {
        "n": T,
        "sharpe_annual": sr_annual,
        "sharpe_period": sr_period,
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": n_trials,
        "sr0_period": float(sr0),
        "sr0_annual": float(sr0 * np.sqrt(freq)),
        "deflated_sharpe_prob": float(stats.norm.cdf(z)),
    }


# ── Сборка текущего конвейера ────────────────────────────────────────────────

def build_current_pipeline() -> pd.DataFrame:
    """Реплей + контекст, который сейчас считает production, + текущий скоринг."""
    wf = pd.read_csv(os.path.join(OUT, "walkforward.csv"), parse_dates=["asof_date"])
    comp = ic.build_components(wf[wf.fold_primary].copy())

    daily = data.load_daily()
    shares = daily[daily.ticker != "IMOEX"].sort_values(["ticker", "date"]).copy()

    # ret1 — доходность последнего закрытого дня (market._ret1)
    shares["ret1"] = shares.groupby("ticker")["close"].pct_change() * 100.0
    comp = comp.merge(
        shares[["ticker", "date", "ret1"]].rename(columns={"date": "asof_date"}),
        on=["ticker", "asof_date"], how="left")

    # market_atr_pctl — медиана ATR-перцентиля по вселенной (market.atr_pctl_market)
    comp = comp.merge(
        comp.groupby("asof_date")["atr_pctl"].median().rename("market_atr_pctl"),
        on="asof_date", how="left")

    # index_above_ema50 — предохранитель от шорт-сквиза (market._above_ema)
    idx = daily[daily.ticker == "IMOEX"].sort_values("date").copy()
    idx["index_above_ema50"] = (
        idx["close"] > idx["close"].ewm(span=50, adjust=False).mean())
    comp = comp.merge(
        idx[["date", "index_above_ema50"]].rename(columns={"date": "asof_date"}),
        on="asof_date", how="left")

    # Текущий production-скоринг, вызванный БЕЗ аргументов — как в бою.
    recs = comp.to_dict("records")
    scores, allowed = np.empty(len(recs)), np.empty(len(recs), dtype=bool)
    for i, r in enumerate(recs):
        row = dict(r)
        row["direction"] = DIR.get(r["strategy"], "LONG")
        row["verdict"] = None
        for k in ("exp_pnl", "prob_profit", "liq_score", "rs", "vol_spike",
                  "gap_down_prob", "atr_pct", "range_pct", "cost_rt",
                  "ret1", "market_atr_pctl"):
            v = row.get(k)
            if v is not None and isinstance(v, float) and v != v:
                row[k] = None
        if not isinstance(row.get("regime"), str):
            row["regime"] = None
        s, a, _ = _score_row(row, False)
        scores[i], allowed[i] = s, a
    comp["score"] = scores
    comp["allowed"] = allowed

    # Специализация Пути А — тот же вызов, что в select_top_rows.
    kept = apply_strategy_specialisation(recs, verbose=False)
    kept_ids = {id(r) for r in kept}
    comp["specialised"] = [id(r) in kept_ids for r in recs]
    return comp


def stage1_gates(perf: dict, alpha: dict) -> dict:
    need = {"sharpe": 1.30, "profit_factor": 1.45,
            "max_drawdown_pct": -12.0, "alpha_annual_pct": 15.0}
    got = {"sharpe": perf.get("sharpe"), "profit_factor": perf.get("profit_factor"),
           "max_drawdown_pct": perf.get("max_drawdown_pct"),
           "alpha_annual_pct": alpha.get("alpha_annual_pct")}
    out = {}
    for k, v in need.items():
        g = got.get(k)
        out[k] = {"факт": None if g is None else round(float(g), 3),
                  "порог": v,
                  "пройден": bool(g is not None and np.isfinite(g) and g >= v)}
    out["_пройдено"] = sum(1 for k, v in out.items()
                           if not k.startswith("_") and v["пройден"])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import config

    comp = build_current_pipeline()
    daily = data.load_daily()
    bench = backtest.benchmark_returns(daily, "IMOEX", forward=True)
    instr = costs.load_instruments()
    sb = set(instr.loc[instr["short_enabled"] != True, "ticker"])  # noqa: E712
    ALL = pd.DatetimeIndex(sorted(comp.asof_date.unique()))
    dts = ALL.to_numpy()
    mid = dts[len(dts) // 2]
    res: dict = {"config": {
        "SCORE_MODE": config.SCORE_MODE,
        "APPLY_RISK_PENALTIES": bool(config.APPLY_RISK_PENALTIES),
        "TRADING_STRATEGIES": list(config.TRADING_STRATEGIES),
        "SELLER_MOMENTUM_SHORT_ENABLED": bool(config.SELLER_MOMENTUM_SHORT_ENABLED),
        "SHORT_IMOEX_MAX_TREND": config.SHORT_IMOEX_MAX_TREND,
        "OVERNIGHT_MIN_EDGE_X_COST": config.OVERNIGHT_MIN_EDGE_X_COST,
        "BEST_TRADES_TOP_N": config.BEST_TRADES_TOP_N,
    }}

    def curve(df, top_n):
        dr, picks = backtest.signal_backtest(
            df.dropna(subset=["score"]), "score", top_n=top_n,
            ret_col="realized_net", short_blocked=sb)
        return dr.reindex(ALL).fillna(0.0), picks

    PROD3 = {"long_overnight", "intraday_short", "intraday_long"}
    variants = {
        "исходная (3 стратегии, top-10)": (comp[comp.strategy.isin(PROD3)], 10),
        "ТЕКУЩАЯ (специализация, top-5)": (
            comp[comp.specialised & comp.allowed], config.BEST_TRADES_TOP_N),
    }
    res["backtest"] = {}
    for label, (df, n) in variants.items():
        s, picks = curve(df, n)
        p = backtest.performance(s)
        a = backtest.alpha_vs_benchmark(s, bench)
        h1 = backtest.performance(s[s.index <= mid])
        h2 = backtest.performance(s[s.index > mid])
        res["backtest"][label] = {
            "активных_дней": int((s != 0).sum()),
            "CAGR_%": round(p["cagr_pct"], 3),
            "Sharpe": round(p["sharpe"], 3),
            "PF": round(p["profit_factor"], 3),
            "MDD_%": round(p["max_drawdown_pct"], 3),
            "альфа_%год": round(a.get("alpha_annual_pct", float("nan")), 3),
            "бета": round(a.get("beta", float("nan")), 3),
            "Sharpe_H1": round(h1.get("sharpe", float("nan")), 3),
            "Sharpe_H2": round(h2.get("sharpe", float("nan")), 3),
            "gates": stage1_gates(p, a),
            "dsr": deflated_sharpe(s[s != 0], N_TRIALS),
        }

    # Rank IC текущего рейтинга
    res["rank_ic"] = {}
    for label, df in (("по всей вселенной", comp),
                      ("после специализации", comp[comp.specialised])):
        r = ic.rank_ic(df, "score", "realized_net")
        res["rank_ic"][label] = {
            "ic": round(r.get("ic", float("nan")), 4),
            "t": round(r.get("t_stat", float("nan")), 2),
            "дней": r.get("n_days"),
        }

    res["воронка_сигналов"] = {
        "всего_в_реплее": int(len(comp)),
        "три_стратегии": int(comp.strategy.isin(PROD3).sum()),
        "после_специализации": int(comp.specialised.sum()),
        "состав": comp[comp.specialised].strategy.value_counts().to_dict(),
    }

    with open(os.path.join(OUT, "reaudit.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)

    pd.set_option("display.width", 240)
    print("\n=== ПЕРЕАУДИТ: текущая конфигурация ===")
    print(json.dumps(res["config"], ensure_ascii=False, indent=1))
    rows = []
    for k, v in res["backtest"].items():
        rows.append({"вариант": k, **{kk: vv for kk, vv in v.items()
                                      if kk not in ("gates", "dsr")}})
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("\n=== Критерии Этапа 1 (текущая) ===")
    g = res["backtest"]["ТЕКУЩАЯ (специализация, top-5)"]["gates"]
    for k, v in g.items():
        if k.startswith("_"):
            continue
        print(f"  {k:20} факт={v['факт']:>9}  порог={v['порог']:>7}  "
              f"{'ПРОЙДЕН' if v['пройден'] else 'НЕ ПРОЙДЕН'}")
    print(f"  ИТОГО: {g['_пройдено']} из 4")
    d = res["backtest"]["ТЕКУЩАЯ (специализация, top-5)"]["dsr"]
    print(f"\n=== Deflated Sharpe (испытаний: {N_TRIALS}) ===")
    print(f"  наблюдаемый Sharpe (годовой): {d.get('sharpe_annual', float('nan')):.3f}")
    print(f"  планка от одного лишь отбора: {d.get('sr0_annual', float('nan')):.3f}")
    print(f"  асимметрия={d.get('skew', float('nan')):.2f}  "
          f"эксцесс={d.get('kurtosis', float('nan')):.2f}")
    print(f"  DSR (вероятность истинного Sharpe > 0): "
          f"{d.get('deflated_sharpe_prob', float('nan')):.3f}")
    print("\nСохранено → audit/out/reaudit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
