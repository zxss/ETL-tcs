"""
Направленный прогноз: ожидаемый PnL стратегии на следующий день.

В отличие от ценового коридора (low/high), направленный PnL зависит от
стратегии. Чтобы оценить его честно, мы НЕ выводим распределение PnL из
маргинальных квантилей low/high (это статистически некорректно — нужна
совместная плотность). Вместо этого TFT предсказывает квантили самих
ПРИМИТИВНЫХ доходностей следующего дня (overnight / intraday / total),
а стратегия отображается на них со своим знаком:

    long_overnight  →  +next_overnight
    long_intraday   →  +next_intraday
    intraday_short  →  -next_intraday
    short_hold      →  -next_total

Из предсказанной квантильной функции считаем:
    ExpPnL    — медианная чистая доходность (за вычетом издержек round-trip),
    Downside  — q0.1 (нетто), Upside — q0.9 (нетто),
    ProbProfit — P(доходность > издержек), интерполяцией квантильной функции.
"""

from __future__ import annotations

import numpy as np

# Карта стратегия → (индекс цели в TARGET_COLS, знак)
# Индексы должны совпадать с forecast.T_OVN / T_INTRA / T_TOTAL.
STRATEGY_MAP = {
    "long_overnight": (2, +1.0),   # T_OVN
    "long_intraday":  (3, +1.0),   # T_INTRA  (внутренний синоним intraday_long)
    "intraday_long":  (3, +1.0),   # T_INTRA  — лонг внутри дня
    "intraday_short": (3, -1.0),   # T_INTRA  — шорт внутри дня
    "short_hold":     (4, -1.0),   # T_TOTAL
}

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _signed_quantiles(qvals: np.ndarray, sign: float) -> np.ndarray:
    """qvals — квантили примитива (по возрастанию уровней QUANTILES).
    Возвращает квантили знаковой доходности, отсортированные по возрастанию."""
    s = sign * qvals
    return np.sort(s)


def _prob_above(levels, vals, threshold: float) -> float:
    """P(R > threshold) по кусочно-линейной интерполяции квантильной функции.
    levels — уровни вероятности (возр.), vals — значения квантилей (возр.)."""
    vals = np.asarray(vals, dtype=np.float64)
    # гарантируем строгую монотонность для интерполяции
    vals = vals + np.linspace(0, 1e-9, len(vals))
    cdf = float(np.interp(threshold, vals, levels))
    cdf = min(max(cdf, 0.02), 0.98)  # хвосты за пределами сетки — неопределённость
    return 1.0 - cdf


def strategy_pnl(pred_targets: np.ndarray, strategy: str, cost_rt: float) -> dict | None:
    """
    pred_targets: (n_targets, n_quantiles) — предсказанные квантили целей одного тикера.
    Возвращает метрики направленного PnL стратегии или None если стратегия неизвестна.
    """
    if strategy not in STRATEGY_MAP:
        return None
    tgt_i, sign = STRATEGY_MAP[strategy]
    qvals = pred_targets[tgt_i]                 # квантили примитива (возр. по уровню)
    sq = _signed_quantiles(qvals, sign)         # знаковая доходность, возр.

    med = float(np.median(sq))                  # центр = q0.5 после сортировки
    q10, q90 = float(sq[0]), float(sq[-1])

    exp_net = med - cost_rt
    downside_net = q10 - cost_rt
    upside_net = q90 - cost_rt
    prob_profit = _prob_above(QUANTILES, sq, cost_rt)

    return {
        "ExpPnL": exp_net,
        "Downside": downside_net,
        "Upside": upside_net,
        "ProbProfit": prob_profit,
        "gross_median": med,
    }


def build_rows(infer_pred: np.ndarray, infer_tickers: list[str],
               tickers: list[str], strats: list[str], cost_rt: float,
               coridor: dict | None = None) -> list[dict]:
    """
    Собирает строки направленного прогноза по (ticker, strategy).
    infer_pred: (M, n_targets, n_quantiles); infer_tickers — порядок строк.
    coridor: {ticker: {ForecastLow, ForecastHigh, ...}} для контекста.
    """
    idx = {tk: i for i, tk in enumerate(infer_tickers)}
    rows = []
    for tk in tickers:
        key = tk.upper() if tk.upper() in idx else tk
        if key not in idx:
            continue
        pt = infer_pred[idx[key]]
        cor = (coridor or {}).get(key) or (coridor or {}).get(tk.upper()) or {}
        for st in strats:
            m = strategy_pnl(pt, st, cost_rt)
            if m is None:
                continue
            rows.append({
                "ticker": tk.upper(),
                "strategy": st,
                "ExpPnL": m["ExpPnL"],
                "Downside": m["Downside"],
                "Upside": m["Upside"],
                "ProbProfit": m["ProbProfit"],
                "ForecastLow": cor.get("ForecastLow"),
                "ForecastHigh": cor.get("ForecastHigh"),
            })
    return rows


def print_table(rows: list[dict], cost_rt: float):
    print("\n  НАПРАВЛЕННЫЙ ПРОГНОЗ: ОЖИДАЕМЫЙ PnL СТРАТЕГИИ (нетто, round-trip cost=%.3f%%)" % cost_rt)
    print("  " + "-" * 86)
    print(f"  {'ticker':<8}{'strategy':<17}{'ExpPnL%':>9}{'Down%':>9}{'Up%':>9}"
          f"{'ProbProfit':>12}{'F.Low':>11}{'F.High':>11}")
    print("  " + "-" * 86)
    # сортировка по ожидаемому PnL убыв.
    for r in sorted(rows, key=lambda x: x["ExpPnL"], reverse=True):
        fl = f"{r['ForecastLow']:.2f}" if r.get("ForecastLow") is not None else "—"
        fh = f"{r['ForecastHigh']:.2f}" if r.get("ForecastHigh") is not None else "—"
        flag = "  ✓" if r["ExpPnL"] > 0 and r["ProbProfit"] > 0.5 else ""
        print(f"  {r['ticker']:<8}{r['strategy']:<17}{r['ExpPnL']:>+8.3f}%{r['Downside']:>+8.3f}%"
              f"{r['Upside']:>+8.3f}%{r['ProbProfit'] * 100:>11.0f}%{fl:>11}{fh:>11}{flag}")
    print("  " + "-" * 86)
    print("  ✓ — ExpPnL>0 и ProbProfit>50%. Прогноз сырой (не означает наличие устойчивого edge:")
    print("     см. контур валидации — White RC / SPA / BH-FDR).")
