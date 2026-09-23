"""
Издержки сделки для исследований — одна точка правды (с 15.09.2026).

Тариф пользователя — «Премиум» T-Инвестиций (подтверждено пользователем
15.09.2026, сверено с tbank.ru/bank/help/general/premium/services/investment):
  * комиссия 0,04 % от суммы КАЖДОЙ сделки — покупка и продажа отдельно,
    круг = 0,08 %;
  * непокрытая позиция, закрытая внутри дня, — без платы;
  * непокрытая позиция на конец дня до 5 000 ₽ — без платы, свыше — «от 45 ₽
    в день» (за календарный день). Точной сетки для крупных позиций на странице
    «Премиум» нет: берётся max(45 ₽, сетка тарифа «Инвестор») — 10 000 ₽ →
    45 ₽, 200 000 ₽ → 190 ₽.

Кроме комиссии, рыночная заявка платит спред и ценовое воздействие. В тарифе
их нет, они оцениваются по 5-минуткам: audit/costs.py → audit/out/cost_matrix.csv
(Corwin–Schultz, не уже одного шага цены; Amihud при заявке 10 000 ₽).

Сценарии круга на одну ногу:
  fee    — только комиссия 0,08 %: нижняя граница (лимитные заявки без
           проскальзывания; на свечах не проверяется — стакана нет);
  base   — комиссия + медианный спред + 2 × воздействие по бумаге: основной;
  stress — комиссия + 95-й перцентиль спреда + 2 × воздействие: чувствительность.
Прежняя плоская 0,128 % = 0,08 комиссии + медианный по вселенной спред 0,048.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEE_SIDE_PCT = 0.04                      # тариф «Премиум», % за сделку
SCENARIOS = ("fee", "base", "stress")
PRIMARY, SENSITIVITY = "base", "stress"
LABELS = {"fee": "только комиссия 0,08 % (лимитки, нижняя граница)",
          "base": "комиссия 0,08 % + спред бумаги (рыночные, основной)",
          "stress": "комиссия 0,08 % + стресс-спред"}

CARRY_FREE_UPTO_RUB = 5_000.0
CARRY_MIN_RUB_DAY = 45.0
_INVESTOR_GRID = ((50_000, 40.0), (100_000, 80.0), (250_000, 190.0),
                  (500_000, 375.0), (1_000_000, 750.0))
MATRIX_PATH = os.path.join(ROOT, "audit", "out", "cost_matrix.csv")
_FALLBACK = "__fallback__"


def load_spreads(path: str = MATRIX_PATH) -> dict:
    """ticker → (спред %, спред p95 %, воздействие %). Бумаги вне матрицы (T,
    NLMK) — медиана голубых фишек."""
    import pandas as pd
    m = pd.read_csv(path)
    out = {r.ticker: (float(r.spread_pct), float(r.spread_p95_pct), float(r.impact_pct))
           for r in m.itertuples(index=False)}
    blue = m[m["segment"] == "blue_chip"]
    out[_FALLBACK] = (float(blue["spread_pct"].median()), float(blue["spread_p95_pct"].median()),
                      float(blue["impact_pct"].median()))
    return out


def round_trip(ticker: str, scenario: str, spreads: dict) -> float:
    """Круг (вход + выход) одной ноги, % от позиции."""
    fee = 2.0 * FEE_SIDE_PCT
    if scenario == "fee":
        return fee
    sp, sp95, imp = spreads.get(ticker) or spreads[_FALLBACK]
    if scenario == "base":
        return fee + sp + 2.0 * imp
    if scenario == "stress":
        return fee + sp95 + 2.0 * imp
    raise ValueError(scenario)


def trade_cost(tickers: str, scenario: str, spreads: dict) -> float:
    """«SBER» или пара «LKOH/ROSN» — сумма кругов по ногам."""
    return sum(round_trip(t, scenario, spreads) for t in str(tickers).split("/"))


def carry_pct(position_rub: float, nights: int) -> float:
    """Плата за перенос непокрытой позиции на тарифе «Премиум», % позиции."""
    if position_rub <= CARRY_FREE_UPTO_RUB:
        return 0.0
    for cap, fee in _INVESTOR_GRID:
        if position_rub <= cap:
            return max(CARRY_MIN_RUB_DAY, fee) * nights / position_rub * 100.0
    raise ValueError(f"позиция {position_rub} ₽ вне таблицы тарифа")
