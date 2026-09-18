"""
Модель издержек стратегического исследования (ТЗ 18.09.2026, раздел 1.1).

Пишется ДО стратегий и наследуется всеми модулями H1–H5. Нулевого
проскальзывания не бывает: каждая нога платит комиссию, спред и ценовое
воздействие.

  комиссия  = 0,04 % (брокер «Премиум») + 0,03 % (тейкер биржи) за сторону,
              круг 0,14 %;
  спред     = Корвин–Шульц (2012) по high/low, но не ниже измеренного по
              5-минуткам медианного спреда бумаги (audit/out/cost_matrix.csv):
              оценка CS на редких барах занижает спред неликвида;
  воздействие = закон квадратного корня I = Y·σ·√(Q/V), Y = 1 (консервативно),
              σ — дневная волатильность бумаги, Q — сумма заявки, V — дневной
              оборот. Платится на каждой ноге отдельно.

Ёмкость. Максимум чистой прибыли (α − Y·σ·√(Q/V))·Q по Q даёт
    Q_opt/V = 4/9 · (α/(Y·σ))²,
та самая формула из ТЗ (она и доказывает, что в законе воздействия стоит
корень: без корня оптимума не существует). Стратегия допускается к боевому
контуру, только если Q_opt на задействованных бумагах ≥ 5 млн ₽.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                      # noqa: E402

FEE_BROKER_SIDE_PCT = 0.04
FEE_EXCHANGE_SIDE_PCT = 0.03
IMPACT_Y = 1.0


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Спред Корвина–Шульца (2012) по high/low двух соседних дней, % цены."""
    b = np.log(high / low) ** 2 + np.log(high.shift(1) / low.shift(1)) ** 2
    gm = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    k = 3.0 - 2.0 * math.sqrt(2.0)
    a = (np.sqrt(2.0 * b) - np.sqrt(b)) / k - np.sqrt(gm / k)
    return (2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))).clip(lower=0.0) * 100.0


@dataclass
class CostModel:
    """Издержки одной ноги. Все величины — проценты от суммы позиции."""

    impact_y: float = IMPACT_Y
    fee_side_pct: float = FEE_BROKER_SIDE_PCT + FEE_EXCHANGE_SIDE_PCT
    spread_floor: dict = field(default_factory=cm.load_spreads)

    # ── составляющие ────────────────────────────────────────────────────────
    def fee_round_trip_pct(self) -> float:
        return 2.0 * self.fee_side_pct

    def spread_pct(self, ticker: str, cs_pct: float | None = None) -> float:
        """Спред за круг: CS-оценка, но не ниже измеренной по 5-минуткам."""
        floor = (self.spread_floor.get(ticker) or self.spread_floor[cm._FALLBACK])[0]
        if cs_pct is None or not np.isfinite(cs_pct):
            return float(floor)
        return float(max(cs_pct, floor))

    def impact_pct(self, sigma_pct: float, notional_rub: float, adv_rub: float) -> float:
        """Закон квадратного корня на одну ногу."""
        if not (adv_rub and adv_rub > 0 and sigma_pct and sigma_pct > 0) or notional_rub <= 0:
            return float("inf")
        return float(self.impact_y * sigma_pct * math.sqrt(notional_rub / adv_rub))

    def round_trip_pct(self, ticker: str, sigma_pct: float, notional_rub: float,
                       adv_rub: float, cs_pct: float | None = None) -> float:
        """Полные издержки круга: комиссия + спред + воздействие на двух ногах."""
        return (self.fee_round_trip_pct() + self.spread_pct(ticker, cs_pct)
                + 2.0 * self.impact_pct(sigma_pct, notional_rub, adv_rub))

    # ── ёмкость ─────────────────────────────────────────────────────────────
    def capacity_rub(self, alpha_pct: float, sigma_pct: float, adv_rub: float) -> float:
        """Q_opt = 4/9·(α/(Y·σ))²·V — сколько рублей бумага вынесет за сделку."""
        if alpha_pct is None or alpha_pct <= 0 or not sigma_pct or sigma_pct <= 0 or adv_rub <= 0:
            return 0.0
        return float(4.0 / 9.0 * (alpha_pct / (self.impact_y * sigma_pct)) ** 2 * adv_rub)

    def max_notional_rub(self, alpha_pct: float, sigma_pct: float, adv_rub: float) -> float:
        """То же, но как предел размера заявки (для сайзинга в плагине)."""
        return self.capacity_rub(alpha_pct, sigma_pct, adv_rub)

    def carry_pct(self, notional_rub: float, nights: int) -> float:
        """Плата за перенос непокрытой позиции (только шорты)."""
        return cm.carry_pct(notional_rub, nights) if nights > 0 else 0.0


def net_excess_pct(gross_pct: float, cost_pct: float, fund_pct: float) -> float:
    """Чистая доходность сверх ставки фонда TMON@ за то же время."""
    return gross_pct - cost_pct - fund_pct
