"""
Плагин боевой стратегии long_overnight (ТЗ 17.09.2026, шаг 2).

Расчёт не переписан — обёртка над тем же build_orders (см. intraday_short).
Финансирование — реальный кэш: паи TMON@ продаются в фазе OVERNIGHT 18:35,
выход утром в фазе CLOSE 09:10, деньги возвращаются в фонд.
"""
from __future__ import annotations

import datetime as dt

from services.strategies.base import BaseStrategy, FundingType, HoldingHorizon, TradeSignal
from services.strategies.plugins.intraday_short import signals_from_orders

STRATEGY_TYPE = "long_overnight"


class LongOvernightStrategy(BaseStrategy):
    strategy_id = STRATEGY_TYPE
    version = "r4.1"
    is_active = True

    def generate_signals(self, asof_date: dt.date, market_data: dict) -> list[TradeSignal]:
        return signals_from_orders(self.strategy_id, STRATEGY_TYPE, FundingType.CASH_UNPARK_TMON,
                                   HoldingHorizon.OVERNIGHT, market_data, max_holding_days=1)

    def should_exit(self, position: dict, current_price: float, days_held: int) -> tuple[bool, str]:
        """Ночная позиция закрывается на следующем открытии (фаза CLOSE 09:10)."""
        if days_held >= 1:
            return True, "утренний выход 09:10"
        return False, ""
