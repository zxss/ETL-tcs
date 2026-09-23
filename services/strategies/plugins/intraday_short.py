"""
Плагин боевой стратегии intraday_short (ТЗ 17.09.2026, шаг 2).

Расчёт НЕ переписан: плагин берёт готовые строки рейтинга и заявки из
tft_forecast.combined.build_orders (как фаза PREP сейчас) и оформляет их в
TradeSignal. Запись плана переносится без изменений в plan_order — поэтому
поведение прогона r4 остаётся тем же.

Финансирование — внутридневная маржа под залог паёв (кэш из фонда не нужен),
горизонт — интрадей, выход в фазе CLEANUP 18:20.
"""
from __future__ import annotations

import datetime as dt

from services.strategies.base import BaseStrategy, FundingType, HoldingHorizon, TradeSignal

STRATEGY_TYPE = "intraday_short"


def priority_from_row(row: dict, cost_pct: float) -> float:
    """Ожидаемый перевес на рубль издержек: (|ожидаемый ход| − издержки) / издержки."""
    exp = row.get("exp_pnl")
    if exp is None or cost_pct <= 0:
        return 0.0
    return (abs(float(exp)) - cost_pct) / cost_pct


def signals_from_orders(strategy_id: str, strategy_type: str, funding: FundingType,
                        horizon: HoldingHorizon, market_data: dict, max_holding_days: int) -> list[TradeSignal]:
    """Общая сборка сигналов из orders/rows фазы PREP (используют оба боевых плагина)."""
    rows = market_data.get("rows") or []
    orders = market_data.get("orders") or []
    plan_orders = market_data.get("plan_orders") or []
    cost_pct = float(market_data.get("cost_rt_pct") or 0.128)
    out = []
    for i, o in enumerate(orders):
        if getattr(o, "strategy", None) != strategy_type or not getattr(o, "is_placeable", False):
            continue
        row = rows[i] if i < len(rows) else {}
        plan = plan_orders[i] if i < len(plan_orders) else None
        out.append(TradeSignal(
            strategy_id=strategy_id, ticker=o.ticker, side=o.order_direction,
            priority_score=priority_from_row(row, cost_pct), funding=funding, horizon=horizon,
            target_budget_rub=float(o.total_rub or 0.0), entry_price_type="LIMIT",
            limit_entry_fraction=float(market_data.get("limit_entry_fraction") or 0.0),
            stop_loss_pct=o.stop_pct, take_profit_pct=o.tp_pct, max_holding_days=max_holding_days,
            plan_order=plan, meta={"row_index": i, "verdict": row.get("verdict")}))
    return out


class IntradayShortStrategy(BaseStrategy):
    strategy_id = STRATEGY_TYPE
    version = "r4.1"
    is_active = True

    def generate_signals(self, asof_date: dt.date, market_data: dict) -> list[TradeSignal]:
        return signals_from_orders(self.strategy_id, STRATEGY_TYPE, FundingType.MARGIN_FREE,
                                   HoldingHorizon.INTRADAY, market_data, max_holding_days=1)

    def should_exit(self, position: dict, current_price: float, days_held: int) -> tuple[bool, str]:
        """Интрадей: закрытие делает фаза CLEANUP 18:20. Здесь — страховка от
        переноса через ночь: позиция старше дня входа закрывается безусловно."""
        if days_held >= 1:
            return True, "интрадей не переносится через ночь"
        return False, ""
