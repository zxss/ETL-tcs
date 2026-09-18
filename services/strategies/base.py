"""
Контракт плагина торговой стратегии (ТЗ пользователя 17.09.2026, раздел 3).

Каждая торговая идея — изолированный плагин: выдаёт сигналы и решает, когда
выходить. Размер позиции плагин только ЗАПРАШИВАЕТ (target_budget_rub); сколько
дать и давать ли вообще, решает services/strategies/allocator.py.

Поведение боевых стратегий этот слой не меняет: плагины intraday_short и
long_overnight оборачивают тот же расчёт (tft_forecast.combined.build_orders) и
несут в plan_order готовую запись плана — байт в байт как сейчас.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FundingType(Enum):
    CASH_UNPARK_TMON = "cash_unpark_tmon"   # нужен реальный кэш: продажа паёв TMON@
    MARGIN_FREE = "margin_free"             # внутридневная маржа под залог паёв
    DERIVATIVE_COLLATERAL = "derivative"    # ГО фьючерса под залог паёв


class HoldingHorizon(Enum):
    INTRADAY = "intraday"                   # жёсткое закрытие в 18:20
    OVERNIGHT = "overnight"                 # 18:35 → 09:10
    MULTI_DAY = "multi_day"                 # 2–30 дней


@dataclass
class TradeSignal:
    strategy_id: str
    ticker: str
    side: str                               # "BUY" | "SELL"
    priority_score: float                   # ожидаемый перевес на рубль издержек
    funding: FundingType
    horizon: HoldingHorizon
    target_budget_rub: float
    entry_price_type: str = "LIMIT"         # "LIMIT" | "MARKET"
    limit_entry_fraction: float = 0.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_holding_days: int = 1
    # Готовая запись плана боевого контура (services/stage2_demo._plan_order).
    # Нужна, чтобы движок не пересчитывал заявку и не менял поведение r4.
    plan_order: Optional[dict] = None
    meta: dict = field(default_factory=dict)

    @property
    def needs_cash(self) -> bool:
        return self.funding is FundingType.CASH_UNPARK_TMON

    def __post_init__(self):
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side={self.side!r}: ожидается BUY или SELL")
        if self.target_budget_rub < 0:
            raise ValueError("target_budget_rub не может быть отрицательным")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days ≥ 1")


class BaseStrategy(ABC):
    """Интерфейс плагина. Ошибка внутри плагина не должна ронять остальные —
    вызовы оборачивает реестр (registry.collect_signals)."""

    strategy_id: str = ""
    version: str = "0"
    is_active: bool = False

    @abstractmethod
    def generate_signals(self, asof_date: dt.date, market_data: dict) -> list[TradeSignal]:
        """Сигналы на день по рыночному контексту.

        market_data — общий контекст фазы PREP: rows (строки рейтинга), orders
        (результат build_orders), day, конфигурация. Плагин берёт только своё.
        """

    @abstractmethod
    def should_exit(self, position: dict, current_price: float, days_held: int) -> tuple[bool, str]:
        """Пора ли закрывать позицию: (да/нет, причина)."""

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "is_active": self.is_active}
