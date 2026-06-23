"""Брокерская интеграция для services/place_orders.py.

Структура:
    base.py            — абстрактный BrokerClient + dataclass'ы Quotation,
                         Instrument, OrderState, Position. Бизнес-логика
                         place_orders разговаривает ТОЛЬКО с этим интерфейсом.
    tinkoff_base.py    — общий REST-транспорт (urllib, ssl=False), find_instrument,
                         стоп-заявки. База для обоих контуров.
    tinkoff_sandbox.py — TinkoffSandboxClient: тестовый контур (виртуальные деньги).
    tinkoff_prod.py    — TinkoffProdClient: БОЕВОЙ счёт, реальные деньги.

Сменить контур/брокера → новый класс с тем же интерфейсом, бизнес-логику не трогаем.
"""
from services.broker.base import (
    ActiveOrder,
    BrokerClient,
    Instrument,
    OrderState,
    Position,
    Quotation,
    StopOrderInfo,
    BrokerError,
    NotSupportedError,
)
from services.broker.tinkoff_base import new_order_id
from services.broker.tinkoff_sandbox import TinkoffSandboxClient
from services.broker.tinkoff_prod import TinkoffProdClient

__all__ = [
    "ActiveOrder", "BrokerClient", "Instrument", "OrderState", "Position",
    "Quotation", "StopOrderInfo", "BrokerError", "NotSupportedError",
    "TinkoffSandboxClient", "TinkoffProdClient", "new_order_id",
]
