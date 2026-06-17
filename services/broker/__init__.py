"""Брокерская интеграция для services/place_orders.py.

Структура:
    base.py             — абстрактный BrokerClient + dataclass'ы Quotation,
                          Instrument, OrderState. Бизнес-логика place_orders
                          разговаривает ТОЛЬКО с этим интерфейсом.
    tinkoff_sandbox.py  — реализация для T-Invest sandbox-эндпоинта
                          (sandbox-invest-public-api.tbank.ru) на stdlib urllib,
                          без новых зависимостей. Тот же приём с ssl=False, что
                          в loaders/list_accounts (MITM на машине пользователя).

Чтобы переключиться на prod или другого брокера — реализовать новый класс
с тем же интерфейсом, бизнес-логику править не нужно.
"""
from services.broker.base import (
    BrokerClient,
    Instrument,
    OrderState,
    Position,
    Quotation,
    BrokerError,
    NotSupportedError,
)
from services.broker.tinkoff_sandbox import TinkoffSandboxClient

__all__ = [
    "BrokerClient", "Instrument", "OrderState", "Position", "Quotation",
    "BrokerError", "NotSupportedError", "TinkoffSandboxClient",
]
