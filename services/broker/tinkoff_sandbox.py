"""
T-Invest Sandbox реализация BrokerClient (тестовый контур).

Виртуальные счета/деньги, исполнение по last price. Контур-специфичные методы
работают через SandboxService.*. Общая часть (HTTP, find_instrument, стоп-заявки)
— в TinkoffRestBase.
"""
from __future__ import annotations

import logging

import config
from services.broker.base import (
    ActiveOrder,
    BrokerError,
    Instrument,
    OrderState,
    Position,
    Quotation,
)
from services.broker.tinkoff_base import TinkoffRestBase, new_order_id, parse_order_state

log = logging.getLogger("broker.tinkoff_sandbox")

__all__ = ["TinkoffSandboxClient", "new_order_id"]


class TinkoffSandboxClient(TinkoffRestBase):
    """BrokerClient против sandbox-эндпоинта T-Invest API."""

    DEFAULT_BASE = config.SANDBOX_API_BASE_URL

    # ── Счёт ─────────────────────────────────────────────────────────────────

    def open_or_get_account(self, preferred_id: str = "") -> str:
        """preferred_id найден в GetSandboxAccounts → используем; иначе создаём."""
        try:
            data = self._post("SandboxService/GetSandboxAccounts")
            existing = [a.get("id") for a in (data.get("accounts") or []) if a.get("id")]
        except BrokerError as e:
            log.warning("GetSandboxAccounts failed (создадим новый счёт): %s", e)
            existing = []

        if preferred_id and preferred_id in existing:
            log.info("Sandbox account %s найден — переиспользуем.", preferred_id)
            return preferred_id
        if preferred_id:
            log.warning("Sandbox account %s не найден (счета песочницы живут 3 мес.) "
                        "— создаю новый.", preferred_id)

        data = self._post("SandboxService/OpenSandboxAccount")
        new_id = data.get("accountId")
        if not new_id:
            raise BrokerError(f"OpenSandboxAccount: пустой accountId, ответ={data}")
        log.info("Sandbox account создан: %s", new_id)
        return new_id

    def pay_in(self, account_id: str, amount_rub: float, currency: str = "rub") -> None:
        """SandboxService.SandboxPayIn — виртуальное пополнение."""
        amount = Quotation.from_float(amount_rub)
        self._post("SandboxService/SandboxPayIn", {
            "accountId": account_id,
            "amount": {"currency": currency,
                       "units": str(amount.units), "nano": int(amount.nano)},
        })
        log.info("Sandbox account %s пополнен на %.2f %s.",
                 account_id, amount_rub, currency.upper())

    # ── Торговля ─────────────────────────────────────────────────────────────

    def post_limit_order(self, *, account_id: str, instrument: Instrument,
                         direction: str, quantity_lots: int,
                         price: Quotation, order_id: str) -> OrderState:
        """SandboxService.PostSandboxOrder — лимитная заявка."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        data = self._post("SandboxService/PostSandboxOrder", {
            "accountId":    account_id,
            "instrumentId": instrument.instrument_uid,
            "quantity":     str(int(quantity_lots)),
            "price":        price.to_payload(),
            "direction":    f"ORDER_DIRECTION_{direction}",
            "orderType":    "ORDER_TYPE_LIMIT",
            "orderId":      order_id,
        })
        return parse_order_state(data)

    def post_market_order(self, *, account_id: str, instrument: Instrument,
                          direction: str, quantity_lots: int,
                          order_id: str) -> OrderState:
        """SandboxService.PostSandboxOrder с ORDER_TYPE_MARKET — закрытие по рынку."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        data = self._post("SandboxService/PostSandboxOrder", {
            "accountId":    account_id,
            "instrumentId": instrument.instrument_uid,
            "quantity":     str(int(quantity_lots)),
            "direction":    f"ORDER_DIRECTION_{direction}",
            "orderType":    "ORDER_TYPE_MARKET",
            "orderId":      order_id,
        })
        return parse_order_state(data)

    def get_order_state(self, *, account_id: str, order_id: str) -> OrderState:
        data = self._post("SandboxService/GetSandboxOrderState",
                          {"accountId": account_id, "orderId": order_id})
        return parse_order_state(data)

    def cancel_order(self, *, account_id: str, order_id: str) -> None:
        self._post("SandboxService/CancelSandboxOrder",
                   {"accountId": account_id, "orderId": order_id})

    # ── Портфель ─────────────────────────────────────────────────────────────

    def get_positions(self, account_id: str) -> list[Position]:
        data = self._post("SandboxService/GetSandboxPositions", {"accountId": account_id})
        out: list[Position] = []
        for s in (data.get("securities") or []):
            uid = s.get("instrumentUid") or ""
            if uid:
                out.append(Position(instrument_uid=uid,
                                    balance_shares=float(s.get("balance", 0) or 0),
                                    blocked_shares=float(s.get("blocked", 0) or 0)))
        return out

    def get_active_orders(self, account_id: str) -> list[ActiveOrder]:
        data = self._post("SandboxService/GetSandboxOrders", {"accountId": account_id})
        out: list[ActiveOrder] = []
        for o in (data.get("orders") or []):
            uid = o.get("instrumentUid")
            if not uid:
                continue
            out.append(ActiveOrder(
                instrument_uid=uid,
                order_id=o.get("orderId", ""),
                direction=(o.get("direction", "") or "").replace("ORDER_DIRECTION_", ""),
                lots=int(o.get("lotsRequested", 0) or 0),
            ))
        return out
