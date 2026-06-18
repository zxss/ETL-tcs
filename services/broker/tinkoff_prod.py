"""
T-Invest PROD реализация BrokerClient — РЕАЛЬНЫЙ счёт, РЕАЛЬНЫЕ деньги.

Контур-специфичные методы работают через боевые сервисы:
  OrdersService.PostOrder / GetOrderState / GetOrders / CancelOrder
  OperationsService.GetPositions
  UsersService.GetAccounts (верификация счёта; счёт НЕ создаётся и НЕ пополняется)

Общая часть (HTTP, find_instrument, стоп-заявки через StopOrdersService) — в
TinkoffRestBase. Эндпоинт — боевой config.API_BASE_URL.
"""
from __future__ import annotations

import logging

import config
from services.broker.base import (
    BrokerError,
    Instrument,
    OrderState,
    Position,
    Quotation,
)
from services.broker.tinkoff_base import TinkoffRestBase, parse_order_state

log = logging.getLogger("broker.tinkoff_prod")

__all__ = ["TinkoffProdClient"]

_FULL = "ACCOUNT_ACCESS_LEVEL_FULL_ACCESS"


class TinkoffProdClient(TinkoffRestBase):
    """BrokerClient против БОЕВОГО эндпоинта T-Invest. Реальные сделки."""

    DEFAULT_BASE = config.API_BASE_URL

    # ── Счёт ─────────────────────────────────────────────────────────────────

    def open_or_get_account(self, preferred_id: str = "") -> str:
        """Верифицирует боевой счёт: он должен существовать и иметь FULL access.
        Реальный счёт НЕ создаётся скриптом — preferred_id обязателен."""
        if not preferred_id:
            raise BrokerError("PROD: не задан номер счёта (PROD_ACCOUNT_ID).")
        data = self._post("UsersService/GetAccounts")
        accounts = {a.get("id"): a for a in (data.get("accounts") or [])}
        acc = accounts.get(preferred_id)
        if acc is None:
            raise BrokerError(
                f"PROD: счёт {preferred_id} не найден среди счетов токена "
                f"({list(accounts)}).")
        access = acc.get("accessLevel")
        if access != _FULL:
            raise BrokerError(
                f"PROD: счёт {preferred_id} имеет доступ {access}, нужен {_FULL}. "
                f"Торговля невозможна (проверьте права токена).")
        log.info("PROD account %s верифицирован (FULL access).", preferred_id)
        return preferred_id

    def pay_in(self, account_id: str, amount_rub: float, currency: str = "rub") -> None:
        """На боевом счёте пополнения через API нет — это no-op с предупреждением."""
        raise BrokerError("pay_in недоступен на боевом счёте (только sandbox).")

    # ── Торговля ─────────────────────────────────────────────────────────────

    def post_limit_order(self, *, account_id: str, instrument: Instrument,
                         direction: str, quantity_lots: int,
                         price: Quotation, order_id: str) -> OrderState:
        """OrdersService.PostOrder — РЕАЛЬНАЯ лимитная заявка."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        data = self._post("OrdersService/PostOrder", {
            "accountId":    account_id,
            "instrumentId": instrument.instrument_uid,
            "quantity":     str(int(quantity_lots)),
            "price":        price.to_payload(),
            "direction":    f"ORDER_DIRECTION_{direction}",
            "orderType":    "ORDER_TYPE_LIMIT",
            "orderId":      order_id,
        })
        return parse_order_state(data)

    def get_order_state(self, *, account_id: str, order_id: str) -> OrderState:
        data = self._post("OrdersService/GetOrderState",
                          {"accountId": account_id, "orderId": order_id})
        return parse_order_state(data)

    def cancel_order(self, *, account_id: str, order_id: str) -> None:
        self._post("OrdersService/CancelOrder",
                   {"accountId": account_id, "orderId": order_id})

    # ── Портфель ─────────────────────────────────────────────────────────────

    def get_positions(self, account_id: str) -> list[Position]:
        data = self._post("OperationsService/GetPositions", {"accountId": account_id})
        out: list[Position] = []
        for s in (data.get("securities") or []):
            uid = s.get("instrumentUid") or ""
            if uid:
                out.append(Position(instrument_uid=uid,
                                    balance_shares=float(s.get("balance", 0) or 0),
                                    blocked_shares=float(s.get("blocked", 0) or 0)))
        return out

    def get_active_order_instrument_uids(self, account_id: str) -> set[str]:
        data = self._post("OrdersService/GetOrders", {"accountId": account_id})
        return {o.get("instrumentUid") for o in (data.get("orders") or [])
                if o.get("instrumentUid")}
