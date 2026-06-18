"""
Общая база для T-Invest брокерских клиентов (sandbox и prod).

Содержит то, что ИДЕНТИЧНО в обоих контурах:
  • HTTP-транспорт (_post) на stdlib urllib + ssl=False (MITM на машине);
  • find_instrument (InstrumentsService.ShareBy — один и тот же эндпоинт);
  • стоп-заявки (StopOrdersService — общий сервис для sandbox и prod);
  • разбор OrderState, генерация order_id.

Контур-специфичные методы (счёт, лимитки, позиции, список заявок) реализуют
подклассы: TinkoffSandboxClient и TinkoffProdClient.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
import uuid
from typing import Any

import config
from services.broker.base import (
    BrokerClient,
    BrokerError,
    Instrument,
    NotSupportedError,
    OrderState,
    Quotation,
)

log = logging.getLogger("broker.tinkoff")


def _ssl_context(verify: bool) -> ssl.SSLContext:
    """Тот же контекст, что в services/list_accounts.py: по умолчанию verify=OFF
    (MITM-перехват TLS на машине пользователя). INVEST_TLS_VERIFY=1 — вернуть."""
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def parse_order_state(d: dict[str, Any]) -> OrderState:
    return OrderState(
        order_id=d.get("orderId", ""),
        execution_report_status=d.get("executionReportStatus", ""),
        lots_requested=int(d.get("lotsRequested", 0) or 0),
        lots_executed=int(d.get("lotsExecuted", 0) or 0),
        raw=d,
    )


def new_order_id() -> str:
    """UUID v4 — ключ идемпотентности заявки."""
    return str(uuid.uuid4())


class TinkoffRestBase(BrokerClient):
    """Базовый REST-клиент T-Invest. Подклассы задают base_url и контур-методы."""

    SVC = "tinkoff.public.invest.api.contract.v1"
    DEFAULT_BASE = config.API_BASE_URL  # переопределяется подклассом

    def __init__(self, *,
                 token: str | None = None,
                 base_url: str | None = None,
                 verify_tls: bool | None = None,
                 timeout: float = 15.0):
        self.token = token or config.INVEST_TOKEN
        self.base  = base_url or self.DEFAULT_BASE
        if verify_tls is None:
            verify_tls = os.getenv("INVEST_TLS_VERIFY", "0") not in ("0", "false", "False")
        self.ssl_ctx = _ssl_context(verify_tls)
        self.timeout = timeout

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _post(self, method: str, payload: dict | None = None) -> dict:
        """POST {base}/{SVC}.{method} с Bearer-токеном. Возвращает JSON."""
        url = f"{self.base}/{self.SVC}.{method}"
        body = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self.ssl_ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (404, 501) or "UNIMPLEMENTED" in err_body.upper():
                raise NotSupportedError(
                    f"{method} не поддержан текущим контуром "
                    f"({self.base}): HTTP {e.code} {err_body}") from e
            raise BrokerError(f"{method}: HTTP {e.code} {e.reason}\n{err_body}") from e
        except urllib.error.URLError as e:
            raise BrokerError(f"{method}: сетевая ошибка: {e}") from e

    # ── Справочник инструментов (общий для sandbox и prod) ───────────────────

    def find_instrument(self, ticker: str) -> Instrument:
        """InstrumentsService.ShareBy по классу TQBR (основной режим MOEX)."""
        payload = {"idType": "INSTRUMENT_ID_TYPE_TICKER",
                   "classCode": "TQBR", "id": ticker.upper()}
        try:
            data = self._post("InstrumentsService/ShareBy", payload)
        except BrokerError as e:
            raise BrokerError(f"ShareBy {ticker}: {e}") from e
        inst = data.get("instrument") or {}
        if not inst.get("uid"):
            raise BrokerError(f"ShareBy {ticker}: инструмент не найден (ответ={data})")
        if not inst.get("apiTradeAvailableFlag", False):
            raise BrokerError(f"{ticker}: торговля через API недоступна "
                              f"(apiTradeAvailableFlag=false)")
        return Instrument(
            ticker=ticker.upper(),
            instrument_uid=inst["uid"],
            figi=inst.get("figi", ""),
            lot=int(inst.get("lot", 1)),
            min_price_increment=Quotation.from_payload(inst.get("minPriceIncrement")),
            currency=inst.get("currency", "rub"),
            trading_status=inst.get("tradingStatus", ""),
            api_trade_available=bool(inst.get("apiTradeAvailableFlag", False)),
        )

    # ── Стоп-заявки (StopOrdersService — общий сервис) ───────────────────────

    def post_stop_order(self, *, account_id: str, instrument: Instrument,
                        direction: str, quantity_lots: int,
                        stop_price: Quotation, order_id: str) -> str:
        """StopOrdersService.PostStopOrder — условная стоп-заявка STOP_LOSS."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        payload = {
            "accountId":      account_id,
            "instrumentId":   instrument.instrument_uid,
            "quantity":       str(int(quantity_lots)),
            "stopPrice":      stop_price.to_payload(),
            "price":          stop_price.to_payload(),
            "direction":      f"STOP_ORDER_DIRECTION_{direction}",
            "expirationType": "STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL",
            "stopOrderType":  "STOP_ORDER_TYPE_STOP_LOSS",
            "orderId":        order_id,
        }
        data = self._post("StopOrdersService/PostStopOrder", payload)
        stop_id = data.get("stopOrderId") or data.get("orderId")
        if not stop_id:
            raise BrokerError(f"PostStopOrder: пустой stopOrderId, ответ={data}")
        return stop_id

    def get_active_stop_instrument_uids(self, account_id: str) -> set[str]:
        """StopOrdersService.GetStopOrders — активные стоп-заявки."""
        data = self._post("StopOrdersService/GetStopOrders", {"accountId": account_id})
        return {o.get("instrumentUid") for o in (data.get("stopOrders") or [])
                if o.get("instrumentUid")}
