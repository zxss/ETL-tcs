"""
T-Invest Sandbox реализация BrokerClient.

На stdlib (urllib) — без новых зависимостей. Тот же приём с ssl=False, что
в loaders / list_accounts.py: на машине пользователя MITM-перехват TLS,
иначе urlopen падает с CERTIFICATE_VERIFY_FAILED.

Эндпоинты sandbox (документация T-Invest):
  SandboxService.OpenSandboxAccount      — создать виртуальный счёт
  SandboxService.GetSandboxAccounts      — список существующих
  SandboxService.SandboxPayIn            — пополнить виртуальными деньгами
  SandboxService.PostSandboxOrder        — выставить заявку (лимитную/рыночную)
  SandboxService.GetSandboxOrderState    — статус заявки (polling)
  InstrumentsService.ShareBy             — спецификация акции (lot, шаг цены)
  StopOrdersService.PostStopOrder        — стоп-заявка
       (⚠️ может быть НЕ поддержано на sandbox-эндпоинте — тогда выбрасываем
       NotSupportedError; place_orders НЕ эмулирует стоп программно, по ТЗ §9.4).
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
    Position,
    Quotation,
)

log = logging.getLogger("broker.tinkoff_sandbox")


def _ssl_context(verify: bool) -> ssl.SSLContext:
    """Тот же контекст, что в services/list_accounts.py: по умолчанию verify=OFF
    (MITM-перехват TLS на машине пользователя). INVEST_TLS_VERIFY=1 — вернуть."""
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TinkoffSandboxClient(BrokerClient):
    """Реализация BrokerClient против sandbox-эндпоинта T-Invest API.

    Параметры:
        token         — Bearer-токен (по умолч. config.INVEST_TOKEN);
        base_url      — REST-base (по умолч. config.SANDBOX_API_BASE_URL);
                        для prod передать config.API_BASE_URL ЯВНО, иначе случайных
                        запусков не будет.
        verify_tls    — проверка TLS-сертификата. По умолчанию False (см. выше).
        timeout       — таймаут HTTP-запроса (сек).
    """

    SVC = "tinkoff.public.invest.api.contract.v1"  # неймспейс методов

    def __init__(self, *,
                 token: str | None = None,
                 base_url: str | None = None,
                 verify_tls: bool | None = None,
                 timeout: float = 15.0):
        self.token   = token    or config.INVEST_TOKEN
        self.base    = base_url or config.SANDBOX_API_BASE_URL
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
            # T-Invest на UNIMPLEMENTED отдаёт HTTP 404 или 501 + код в JSON
            if e.code in (404, 501) or "UNIMPLEMENTED" in err_body.upper():
                raise NotSupportedError(
                    f"{method} не поддержан текущим контуром "
                    f"({self.base}): HTTP {e.code} {err_body}"
                ) from e
            raise BrokerError(f"{method}: HTTP {e.code} {e.reason}\n{err_body}") from e
        except urllib.error.URLError as e:
            raise BrokerError(f"{method}: сетевая ошибка: {e}") from e

    # ── Счёт ─────────────────────────────────────────────────────────────────

    def open_or_get_account(self, preferred_id: str = "") -> str:
        """Стратегия по ТЗ §2:
          1) если preferred_id задан и найден в GetSandboxAccounts → используем;
          2) иначе OpenSandboxAccount → новый id (нужно потом SandboxPayIn).
        """
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
            log.warning(
                "Sandbox account %s не найден (счета песочницы живут 3 мес.) — "
                "создаю новый.", preferred_id,
            )

        data = self._post("SandboxService/OpenSandboxAccount")
        new_id = data.get("accountId")
        if not new_id:
            raise BrokerError(f"OpenSandboxAccount: пустой accountId, ответ={data}")
        log.info("Sandbox account создан: %s", new_id)
        return new_id

    def pay_in(self, account_id: str, amount_rub: float, currency: str = "rub") -> None:
        """SandboxService.SandboxPayIn — виртуальное пополнение."""
        amount = Quotation.from_float(amount_rub)
        payload = {
            "accountId": account_id,
            "amount": {
                "currency": currency,
                "units":    str(amount.units),
                "nano":     int(amount.nano),
            },
        }
        self._post("SandboxService/SandboxPayIn", payload)
        log.info("Sandbox account %s пополнен на %.2f %s.",
                 account_id, amount_rub, currency.upper())

    # ── Справочник инструментов ──────────────────────────────────────────────

    def find_instrument(self, ticker: str) -> Instrument:
        """InstrumentsService.ShareBy по классу TQBR (основной режим MOEX).

        TQBR подходит для всех акций из config.TICKERS. Для других режимов
        (TQTF — ETF, например) метод нужно будет расширить.
        """
        payload = {
            "idType":    "INSTRUMENT_ID_TYPE_TICKER",
            "classCode": "TQBR",
            "id":        ticker.upper(),
        }
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

    # ── Торговля ─────────────────────────────────────────────────────────────

    def post_limit_order(self, *, account_id: str, instrument: Instrument,
                         direction: str, quantity_lots: int,
                         price: Quotation, order_id: str) -> OrderState:
        """SandboxService.PostSandboxOrder — лимитная заявка."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        payload = {
            "accountId":     account_id,
            "instrumentId":  instrument.instrument_uid,
            "quantity":      str(int(quantity_lots)),
            "price":         price.to_payload(),
            "direction":     f"ORDER_DIRECTION_{direction}",
            "orderType":     "ORDER_TYPE_LIMIT",
            "orderId":       order_id,
        }
        data = self._post("SandboxService/PostSandboxOrder", payload)
        return _parse_order_state(data)

    def post_stop_order(self, *, account_id: str, instrument: Instrument,
                        direction: str, quantity_lots: int,
                        stop_price: Quotation, order_id: str) -> str:
        """StopOrdersService.PostStopOrder — условная стоп-заявка STOP_LOSS.

        Если sandbox-эндпоинт не поддерживает PostStopOrder, _post бросит
        NotSupportedError — пробрасываем наверх (place_orders обработает).
        """
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        payload = {
            "accountId":      account_id,
            "instrumentId":   instrument.instrument_uid,
            "quantity":       str(int(quantity_lots)),
            "stopPrice":      stop_price.to_payload(),
            # для STOP_LOSS поле price можно не передавать — конвертация в рынок;
            # но API требует поле — отдаём ту же stop_price для совместимости
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

    def get_order_state(self, *, account_id: str, order_id: str) -> OrderState:
        """SandboxService.GetSandboxOrderState — статус активной/закрытой заявки."""
        payload = {"accountId": account_id, "orderId": order_id}
        data = self._post("SandboxService/GetSandboxOrderState", payload)
        return _parse_order_state(data)

    def cancel_order(self, *, account_id: str, order_id: str) -> None:
        """SandboxService.CancelSandboxOrder — снять активную заявку."""
        self._post("SandboxService/CancelSandboxOrder",
                   {"accountId": account_id, "orderId": order_id})

    # ── Состояние портфеля ────────────────────────────────────────────────────

    def get_positions(self, account_id: str) -> list[Position]:
        """SandboxService.GetSandboxPositions — открытые позиции."""
        data = self._post("SandboxService/GetSandboxPositions",
                          {"accountId": account_id})
        out: list[Position] = []
        for s in (data.get("securities") or []):
            uid = s.get("instrumentUid") or s.get("instrument_uid") or ""
            if not uid:
                continue
            out.append(Position(
                instrument_uid=uid,
                balance_shares=float(s.get("balance", 0) or 0),
                blocked_shares=float(s.get("blocked", 0) or 0),
            ))
        return out

    def get_active_stop_instrument_uids(self, account_id: str) -> set[str]:
        """StopOrdersService.GetStopOrders — активные стоп-заявки.

        В sandbox может быть не поддержано — тогда _post бросит NotSupportedError,
        пробрасываем (вызывающий решит, можно ли продолжать)."""
        data = self._post("StopOrdersService/GetStopOrders",
                          {"accountId": account_id})
        uids: set[str] = set()
        for o in (data.get("stopOrders") or []):
            uid = o.get("instrumentUid") or o.get("instrument_uid")
            if uid:
                uids.add(uid)
        return uids

    def get_active_order_instrument_uids(self, account_id: str) -> set[str]:
        """SandboxService.GetSandboxOrders — активные (неисполненные) заявки."""
        data = self._post("SandboxService/GetSandboxOrders",
                          {"accountId": account_id})
        uids: set[str] = set()
        for o in (data.get("orders") or []):
            uid = o.get("instrumentUid") or o.get("instrument_uid")
            if uid:
                uids.add(uid)
        return uids


def _parse_order_state(d: dict[str, Any]) -> OrderState:
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
