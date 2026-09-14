"""
Общая база для T-Invest брокерских клиентов (sandbox и prod).

Содержит то, что ИДЕНТИЧНО в обоих контурах:
  • HTTP-транспорт (_post) на stdlib urllib с проверкой TLS (см. tls.py);
  • find_instrument (InstrumentsService.ShareBy — один и тот же эндпоинт);
  • стоп-заявки (StopOrdersService — общий сервис для sandbox и prod);
  • разбор OrderState, генерация order_id.

Контур-специфичные методы (счёт, лимитки, позиции, список заявок) реализуют
подклассы: TinkoffSandboxClient и TinkoffProdClient.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import config
import tls
from services.broker.base import (
    BrokerClient,
    BrokerError,
    Instrument,
    NotSupportedError,
    OrderState,
    Quotation,
    StopOrderInfo,
)

log = logging.getLogger("broker.tinkoff")

# Методы, меняющие заявки. Между ними — минимальный интервал: 10.09 четыре
# PostSandboxOrder ушли в одну секунду, и брокер отбил два по HTTP 429.
_ORDER_METHODS = ("PostSandboxOrder", "PostOrder", "PostStopOrder", "PostSandboxStopOrder",
                  "CancelSandboxOrder", "CancelOrder", "CancelStopOrder",
                  "CancelSandboxStopOrder")
_last_order_call = 0.0


def _throttle(method: str) -> None:
    """Пауза перед заявкой, если предыдущая ушла ближе BROKER_ORDER_MIN_INTERVAL_SEC.
    Чтение (позиции, состояния) не тормозится."""
    global _last_order_call
    if method.rsplit("/", 1)[-1] not in _ORDER_METHODS:
        return
    gap = float(getattr(config, "BROKER_ORDER_MIN_INTERVAL_SEC", 0.25))
    wait = _last_order_call + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_order_call = time.monotonic()


def _retry_delay(err, attempt: int) -> float:
    """Пауза перед повтором после 429: Retry-After, если брокер его прислал,
    иначе 1 → 2 → 4 с. Потолок 10 с — фаза не должна висеть на одной заявке."""
    hdrs = getattr(err, "headers", None) or {}
    for h in ("Retry-After", "x-ratelimit-reset"):
        try:
            v = float(hdrs.get(h))
            if v > 0:
                return min(v, 10.0)
        except (TypeError, ValueError):
            pass
    return float(2 ** attempt)


def _money(d: dict[str, Any], key: str) -> float | None:
    """MoneyValue/Quotation → float. None, если поля нет или оно нулевое.

    Ноль трактуется как «не заполнено», а не как «цена ноль»: T-Invest
    возвращает нулевой MoneyValue для неисполненной заявки, и записать такую
    цену в журнал исполнения означало бы посчитать проскальзывание −100%.
    """
    v = d.get(key)
    if not isinstance(v, dict):
        return None
    f = Quotation.from_payload(v).as_float()
    return f if f else None


def parse_order_state(d: dict[str, Any]) -> OrderState:
    return OrderState(
        order_id=d.get("orderId", ""),
        execution_report_status=d.get("executionReportStatus", ""),
        lots_requested=int(d.get("lotsRequested", 0) or 0),
        lots_executed=int(d.get("lotsExecuted", 0) or 0),
        raw=d,
        executed_price=_money(d, "averagePositionPrice"),
        # Только totalOrderAmount: executedOrderPrice в GetOrderState — сумма
        # заявки, а в ответе PostOrder — цена за штуку. 11.09 на этом
        # разночтении упал учёт ручной продажи ENPG.
        executed_amount=_money(d, "totalOrderAmount"),
        executed_commission=_money(d, "executedCommission"),
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
        # Валидация токена — здесь, при инициализации сетевого клиента:
        # config импортируется без токена (аналитика, работа с БД).
        self.token = token or config.require_invest_token()
        self.base  = base_url or self.DEFAULT_BASE
        # verify_tls=None → политика из config.INVEST_TLS_VERIFY (по умолч. ВКЛ).
        self.ssl_ctx = tls.ssl_context(verify_tls)
        self.timeout = timeout

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _post(self, method: str, payload: dict | None = None) -> dict:
        """POST {base}/{SVC}.{method} с Bearer-токеном. Возвращает JSON.

        HTTP 429 (лимит запросов брокера) повторяется до BROKER_429_RETRIES раз.
        Повтор заявки безопасен: orderId в теле — ключ идемпотентности, брокер
        не создаст вторую заявку с тем же ключом.
        """
        url = f"{self.base}/{self.SVC}.{method}"
        body = json.dumps(payload or {}).encode("utf-8")
        retries = int(getattr(config, "BROKER_429_RETRIES", 3))
        for attempt in range(retries + 1):
            _throttle(method)
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
                if e.code == 429 and attempt < retries:
                    delay = _retry_delay(e, attempt)
                    log.warning("%s: HTTP 429 — повтор %d/%d через %.1f с",
                                method, attempt + 1, retries, delay)
                    time.sleep(delay)
                    continue
                if e.code in (404, 501) or "UNIMPLEMENTED" in err_body.upper():
                    raise NotSupportedError(
                        f"{method} не поддержан текущим контуром "
                        f"({self.base}): HTTP {e.code} {err_body}") from e
                raise BrokerError(f"{method}: HTTP {e.code} {e.reason}\n{err_body}") from e
            except urllib.error.URLError as e:
                raise BrokerError(f"{method}: сетевая ошибка: {e}") from e
        raise BrokerError(f"{method}: исчерпаны повторы после HTTP 429")

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
            short_enabled=bool(inst.get("shortEnabledFlag", False)),
        )

    def find_instrument_by_uid(self, uid: str) -> Instrument:
        """InstrumentsService.GetInstrumentBy по UID — для позиций, где известен
        только instrument_uid (нужны тикер и размер лота, чтобы закрыть по рынку)."""
        try:
            data = self._post("InstrumentsService/GetInstrumentBy",
                              {"idType": "INSTRUMENT_ID_TYPE_UID", "id": uid})
        except BrokerError as e:
            raise BrokerError(f"GetInstrumentBy {uid[:8]}…: {e}") from e
        inst = data.get("instrument") or {}
        if not inst.get("uid"):
            raise BrokerError(f"GetInstrumentBy {uid[:8]}…: инструмент не найден")
        return Instrument(
            ticker=inst.get("ticker", uid[:8]),
            instrument_uid=inst["uid"],
            figi=inst.get("figi", ""),
            lot=int(inst.get("lot", 1)),
            min_price_increment=Quotation.from_payload(inst.get("minPriceIncrement")),
            currency=inst.get("currency", "rub"),
            trading_status=inst.get("tradingStatus", ""),
            api_trade_available=bool(inst.get("apiTradeAvailableFlag", False)),
            short_enabled=bool(inst.get("shortEnabledFlag", False)),
            class_code=inst.get("classCode", ""),
        )

    def find_instrument_listings(self, query: str) -> list[dict]:
        """InstrumentsService.FindInstrument — все листинги по запросу.

        Нужен казначейству: один фонд (TMON) торгуется в нескольких классах, и
        доступ через API у них разный — ShareBy по TQBR этого не покажет.
        """
        data = self._post("InstrumentsService/FindInstrument", {"query": query})
        return list(data.get("instruments") or [])

    def get_last_price(self, instrument_uid: str) -> float | None:
        """MarketDataService.GetLastPrices — цена последней сделки, за штуку."""
        data = self._post("MarketDataService/GetLastPrices",
                          {"instrumentId": [instrument_uid]})
        for p in data.get("lastPrices") or []:
            if p.get("instrumentUid") in (None, "", instrument_uid):
                return _money(p, "price")
        return None

    # ── Стоп-заявки (StopOrdersService — общий сервис) ───────────────────────

    def post_stop_order(self, *, account_id: str, instrument: Instrument,
                        direction: str, quantity_lots: int,
                        stop_price: Quotation, order_id: str,
                        order_type: str = "STOP_LOSS") -> str:
        """StopOrdersService.PostStopOrder — стоп-лосс или тейк-профит."""
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY|SELL, got {direction!r}")
        if order_type not in ("STOP_LOSS", "TAKE_PROFIT"):
            raise ValueError(f"order_type must be STOP_LOSS|TAKE_PROFIT, got {order_type!r}")
        payload = {
            "accountId":      account_id,
            "instrumentId":   instrument.instrument_uid,
            "quantity":       str(int(quantity_lots)),
            "stopPrice":      stop_price.to_payload(),
            "price":          stop_price.to_payload(),
            "direction":      f"STOP_ORDER_DIRECTION_{direction}",
            "expirationType": "STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL",
            "stopOrderType":  f"STOP_ORDER_TYPE_{order_type}",
            "orderId":        order_id,
        }
        data = self._post("StopOrdersService/PostStopOrder", payload)
        stop_id = data.get("stopOrderId") or data.get("orderId")
        if not stop_id:
            raise BrokerError(f"PostStopOrder: пустой stopOrderId, ответ={data}")
        return stop_id

    def cancel_stop_order(self, *, account_id: str, stop_order_id: str) -> None:
        """StopOrdersService.CancelStopOrder — снять стоп/тейк по id."""
        self._post("StopOrdersService/CancelStopOrder",
                   {"accountId": account_id, "stopOrderId": stop_order_id})

    def get_active_stop_orders(self, account_id: str) -> list[StopOrderInfo]:
        """StopOrdersService.GetStopOrders — активные стопы и тейки с типом/id."""
        data = self._post("StopOrdersService/GetStopOrders", {"accountId": account_id})
        out: list[StopOrderInfo] = []
        for o in (data.get("stopOrders") or []):
            uid = o.get("instrumentUid")
            if not uid:
                continue
            # тип отдаётся как orderType или stopOrderType (в зависимости от версии)
            raw_kind = (o.get("orderType") or o.get("stopOrderType") or "")
            kind = raw_kind.replace("STOP_ORDER_TYPE_", "")
            out.append(StopOrderInfo(
                instrument_uid=uid, kind=kind,
                stop_order_id=o.get("stopOrderId") or o.get("orderId") or "",
            ))
        return out
