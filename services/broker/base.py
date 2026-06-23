"""
Абстракция брокера для services/place_orders.py.

Зачем:
  Бизнес-логика автозаявок (парсинг top-N, расчёт цен/стопов, подтверждение
  пользователем, логирование) НЕ должна знать про T-Invest конкретно. Если
  завтра переключимся на prod или сменим брокера, правится ТОЛЬКО реализация
  BrokerClient — не код в place_orders.

Что отсюда импортирует place_orders:
  - класс BrokerClient (интерфейс)
  - dataclass'ы Quotation, Instrument, OrderState
  - исключения BrokerError, NotSupportedError
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


# ── Ошибки ────────────────────────────────────────────────────────────────────


class BrokerError(RuntimeError):
    """Любая ошибка брокерского API (HTTP, бизнес-код, валидация заявки)."""


class NotSupportedError(BrokerError):
    """Метод не поддержан текущим контуром (например, PostStopOrder в sandbox).

    place_orders ловит ИМЕННО это исключение, чтобы выдать понятный [ERROR] и
    не оставить висячий вход без стопа (по ТЗ — без программной эмуляции)."""


# ── Quotation (T-Invest формат цены) ─────────────────────────────────────────


@dataclass(frozen=True)
class Quotation:
    """Представление цены в формате T-Invest API: units + nano.

    1 nano = 10⁻⁹. Полная цена = units + nano / 1e9.
    Точное представление без float-шумов: цена 142.03 = units=142, nano=30_000_000.

    Используется в Request-телах PostOrder/PostStopOrder. Создаётся из float
    через `from_float()` с округлением к шагу инструмента (min_price_increment).
    """
    units: int
    nano:  int

    @classmethod
    def from_float(cls, value: float, step: Optional["Quotation"] = None) -> "Quotation":
        """Сконвертировать float-цену в Quotation, опционально округлив к шагу.

        step — min_price_increment инструмента (тоже Quotation). Округление —
        к ближайшему кратному шагу (банковским round half up), как требует MOEX.
        Если step не передан — округляем к 9 знакам после запятой (это native
        точность Quotation, шумы float-арифметики срезаются).
        """
        d = Decimal(str(value))
        if step is not None:
            step_d = step.as_decimal()
            if step_d > 0:
                # ближайший шаг
                n_steps = (d / step_d).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                d = n_steps * step_d
        # разбираем на units + nano
        units = int(d)  # truncate, но для положительных = floor
        if d < 0 and d != units:
            units -= 1
        nano_d = (d - Decimal(units)) * Decimal(1_000_000_000)
        nano = int(nano_d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return cls(units=units, nano=nano)

    def as_decimal(self) -> Decimal:
        return Decimal(self.units) + Decimal(self.nano) / Decimal(1_000_000_000)

    def as_float(self) -> float:
        return float(self.as_decimal())

    def to_payload(self) -> dict:
        """Словарь для JSON-тела запроса T-Invest API."""
        return {"units": str(self.units), "nano": int(self.nano)}

    @classmethod
    def from_payload(cls, p: dict | None) -> "Quotation":
        if not p:
            return cls(0, 0)
        # API отдаёт units как строку, nano как int — нормализуем
        return cls(units=int(p.get("units", 0)), nano=int(p.get("nano", 0)))


# ── Инструмент ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Instrument:
    """Спецификация торгуемого инструмента, нужная для постановки заявки."""
    ticker:               str
    instrument_uid:       str   # UID T-Invest (для PostOrder.instrument_id)
    figi:                 str
    lot:                  int   # размер лота (акций/контрактов в 1 лоте)
    min_price_increment:  Quotation
    currency:             str   # "rub", "usd"...
    trading_status:       str   # "SECURITY_TRADING_STATUS_NORMAL_TRADING" — норма
    api_trade_available:  bool  # True → можно торговать через API


# ── Состояние заявки ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Position:
    """Открытая позиция по инструменту (из GetPositions)."""
    instrument_uid: str
    balance_shares: float   # +long / -short, в ШТУКАХ (не лотах)
    blocked_shares: float

    @property
    def is_open(self) -> bool:
        return abs(self.balance_shares) > 1e-9


@dataclass(frozen=True)
class StopOrderInfo:
    """Активная условная заявка (стоп-лосс или тейк-профит) из GetStopOrders."""
    instrument_uid: str
    kind:           str   # "STOP_LOSS" | "TAKE_PROFIT" | "" (нераспознан)
    stop_order_id:  str


@dataclass(frozen=True)
class ActiveOrder:
    """Активная (неисполненная) лимитная заявка из GetOrders.
    Нужна для синхронизации портфеля: снять заявку по тикеру, сигнал которого
    исчез/сменил направление."""
    instrument_uid: str
    order_id:       str
    direction:      str   # "BUY" | "SELL"
    lots:           int


@dataclass(frozen=True)
class OrderState:
    """Состояние заявки от GetOrderState (упрощённое до нужных полей)."""
    order_id:               str
    execution_report_status: str  # _FILL / _PARTIALLYFILL / _NEW / _REJECTED / _CANCELLED
    lots_requested:         int
    lots_executed:          int
    raw:                    dict  # полный ответ для логов

    @property
    def is_filled(self) -> bool:
        return self.execution_report_status == "EXECUTION_REPORT_STATUS_FILL"

    @property
    def is_terminal(self) -> bool:
        return self.execution_report_status in {
            "EXECUTION_REPORT_STATUS_FILL",
            "EXECUTION_REPORT_STATUS_REJECTED",
            "EXECUTION_REPORT_STATUS_CANCELLED",
        }


# ── Интерфейс брокера ─────────────────────────────────────────────────────────


class BrokerClient(ABC):
    """Минимальный интерфейс, нужный services/place_orders.py.

    Конкретные реализации:
      - TinkoffSandboxClient — sandbox-контур T-Invest (по умолчанию).
    """

    # ----- Счёт -----

    @abstractmethod
    def open_or_get_account(self, preferred_id: str = "") -> str:
        """Вернуть id рабочего счёта.

        Если preferred_id задан и существует — вернуть его.
        Иначе создать новый (sandbox) и вернуть его id.
        """

    @abstractmethod
    def pay_in(self, account_id: str, amount_rub: float, currency: str = "rub") -> None:
        """Пополнить счёт виртуальными деньгами (только sandbox)."""

    # ----- Справочник -----

    @abstractmethod
    def find_instrument(self, ticker: str) -> Instrument:
        """Найти инструмент по тикеру (MOEX-класс TQBR).

        Бросает BrokerError, если не найден или недоступен через API.
        """

    @abstractmethod
    def find_instrument_by_uid(self, uid: str) -> Instrument:
        """Найти инструмент по instrument_uid (для позиций, где известен только
        uid: нужно узнать тикер и размер лота для закрытия по рынку)."""

    # ----- Торговля -----

    @abstractmethod
    def post_limit_order(self, *, account_id: str, instrument: Instrument,
                         direction: str, quantity_lots: int,
                         price: Quotation, order_id: str) -> OrderState:
        """Выставить лимитную заявку. direction = "BUY" | "SELL"."""

    @abstractmethod
    def post_market_order(self, *, account_id: str, instrument: Instrument,
                          direction: str, quantity_lots: int,
                          order_id: str) -> OrderState:
        """Выставить РЫНОЧНУЮ заявку (для немедленного закрытия позиции).
        direction = "BUY" (закрыть шорт) | "SELL" (закрыть лонг)."""

    @abstractmethod
    def post_stop_order(self, *, account_id: str, instrument: Instrument,
                        direction: str, quantity_lots: int,
                        stop_price: Quotation, order_id: str,
                        order_type: str = "STOP_LOSS") -> str:
        """Выставить условную заявку. order_type = "STOP_LOSS" | "TAKE_PROFIT".

        STOP_LOSS  срабатывает при движении против позиции (защита от убытка);
        TAKE_PROFIT — при движении в прибыль (фиксация по цели диапазона).
        Направление одинаковое (выход из позиции): SHORT→BUY, LONG→SELL.

        Возвращает stop_order_id. Если контур не поддерживает PostStopOrder —
        бросает NotSupportedError.
        """

    @abstractmethod
    def cancel_stop_order(self, *, account_id: str, stop_order_id: str) -> None:
        """Снять условную (стоп/тейк) заявку по id."""

    @abstractmethod
    def get_order_state(self, *, account_id: str, order_id: str) -> OrderState:
        """Текущее состояние заявки (для polling в режиме wait_fill)."""

    @abstractmethod
    def cancel_order(self, *, account_id: str, order_id: str) -> None:
        """Снять активную заявку по id."""

    # ----- Состояние портфеля (для двухфазной привязки стопов) -----

    @abstractmethod
    def get_positions(self, account_id: str) -> list[Position]:
        """Открытые позиции по инструментам (нужно, чтобы понять, какие
        лимитки уже залились и требуют стопа)."""

    @abstractmethod
    def get_active_stop_orders(self, account_id: str) -> list[StopOrderInfo]:
        """Активные условные заявки (стоп-лосс и тейк-профит) с типом и id.
        Может бросить NotSupportedError, если контур не отдаёт список стопов."""

    def get_active_stop_instrument_uids(self, account_id: str) -> set[str]:
        """instrument_uid инструментов с любым активным стопом/тейком
        (для защиты от задвоения в фазе 1). Производное от get_active_stop_orders."""
        return {s.instrument_uid for s in self.get_active_stop_orders(account_id)}

    @abstractmethod
    def get_active_orders(self, account_id: str) -> list["ActiveOrder"]:
        """Активные (неисполненные) лимитные заявки с id и направлением —
        для синхронизации портфеля (снять заявки по исчезнувшим сигналам)."""

    def get_active_order_instrument_uids(self, account_id: str) -> set[str]:
        """instrument_uid инструментов с активной (неисполненной) заявкой —
        для защиты от задвоения лимиток. Производное от get_active_orders."""
        return {o.instrument_uid for o in self.get_active_orders(account_id)}
