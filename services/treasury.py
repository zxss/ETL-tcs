"""
Казначейство: парковка свободной ликвидности в фонд денежного рынка (ТЗ Treasury).

Свободный рублёвый кэш в дни простоя лежит на счёте с нулевой доходностью.
Сервис держит его в паях фонда денежного рынка (по умолчанию TMON) и
высвобождает ровно столько, сколько нужно под покупки:

    CLOSE     10:00  park_idle_cash()                  весь кэш сверх буфера → фонд
    CLEANUP   18:20  restore_buffer()                  рубли ушли в минус → продать паи до буфера
    OVERNIGHT 18:35  release_cash_for_overnight(сумма) продать паи под ночные лонги

Интрадей-шорты фонд не трогают: они открываются под залог паёв и закрываются
до клиринга, плата за перенос не возникает (ТЗ §1).

Какой листинг торговать. У TMON на Мосбирже (классы TQBR и TQTF)
apiTradeAvailableFlag = false — выставить заявку через API нельзя (проверено
14.09). Через API доступен листинг TMON@ на СПБ бирже — тот же фонд. Поэтому
инструмент выбирается среди листингов с тикером TMON и TMON@: первый доступный
через API, либо класс из TREASURY_CLASS_CODE, если он задан явно.

Песочница (ТЗ, задача 4). Если брокер отказал в покупке паёв (листинг
недоступен через API, нет котировок или ликвидности), владение ведётся
виртуально в таблице treasury_ledger: реальные рубли остаются на счёте, но
считаются припаркованными. Доход начисляется через рыночную цену пая: фонд
денежного рынка дорожает на RUONIA за вычетом комиссии фонда, поэтому
отдельная ставка не нужна и доход не считается дважды. На боевом контуре
виртуального режима нет — отказ брокера там ошибка.
"""
from __future__ import annotations

import logging
import math
import time

import config
from services.broker.base import BrokerError, Instrument
from services.broker.tinkoff_base import new_order_id

log = logging.getLogger("treasury")

SELL_RESERVE = 0.002        # запас на проскальзывание и комиссию при продаже паёв (ТЗ: ×1.002)
FILL_TIMEOUT_SEC = 15.0     # ожидание исполнения заявки по фонду (ТЗ: до 15 секунд)

_NORMAL = "SECURITY_TRADING_STATUS_NORMAL_TRADING"
_FILL = "EXECUTION_REPORT_STATUS_FILL"
_TERMINAL = (_FILL, "EXECUTION_REPORT_STATUS_REJECTED", "EXECUTION_REPORT_STATUS_CANCELLED")


# ── Чистая арифметика (покрыта tests/test_treasury.py) ───────────────────────

def sweep_lots(free_cash_rub: float, lot_price_rub: float, *,
               buffer_rub: float, min_sweep_rub: float) -> int:
    """Сколько лотов фонда купить на кэш сверх буфера. 0 — сумма ниже порога:
    не гонять заявки ради сотни рублей."""
    if not lot_price_rub or lot_price_rub <= 0:
        return 0
    allocatable = free_cash_rub - buffer_rub
    if allocatable < min_sweep_rub:
        return 0
    return int(allocatable // lot_price_rub)


def lots_to_sell(deficit_rub: float, lot_price_rub: float) -> int:
    """Лотов к продаже под дефицит кэша — с запасом SELL_RESERVE, вверх."""
    if deficit_rub <= 0 or not lot_price_rub or lot_price_rub <= 0:
        return 0
    return math.ceil(deficit_rub * (1.0 + SELL_RESERVE) / lot_price_rub)


def order_cost_rub(o) -> float:
    """Стоимость заявки: цена входа × лоты × размер лота. Принимает и Order,
    и запись плана (dict)."""
    get = o.get if isinstance(o, dict) else (lambda k: getattr(o, k, None))
    return float(get("entry_price") or 0.0) * int(get("quantity_lots") or 0) \
        * int(get("lot_size") or 1)


def fit_budget(orders: list, available_rub: float) -> tuple[list, list]:
    """Урезает ночную корзину под фактически доступный кэш.

    Заявки идут в порядке ранга. Та, что не влезает в остаток, пропускается;
    следующие (возможно, дешевле) ещё пробуются. В плечо не идём никогда.
    """
    kept, dropped = [], []
    left = float(available_rub)
    for o in orders:
        cost = order_cost_rub(o)
        if cost <= left + 1e-9:
            kept.append(o)
            left -= cost
        else:
            dropped.append(o)
    return kept, dropped


def choose_listing(listings: list[dict], ticker: str, class_code: str = "") -> dict | None:
    """Листинг фонда из выдачи FindInstrument.

    Тикеры TMON и TMON@ — один фонд на разных биржах. Явный class_code
    выигрывает; иначе первый доступный через API, иначе первый найденный
    (тогда в песочнице включится виртуальный режим).
    """
    base = ticker.upper().rstrip("@")
    cands = [i for i in listings if str(i.get("ticker", "")).upper().rstrip("@") == base]
    if class_code:
        cands = [i for i in cands if i.get("classCode") == class_code]
        return cands[0] if cands else None
    tradable = [i for i in cands if i.get("apiTradeAvailableFlag")]
    return (tradable or cands or [None])[0]


def card_line(state: dict) -> str:
    """Строка для карточки аудита в Telegram (формат из ТЗ, §5 п.5)."""
    tk = str(getattr(config, "TREASURY_TICKER", "TMON")).upper()
    line = (f"Казначейство: {tk} {state['tmon_value_rub']:.0f} ₽ | "
            f"Свободный кэш {state['free_cash_rub']:.0f} ₽")
    if state.get("virtual_lots"):
        line += " (виртуально)"
    return line


# ── Виртуальный реестр (только песочница) ────────────────────────────────────

class VirtualLedger:
    """Владение паями в таблице treasury_ledger.

    Пишутся и реальные сделки по фонду (mode='broker') — ради следа в аудите,
    но позицию складывают только виртуальные (mode='virtual'): реальные паи
    брокер и так показывает в портфеле.
    """

    def __init__(self, conn):
        self.conn = conn

    def init(self) -> None:
        from models.market_data import CREATE_TREASURY_LEDGER_SQL
        with self.conn.cursor() as cur:
            cur.execute(CREATE_TREASURY_LEDGER_SQL)
        self.conn.commit()

    def position(self, account_id: str, ticker: str) -> tuple[int, float]:
        """(лотов, чистая стоимость покупок в ₽) виртуального владения."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(CASE WHEN side = 'BUY' THEN lots ELSE -lots END), 0),"
                "       COALESCE(SUM(CASE WHEN side = 'BUY' THEN amount_rub ELSE -amount_rub END), 0)"
                "  FROM treasury_ledger"
                " WHERE account_id = %s AND ticker = %s AND mode = 'virtual';",
                (account_id, ticker))
            lots, cost = cur.fetchone()
        return int(lots or 0), float(cost or 0.0)

    def record(self, *, account_env: str, account_id: str, ticker: str, mode: str,
               side: str, lots: int, price: float, amount_rub: float,
               reason: str, run_id: str | None = None, order_id: str | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO treasury_ledger (account_env, account_id, ticker, mode, side,"
                " lots, price, amount_rub, reason, run_id, order_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                (account_env, account_id, ticker, mode, side, int(lots), float(price),
                 round(float(amount_rub), 4), reason, run_id, order_id))
        self.conn.commit()


# ── Сервис ───────────────────────────────────────────────────────────────────

_UIDS_CACHE: set[str] | None = None


def treasury_uids(broker) -> set[str]:
    """instrument_uid всех листингов фонда — чтобы паи не считались торговой
    позицией в сводке баланса. Кэш на процесс; сбой поиска — пустое множество."""
    global _UIDS_CACHE
    if _UIDS_CACHE is None:
        try:
            tk = str(getattr(config, "TREASURY_TICKER", "TMON")).upper().rstrip("@")
            _UIDS_CACHE = {i["uid"] for i in broker.find_instrument_listings(tk)
                           if str(i.get("ticker", "")).upper().rstrip("@") == tk and i.get("uid")}
        except Exception:                        # noqa: BLE001 — сводка не должна падать
            return set()
    return set(_UIDS_CACHE)


class TreasuryService:
    """Парковка кэша в фонд и высвобождение под ночные покупки."""

    def __init__(self, broker, account_id: str, *, env: str,
                 ledger: VirtualLedger | None = None, writer=None, run_id: str | None = None):
        self.broker, self.account_id, self.env = broker, account_id, env
        self.ledger, self.writer, self.run_id = ledger, writer, run_id
        self._inst: Instrument | None = None
        self.last_release: dict = {}

    # ── параметры ──

    @property
    def ticker(self) -> str:
        return str(getattr(config, "TREASURY_TICKER", "TMON")).upper()

    @property
    def buffer(self) -> float:
        return float(getattr(config, "TREASURY_CASH_BUFFER_RUB", 1000.0))

    @property
    def min_sweep(self) -> float:
        return float(getattr(config, "TREASURY_MIN_SWEEP_RUB", 2000.0))

    @property
    def sandbox(self) -> bool:
        return self.env == "SANDBOX"

    # ── инструмент и цена ──

    def instrument(self) -> Instrument:
        if self._inst is None:
            listings = self.broker.find_instrument_listings(self.ticker.rstrip("@"))
            pick = choose_listing(listings, self.ticker,
                                  str(getattr(config, "TREASURY_CLASS_CODE", "") or ""))
            if not pick:
                raise BrokerError(f"{self.ticker}: листинг фонда не найден")
            self._inst = self.broker.find_instrument_by_uid(pick["uid"])
            log.info("[TREASURY] инструмент %s/%s, лот %d, через API: %s",
                     self._inst.ticker, getattr(self._inst, "class_code", "") or pick.get("classCode"),
                     self._inst.lot, "да" if self.tradable() else "нет")
        return self._inst

    def tradable(self) -> bool:
        i = self.instrument()
        return bool(i.api_trade_available) and i.trading_status == _NORMAL

    def lot_price(self) -> float:
        """Цена одного лота фонда, ₽ (последняя сделка)."""
        inst = self.instrument()
        p = self.broker.get_last_price(inst.instrument_uid)
        if not p:
            raise BrokerError(f"{inst.ticker}: нет последней цены")
        return float(p) * int(inst.lot or 1)

    # ── состояние ──

    def _real_lots(self) -> int:
        inst = self.instrument()
        shares = sum(p.balance_shares for p in self.broker.get_positions(self.account_id)
                     if p.instrument_uid == inst.instrument_uid)
        return int(shares // int(inst.lot or 1))

    def _virtual(self) -> tuple[int, float]:
        if self.ledger is None or not self.sandbox:
            return 0, 0.0
        return self.ledger.position(self.account_id, self.ticker)

    def get_treasury_state(self) -> dict:
        """free_cash_rub, tmon_lots, tmon_price, tmon_value_rub (ТЗ, задача 1 п.2).

        free_cash_rub — свободные рубли за вычетом виртуально припаркованных:
        в песочнице с виртуальным реестром реальный кэш на счёте есть, но
        считается вложенным в фонд. Поле money брокера уже без рублей,
        заблокированных под заявки.
        """
        cash = float(self.broker.get_money_rub(self.account_id))
        price = self.lot_price()
        real = self._real_lots()
        vlots, vcost = self._virtual()
        lots = real + vlots
        return {
            "ticker": self.instrument().ticker,
            "free_cash_rub": round(cash - vcost, 2),
            "cash_rub": round(cash, 2),
            "tmon_lots": lots,
            "tmon_price": round(price, 6),
            "tmon_value_rub": round(lots * price, 2),
            "real_lots": real,
            "virtual_lots": vlots,
            "virtual_cost_rub": round(vcost, 2),
            "mode": "virtual" if vlots else "broker",
        }

    # ── SWEEP ──

    def park_idle_cash(self) -> dict:
        """Весь свободный кэш сверх буфера → паи фонда (ТЗ, задача 1 п.3)."""
        st = self.get_treasury_state()
        lots = sweep_lots(st["free_cash_rub"], st["tmon_price"],
                          buffer_rub=self.buffer, min_sweep_rub=self.min_sweep)
        out = {"lots": 0, "amount_rub": 0.0, "mode": None}
        if lots <= 0:
            log.info("[TREASURY] свободно %.2f ₽ — сверх буфера %.0f ₽ меньше порога %.0f ₽, "
                     "парковка не нужна", st["free_cash_rub"], self.buffer, self.min_sweep)
            return out
        done = self._trade("BUY", lots, st["tmon_price"], reason="sweep", allow_virtual=True)
        out.update(lots=done["lots"], amount_rub=round(done["lots"] * done["price"], 2),
                   mode=done["mode"])
        log.info("[TREASURY] Припарковано %.2f ₽ в %s (%d шт)%s",
                 out["amount_rub"], self.ticker, done["lots"],
                 " — виртуально" if done["mode"] == "virtual" else "")
        return out

    # ── UNPARK ──

    def release_cash_for_overnight(self, required_rub: float, *, reason: str = "overnight") -> bool:
        """Высвободить кэш под ночные покупки (ТЗ, задача 1 п.4).

        True — живого кэша сверх буфера хватает на required_rub. False — паёв не
        хватило (проданы все) или продажа не исполнилась: вызывающий обязан
        урезать корзину под фактически доступный кэш, в маржу не влезать.
        """
        st = self.get_treasury_state()
        available = st["free_cash_rub"] - self.buffer
        self.last_release = {"required_rub": round(required_rub, 2),
                             "available_before_rub": round(available, 2),
                             "sold_lots": 0, "enough_units": True}
        if available >= required_rub:
            log.info("[TREASURY] кэша хватает: доступно %.2f ₽, нужно %.2f ₽ — паи не продаются",
                     available, required_rub)
            return True

        deficit = required_rub - available
        need = lots_to_sell(deficit, st["tmon_price"])
        held = st["tmon_lots"]
        lots = min(need, held)
        self.last_release["enough_units"] = need <= held
        if lots <= 0:
            log.warning("[TREASURY] паёв %s нет, дефицит %.2f ₽ — ночной бюджет будет урезан",
                        self.ticker, deficit)
            return False
        if need > held:
            log.warning("[TREASURY] паёв %s не хватает: нужно %d, есть %d — продаю все, "
                        "ночной бюджет будет урезан", self.ticker, need, held)

        sold = 0
        real = min(lots, st["real_lots"])
        if real:
            try:
                sold += self._trade("SELL", real, st["tmon_price"], reason=reason,
                                    allow_virtual=False)["lots"]
            except BrokerError as e:
                log.error("[TREASURY] продажа %d шт %s не прошла: %s", real, self.ticker, e)
        virt = min(lots - real, st["virtual_lots"])
        if virt:
            self._journal("virtual", "SELL", virt, st["tmon_price"], reason)
            sold += virt
        self.last_release["sold_lots"] = sold
        log.info("[TREASURY] Продано %d шт %s на сумму ~%.2f ₽ под ночные лонги.",
                 sold, self.ticker, deficit)
        return self.last_release["enough_units"] and sold == lots

    def restore_buffer(self) -> bool:
        """Рубли ушли в минус (убыток интрадея, комиссии) — продать паи до
        буфера. Без этого после CLEANUP остался бы маржинальный долг, а критерий
        приёмки требует ровно 0 ₽ (ТЗ, §5 п.2)."""
        st = self.get_treasury_state()
        if st["free_cash_rub"] >= 0:
            return True
        log.warning("[TREASURY] свободный кэш %.2f ₽ < 0 — продаю паи до буфера",
                    st["free_cash_rub"])
        return self.release_cash_for_overnight(0.0, reason="cover")

    # ── исполнение ──

    def _trade(self, side: str, lots: int, ref_price: float, *, reason: str,
               allow_virtual: bool) -> dict:
        """Рыночная заявка по фонду. В песочнице при отказе брокера — виртуальная
        запись (только для покупки: продать виртуально реально лежащие паи
        значило бы рассинхронизировать счёт и реестр)."""
        err: Exception | None = None
        if self.tradable():
            try:
                done, price, oid = self._market(side, lots)
            except BrokerError as e:
                done, price, oid, err = 0, 0.0, None, e
            if done > 0:
                self._journal("broker", side, done, price, reason, order_id=oid)
                return {"mode": "broker", "lots": done, "price": price}
        else:
            err = BrokerError(f"{self.instrument().ticker}: листинг недоступен через API")

        if not (allow_virtual and self.sandbox):
            raise err or BrokerError(f"{side} {lots} {self.ticker}: не исполнена")
        if self.ledger is None:
            raise BrokerError(f"{err}; виртуальный реестр недоступен (нет подключения к БД)")
        log.warning("[TREASURY] брокер не исполнил %s %s (%s) — виртуальная запись "
                    "в treasury_ledger", side, self.ticker, err)
        self._journal("virtual", side, lots, ref_price, reason)
        return {"mode": "virtual", "lots": lots, "price": ref_price}

    def _market(self, side: str, lots: int) -> tuple[int, float, str]:
        """Рыночная заявка и ожидание исполнения до FILL_TIMEOUT_SEC.

        Возвращает (исполнено лотов, цена лота, id заявки у брокера). Состояние
        запрашивается по id из ответа брокера, а не по нашему ключу
        идемпотентности: по ключу GetOrderState отвечает 404 (песочница 14.09).
        """
        inst = self.instrument()
        key = new_order_id()
        st = self.broker.post_market_order(account_id=self.account_id, instrument=inst,
                                           direction=side, quantity_lots=lots, order_id=key)
        oid = getattr(st, "order_id", None) or key
        deadline = time.monotonic() + FILL_TIMEOUT_SEC
        while True:
            try:
                st = self.broker.get_order_state(account_id=self.account_id, order_id=oid)
            except BrokerError as e:
                log.warning("[TREASURY] состояние заявки %s недоступно: %s", oid, e)
            if st.execution_report_status in _TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(1.0)
        done = int(st.lots_executed or 0)
        if st.execution_report_status not in _TERMINAL:
            try:
                self.broker.cancel_order(account_id=self.account_id, order_id=oid)
            except Exception as e:               # noqa: BLE001
                log.warning("[TREASURY] остаток заявки %s не снят: %s", oid, e)
        if done <= 0:
            raise BrokerError(f"{side} {lots} {inst.ticker}: не исполнена "
                              f"({st.execution_report_status})")
        lot = int(inst.lot or 1)
        if st.executed_price:
            price = float(st.executed_price) * lot
        elif st.executed_amount:
            price = float(st.executed_amount) / done
        else:
            price = self.lot_price()
        return done, price, oid

    def _journal(self, mode: str, side: str, lots: int, lot_price: float, reason: str,
                 order_id: str | None = None) -> None:
        amount = lots * lot_price
        if self.ledger is not None:
            try:
                self.ledger.record(account_env=self.env, account_id=self.account_id,
                                   ticker=self.ticker, mode=mode, side=side, lots=lots,
                                   price=lot_price, amount_rub=amount, reason=reason,
                                   run_id=self.run_id, order_id=order_id)
            except Exception as e:               # noqa: BLE001
                try:
                    self.ledger.conn.rollback()
                except Exception:                # noqa: BLE001
                    pass
                if mode == "virtual":
                    raise BrokerError(f"виртуальная запись не сохранена: {e}") from e
                log.warning("[TREASURY] след сделки не записан в treasury_ledger: %s", e)
        if self.writer is not None:
            from services.place_orders import _logrow
            _logrow(self.writer, env=self.env, account_id=self.account_id,
                    ticker=self.ticker, direction=side, action=f"treasury_{reason}",
                    order_id=order_id or "", qty_lots_api=lots, price=round(lot_price, 6),
                    status=mode, info=f"{amount:.2f} RUB")
