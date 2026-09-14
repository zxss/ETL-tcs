"""
Казначейство (ТЗ Treasury): парковка кэша в фонд денежного рынка и
высвобождение под ночные лонги. Без сети, без БД и без реального брокера.

Критерии приёмки ТЗ §5 п.4:
  • объём продажи паёв под дефицит кэша округляется вверх;
  • покупки паёв на сумму меньше TREASURY_MIN_SWEEP_RUB нет;
  • ночная корзина урезается, если паёв не хватает на все сигналы.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

import config                                            # noqa: E402
from services import place_orders as po                  # noqa: E402
from services import stage2_demo as s2                   # noqa: E402
from services import treasury as tr                      # noqa: E402
from services.broker.base import (                       # noqa: E402
    BrokerError, Instrument, OrderState, Position, Quotation,
)
from tft_forecast.combined import Order                  # noqa: E402

FILL = "EXECUTION_REPORT_STATUS_FILL"
NORMAL = "SECURITY_TRADING_STATUS_NORMAL_TRADING"
ACC = "acc-1"
PRICE = 164.25

LISTINGS = [
    {"ticker": "TMON", "classCode": "TQBR", "uid": "u-tqbr", "apiTradeAvailableFlag": False},
    {"ticker": "TMON@", "classCode": "SPBRU", "uid": "u-spb", "apiTradeAvailableFlag": True},
]


class MemoryLedger:
    """treasury_ledger в памяти — тот же интерфейс, что у VirtualLedger."""

    def __init__(self, conn=None):
        self.conn = conn
        self.rows: list[dict] = []

    def init(self):
        pass

    def position(self, account_id, ticker):
        v = [r for r in self.rows if r["mode"] == "virtual"
             and r["account_id"] == account_id and r["ticker"] == ticker]
        sign = lambda r: 1 if r["side"] == "BUY" else -1       # noqa: E731
        return (sum(sign(r) * r["lots"] for r in v),
                sum(sign(r) * r["amount_rub"] for r in v))

    def record(self, **kw):
        self.rows.append(kw)


class FundBroker:
    """Брокер с одним фондом по фиксированной цене. Рыночные заявки исполняются
    сразу; состояние заявки — только по id брокера, как в песочнице."""

    def __init__(self, cash=2_000_000.0, lots=0, price=PRICE, reject=False, events=None):
        self.cash, self.lots, self.price, self.reject = cash, lots, price, reject
        self.orders: list[tuple[str, int]] = []
        self.events = events if events is not None else []
        self._states: dict[str, OrderState] = {}

    def find_instrument_listings(self, query):
        return LISTINGS

    def find_instrument_by_uid(self, uid):
        spb = uid == "u-spb"
        return Instrument(ticker="TMON@" if spb else "TMON", instrument_uid=uid, figi="",
                          lot=1, min_price_increment=Quotation.from_float(0.01),
                          currency="rub", trading_status=NORMAL, api_trade_available=spb,
                          class_code="SPBRU" if spb else "TQBR")

    def get_last_price(self, uid):
        return self.price

    def get_money_rub(self, account_id):
        return self.cash

    def get_positions(self, account_id):
        return [Position("u-spb", float(self.lots), 0.0)] if self.lots else []

    def get_active_orders(self, account_id):
        return []

    def cancel_order(self, *, account_id, order_id):
        pass

    def post_market_order(self, *, account_id, instrument, direction, quantity_lots, order_id):
        self.events.append(("fund", direction, quantity_lots))
        if self.reject:
            raise BrokerError("HTTP 400: нет ликвидности")
        self.orders.append((direction, quantity_lots))
        sign = 1 if direction == "BUY" else -1
        self.lots += sign * quantity_lots
        self.cash -= sign * quantity_lots * self.price
        oid = "exch-" + order_id
        self._states[oid] = OrderState(order_id=oid, execution_report_status=FILL,
                                       lots_requested=quantity_lots,
                                       lots_executed=quantity_lots, raw={},
                                       executed_price=self.price,
                                       executed_amount=quantity_lots * self.price)
        return OrderState(order_id=oid, execution_report_status=FILL,
                          lots_requested=quantity_lots, lots_executed=quantity_lots,
                          raw={}, executed_amount=quantity_lots * self.price)

    def get_order_state(self, *, account_id, order_id):
        if order_id not in self._states:
            raise BrokerError("HTTP 404 Order not found")
        return self._states[order_id]


def _patch_config(case, **extra):
    vals = {"TREASURY_ENABLED": True, "TREASURY_TICKER": "TMON", "TREASURY_CLASS_CODE": "",
            "TREASURY_CASH_BUFFER_RUB": 1000.0, "TREASURY_MIN_SWEEP_RUB": 2000.0}
    vals.update(extra)
    for k, v in vals.items():
        p = mock.patch.object(config, k, v)
        p.start()
        case.addCleanup(p.stop)
    p = mock.patch("time.sleep")
    p.start()
    case.addCleanup(p.stop)


def service(broker, *, env="SANDBOX", ledger=None):
    return tr.TreasuryService(broker, ACC, env=env, ledger=ledger)


def night(ticker, cost=10_000.0):
    return {"ticker": ticker, "entry_price": 100.0, "quantity_lots": int(cost // 100),
            "lot_size": 1}


# ── Арифметика ───────────────────────────────────────────────────────────────

class TestArithmetic(unittest.TestCase):

    def test_sell_volume_rounds_up_with_reserve(self):
        """Критерий 4.1: продажа под дефицит — вверх и с запасом 0,2%."""
        self.assertEqual(tr.lots_to_sell(30_000, PRICE), 184)
        self.assertEqual(tr.lots_to_sell(30_000, PRICE),
                         math.ceil(30_000 * 1.002 / PRICE))
        # ровно один лот дефицита → два: запас 0,2% не влезает в один
        self.assertEqual(tr.lots_to_sell(PRICE, PRICE), 2)
        self.assertEqual(tr.lots_to_sell(0, PRICE), 0)
        self.assertGreaterEqual(tr.lots_to_sell(30_000, PRICE) * PRICE, 30_000)

    def test_no_sweep_below_minimum(self):
        """Критерий 4.2: сверх буфера меньше 2000 ₽ — не покупаем."""
        self.assertEqual(tr.sweep_lots(2_999.99, PRICE, buffer_rub=1000, min_sweep_rub=2000), 0)
        self.assertEqual(tr.sweep_lots(3_000.00, PRICE, buffer_rub=1000, min_sweep_rub=2000),
                         int(2000 // PRICE))

    def test_sweep_leaves_buffer(self):
        lots = tr.sweep_lots(2_000_000, PRICE, buffer_rub=1000, min_sweep_rub=2000)
        left = 2_000_000 - lots * PRICE
        self.assertEqual(lots, 12170)
        self.assertGreaterEqual(left, 1000)
        self.assertLess(left, 1000 + PRICE)

    def test_fit_budget_cuts_basket_in_rank_order(self):
        """Критерий 4.3: корзина урезается под кэш, в плечо не идём."""
        orders = [night("A", 10_000), night("B", 9_000), night("C", 8_000)]
        kept, dropped = tr.fit_budget(orders, 18_500)
        self.assertEqual([o["ticker"] for o in kept], ["A", "C"])
        self.assertEqual([o["ticker"] for o in dropped], ["B"])
        self.assertLessEqual(sum(tr.order_cost_rub(o) for o in kept), 18_500)
        self.assertEqual(tr.fit_budget(orders, 0)[0], [])

    def test_order_cost_accepts_order_objects(self):
        o = Order(ticker="ENPG", strategy="long_overnight", direction="LONG",
                  anchor_price=308, f_low=None, f_high=None, down_pct=None,
                  entry_price=307.96, better_pct=None, stop_price=304.88, stop_pct=1.0,
                  tp_price=316.87, tp_pct=2.9, lot_size=1, lot_known=True,
                  quantity_lots=32, total_rub=9854.72, unavailable=False)
        self.assertAlmostEqual(tr.order_cost_rub(o), 307.96 * 32)


class TestListing(unittest.TestCase):

    def test_picks_api_tradable_listing(self):
        """TMON на Мосбирже закрыт для API — берётся TMON@ на СПБ."""
        self.assertEqual(tr.choose_listing(LISTINGS, "TMON")["classCode"], "SPBRU")

    def test_explicit_class_wins(self):
        self.assertEqual(tr.choose_listing(LISTINGS, "TMON", "TQBR")["uid"], "u-tqbr")
        self.assertIsNone(tr.choose_listing(LISTINGS, "TMON", "TQTF"))

    def test_other_tickers_ignored(self):
        self.assertIsNone(tr.choose_listing([{"ticker": "LQDT", "uid": "x"}], "TMON"))


class TestCardLine(unittest.TestCase):

    def test_format_from_spec(self):
        with mock.patch.object(config, "TREASURY_TICKER", "TMON"):
            self.assertEqual(
                tr.card_line({"tmon_value_rub": 1_998_922.4, "free_cash_rub": 1077.6}),
                "Казначейство: TMON 1998922 ₽ | Свободный кэш 1078 ₽")
            self.assertTrue(tr.card_line({"tmon_value_rub": 1, "free_cash_rub": 1,
                                          "virtual_lots": 5}).endswith("(виртуально)"))


# ── SWEEP / UNPARK на брокере ────────────────────────────────────────────────

class TestPark(unittest.TestCase):
    def setUp(self):
        _patch_config(self)

    def test_park_buys_all_above_buffer(self):
        b = FundBroker(cash=2_000_000)
        out = service(b).park_idle_cash()
        self.assertEqual(b.orders, [("BUY", 12170)])
        self.assertEqual(out["mode"], "broker")
        self.assertGreaterEqual(b.cash, 1000)
        self.assertLess(b.cash, 1000 + PRICE)

    def test_no_order_below_min_sweep(self):
        b = FundBroker(cash=2_900)
        out = service(b).park_idle_cash()
        self.assertEqual(b.orders, [])
        self.assertEqual(out["lots"], 0)

    def test_state_fields(self):
        b = FundBroker(cash=1077.5, lots=12170)
        st = service(b).get_treasury_state()
        self.assertEqual(st["tmon_lots"], 12170)
        self.assertEqual(st["free_cash_rub"], 1077.5)
        self.assertAlmostEqual(st["tmon_value_rub"], 12170 * PRICE, places=2)
        self.assertEqual(st["tmon_price"], PRICE)


class TestRelease(unittest.TestCase):
    def setUp(self):
        _patch_config(self)

    def test_enough_cash_sells_nothing(self):
        b = FundBroker(cash=50_000, lots=100)
        self.assertTrue(service(b).release_cash_for_overnight(30_000))
        self.assertEqual(b.orders, [])

    def test_deficit_sold_with_reserve(self):
        b = FundBroker(cash=1077.5, lots=12170)
        s = service(b)
        self.assertTrue(s.release_cash_for_overnight(30_000))
        need = math.ceil((30_000 - 77.5) * 1.002 / PRICE)
        self.assertEqual(b.orders, [("SELL", need)])
        self.assertGreaterEqual(b.cash - 1000, 30_000)
        self.assertEqual(s.last_release["sold_lots"], need)

    def test_not_enough_units_sells_all_and_reports(self):
        b = FundBroker(cash=1000, lots=10)
        self.assertFalse(service(b).release_cash_for_overnight(30_000))
        self.assertEqual(b.orders, [("SELL", 10)])
        self.assertEqual(b.lots, 0)

    def test_basket_cut_when_fund_is_short(self):
        """Критерий 4.3 сквозной: паёв на 16 425 ₽, сигналов на 30 000 ₽ —
        выставляется только то, что покрыто реальным кэшем."""
        b = FundBroker(cash=1000, lots=100)
        s = service(b)
        basket = [night("A"), night("B"), night("C")]
        ok = s.release_cash_for_overnight(sum(tr.order_cost_rub(o) for o in basket))
        self.assertFalse(ok)
        available = s.get_treasury_state()["free_cash_rub"] - s.buffer
        kept, dropped = tr.fit_budget(basket, available)
        self.assertEqual([o["ticker"] for o in kept], ["A"])
        self.assertEqual(len(dropped), 2)
        self.assertGreaterEqual(b.cash - sum(tr.order_cost_rub(o) for o in kept), 0)

    def test_restore_buffer_after_negative_cash(self):
        """Критерий 2: минус по рублям после интрадея гасится продажей паёв."""
        b = FundBroker(cash=-500, lots=100)
        self.assertTrue(service(b).restore_buffer())
        self.assertEqual(b.orders, [("SELL", math.ceil(1500 * 1.002 / PRICE))])
        self.assertGreaterEqual(b.cash, 1000)

    def test_restore_buffer_noop_when_cash_positive(self):
        b = FundBroker(cash=10, lots=100)
        self.assertTrue(service(b).restore_buffer())
        self.assertEqual(b.orders, [])


class TestSandboxFallback(unittest.TestCase):
    """ТЗ, задача 4: брокер отказал по фонду — виртуальный реестр."""

    def setUp(self):
        _patch_config(self)

    def test_rejected_buy_goes_to_virtual_ledger(self):
        b, led = FundBroker(cash=2_000_000, reject=True), MemoryLedger()
        out = service(b, ledger=led).park_idle_cash()
        self.assertEqual(out["mode"], "virtual")
        self.assertEqual(b.cash, 2_000_000)                   # реальные рубли на месте
        st = service(b, ledger=led).get_treasury_state()
        self.assertEqual(st["virtual_lots"], 12170)
        self.assertGreaterEqual(st["free_cash_rub"], 1000)    # но считаются припаркованными
        self.assertLess(st["free_cash_rub"], 1000 + PRICE)
        self.assertEqual(led.rows[0]["mode"], "virtual")

    def test_listing_closed_for_api_goes_virtual_without_orders(self):
        with mock.patch.object(config, "TREASURY_CLASS_CODE", "TQBR"):
            b, led = FundBroker(cash=10_000), MemoryLedger()
            out = service(b, ledger=led).park_idle_cash()
        self.assertEqual(out["mode"], "virtual")
        self.assertEqual(b.events, [])

    def test_virtual_units_released_without_broker(self):
        b, led = FundBroker(cash=2_000_000, reject=True), MemoryLedger()
        s = service(b, ledger=led)
        s.park_idle_cash()
        self.assertTrue(s.release_cash_for_overnight(30_000))
        sells = [r for r in led.rows if r["side"] == "SELL"]
        self.assertEqual(sells[0]["mode"], "virtual")
        self.assertGreaterEqual(s.get_treasury_state()["free_cash_rub"] - 1000, 30_000)
        self.assertEqual(b.orders, [])

    def test_virtual_income_follows_fund_price(self):
        """Доход по ставке денежного рынка — через рост цены пая."""
        b, led = FundBroker(cash=2_000_000, reject=True), MemoryLedger()
        s = service(b, ledger=led)
        s.park_idle_cash()
        before = s.get_treasury_state()
        b.price = PRICE * 1.0005                              # ~день при 18% годовых
        after = s.get_treasury_state()
        self.assertAlmostEqual(after["tmon_value_rub"] - before["tmon_value_rub"],
                               12170 * PRICE * 0.0005, places=2)
        self.assertEqual(after["free_cash_rub"], before["free_cash_rub"])

    def test_prod_has_no_virtual_mode(self):
        b, led = FundBroker(cash=2_000_000, reject=True), MemoryLedger()
        with self.assertRaises(BrokerError):
            service(b, env="PROD", ledger=led).park_idle_cash()
        self.assertEqual(led.rows, [])


# ── Интеграция в фазы Этапа 2 ────────────────────────────────────────────────

def _calls(func) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


class TestShortsDoNotTouchFund(unittest.TestCase):
    """Критерий 1: выставление и закрытие intraday_short паи не трогают."""

    def test_intraday_paths_have_no_treasury_calls(self):
        for fn in (s2.phase_order, po.square_off_intraday, po.place_limits):
            used = _calls(fn)
            for forbidden in ("_treasury", "park_idle_cash", "release_cash_for_overnight",
                              "restore_buffer", "TreasuryService"):
                self.assertNotIn(forbidden, used, f"{fn.__name__} вызывает {forbidden}")


class PhaseBase(unittest.TestCase):
    DAY = dt.date(2026, 9, 15)

    def setUp(self):
        _patch_config(self)
        self.tmp = tempfile.mkdtemp(prefix="treasury-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for k, v in (("STAGE2_DIR", self.tmp), ("STAGE2_ENABLED", True),
                     ("TELEGRAM_ENABLED", False), ("STAGE2_START_DATE", ""),
                     ("STAGE2_HALT_ON_FAIL", True), ("TRADING_MODE", "sandbox"),
                     ("PROD_ACCOUNT_ID", "")):
            p = mock.patch.object(config, k, v)
            p.start()
            self.addCleanup(p.stop)
        self.ledger = MemoryLedger()
        for target, value in (
                (mock.patch.object(po, "_PENDING_PATH", Path(self.tmp) / "pending.json"), None),
                (mock.patch.object(po, "_open_log",
                                   return_value=(mock.MagicMock(), mock.MagicMock())), None),
                (mock.patch.object(s2, "_snapshot_account"), None),
                (mock.patch.object(s2.database, "get_connection",
                                   return_value=mock.MagicMock()), None),
                (mock.patch.object(tr, "VirtualLedger", lambda conn: self.ledger), None)):
            target.start()
            self.addCleanup(target.stop)

    def broker(self, b):
        p = mock.patch.object(po, "_make_broker_and_account", return_value=(b, ACC, "SANDBOX"))
        p.start()
        self.addCleanup(p.stop)
        return b

    def at(self, hh, mm):
        return dt.datetime(2026, 9, 15, hh, mm, tzinfo=s2.MSK)

    def meta(self, phase):
        d = s2._run_dirs_for_day(self.DAY)[phase]
        return json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))

    def fake_run(self, phase, hhmmss, **data):
        d = os.path.join(self.tmp, "runs", f"20260915-{hhmmss}-{phase}")
        os.makedirs(d)
        json.dump({"run_id": os.path.basename(d), "verdict": s2.PASS, **data},
                  open(os.path.join(d, "run_meta.json"), "w", encoding="utf-8"))


class TestPhaseClose(PhaseBase):

    def test_close_parks_idle_cash(self):
        b = self.broker(FundBroker(cash=2_000_000))
        rc = s2.phase_close(now=self.at(10, 0))
        self.assertEqual(rc, 0)
        self.assertEqual(b.orders, [("BUY", 12170)])
        m = self.meta("CLOSE")
        self.assertEqual(m["treasury"]["tmon_lots"], 12170)
        self.assertAlmostEqual(m["treasury_parked_rub"], 12170 * PRICE, places=2)
        self.assertEqual(m["verdict"], s2.PASS)

    def test_close_dry_run_parks_nothing(self):
        b = self.broker(FundBroker(cash=2_000_000))
        s2.phase_close(now=self.at(10, 0), dry_run=True)
        self.assertEqual(b.orders, [])


class TestPhaseCleanup(PhaseBase):

    def setUp(self):
        super().setUp()
        p = mock.patch.object(po, "square_off_intraday", return_value=0)
        p.start()
        self.addCleanup(p.stop)

    def test_negative_cash_restored_and_no_margin_debt(self):
        self.fake_run("CLOSE", "100000", treasury={"tmon_lots": 100})
        b = self.broker(FundBroker(cash=-500, lots=100))
        rc = s2.phase_cleanup(now=self.at(18, 20))
        self.assertEqual(rc, 0)
        self.assertEqual(b.orders, [("SELL", 10)])
        self.assertGreaterEqual(b.cash, 0)
        self.assertEqual(self.meta("CLEANUP")["verdict"], s2.PASS)

    def test_fund_moved_during_day_is_flagged(self):
        self.fake_run("CLOSE", "100000", treasury={"tmon_lots": 120})
        self.broker(FundBroker(cash=5000, lots=100))
        s2.phase_cleanup(now=self.at(18, 20))
        m = self.meta("CLEANUP")
        self.assertTrue(any("изменились за день" in w for w in m["warnings"]), m["warnings"])

    def test_margin_debt_left_is_critical(self):
        """Паёв нет, рубли в минусе — долг остаётся, это FAIL."""
        self.fake_run("CLOSE", "100000", treasury={"tmon_lots": 0})
        self.broker(FundBroker(cash=-500, lots=0))
        rc = s2.phase_cleanup(now=self.at(18, 20))
        self.assertEqual(rc, 1)
        self.assertTrue(any("маржинальный долг" in e for e in self.meta("CLEANUP")["errors"]))


class TestPhaseOvernight(PhaseBase):

    def setUp(self):
        super().setUp()
        self.fake_run("CLEANUP", "182000")
        self.events: list = []

        def _place(broker, account_id, orders, **kw):
            self.events.append(("place", [o.ticker for o in orders]))
            return [{"ticker": o.ticker, "order_id": f"x-{o.ticker}"} for o in orders]

        for p in (mock.patch.object(po, "compute_orders",
                                    return_value=([self.order("AAA"), self.order("BBB"),
                                                   self.order("CCC")], {})),
                  mock.patch.object(po, "place_limits", side_effect=_place),
                  mock.patch.object(s2, "ensure_execution_audit", return_value=True),
                  mock.patch.object(s2, "_record_intents", return_value=0),
                  mock.patch.object(s2, "_close_trading_day", side_effect=lambda d, st, r: st)):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def order(tk):
        return Order(ticker=tk, strategy="long_overnight", direction="LONG",
                     anchor_price=100.0, f_low=None, f_high=None, down_pct=None,
                     entry_price=100.0, better_pct=None, stop_price=99.0, stop_pct=1.0,
                     tp_price=102.0, tp_pct=2.0, lot_size=1, lot_known=True,
                     quantity_lots=100, total_rub=10_000.0, unavailable=False)

    def test_fund_sold_before_stock_orders(self):
        b = self.broker(FundBroker(cash=1077.5, lots=12170, events=self.events))
        rc = s2.phase_overnight(now=self.at(18, 35))
        self.assertEqual(rc, 0)
        self.assertEqual(self.events[0][:2], ("fund", "SELL"))
        self.assertEqual(self.events[1], ("place", ["AAA", "BBB", "CCC"]))
        self.assertGreaterEqual(b.cash - 30_000, 0)            # критерий 3: без плеча
        m = self.meta("OVERNIGHT")
        self.assertEqual(m["verdict"], s2.PASS)
        self.assertEqual(m["treasury_released_lots"], self.events[0][2])

    def test_basket_cut_to_real_cash(self):
        self.broker(FundBroker(cash=1000, lots=100, events=self.events))
        s2.phase_overnight(now=self.at(18, 35))
        self.assertEqual(self.events[-1], ("place", ["AAA"]))
        d = s2._run_dirs_for_day(self.DAY)["OVERNIGHT"]
        night_orders = json.load(open(os.path.join(d, "overnight_orders.json"), encoding="utf-8"))
        cut = [r["ticker"] for r in night_orders["rejected"]
               if "не хватило кэша" in r.get("skip_reason", "")]
        self.assertEqual(cut, ["BBB", "CCC"])
        self.assertEqual(self.meta("OVERNIGHT")["verdict"], s2.PASS_WARN)

    def test_no_orders_no_fund_sale(self):
        with mock.patch.object(po, "compute_orders", return_value=([], {})):
            b = self.broker(FundBroker(cash=1077.5, lots=12170, events=self.events))
            s2.phase_overnight(now=self.at(18, 35))
        self.assertEqual(b.orders, [])
        self.assertEqual(self.meta("OVERNIGHT")["treasury"]["tmon_lots"], 12170)


if __name__ == "__main__":
    unittest.main()
