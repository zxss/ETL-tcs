"""
Хотфикс 11.09, оркестратор Этапа 2: утреннее закрытие овернайта (CLOSE),
вечерний монитор стопов (protect), PARTIAL_FAILURE, дата старта теста, дневной
аудит с вердиктом закрывающей фазы. Без сети и без реального брокера.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import itertools
import json
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                            # noqa: E402
from audit import execution_audit as ea                  # noqa: E402
from services import notify                              # noqa: E402
from services import place_orders as po                  # noqa: E402
from services import stage2_demo as s2                   # noqa: E402
from services.broker.base import OrderState              # noqa: E402

FILL = "EXECUTION_REPORT_STATUS_FILL"
ACC = "acc-1"


def _calls(func) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


class Pos:
    def __init__(self, uid, shares):
        self.instrument_uid, self.balance_shares = uid, shares

    @property
    def is_open(self):
        return self.balance_shares != 0


class Broker:
    def __init__(self, positions=(), stops=(), active=(), fill=True):
        self.calls, self._pos, self._stops = [], list(positions), list(stops)
        self._active, self.fill = list(active), fill

    def get_positions(self, a):
        return self._pos

    def get_active_stop_orders(self, a):
        return self._stops

    def get_active_orders(self, a):
        return self._active

    def cancel_stop_order(self, *, account_id, stop_order_id):
        self.calls.append(("cancel_stop", stop_order_id))

    def cancel_order(self, *, account_id, order_id):
        self.calls.append(("cancel_order", order_id))

    def find_instrument_by_uid(self, uid):
        return types.SimpleNamespace(instrument_uid=uid, lot=1)

    def post_market_order(self, *, account_id, instrument, direction, quantity_lots, order_id):
        self.calls.append(("market", instrument.instrument_uid, direction, quantity_lots))

    def get_order_state(self, *, account_id, order_id):
        ok = self.fill
        return OrderState(order_id=order_id,
                          execution_report_status=FILL if ok else "EXECUTION_REPORT_STATUS_NEW",
                          lots_requested=32, lots_executed=32 if ok else 0,
                          raw={"orderDate": "2026-09-14T07:00:05Z"},
                          executed_price=308.65, executed_commission=4.94)


def _stop(uid, kind, sid):
    return types.SimpleNamespace(instrument_uid=uid, kind=kind, stop_order_id=sid)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2hf-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for k, v in (("STAGE2_DIR", self.tmp), ("STAGE2_ENABLED", True),
                     ("TELEGRAM_ENABLED", False), ("STAGE2_START_DATE", ""),
                     ("STAGE2_HALT_ON_FAIL", True), ("TRADING_MODE", "sandbox")):
            p = mock.patch.object(config, k, v)
            p.start()
            self.addCleanup(p.stop)
        self.pending = Path(self.tmp) / "pending_stops.json"
        p = mock.patch.object(po, "_PENDING_PATH", self.pending)
        p.start()
        self.addCleanup(p.stop)

    def registry(self, *recs):
        self.pending.write_text(json.dumps({ACC: list(recs)}), encoding="utf-8")

    def reg(self):
        return json.loads(self.pending.read_text(encoding="utf-8"))[ACC]

    @staticmethod
    def rec(uid, ticker="ENPG", strategy="long_overnight", oid="e1"):
        return {"order_id": oid, "ticker": ticker, "strategy": strategy,
                "instrument_uid": uid, "lot": 1, "exit_direction": "SELL",
                "closed": False}

    def close(self, broker, dry_run=False):
        with mock.patch.object(ea, "record_exit") as rx, \
             mock.patch.object(ea, "record_fill"), \
             mock.patch("time.sleep"), \
             mock.patch("time.monotonic", side_effect=itertools.count(0, 30)):
            out = s2.close_overnight_positions(broker, ACC, mock.MagicMock(),
                                               dry_run=dry_run, writer=mock.MagicMock(),
                                               env="SANDBOX")
        return out, rx


class TestOvernightClose(Base):
    def test_closes_overnight_long_after_cancelling_its_stops(self):
        """Дефект 1: выход long_overnight утром. Стопы снимаются ДО продажи —
        иначе оставшийся стоп при касании откроет обратную позицию."""
        b = Broker(positions=[Pos("u1", 32)],
                   stops=[_stop("u1", "STOP_LOSS", "s1"), _stop("u1", "TAKE_PROFIT", "t1")])
        self.registry(self.rec("u1"))
        out, rx = self.close(b)
        self.assertEqual(b.calls, [("cancel_stop", "s1"), ("cancel_stop", "t1"),
                                   ("market", "u1", "SELL", 32)])
        self.assertEqual(out["closed"], ["ENPG"])
        self.assertTrue(self.reg()[0]["closed"])
        self.assertEqual(self.reg()[0]["closed_reason"], "overnight_close")
        self.assertEqual(rx.call_args.kwargs["exit_reason"], "overnight_close")

    def test_intraday_and_unregistered_positions_untouched(self):
        b = Broker(positions=[Pos("u1", 10), Pos("u2", 5)])
        self.registry(self.rec("u1", ticker="SBER", strategy="intraday_short"))
        out, _ = self.close(b)
        self.assertEqual(b.calls, [])
        self.assertEqual(out["overnight_before"], 0)

    def test_unfilled_overnight_limit_is_cancelled(self):
        """Висящая ночная лимитка залилась бы днём и дала позицию без выхода."""
        b = Broker(active=[types.SimpleNamespace(order_id="e1", instrument_uid="u1")])
        self.registry(self.rec("u1"))
        out, _ = self.close(b)
        self.assertIn(("cancel_order", "e1"), b.calls)
        self.assertEqual(out["orders_cancelled"], 1)
        self.assertTrue(self.reg()[0]["closed"])

    def test_leftover_stop_cancelled_when_position_gone(self):
        """Позицию ночью закрыл стоп, а тейк остался: SELL-тейк на пустой позиции
        при открытии открыл бы шорт."""
        b = Broker(stops=[_stop("u1", "TAKE_PROFIT", "t1")])
        self.registry(self.rec("u1"))
        self.close(b)
        self.assertIn(("cancel_stop", "t1"), b.calls)

    def test_not_filled_close_is_reported_and_record_kept(self):
        b = Broker(positions=[Pos("u1", 32)], fill=False)
        self.registry(self.rec("u1"))
        out, _ = self.close(b)
        self.assertEqual(len(out["failed"]), 1)
        self.assertFalse(self.reg()[0]["closed"])

    def test_dry_run_sends_nothing(self):
        b = Broker(positions=[Pos("u1", 32)], stops=[_stop("u1", "STOP_LOSS", "s1")])
        self.registry(self.rec("u1"))
        out, _ = self.close(b, dry_run=True)
        self.assertEqual(b.calls, [])
        self.assertFalse(self.reg()[0]["closed"])

    def test_phase_close_only_exits(self):
        used = _calls(s2.phase_close)
        self.assertIn("close_overnight_positions", used)
        for forbidden in ("place_limits", "compute_orders", "post_limit_order"):
            self.assertNotIn(forbidden, used)

    def test_cli_knows_new_commands(self):
        src = inspect.getsource(s2.main)
        self.assertIn("phase_close", src)
        self.assertIn("cmd_protect", src)
        self.assertIn("CLOSE", s2.PHASES)


class StrictIdBroker(Broker):
    """Как песочница 14.09: состояние заявки отдаётся только по id брокера,
    по нашему ключу идемпотентности — 404."""

    def post_market_order(self, *, account_id, instrument, direction, quantity_lots, order_id):
        super().post_market_order(account_id=account_id, instrument=instrument,
                                  direction=direction, quantity_lots=quantity_lots,
                                  order_id=order_id)
        return OrderState(order_id="exch-" + order_id,
                          execution_report_status="EXECUTION_REPORT_STATUS_NEW",
                          lots_requested=quantity_lots, lots_executed=0, raw={})

    def get_order_state(self, *, account_id, order_id):
        if not str(order_id).startswith("exch-"):
            raise po.BrokerError("HTTP 404 Order not found")
        return super().get_order_state(account_id=account_id, order_id=order_id)


class TestCloseUsesBrokerOrderId(Base):
    def test_close_polls_state_by_broker_id(self):
        """Регрессия 14.09: CLOSE ждал исполнения по ключу идемпотентности,
        брокер отвечал 404, и закрытие считалось несостоявшимся → FAIL."""
        b = StrictIdBroker(positions=[Pos("u1", 32)])
        self.registry(self.rec("u1"))
        out, _ = self.close(b)
        self.assertEqual(out["failed"], [])
        self.assertEqual(out["closed"], ["ENPG"])


class TestProtect(Base):
    def test_protect_attaches_stops_and_places_no_entries(self):
        used = _calls(s2.cmd_protect)
        self.assertIn("attach_stops", used)
        for forbidden in ("place_limits", "post_limit_order", "post_market_order"):
            self.assertNotIn(forbidden, used)


class TestPartialFailure(Base):
    def res(self, phase="OVERNIGHT"):
        d = s2.make_run_dir(s2.new_run_id(phase))
        return s2.PhaseResult(phase, os.path.basename(d), d)

    def test_broker_rejection_is_partial_not_pass(self):
        r = self.res()
        s2._check_placement(r, [("OGKB", "placed"), ("AFLT", "error_broker"),
                                ("ENPG", "skip_duplicate")], [1], [1, 2, 3])
        self.assertEqual(r.verdict, s2.PARTIAL)
        self.assertIn("AFLT", r.partial[0])
        self.assertNotIn("ENPG", r.partial[0])
        self.assertEqual(r.exit_code(), 0)

    def test_error_beats_partial(self):
        r = self.res()
        r.partial_fail("x")
        r.check(False, "y")
        self.assertEqual(r.verdict, s2.FAIL)

    def test_partial_does_not_halt_the_test(self):
        r = self.res()
        r.partial_fail("отказ по AFLT")
        st = s2.load_state()
        self.assertEqual(s2._finish(r, st), 0)
        self.assertNotEqual(s2.load_state().get("status"), "halted")
        meta = json.load(open(os.path.join(r.run_dir, "run_meta.json"), encoding="utf-8"))
        self.assertEqual(meta["verdict"], s2.PARTIAL)
        self.assertEqual(meta["partial"], ["отказ по AFLT"])

    def test_partial_card_is_yellow_and_audible(self):
        r = self.res()
        r.partial_fail("отказ по AFLT")
        with mock.patch.object(config, "TELEGRAM_ENABLED", True), \
             mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "1:x"), \
             mock.patch.object(config, "TELEGRAM_CHAT_ID", "2"), \
             mock.patch.object(notify, "send") as snd:
            s2._notify_phase(r, {})
        self.assertIn("\U0001f7e1", snd.call_args.args[0])
        self.assertFalse(snd.call_args.kwargs["silent"])


class TestStartDate(Base):
    def test_phase_before_start_does_nothing(self):
        """Пауза 11.09 и старт 14.09: до даты старта фаза не выполняется и не
        создаёт даже каталог пропуска."""
        with mock.patch.object(config, "STAGE2_START_DATE", "2026-09-14"):
            rc = s2.phase_overnight(now=dt.datetime(2026, 9, 11, 18, 35, tzinfo=s2.MSK))
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, "runs")))

    def test_bad_date_fails_loudly(self):
        with mock.patch.object(config, "STAGE2_START_DATE", "14.09.2026"):
            with self.assertRaises(ValueError):
                s2._start_date()


class TestDailyAuditSeesClosingPhase(Base):
    def test_overnight_verdict_and_run_id_reach_the_audit(self):
        """10.09 overnight_run_id был null: аудит собирался до run_meta
        OVERNIGHT. Частичный отказ брокера тоже терялся."""
        day = dt.date(2026, 9, 14)
        runs = os.path.join(self.tmp, "runs")
        for ph, t in (("PREP", "094500"), ("CLOSE", "100000"), ("ORDER", "100500"),
                      ("CLEANUP", "182000")):
            d = os.path.join(runs, f"20260914-{t}-{ph}")
            os.makedirs(d)
            json.dump({"run_id": os.path.basename(d), "verdict": s2.PASS},
                      open(os.path.join(d, "run_meta.json"), "w"))
        od = os.path.join(runs, "20260914-183500-OVERNIGHT")
        os.makedirs(od)
        r = s2.PhaseResult("OVERNIGHT", "20260914-183500-OVERNIGHT", od)
        r.partial_fail("отказ по AFLT")
        with mock.patch.object(s2, "_execution_stats",
                               return_value={"available": True, "orders": 0, "filled": 0}):
            audit = s2.build_daily_audit(day, r)
        self.assertEqual(audit["overnight_run_id"], "20260914-183500-OVERNIGHT")
        self.assertEqual(audit["verdict"], s2.PARTIAL)
        self.assertEqual(audit["close_run_id"], "20260914-100000-CLOSE")


if __name__ == "__main__":
    unittest.main()
