"""
Тесты исправлений по аудиту r4 (AUDIT-R4-VERDICT.md, 17.09.2026).

  1. OCO: сработал SL → TP снимается (кейс SNGS: стоп 07:06, тейк висел до 09:10).
  2. Журнал исполнения: заливки и выходы по стопу попадают в execution_audit.
  3. CLEANUP: порядок и ожидание нуля — в tests/test_square_off.py (TestCleanupOrder).
  4. Реестр стопов: атомарная запись, блокировка, нечитаемый файл — отказ.
  5. Предохранитель ликвидности при фиксированной сумме позиции (AKRN).
  6. Цена уже за стопом → аварийный выход по рынку (кейс SMLT 15.09).
  +  Дневной аудит не теряет сбой перезапущенной фазы (PARK 16.09).
"""
from __future__ import annotations

import datetime as dt
import io
import contextlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

import config                                               # noqa: E402
from services import exec_journal                           # noqa: E402
from services import place_orders as po                     # noqa: E402
from services import stage2_demo as s2                      # noqa: E402
from services.broker.base import (BrokerError, OrderState, StopOrderRecord,  # noqa: E402
                                  Trade)
from tests.conftest import make_dashboard_row, make_instrument  # noqa: E402

FILL = "EXECUTION_REPORT_STATUS_FILL"
ACC = "ACC1"


class Pos:
    def __init__(self, uid, shares):
        self.instrument_uid, self.balance_shares = uid, shares

    @property
    def is_open(self):
        return self.balance_shares != 0


class Broker:
    """Брокер для attach_stops: позиции, стопы (активные и история), цена, заявки."""

    def __init__(self, *, positions=(), active_stops=(), history=(), last=None,
                 reject_stop=False, cancel_fails=False):
        self.positions = list(positions)
        self.active_stops = list(active_stops)
        self.history = list(history)
        self.last = last
        self.reject_stop, self.cancel_fails = reject_stop, cancel_fails
        self.calls: list[tuple] = []

    def get_positions(self, a):
        return self.positions

    def get_active_stop_orders(self, a):
        return list(self.active_stops)

    def get_stop_order_history(self, a, since):
        self.calls.append(("history", since))
        return list(self.history)

    def find_instrument(self, tk):
        return make_instrument(lot=1, ticker=tk)

    def get_last_price(self, uid):
        return self.last

    def post_stop_order(self, *, account_id, instrument, direction, quantity_lots,
                        stop_price, order_id, order_type="STOP_LOSS"):
        if self.reject_stop and order_type == "STOP_LOSS":
            raise BrokerError('PostStopOrder: HTTP 400 {"code":3,"message":"The price is '
                              'outside the limits for this instrument","description":"30099"}')
        self.calls.append(("stop", order_type, stop_price.as_float()))
        return f"new-{order_type}"

    def cancel_stop_order(self, *, account_id, stop_order_id):
        if self.cancel_fails:
            raise BrokerError("сеть")
        self.calls.append(("cancel_stop", stop_order_id))
        self.active_stops = [s for s in self.active_stops if s.stop_order_id != stop_order_id]

    def post_market_order(self, *, account_id, instrument, direction, quantity_lots, order_id):
        self.calls.append(("market", instrument.ticker, direction, quantity_lots))
        return OrderState(order_id="mkt-1", execution_report_status="EXECUTION_REPORT_STATUS_NEW",
                          lots_requested=quantity_lots, lots_executed=0, raw={})

    def get_order_state(self, *, account_id, order_id):
        return OrderState(order_id=order_id, execution_report_status=FILL,
                          lots_requested=36, lots_executed=36, raw={},
                          executed_price=264.0, executed_commission=3.8)


def _stop(uid, kind, sid):
    return types.SimpleNamespace(instrument_uid=uid, kind=kind, stop_order_id=sid)


def _rec(ticker="SNGS", uid="u1", **over):
    r = {"order_id": f"o-{ticker}", "ticker": ticker, "strategy": "long_overnight",
         "instrument_uid": uid, "lot": 1, "api_qty": 36, "exit_direction": "SELL",
         "stop_units": 268, "stop_nano": 600_000_000, "tp_units": 282, "tp_nano": 600_000_000,
         "created": "2026-09-16T18:41:44", "stop_placed": False, "stop_order_id": None,
         "tp_placed": False, "tp_order_id": None, "closed": False}
    r.update(over)
    return r


def run_attach(broker, records, **kw):
    saved = {}
    with mock.patch.object(po, "_load_pending", return_value={ACC: records}), \
         mock.patch.object(po, "_save_pending", side_effect=saved.update), \
         mock.patch.object(po.time, "sleep"), \
         contextlib.redirect_stdout(io.StringIO()):
        rep: dict = {}
        po.attach_stops(broker, ACC, dry_run=False, writer=mock.MagicMock(),
                        env="SANDBOX", report=rep, **kw)
    return rep


# ── 1. OCO ────────────────────────────────────────────────────────────────────

class TestOco(unittest.TestCase):

    def _bracketed(self):
        return _rec(stop_placed=True, stop_order_id="sl-1", tp_placed=True, tp_order_id="tp-1")

    def test_tp_canceled_when_sl_filled(self):
        """SNGS 17.09: SL исполнен → TP снимается в том же прогоне, запись закрыта."""
        rec = self._bracketed()
        b = Broker(positions=[],
                   active_stops=[_stop("u1", "TAKE_PROFIT", "tp-1")],
                   history=[StopOrderRecord("sl-1", "u1", "STOP_LOSS", "EXECUTED",
                                            "2026-09-17T04:06:20Z"),
                            StopOrderRecord("tp-1", "u1", "TAKE_PROFIT", "ACTIVE")])
        rep = run_attach(b, [rec])
        self.assertIn(("cancel_stop", "tp-1"), b.calls)
        self.assertTrue(rec["closed"])
        self.assertEqual(rec["closed_reason"], "stop")
        self.assertEqual(rep["oco_closed"], ["SNGS"])

    def test_sl_canceled_when_tp_filled_even_if_position_still_shown(self):
        """Нога исполнена, а позиция ещё видна — парную снять, но запись НЕ закрывать.

        25.09.2026, боевой счёт: вход 51 лот HYDR, стопы поставлены на 33 (остальное
        долилось позже), стоп сработал на 33 000 шт — 18 000 остались БЕЗ ЗАЩИТЫ, а
        запись была закрыта. CLOSE их больше не трогал, и тест вставал в halt каждое
        утро. Теперь запись остаётся открытой, флаги защиты сброшены, и следующий
        прогон PROTECT ставит стопы на фактический остаток по свежему балансу.
        """
        rec = self._bracketed()
        b = Broker(positions=[Pos("u1", 36)],
                   active_stops=[_stop("u1", "STOP_LOSS", "sl-1")],
                   history=[StopOrderRecord("tp-1", "u1", "TAKE_PROFIT", "EXECUTED")])
        rep = run_attach(b, [rec])
        self.assertIn(("cancel_stop", "sl-1"), b.calls)
        self.assertFalse(rec["closed"], "запись с живой позицией закрывать нельзя")
        self.assertFalse(rec["stop_placed"])
        self.assertFalse(rec["tp_placed"])
        self.assertEqual(rec["pending_exit_reason"], "target")
        self.assertEqual(rep.get("partial_exits"), ["SNGS"])
        self.assertFalse([c for c in b.calls if c[0] == "stop"])   # в этом прогоне не ставим

    def test_full_exit_still_closes_record(self):
        """Позиции не осталось — поведение прежнее: запись закрывается с причиной."""
        rec = self._bracketed()
        b = Broker(positions=[],
                   active_stops=[_stop("u1", "STOP_LOSS", "sl-1")],
                   history=[StopOrderRecord("tp-1", "u1", "TAKE_PROFIT", "EXECUTED")])
        run_attach(b, [rec])
        self.assertTrue(rec["closed"])
        self.assertEqual(rec["closed_reason"], "target")

    def test_record_stays_open_if_sibling_cancel_fails(self):
        rec = self._bracketed()
        b = Broker(active_stops=[_stop("u1", "TAKE_PROFIT", "tp-1")], cancel_fails=True,
                   history=[StopOrderRecord("sl-1", "u1", "STOP_LOSS", "EXECUTED")])
        run_attach(b, [rec])
        self.assertFalse(rec["closed"])                          # следующий прогон повторит

    def test_close_phase_cancels_both_legs_before_market_exit(self):
        """CLOSE 09:10: обе условные заявки снимаются до рыночной продажи."""
        src = Path(s2.__file__).read_text(encoding="utf-8")
        body = src[src.index("def close_overnight_positions("):src.index("def phase_close(")]
        self.assertLess(body.index("if not _cancel_stops(tk, uid):\n            continue"
                                   "                             # без снятых стопов"),
                        body.index("broker.post_market_order("))


# ── 6. цена уже за стопом ─────────────────────────────────────────────────────

class TestStopBreach(unittest.TestCase):

    def test_market_exit_when_price_below_stop(self):
        """SMLT 15.09: лонг залит, цена 264,2 ниже стопа 268,6 — стоп не ставится,
        позиция закрывается по рынку, запись закрыта с причиной stop_breach."""
        rec = _rec(ticker="SMLT")
        b = Broker(positions=[Pos("u1", 36)], last=264.2)
        rep = run_attach(b, [rec])
        self.assertIn(("market", "SMLT", "SELL", 36), b.calls)
        self.assertFalse([c for c in b.calls if c[0] == "stop"])
        self.assertTrue(rec["closed"])
        self.assertEqual(rec["closed_reason"], "stop_breach")
        self.assertEqual(rep["breach_exits"], ["SMLT"])

    def test_short_exit_when_price_above_stop(self):
        rec = _rec(ticker="SHRT", exit_direction="BUY", stop_units=105, stop_nano=0,
                   tp_units=95, tp_nano=0, strategy="intraday_short")
        b = Broker(positions=[Pos("u1", -10)], last=106.0)
        run_attach(b, [rec])
        self.assertIn(("market", "SHRT", "BUY", 10), b.calls)

    def test_broker_rejection_rechecks_price_and_exits(self):
        """Цена прошла стоп между проверкой и постановкой: отказ 30099 → выход по рынку."""
        rec = _rec(ticker="SMLT")
        b = Broker(positions=[Pos("u1", 36)], last=270.0, reject_stop=True)
        prices = iter([270.0, 264.2])
        b.get_last_price = lambda uid: next(prices)
        run_attach(b, [rec])
        self.assertIn(("market", "SMLT", "SELL", 36), b.calls)
        self.assertEqual(rec["closed_reason"], "stop_breach")

    def test_normal_stop_when_price_above_stop(self):
        rec = _rec(ticker="SMLT")
        b = Broker(positions=[Pos("u1", 36)], last=271.0)
        run_attach(b, [rec])
        self.assertIn(("stop", "STOP_LOSS", 268.6), b.calls)
        self.assertFalse([c for c in b.calls if c[0] == "market"])
        self.assertTrue(rec["stop_placed"])

    def test_no_market_exit_while_sibling_stop_cannot_be_cancelled(self):
        """Тейк на пустой позиции откроет шорт — без его снятия по рынку не выходим."""
        rec = _rec(ticker="SMLT", tp_placed=True, tp_order_id="tp-1")
        b = Broker(positions=[Pos("u1", 36)], last=264.2, cancel_fails=True,
                   active_stops=[_stop("u1", "TAKE_PROFIT", "tp-1")])
        rep = run_attach(b, [rec])
        self.assertFalse([c for c in b.calls if c[0] == "market"])
        self.assertEqual(rep["breach_failed"], ["SMLT"])
        self.assertFalse(rec["closed"])


# ── 4. реестр ─────────────────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="reg-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name, val in (("_LOG_DIR", self.tmp),
                          ("_PENDING_PATH", self.tmp / "pending_stops.json"),
                          ("_LOCK_PATH", self.tmp / "pending_stops.lock")):
            p = mock.patch.object(po, name, val)
            p.start()
            self.addCleanup(p.stop)

    def test_atomic_registry_write(self):
        """Обрыв посреди записи не портит прежний реестр и не оставляет мусора."""
        po._save_pending({ACC: [{"ticker": "OLD"}]})
        real_dumps = json.dumps

        def broken(obj, **kw):
            raise OSError("диск кончился посреди записи")

        with mock.patch.object(po.json, "dumps", side_effect=broken):
            with self.assertRaises(OSError):
                po._save_pending({ACC: [{"ticker": "NEW"}]})
        self.assertEqual(po._load_pending(), {ACC: [{"ticker": "OLD"}]})
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         ["pending_stops.json"])                 # ни .tmp, ни порчи
        self.assertIs(json.dumps, real_dumps)

    def test_replace_failure_keeps_old_registry(self):
        po._save_pending({ACC: [{"ticker": "OLD"}]})
        with mock.patch.object(po.os, "replace", side_effect=OSError("сбой")):
            with self.assertRaises(OSError):
                po._save_pending({ACC: [{"ticker": "NEW"}]})
        self.assertEqual(po._load_pending(), {ACC: [{"ticker": "OLD"}]})
        self.assertFalse(list(self.tmp.glob(".*.tmp")))

    def test_corrupt_registry_raises_instead_of_empty(self):
        (self.tmp / "pending_stops.json").write_text('{"ACC1": [{"tick', encoding="utf-8")
        with self.assertRaises(po.RegistryError):
            po._load_pending()

    def test_lock_is_exclusive(self):
        import subprocess
        code = ("import fcntl,sys,time; f=open(sys.argv[1],'a+'); "
                "fcntl.flock(f, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(5)")
        proc = subprocess.Popen([sys.executable, "-c", code, str(self.tmp / "pending_stops.lock")],
                                stdout=subprocess.PIPE, text=True)
        self.addCleanup(proc.kill)
        self.assertEqual(proc.stdout.readline().strip(), "locked")
        with self.assertRaises(po.RegistryBusy):
            with po.registry_lock(wait=False):
                pass
        proc.kill()
        proc.wait()
        with po.registry_lock(wait=False):
            pass                                              # освободился — берётся

    def test_protect_skips_when_registry_busy(self):
        with mock.patch.object(po, "registry_lock", side_effect=po.RegistryBusy("занят")), \
             mock.patch.object(s2, "cmd_protect") as protect:
            self.assertEqual(s2.main(["protect"]), 0)
        protect.assert_not_called()

    def test_trading_phase_runs_under_lock(self):
        entered = []

        @contextlib.contextmanager
        def lock(**kw):
            entered.append(kw)
            yield

        with mock.patch.object(po, "registry_lock", side_effect=lock), \
             mock.patch.object(s2, "phase_cleanup", return_value=0) as ph:
            self.assertEqual(s2.main(["cleanup"]), 0)
        ph.assert_called_once()
        self.assertEqual(entered, [{}])


# ── 2. журнал исполнения ─────────────────────────────────────────────────────

class FakeConn:
    """execution_audit в памяти: SELECT по order_id и UPDATE заливки/выхода."""

    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                self.sql, self.params = sql, params
                if sql.lstrip().startswith("SELECT"):
                    r = conn.rows.get(params[0])
                    self._one = None if r is None else (
                        r["requested_price"], r["filled"], r.get("filled_price"),
                        r.get("filled_at"), r.get("fee_rub"))
                elif "filled = TRUE" in sql:
                    conn.rows[params["order_id"]].update(
                        filled=True, filled_price=params["filled_price"],
                        filled_at=params["filled_at"], fee_rub=params["fee_rub"])
                elif "exit_reason" in sql:
                    conn.rows[params["order_id"]].update(
                        exit_price=params["exit_price"], exit_reason=params["exit_reason"],
                        pnl_net_rub=params["pnl_net_rub"])

            def fetchone(self):
                return self._one
        return Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


class JournalBroker:
    """SNGS 16–17.09 по фактическим операциям песочницы."""

    def get_order_state(self, *, account_id, order_id):
        return OrderState(order_id=order_id, execution_report_status=FILL,
                          lots_requested=61, lots_executed=61, raw={},
                          executed_price=16.14, executed_commission=49.227)

    def get_trades(self, account_id, since, uid=None):
        trades = [Trade("u1", "BUY", 16.14, 6100, "2026-09-16T17:56:43Z"),
                  Trade("u1", "SELL", 16.015, 6100, "2026-09-17T04:06:20Z")]
        out = [t for t in trades if t.at >= since]
        fees = 49.227 * (since <= "2026-09-16T17:56:43Z") + 48.84575
        return out, fees


class TestExecutionJournal(unittest.TestCase):

    def test_stop_exit_fills_row_that_was_left_unfilled(self):
        """Строка SNGS была filled=false; после OCO в ней вход, выход и чистый PnL."""
        conn = FakeConn({"o-SNGS": {"requested_price": 16.1404, "filled": False}})
        rec = _rec(lot=100)
        ok = exec_journal.journal_exit(JournalBroker(), ACC, conn, rec, "stop")
        row = conn.rows["o-SNGS"]
        self.assertTrue(ok)
        self.assertTrue(row["filled"])
        self.assertEqual(row["filled_price"], 16.14)
        self.assertEqual(row["filled_at"], dt.datetime(2026, 9, 16, 17, 56, 43,
                                                       tzinfo=dt.timezone.utc))
        self.assertEqual(row["exit_reason"], "stop")
        self.assertAlmostEqual(row["exit_price"], 16.015)
        self.assertAlmostEqual(row["pnl_net_rub"], -860.58, places=1)   # как в кассе 17.09

    def test_protect_journals_fill_when_position_appears(self):
        conn = FakeConn({"o-SNGS": {"requested_price": 16.1404, "filled": False}})
        rec = _rec(lot=100, stop_units=15, stop_nano=980_000_000, tp_units=16, tp_nano=400_000_000)
        b = Broker(positions=[Pos("u1", 6100)], last=16.2)
        b.get_order_state = JournalBroker().get_order_state
        b.get_trades = JournalBroker().get_trades
        rep = run_attach(b, [rec], conn=conn)
        self.assertEqual(rep["fills_journaled"], 1)
        self.assertTrue(rec["fill_journaled"])
        self.assertTrue(conn.rows["o-SNGS"]["filled"])

    def test_no_conn_is_harmless(self):
        self.assertFalse(exec_journal.journal_fill(JournalBroker(), ACC, None, _rec()))

    def test_utc_iso_treats_naive_registry_time_as_msk(self):
        self.assertEqual(exec_journal.utc_iso("2026-09-16T18:41:44"), "2026-09-16T15:41:44Z")


# ── 5. ликвидность ───────────────────────────────────────────────────────────

class TestLiquidityCap(unittest.TestCase):

    @staticmethod
    def _row(**over):
        r = make_dashboard_row(ticker="AKRN", anchor_price=18_000.0, f_low=17_800.0,
                               f_high=18_400.0, down=-2.0)
        r.update(over)
        return r

    def test_position_capped_by_max_pos(self):
        from tft_forecast.combined import build_orders
        o = build_orders([self._row(max_pos=60_000.0)], position_rub=100_000.0,
                         entry_frac=0.2)[0]
        self.assertTrue(o.is_placeable)
        self.assertLessEqual(o.total_rub, 60_000.0)
        uncapped = build_orders([self._row(max_pos=None)], position_rub=100_000.0,
                                entry_frac=0.2)[0]
        self.assertGreater(uncapped.total_rub, 60_000.0)

    def test_skip_when_max_pos_below_one_lot(self):
        from tft_forecast.combined import build_orders
        o = build_orders([self._row(max_pos=10_000.0)], position_rub=100_000.0,
                         entry_frac=0.2)[0]
        self.assertIsNone(o.quantity_lots)
        self.assertFalse(o.is_placeable)

    def test_liquid_name_keeps_position_limit(self):
        from tft_forecast.combined import build_orders
        o = build_orders([self._row(ticker="SBER", anchor_price=300.0, f_low=297.0,
                                    f_high=306.0, max_pos=96_892_000.0)],
                         position_rub=100_000.0, entry_frac=0.2)[0]
        self.assertGreater(o.total_rub, 95_000.0)       # лимит, а не MaxPos


# ── дневной аудит ─────────────────────────────────────────────────────────────

class TestDailyAuditKeepsSupersededFailures(unittest.TestCase):

    def test_first_park_partial_survives_rerun(self):
        tmp = tempfile.mkdtemp(prefix="s2da-")
        self.addCleanup(shutil.rmtree, tmp, True)
        day = dt.date(2026, 9, 16)
        with mock.patch.object(config, "STAGE2_DIR", tmp, create=True), \
             mock.patch.object(s2, "_execution_stats", return_value={"available": True}):
            for name, verdict, partial in (
                    ("20260916-100502-PARK", s2.PARTIAL, ["покупка паёв не исполнена: Not enough balance"]),
                    ("20260916-101647-PARK", s2.PASS, [])):
                d = os.path.join(tmp, "runs", name)
                os.makedirs(d)
                s2._write_json(os.path.join(d, "run_meta.json"),
                               {"run_id": name, "verdict": verdict, "errors": [],
                                "warnings": [], "partial": partial})
            res = s2.PhaseResult("OVERNIGHT", "20260916-183502-OVERNIGHT",
                                 os.path.join(tmp, "runs", "x"))
            audit = s2.build_daily_audit(day, res)
        self.assertTrue(any("20260916-100502-PARK" in w and "Not enough balance" in w
                            for w in audit["warnings"]), audit["warnings"])


if __name__ == "__main__":
    unittest.main()
