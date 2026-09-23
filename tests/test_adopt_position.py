"""Ручная позиция под защитой контура (services/adopt_position.py).

Повод: 23.09.2026 пользователю понадобилось заводить в контур позиции,
открытые вне конвейера (сигналы стороннего аналитика). Без записи в
data/order_log/pending_stops.json PROTECT такую позицию не видит вовсе:
стоп не ставит и в строку «без стопа» не включает — тишина в логе выглядит
как «всё в порядке».
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services import adopt_position as ap
from services import place_orders as po
from services.broker.base import Quotation

ACC = "2018145468"
UID = "8e2b0325-0292-4654-8a18-4f63ed3b0e09"
STEP = Quotation(units=0, nano=10_000_000)        # шаг 0,01


def _rec(balance=-450.0, strategy="manual", take=None, **kw):
    return ap.build_record(
        order_id=kw.pop("order_id", "adopted-test"),
        ticker="VTBR", instrument_uid=UID, lot=1, balance_shares=balance,
        stop_q=Quotation.from_float(52.50, STEP),
        tp_q=Quotation.from_float(take, STEP) if take is not None else None,
        strategy=strategy, **kw)


class TestExitDirection(unittest.TestCase):

    def test_short_closes_with_buy(self):
        self.assertEqual(ap.exit_direction(-450.0), "BUY")

    def test_long_closes_with_sell(self):
        self.assertEqual(ap.exit_direction(450.0), "SELL")


class TestCheckLevels(unittest.TestCase):
    """Стоп с неправильной стороны PROTECT исполнит как «цена уже за стопом»
    и закроет позицию по рынку тем же прогоном — это не опечатка, а убыток."""

    def test_short_stop_must_be_above(self):
        self.assertEqual(ap.check_levels(balance_shares=-450, last=51.06,
                                         stop=52.50, take=None), [])
        errs = ap.check_levels(balance_shares=-450, last=51.06, stop=49.0, take=None)
        self.assertTrue(errs and "ВЫШЕ" in errs[0])

    def test_long_stop_must_be_below(self):
        self.assertEqual(ap.check_levels(balance_shares=450, last=51.06,
                                         stop=49.0, take=None), [])
        errs = ap.check_levels(balance_shares=450, last=51.06, stop=52.5, take=None)
        self.assertTrue(errs and "НИЖЕ" in errs[0])

    def test_short_take_must_be_below(self):
        self.assertEqual(ap.check_levels(balance_shares=-450, last=51.06,
                                         stop=52.5, take=49.8), [])
        self.assertTrue(ap.check_levels(balance_shares=-450, last=51.06,
                                        stop=52.5, take=53.0))

    def test_zero_position_rejected(self):
        self.assertTrue(ap.check_levels(balance_shares=0, last=51.06,
                                        stop=52.5, take=None))


class TestBuildRecord(unittest.TestCase):

    def test_shape_matches_place_limits(self):
        """Ключи записи совпадают с теми, что пишет конвейер."""
        rec = _rec()
        for key in ("order_id", "ticker", "strategy", "instrument_uid", "lot",
                    "api_qty", "exit_direction", "stop_units", "stop_nano",
                    "tp_units", "tp_nano", "created", "stop_placed",
                    "stop_order_id", "tp_placed", "tp_order_id", "closed"):
            self.assertIn(key, rec, key)

    def test_quantity_in_lots(self):
        rec = _rec(balance=-450.0)
        self.assertEqual(rec["api_qty"], 450)
        self.assertEqual(rec["exit_direction"], "BUY")

    def test_no_take_counts_as_nothing_to_place(self):
        self.assertTrue(_rec()["tp_placed"])
        self.assertFalse(_rec(take=49.8)["tp_placed"])

    def test_stop_is_pending_for_protect(self):
        self.assertFalse(_rec()["stop_placed"])

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            _rec(strategy="что-то своё")

    def test_manual_survives_cleanup_intraday_does_not(self):
        """Ровно та развилка, которую выбирает пользователь флагом --strategy."""
        self.assertIs(po._is_intraday(_rec(strategy="manual")), False)
        self.assertIs(po._is_intraday(_rec(strategy="intraday_short")), True)

    def test_record_is_marked_as_adopted(self):
        self.assertTrue(_rec()["adopted"])


class TestRegistry(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="adopt-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name, val in (("_LOG_DIR", self.tmp),
                          ("_PENDING_PATH", self.tmp / "pending_stops.json"),
                          ("_LOCK_PATH", self.tmp / "pending_stops.lock")):
            p = mock.patch.object(po, name, val)
            p.start()
            self.addCleanup(p.stop)

    def test_add_then_visible_to_protect(self):
        ap.add_record(ACC, _rec())
        active = [r for r in po._load_pending().get(ACC, []) if not r.get("closed")]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["ticker"], "VTBR")

    def test_duplicate_instrument_refused(self):
        ap.add_record(ACC, _rec())
        with self.assertRaises(ValueError):
            ap.add_record(ACC, _rec(order_id="adopted-second"))

    def test_forget_closes_record(self):
        ap.add_record(ACC, _rec())
        self.assertEqual(ap.forget(ACC, "vtbr"), 1)
        self.assertEqual(ap.active_records(ACC), [])

    def test_forget_missing_is_noop(self):
        self.assertEqual(ap.forget(ACC, "SBER"), 0)

    def test_other_account_untouched(self):
        ap.add_record(ACC, _rec())
        ap.add_record("1639899c", _rec(order_id="adopted-sandbox"))
        self.assertEqual(len(ap.active_records(ACC)), 1)
        self.assertEqual(len(ap.active_records("1639899c")), 1)


class TestDescribe(unittest.TestCase):

    def test_names_the_cleanup_consequence(self):
        self.assertIn("уходит в ночь", ap.describe(_rec(strategy="manual")))
        self.assertIn("CLEANUP закроет", ap.describe(_rec(strategy="intraday_short")))

    def test_shows_risk_to_stop(self):
        out = ap.describe(_rec(balance=-450.0), last=51.06)
        self.assertIn("риск до стопа", out)


class TestCliGuards(unittest.TestCase):
    """Неполные аргументы не должны стоить похода в боевой контур."""

    def test_missing_stop_fails_before_broker(self):
        with mock.patch.object(ap, "_broker",
                               side_effect=AssertionError("брокер не должен вызываться")):
            with self.assertRaises(SystemExit) as cm:
                ap.main(["--prod", "-t", "VTBR"])
        self.assertEqual(cm.exception.code, 2)

    def test_missing_ticker_fails_before_broker(self):
        with mock.patch.object(ap, "_broker",
                               side_effect=AssertionError("брокер не должен вызываться")):
            with self.assertRaises(SystemExit) as cm:
                ap.main(["--prod", "--stop", "52.5"])
        self.assertEqual(cm.exception.code, 2)

    def test_list_does_not_need_ticker(self):
        # stdout гасим: вывод теста иначе попадает в лог post-receive хука и
        # маскирует строку OK, по которой видно, здоров ли деплой
        with mock.patch.object(ap, "_broker", return_value=(None, ACC, "PROD")), \
             mock.patch.object(ap, "active_records", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ap.main(["--prod", "--list"]), 0)


if __name__ == "__main__":
    unittest.main()
