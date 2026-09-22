"""
Тесты журнала исполнения (audit/execution_audit.py + врезка в stage2_demo).

Что здесь защищается: критерий ТЗ Этапа 2 «расхождение фактического
проскальзывания с расчётным ≤ 20%». Обе половины сравнения раньше были пустыми —
факт не писал никто, расчёт не передавался. Тесты фиксируют, что обе теперь
заполняются, и что сбой журнала виден в вердикте фазы, а не только в логе.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                            # noqa: E402
from audit import execution_audit as ea                  # noqa: E402
from services import stage2_demo as s2                   # noqa: E402
from services.broker.tinkoff_base import parse_order_state  # noqa: E402


class TestOrderStateCapturesFill(unittest.TestCase):
    """Цена заливки приходит от API — раньше парсер её выбрасывал."""

    def _state(self, **over):
        d = {"orderId": "o-1", "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
             "lotsRequested": 2, "lotsExecuted": 2,
             "averagePositionPrice": {"units": "142", "nano": 30_000_000},
             "executedOrderPrice": {"units": "1420", "nano": 300_000_000},
             "executedCommission": {"units": "0", "nano": 568_000_000}}
        d.update(over)
        return parse_order_state(d)

    def test_executed_price_parsed(self):
        self.assertAlmostEqual(self._state().executed_price, 142.03, places=6)

    def test_commission_parsed(self):
        self.assertAlmostEqual(self._state().executed_commission, 0.568, places=6)

    def test_zero_money_means_not_filled_not_price_zero(self):
        """Нулевой MoneyValue у неисполненной заявки нельзя записать как цену:
        проскальзывание вышло бы −100%."""
        st = self._state(averagePositionPrice={"units": "0", "nano": 0})
        self.assertIsNone(st.executed_price)

    def test_missing_fields_do_not_raise(self):
        st = parse_order_state({"orderId": "o", "executionReportStatus": "NEW"})
        self.assertIsNone(st.executed_price)
        self.assertEqual(st.lots_executed, 0)

    def test_raw_response_is_kept(self):
        """Сырой ответ — страховка на случай, если поле цены выбрано неверно."""
        self.assertIn("averagePositionPrice", self._state().raw)


class TestSlippageSign(unittest.TestCase):
    """Знак проскальзывания обязан быть одинаково «плохим» для лонга и шорта."""

    def _capture(self, *, side, requested, filled):
        conn = mock.MagicMock()
        with mock.patch.object(ea, "_UPDATE_FILL_SQL", "SQL"):
            ea.record_fill(conn, order_id="o", filled_price=filled,
                           filled_at=dt.datetime.now(), requested_price=requested,
                           side=side, qty_shares=10)
        return conn.cursor.return_value.__enter__.return_value.execute.call_args[0][1]

    def test_buy_higher_is_positive_slippage(self):
        self.assertGreater(self._capture(side="BUY", requested=100.0,
                                         filled=101.0)["slippage_pct"], 0)

    def test_sell_lower_is_positive_slippage(self):
        """Шорт залили дешевле заявки — это тоже потеря, знак обязан совпасть
        с лонгом, иначе среднее по портфелю схлопнется к нулю."""
        self.assertGreater(self._capture(side="SELL", requested=100.0,
                                         filled=99.0)["slippage_pct"], 0)

    def test_favourable_fill_is_negative(self):
        self.assertLess(self._capture(side="BUY", requested=100.0,
                                      filled=99.0)["slippage_pct"], 0)

    def test_zero_requested_price_is_refused(self):
        """Деление на нулевую цену заявки дало бы бесконечность в журнале."""
        conn = mock.MagicMock()
        ok = ea.record_fill(conn, order_id="o", filled_price=10.0,
                            filled_at=dt.datetime.now(), requested_price=0.0,
                            side="BUY")
        self.assertFalse(ok)


class TestWritersReportFailure(unittest.TestCase):
    """Журнал, который молча не пишется, хуже отсутствующего."""

    def test_record_intent_returns_false_on_error(self):
        conn = mock.MagicMock()
        conn.cursor.side_effect = RuntimeError('relation does not exist')
        self.assertFalse(ea.record_intent(conn, order_id="o", account_env="SANDBOX",
                                          asof_date=dt.date.today(), ticker="SBER",
                                          strategy="s", side="BUY", requested_price=1.0))
        conn.rollback.assert_called_once()

    def test_record_intent_returns_true_on_success(self):
        self.assertTrue(ea.record_intent(mock.MagicMock(), order_id="o",
                                         account_env="SANDBOX", asof_date=dt.date.today(),
                                         ticker="SBER", strategy="s", side="BUY",
                                         requested_price=1.0))


class TestRecordFills(unittest.TestCase):
    PLAN = [{"ticker": "SBER", "direction": "LONG", "entry_price": 100.0,
             "lot_size": 10}]

    def test_filled_order_is_recorded(self):
        fills = [{"order_id": "o", "ticker": "SBER", "lots_executed": 2,
                  "executed_price": 101.0, "executed_commission": 0.8}]
        with mock.patch.object(ea, "record_fill", return_value=True) as rf:
            done, miss = s2._record_fills(mock.MagicMock(), fills, self.PLAN)
        self.assertEqual((done, miss), (1, 0))
        self.assertEqual(rf.call_args[1]["qty_shares"], 20)

    def test_partial_fill_is_recorded_too(self):
        """Заливка на половину объёма — это факт со своей ценой, а не «не исполнено»."""
        fills = [{"order_id": "o", "ticker": "SBER", "lots_executed": 1,
                  "executed_price": 101.0}]
        with mock.patch.object(ea, "record_fill", return_value=True):
            done, _ = s2._record_fills(mock.MagicMock(), fills, self.PLAN)
        self.assertEqual(done, 1)

    def test_unfilled_order_is_skipped(self):
        fills = [{"order_id": "o", "ticker": "SBER", "lots_executed": 0,
                  "executed_price": None}]
        with mock.patch.object(ea, "record_fill") as rf:
            done, miss = s2._record_fills(mock.MagicMock(), fills, self.PLAN)
        self.assertEqual((done, miss), (0, 0))
        rf.assert_not_called()

    def test_short_is_recorded_as_sell(self):
        plan = [{**self.PLAN[0], "direction": "SHORT"}]
        fills = [{"order_id": "o", "ticker": "SBER", "lots_executed": 1,
                  "executed_price": 99.0}]
        with mock.patch.object(ea, "record_fill", return_value=True) as rf:
            s2._record_fills(mock.MagicMock(), fills, plan)
        self.assertEqual(rf.call_args[1]["side"], "SELL")

    def test_write_failure_is_counted(self):
        fills = [{"order_id": "o", "ticker": "SBER", "lots_executed": 1,
                  "executed_price": 101.0}]
        with mock.patch.object(ea, "record_fill", return_value=False):
            done, miss = s2._record_fills(mock.MagicMock(), fills, self.PLAN)
        self.assertEqual((done, miss), (0, 1))


class TestExpectedSlippageIsPassed(unittest.TestCase):
    def test_intent_carries_expected_slippage(self):
        """Без расчётной половины сравнивать факт не с чем."""
        placed = [{"ticker": "SBER", "order_id": "o", "strategy": "long_overnight"}]
        plan = [{"ticker": "SBER", "direction": "LONG", "entry_price": 100.0}]
        with mock.patch.object(ea, "record_intent", return_value=True) as ri:
            s2._record_intents(mock.MagicMock(), placed, plan, run_id="r",
                               phase="ORDER", env="SANDBOX", day=dt.date.today())
        self.assertEqual(ri.call_args[1]["expected_slippage_pct"],
                         config.EXPECTED_SLIPPAGE_PCT)

    def test_default_is_decomposition_of_round_trip_cost(self):
        """0.024% = (0.128 − 2×0.04) / 2 — не новое число, а разложение того,
        которому уже доверяет модель."""
        self.assertAlmostEqual(
            config.EXPECTED_SLIPPAGE_PCT,
            (config.TFT_COST_RT - 2 * config.BROKER_COMMISSION_PCT) / 2, places=4)

    def test_failures_are_counted(self):
        placed = [{"ticker": "SBER", "order_id": "o"}]
        with mock.patch.object(ea, "record_intent", return_value=False):
            miss = s2._record_intents(mock.MagicMock(), placed, [], run_id="r",
                                      phase="ORDER", env="SANDBOX",
                                      day=dt.date.today())
        self.assertEqual(miss, 1)


class TestBrokenJournalIsVisible(unittest.TestCase):
    """Ноль в аудите не должен быть неотличим от честного дня без сделок."""

    def test_unavailable_journal_is_flagged(self):
        with mock.patch.object(s2.database, "get_db_connection",
                               side_effect=RuntimeError("нет таблицы")):
            stats = s2._execution_stats(dt.date.today())
        self.assertFalse(stats["available"])
        self.assertEqual(stats["orders"], 0)

    def test_available_journal_with_no_orders_is_not_flagged(self):
        conn = mock.MagicMock()
        conn.__enter__.return_value.cursor.return_value.__enter__ \
            .return_value.fetchone.return_value = (0, 0, None, None)
        with mock.patch.object(s2.database, "get_db_connection", return_value=conn):
            stats = s2._execution_stats(dt.date.today())
        self.assertTrue(stats["available"])
        self.assertEqual(stats["orders"], 0)


class TestAccountEnvScoping(unittest.TestCase):
    """Параллельный турнирный контур (23.09.2026): песочница и боевой счёт
    пишут в общую execution_audit, поэтому дневная сводка и счётчик сделок
    обязаны фильтроваться по account_env, иначе один запрос на один день
    молча просуммирует SANDBOX и PROD."""

    def _conn(self, row):
        conn = mock.MagicMock()
        cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = row
        return conn, cur

    def test_execution_stats_filters_by_env(self):
        conn, cur = self._conn((3, 2, None, None))
        with mock.patch.object(s2.database, "get_db_connection", return_value=conn):
            stats = s2._execution_stats(dt.date.today(), env="PROD")
        self.assertEqual(stats["orders"], 3)
        self.assertEqual(cur.execute.call_args[0][1][1], "PROD")

    def test_execution_stats_defaults_to_sandbox(self):
        conn, cur = self._conn((0, 0, None, None))
        with mock.patch.object(s2.database, "get_db_connection", return_value=conn):
            s2._execution_stats(dt.date.today())
        self.assertEqual(cur.execute.call_args[0][1][1], "SANDBOX")

    def test_trades_since_counts_filled_from_start_day(self):
        conn, cur = self._conn((14,))
        with mock.patch.object(s2.database, "get_db_connection", return_value=conn):
            n = s2._trades_since(dt.date(2026, 9, 18), env="PROD")
        self.assertEqual(n, 14)
        args = cur.execute.call_args[0][1]
        self.assertEqual(args, (dt.date(2026, 9, 18), "PROD"))

    def test_trades_since_is_quiet_on_db_failure(self):
        """Как и _execution_stats: сбой журнала не должен ронять фазу — счётчик
        просто возвращает 0, а не бросает наружу."""
        with mock.patch.object(s2.database, "get_db_connection",
                               side_effect=RuntimeError("нет таблицы")):
            self.assertEqual(s2._trades_since(dt.date.today(), env="PROD"), 0)


if __name__ == "__main__":
    unittest.main()
