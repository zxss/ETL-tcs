"""
Тесты фазы закрытия внутридневных позиций (services/place_orders --square-off).

Что защищаем:
  * закрываются ТОЛЬКО intraday_long / intraday_short;
  * long_overnight не трогается ни при каких условиях — она держится через ночь
    по замыслу стратегии;
  * записи без поля strategy (созданные до этой версии) не трогаются: закрыть
    по ошибке ночную позицию хуже, чем не закрыть дневную;
  * SL и TP снимаются ДО закрытия позиции, иначе оставшаяся условная заявка
    при касании своей цены откроет обратную позицию;
  * временной гейт и флаг INTRADAY_SQUARE_OFF_ENABLED работают.
"""
from __future__ import annotations

import datetime as dt
import io
import contextlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

from services import place_orders as po  # noqa: E402

MSK = dt.timezone(dt.timedelta(hours=3))
LATE = dt.datetime(2026, 9, 9, 18, 40, tzinfo=MSK)     # после 18:35
EARLY = dt.datetime(2026, 9, 9, 12, 0, tzinfo=MSK)     # задолго до


class FakePosition:
    def __init__(self, uid, shares):
        self.instrument_uid = uid
        self.balance_shares = shares
        self.is_open = shares != 0


class FakeStop:
    def __init__(self, uid, kind, sid):
        self.instrument_uid = uid
        self.kind = kind
        self.stop_order_id = sid


class FakeBroker:
    def __init__(self, positions, stops=()):
        self._positions = positions
        self._stops = list(stops)
        self.cancelled: list[str] = []
        self.market_orders: list[tuple] = []

    def get_positions(self, account_id):
        return self._positions

    def get_active_stop_orders(self, account_id):
        return list(self._stops)

    def cancel_stop_order(self, *, account_id, stop_order_id):
        self.cancelled.append(stop_order_id)

    def find_instrument_by_uid(self, uid):
        from tests.conftest import make_instrument
        return make_instrument(lot=1)


def _rec(ticker, uid, strategy=None, closed=False):
    r = {"order_id": f"o-{ticker}", "ticker": ticker, "instrument_uid": uid,
         "lot": 1, "api_qty": 10, "exit_direction": "SELL",
         "stop_units": 100, "stop_nano": 0, "tp_units": 110, "tp_nano": 0,
         "created": "2026-09-09T10:00:00", "stop_placed": True,
         "stop_order_id": f"sl-{ticker}", "tp_placed": True,
         "tp_order_id": f"tp-{ticker}", "closed": closed}
    if strategy is not None:
        r["strategy"] = strategy
    return r


class SquareOffBase(unittest.TestCase):
    ACCOUNT = "ACC1"

    def run_square_off(self, records, positions, stops=(), *, now=LATE,
                       enabled=True, force=False, dry_run=False):
        broker = FakeBroker(positions, stops)
        saved = {}
        closed_calls = []

        def fake_market_close(b, acc, pos, ticker, writer, env):
            closed_calls.append(ticker)
            return True

        with mock.patch.object(po, "_load_pending",
                               return_value={self.ACCOUNT: records}), \
             mock.patch.object(po, "_save_pending", side_effect=saved.update), \
             mock.patch.object(po, "_market_close", side_effect=fake_market_close), \
             mock.patch.object(po.config, "INTRADAY_SQUARE_OFF_ENABLED", enabled), \
             mock.patch.object(po.config, "INTRADAY_SQUARE_OFF_TIME", "18:35"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                n = po.square_off_intraday(
                    broker, self.ACCOUNT, dry_run=dry_run, no_confirm=True,
                    writer=mock.MagicMock(), env="SANDBOX", force=force, now=now)
        return n, closed_calls, broker, buf.getvalue()


class TestStrategyFiltering(SquareOffBase):

    def test_closes_intraday_long_and_short(self):
        recs = [_rec("AAA", "u1", "intraday_long"),
                _rec("BBB", "u2", "intraday_short")]
        pos = [FakePosition("u1", 10), FakePosition("u2", -10)]
        n, closed, _, _ = self.run_square_off(recs, pos)
        self.assertEqual(n, 2)
        self.assertCountEqual(closed, ["AAA", "BBB"])

    def test_never_touches_long_overnight(self):
        """Ключевой инвариант: ночная стратегия держится через ночь."""
        recs = [_rec("NIGHT", "u1", "long_overnight"),
                _rec("DAY", "u2", "intraday_long")]
        pos = [FakePosition("u1", 10), FakePosition("u2", 10)]
        n, closed, _, out = self.run_square_off(recs, pos)
        self.assertEqual(closed, ["DAY"])
        self.assertNotIn("NIGHT", closed)
        self.assertIn("KEEP", out)

    def test_record_without_strategy_is_skipped(self):
        """Старая запись без стратегии: не знаем — не трогаем."""
        recs = [_rec("OLD", "u1", strategy=None)]
        pos = [FakePosition("u1", 10)]
        n, closed, _, out = self.run_square_off(recs, pos)
        self.assertEqual(n, 0)
        self.assertEqual(closed, [])
        self.assertIn("без стратегии", out)

    def test_position_not_in_registry_is_skipped(self):
        """Позиция, открытая вручную, системе не принадлежит."""
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10), FakePosition("u_manual", 5)]
        n, closed, _, out = self.run_square_off(recs, pos)
        self.assertEqual(closed, ["AAA"])
        self.assertIn("нет в реестре", out)

    def test_closed_records_ignored(self):
        recs = [_rec("DONE", "u1", "intraday_long", closed=True)]
        pos = [FakePosition("u1", 10)]
        n, closed, _, _ = self.run_square_off(recs, pos)
        self.assertEqual(n, 0)

    def test_no_open_positions(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        n, closed, _, out = self.run_square_off(recs, [])
        self.assertEqual(n, 0)
        self.assertIn("Открытых позиций нет", out)


class TestStopCancellation(SquareOffBase):

    def test_stops_cancelled_before_close(self):
        """SL и TP снимаются, иначе оставшийся стоп откроет обратную позицию."""
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        stops = [FakeStop("u1", "STOP_LOSS", "sl1"),
                 FakeStop("u1", "TAKE_PROFIT", "tp1")]
        n, closed, broker, _ = self.run_square_off(recs, pos, stops)
        self.assertEqual(n, 1)
        self.assertCountEqual(broker.cancelled, ["sl1", "tp1"])

    def test_only_own_stops_cancelled(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        stops = [FakeStop("u1", "STOP_LOSS", "mine"),
                 FakeStop("u_other", "STOP_LOSS", "foreign")]
        _, _, broker, _ = self.run_square_off(recs, pos, stops)
        self.assertEqual(broker.cancelled, ["mine"])

    def test_overnight_stops_survive(self):
        recs = [_rec("NIGHT", "u1", "long_overnight")]
        pos = [FakePosition("u1", 10)]
        stops = [FakeStop("u1", "STOP_LOSS", "sl-night")]
        n, _, broker, _ = self.run_square_off(recs, pos, stops)
        self.assertEqual(n, 0)
        self.assertEqual(broker.cancelled, [])


class TestGates(SquareOffBase):

    def test_disabled_flag_blocks(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        n, closed, _, out = self.run_square_off(recs, pos, enabled=False)
        self.assertEqual(n, 0)
        self.assertEqual(closed, [])
        self.assertIn("отключено", out)

    def test_too_early_blocks(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        n, closed, _, out = self.run_square_off(recs, pos, now=EARLY)
        self.assertEqual(n, 0)
        self.assertIn("рано", out)

    def test_force_overrides_time_and_flag(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        n, closed, _, _ = self.run_square_off(recs, pos, now=EARLY,
                                              enabled=False, force=True)
        self.assertEqual(n, 1)
        self.assertEqual(closed, ["AAA"])

    def test_dry_run_changes_nothing(self):
        recs = [_rec("AAA", "u1", "intraday_long")]
        pos = [FakePosition("u1", 10)]
        stops = [FakeStop("u1", "STOP_LOSS", "sl1")]
        n, closed, broker, out = self.run_square_off(recs, pos, stops, dry_run=True)
        self.assertEqual(n, 0)
        self.assertEqual(closed, [])
        self.assertEqual(broker.cancelled, [])
        self.assertIn("DRY-RUN", out)


class TestHelpers(unittest.TestCase):

    def test_square_off_time_parsing(self):
        with mock.patch.object(po.config, "INTRADAY_SQUARE_OFF_TIME", "17:05"):
            self.assertEqual(po._square_off_time(), dt.time(17, 5))

    def test_bad_time_falls_back(self):
        with mock.patch.object(po.config, "INTRADAY_SQUARE_OFF_TIME", "мусор"):
            self.assertEqual(po._square_off_time(), dt.time(18, 35))

    def test_is_intraday(self):
        self.assertIs(po._is_intraday({"strategy": "intraday_long"}), True)
        self.assertIs(po._is_intraday({"strategy": "intraday_short"}), True)
        self.assertIs(po._is_intraday({"strategy": "long_overnight"}), False)
        self.assertIsNone(po._is_intraday({}))
        self.assertIsNone(po._is_intraday({"strategy": ""}))

    def test_intraday_set_matches_config_strategies(self):
        """Множество внутридневных не должно разойтись с боевым меню стратегий."""
        import config
        for st in po.INTRADAY_STRATEGIES:
            self.assertIn(st, config.VALIDATION_STRATS)
        self.assertNotIn("long_overnight", po.INTRADAY_STRATEGIES)


if __name__ == "__main__":
    unittest.main()
