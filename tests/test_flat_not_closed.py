"""Нулевой баланс у брокера — ещё не закрытие позиции.

23.09.2026, боевой счёт «Робот». Тейк по UPRO сработал в 22:57; с 22:58
GetPositions показывал ноль, и PROTECT решил, что сделка закрыта: снял
STOP_LOSS и закрыл запись реестра. В 23:53 позиция вернулась — фактическая
продажа прошла только утром в 09:15. Ночь позиция простояла без стопа и вне
учёта, а утром CLOSE увидел незакрытый UPRO, уронил тест в halt, и фаза ORDER
была пропущена.

Признак закрытия — сделка выхода в операциях, а не нулевой баланс.
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import place_orders as po                    # noqa: E402
from services.broker.base import BrokerError               # noqa: E402

ACC = "2018145468"
UID = "upro-uid"


class Pos:
    def __init__(self, uid, shares):
        self.instrument_uid, self.balance_shares = uid, shares

    @property
    def is_open(self):
        return self.balance_shares != 0


def _trade(side, qty):
    return types.SimpleNamespace(instrument_uid=UID, side=side, quantity=qty,
                                 price=1.0, at="2026-09-23T15:48:53Z")


class Broker:
    """Брокер, у которого баланс и операции задаются раздельно — как в проде."""

    def __init__(self, *, shares=0.0, trades=(), trades_fail=False, stops=()):
        self.positions = [Pos(UID, shares)] if shares else []
        self.trades = list(trades)
        self.trades_fail = trades_fail
        self.active_stops = list(stops)
        self.cancelled: list[str] = []

    def get_positions(self, a):
        return self.positions

    def get_active_stop_orders(self, a):
        return list(self.active_stops)

    def get_stop_order_history(self, a, since):
        return []

    def get_trades(self, account_id, since, instrument_uid=None):
        if self.trades_fail:
            raise BrokerError("OperationsService недоступен")
        return list(self.trades), 0.0

    def cancel_stop_order(self, *, account_id, stop_order_id):
        self.cancelled.append(stop_order_id)
        self.active_stops = [s for s in self.active_stops
                             if s.stop_order_id != stop_order_id]

    def find_instrument(self, tk):
        return types.SimpleNamespace(instrument_uid=UID, ticker=tk, lot=1)


def _stop(sid, kind="STOP_LOSS"):
    return types.SimpleNamespace(instrument_uid=UID, kind=kind, stop_order_id=sid)


def _rec(**over):
    """Запись UPRO в том виде, в каком её оставил OVERNIGHT 23.09."""
    r = {"order_id": "84740502355", "ticker": "UPRO", "strategy": "long_overnight",
         "instrument_uid": UID, "lot": 1000, "api_qty": 19, "exit_direction": "SELL",
         "stop_units": 0, "stop_nano": 993_500_000,
         "tp_units": 1, "tp_nano": 29_000_000,
         "created": "2026-09-23T18:48:52", "closed": False,
         "stop_placed": True, "stop_order_id": "sl-upro",
         "tp_placed": True, "tp_order_id": "tp-upro"}
    r.update(over)
    return r


def run(broker, records):
    saved: dict = {}
    with mock.patch.object(po, "_load_pending", return_value={ACC: records}), \
         mock.patch.object(po, "_save_pending", side_effect=saved.update), \
         contextlib.redirect_stdout(io.StringIO()) as out:
        rep: dict = {}
        po.attach_stops(broker, ACC, dry_run=False, writer=mock.MagicMock(),
                        env="PROD", report=rep)
    return rep, out.getvalue()


class TestExitConfirmed(unittest.TestCase):

    def test_no_exit_trade_is_not_confirmed(self):
        br = Broker(trades=[_trade("BUY", 19000)])          # только вход
        self.assertIs(po.exit_confirmed(br, ACC, _rec()), False)

    def test_exit_trade_confirms(self):
        br = Broker(trades=[_trade("BUY", 19000), _trade("SELL", 19000)])
        self.assertIs(po.exit_confirmed(br, ACC, _rec()), True)

    def test_short_confirms_on_buy_back(self):
        rec = _rec(exit_direction="BUY")
        br = Broker(trades=[_trade("SELL", 19000), _trade("BUY", 19000)])
        self.assertIs(po.exit_confirmed(br, ACC, rec), True)

    def test_operations_unavailable_is_unknown(self):
        # stdout гасим: предупреждение иначе попадает в лог post-receive хука
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(po.exit_confirmed(Broker(trades_fail=True), ACC, _rec()))

    def test_broker_without_operations_is_unknown(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(po.exit_confirmed(object(), ACC, _rec()))


class TestFlatWithoutExitKeepsProtection(unittest.TestCase):
    """Ровно 23.09: баланс ноль, сделки выхода нет."""

    def test_record_survives(self):
        rec = _rec()
        br = Broker(shares=0, trades=[_trade("BUY", 19000)], stops=[_stop("sl-upro")])
        rep, _ = run(br, [rec])
        self.assertFalse(rec["closed"], "запись закрыта при неподтверждённом выходе")
        self.assertEqual(rep["flat_unconfirmed"], ["UPRO"])
        self.assertEqual(rep["closed_externally"], [])

    def test_stop_is_not_cancelled(self):
        br = Broker(shares=0, trades=[_trade("BUY", 19000)], stops=[_stop("sl-upro")])
        run(br, [_rec()])
        self.assertEqual(br.cancelled, [], "снят стоп с живой позиции")

    def test_position_returns_and_is_still_protected(self):
        """Полная вчерашняя последовательность: есть → ноль → вернулась."""
        rec = _rec()
        for shares in (19000, 0, 19000):
            br = Broker(shares=shares, trades=[_trade("BUY", 19000)],
                        stops=[_stop("sl-upro"), _stop("tp-upro", "TAKE_PROFIT")])
            run(br, [rec])
            self.assertFalse(rec["closed"])
            self.assertEqual(br.cancelled, [])
        self.assertTrue(rec["stop_placed"], "защита должна остаться на месте")


class TestGenuineCloseStillWorks(unittest.TestCase):

    def test_exit_trade_closes_record_and_cancels_legs(self):
        rec = _rec()
        br = Broker(shares=0, trades=[_trade("BUY", 19000), _trade("SELL", 19000)],
                    stops=[_stop("sl-upro")])
        rep, _ = run(br, [rec])
        self.assertTrue(rec["closed"], "подтверждённый выход обязан закрывать запись")
        self.assertIn("sl-upro", br.cancelled)
        self.assertEqual(rep["flat_unconfirmed"], [])

    def test_unknown_operations_falls_back_to_old_behaviour(self):
        """Нечем подтвердить — не копим зомби-записи с живыми стопами."""
        rec = _rec()
        br = Broker(shares=0, trades_fail=True, stops=[_stop("sl-upro")])
        rep, _ = run(br, [rec])
        self.assertTrue(rec["closed"])
        self.assertEqual(rep["flat_unconfirmed"], [])


if __name__ == "__main__":
    unittest.main()
