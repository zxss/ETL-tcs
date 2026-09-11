"""
Хотфикс 11.09, брокерский клиент: повтор при HTTP 429, интервал между заявками,
разночтение полей цены в ответах T-Invest. Сеть не трогается.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                            # noqa: E402
from services.broker import tinkoff_base as tb           # noqa: E402
from services.broker.base import BrokerError             # noqa: E402


class _Resp:
    def __init__(self, payload: bytes):
        self._p = payload

    def read(self):
        return self._p

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client():
    return types.SimpleNamespace(base="https://sandbox.example/rest", SVC="svc.v1",
                                 token="t", timeout=1, ssl_ctx=None)


def _http(code, hdrs=None):
    return urllib.error.HTTPError("u", code, "err", hdrs or {}, None)


READ = "SandboxService/GetSandboxPositions"
ORDER = "SandboxService/PostSandboxOrder"


class TestRetryOn429(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(config, "BROKER_429_RETRIES", 3)
        p.start()
        self.addCleanup(p.stop)

    def test_retries_then_succeeds(self):
        """10.09 две ночные заявки из четырёх пропали на 429 без повтора."""
        with mock.patch("urllib.request.urlopen",
                        side_effect=[_http(429), _http(429), _Resp(b'{"ok": 1}')]), \
             mock.patch("time.sleep") as sl:
            self.assertEqual(tb.TinkoffRestBase._post(_client(), READ, {}), {"ok": 1})
        self.assertEqual([c.args[0] for c in sl.call_args_list], [1.0, 2.0])

    def test_gives_up_after_limit(self):
        with mock.patch("urllib.request.urlopen", side_effect=[_http(429)] * 4) as u, \
             mock.patch("time.sleep"):
            with self.assertRaises(BrokerError):
                tb.TinkoffRestBase._post(_client(), READ, {})
        self.assertEqual(u.call_count, 4)

    def test_other_errors_not_retried(self):
        with mock.patch("urllib.request.urlopen", side_effect=[_http(400)]) as u, \
             mock.patch("time.sleep") as sl:
            with self.assertRaises(BrokerError):
                tb.TinkoffRestBase._post(_client(), READ, {})
        self.assertEqual(u.call_count, 1)
        sl.assert_not_called()

    def test_retry_after_header_respected(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=[_http(429, {"Retry-After": "3"}), _Resp(b"{}")]), \
             mock.patch("time.sleep") as sl:
            tb.TinkoffRestBase._post(_client(), READ, {})
        self.assertEqual(sl.call_args_list[0].args[0], 3.0)

    def test_retry_sends_same_order_id(self):
        """Повтор заявки безопасен, только если ключ идемпотентности тот же."""
        bodies = []

        def fake(req, **kw):
            bodies.append(req.data)
            if len(bodies) == 1:
                raise _http(429)
            return _Resp(b"{}")
        with mock.patch("urllib.request.urlopen", side_effect=fake), \
             mock.patch("time.sleep"):
            tb.TinkoffRestBase._post(_client(), ORDER, {"orderId": "abc"})
        self.assertEqual(bodies[0], bodies[1])


class TestThrottle(unittest.TestCase):
    def test_order_calls_are_spaced(self):
        tb._last_order_call = 0.0
        with mock.patch.object(config, "BROKER_ORDER_MIN_INTERVAL_SEC", 0.25), \
             mock.patch("time.monotonic", side_effect=[100.0, 100.0, 100.1, 100.35]), \
             mock.patch("time.sleep") as sl:
            tb._throttle(ORDER)
            tb._throttle(ORDER)
        self.assertEqual(len(sl.call_args_list), 1)
        self.assertAlmostEqual(sl.call_args_list[0].args[0], 0.15, places=6)

    def test_reads_are_not_throttled(self):
        tb._last_order_call = 10 ** 9
        with mock.patch("time.sleep") as sl:
            tb._throttle(READ)
        sl.assert_not_called()


class TestPriceFieldSemantics(unittest.TestCase):
    def test_post_order_response_gives_no_price(self):
        """В ответе PostOrder executedOrderPrice — цена за штуку, а
        averagePositionPrice нет. Цену берём только из GetOrderState."""
        st = tb.parse_order_state({
            "orderId": "o", "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
            "lotsRequested": "32", "lotsExecuted": "32",
            "executedOrderPrice": {"units": "308", "nano": 650000000},
            "totalOrderAmount": {"units": "9876", "nano": 800000000}})
        self.assertIsNone(st.executed_price)
        self.assertAlmostEqual(st.executed_amount, 9876.8, places=6)

    def test_order_state_gives_average_price(self):
        st = tb.parse_order_state({
            "orderId": "o", "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
            "lotsRequested": "32", "lotsExecuted": "32",
            "averagePositionPrice": {"units": "310", "nano": 600000000},
            "executedOrderPrice": {"units": "9939", "nano": 200000000},
            "totalOrderAmount": {"units": "9939", "nano": 200000000}})
        self.assertAlmostEqual(st.executed_price, 310.6, places=6)


if __name__ == "__main__":
    unittest.main()
