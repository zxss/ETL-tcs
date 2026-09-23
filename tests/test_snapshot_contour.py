"""Снимок счёта обязан спрашивать контур у клиента, а не у вызывающего.

23.09.2026, боевой счёт: PARK упал на
`SandboxService/GetSandboxOrders … HTTP 404 {"code":5,"message":"Account not
found"}` — services/stage2_demo.py звал снимок с жёстким sandbox=True. Тот же
вызов стоит в PREP, ORDER, CLOSE, CLEANUP и OVERNIGHT, то есть на боевом
контуре не работала ни одна фаза. Хуже того, фаза списала 404 в
PASS_WITH_WARNINGS и отчиталась успехом.
"""
from __future__ import annotations

import unittest

from services import account_status, stage2_balance
from services.broker import TinkoffProdClient, TinkoffSandboxClient

PROD_M = "OrdersService/GetOrders"
SBX_M = "SandboxService/GetSandboxOrders"


class FakeClient:
    """Минимальный двойник: пишет вызванные методы, возвращает пустые ответы."""

    def __init__(self, orders_method: str):
        self.ORDERS_METHOD = orders_method
        self.calls: list[str] = []

    def _post(self, method: str, payload=None):
        self.calls.append(method)
        if method.endswith("GetOrders") or method.endswith("GetSandboxOrders"):
            return {"orders": []}
        if method.endswith("GetStopOrders"):
            return {"stopOrders": []}
        return {}


class TestOrdersMethodIsAClassAttribute(unittest.TestCase):

    def test_prod_client(self):
        self.assertEqual(TinkoffProdClient.ORDERS_METHOD, PROD_M)

    def test_sandbox_client(self):
        self.assertEqual(TinkoffSandboxClient.ORDERS_METHOD, SBX_M)


class TestSnapshotFollowsClient(unittest.TestCase):

    def test_prod_client_never_calls_sandbox_service(self):
        c = FakeClient(PROD_M)
        snap = account_status.snapshot(c, "2018145468")
        self.assertIn(PROD_M, c.calls)
        self.assertFalse([m for m in c.calls if m.startswith("SandboxService/")])
        self.assertFalse(snap["sandbox"])

    def test_sandbox_client_uses_sandbox_service(self):
        c = FakeClient(SBX_M)
        snap = account_status.snapshot(c, "1639899c")
        self.assertIn(SBX_M, c.calls)
        self.assertTrue(snap["sandbox"])

    def test_explicit_wrong_flag_does_not_win(self):
        """Ровно тот случай, что уронил PARK: боевой клиент + sandbox=True."""
        c = FakeClient(PROD_M)
        with self.assertLogs("services.account_status", level="WARNING"):
            snap = account_status.snapshot(c, "2018145468", sandbox=True)
        self.assertIn(PROD_M, c.calls)
        self.assertFalse([m for m in c.calls if m.startswith("SandboxService/")])
        self.assertFalse(snap["sandbox"])

    def test_balance_capture_defaults_to_client(self):
        c = FakeClient(PROD_M)
        stage2_balance.capture(c, "2018145468", run_id="r", phase="PARK")
        self.assertFalse([m for m in c.calls if m.startswith("SandboxService/")])


class TestPhasesDoNotPinTheContour(unittest.TestCase):

    def test_stage2_demo_passes_no_sandbox_flag(self):
        import inspect
        from services import stage2_demo
        src = inspect.getsource(stage2_demo._snapshot_account)
        self.assertNotIn("sandbox=True", src)


if __name__ == "__main__":
    unittest.main()
