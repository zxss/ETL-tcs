"""Учёт капитала рукава levmom: ГО не задваивается (числа брокера 01.10.2026)."""
import unittest
from types import SimpleNamespace

from research.levmom import sleeve


def q(v):
    u = int(v)
    return {"currency": "rub", "units": str(u), "nano": int(round((v - u) * 1e9))}


class FakeBr:
    def __init__(self, money, blocked, vm, orders=()):
        self.money, self.blocked, self.vm, self.orders = money, blocked, vm, list(orders)

    def _post(self, method, payload=None):
        if method.endswith("GetSandboxPositions"):
            return {"money": [q(self.money)], "blocked": [q(self.blocked)] if self.blocked else []}
        if method.endswith("GetSandboxOrders"):
            return {"orders": self.orders}
        if method.endswith("GetSandboxPortfolio"):
            return {"positions": [{"varMargin": q(self.vm)}]}
        raise AssertionError(method)

    def get_last_price(self, uid):
        return 165.24


class Equity(unittest.TestCase):
    def setUp(self):
        self._fi, self._pos = sleeve.fund_instrument, sleeve.positions
        sleeve.fund_instrument = lambda br: SimpleNamespace(instrument_uid="TMON")
        sleeve.positions = lambda br, acc: {"TMON": 916.0}

    def tearDown(self):
        sleeve.fund_instrument, sleeve.positions = self._fi, self._pos

    def test_go_not_double_counted(self):
        eq = sleeve.equity(FakeBr(851_298, 458_690, -29_062), "acc")
        self.assertAlmostEqual(eq["equity"], 851_298 + 916 * 165.24 - 29_062, places=0)
        self.assertLess(eq["equity"], 1_000_000)               # не 1,46 млн
        self.assertAlmostEqual(eq["go_broker"], 458_690, places=0)

    def test_buy_order_reserve_returned(self):
        order = {"direction": "ORDER_DIRECTION_BUY", "initialOrderPrice": q(20_000)}
        eq = sleeve.equity(FakeBr(80_000, 20_000, 0, [order]), "acc")
        self.assertAlmostEqual(eq["money"], 100_000, places=0)
        self.assertAlmostEqual(eq["go_broker"], 0, places=0)


if __name__ == "__main__":
    unittest.main()
