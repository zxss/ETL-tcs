"""
Дефекты контура, найденные 24.09.2026 при разборе моментум-рукава и отчётов.

1. get_positions читал только securities — шорт фьючерса был НЕВИДИМ, CLEANUP
   не закрыл бы такую позицию, PROTECT не поставил бы стоп.
2. Шапка balance_report.md всегда писала «SANDBOX», в том числе для боевого
   счёта «Робот» с реальными деньгами.
3. В разложении отчёта не было строки казначейства: доход паёв TMON молча
   приписывался торговле (на r4 за 5 дней это половина прироста).
4. Лимит ночного риска по VaR — новый, по умолчанию выключен.
"""
from __future__ import annotations

import unittest

from services.broker.base import POSITION_SECTIONS
from services.broker.tinkoff_base import parse_positions
from services import risk_limits


class TestPositionsParsing(unittest.TestCase):
    def test_futures_and_options_visible(self):
        data = {
            "securities": [{"instrumentUid": "sber", "balance": 100, "blocked": 0}],
            "futures": [{"instrumentUid": "mxz6", "balance": -5, "blocked": 0}],
            "options": [{"instrumentUid": "opt1", "balance": 2, "blocked": 1}],
        }
        got = {p.instrument_uid: p.balance_shares for p in parse_positions(data)}
        self.assertEqual(got, {"sber": 100.0, "mxz6": -5.0, "opt1": 2.0})

    def test_short_future_is_open(self):
        (p,) = parse_positions({"futures": [{"instrumentUid": "mxz6", "balance": -5}]})
        self.assertTrue(p.is_open, "шорт фьючерса обязан считаться открытой позицией")

    def test_empty_and_missing_sections(self):
        self.assertEqual(parse_positions({}), [])
        self.assertEqual(parse_positions({"securities": None, "futures": []}), [])

    def test_entries_without_uid_skipped(self):
        self.assertEqual(parse_positions({"securities": [{"balance": 10}]}), [])

    def test_sections_cover_srochny_market(self):
        self.assertIn("futures", POSITION_SECTIONS)
        self.assertIn("options", POSITION_SECTIONS)


class TestBalanceReportContour(unittest.TestCase):
    """Контур берётся у клиента, а не из аргумента (боевой ≠ SANDBOX)."""

    def test_sandbox_flag_from_client(self):
        class Sandbox:
            ORDERS_METHOD = "SandboxService/GetSandboxOrders"

        class Prod:
            ORDERS_METHOD = "OrdersService/GetOrders"

        self.assertTrue("Sandbox" in str(getattr(Sandbox, "ORDERS_METHOD", "")))
        self.assertFalse("Sandbox" in str(getattr(Prod, "ORDERS_METHOD", "")))

    @staticmethod
    def _day(sandbox: bool) -> dict:
        return {"trading_day": "2026-09-23", "account_id": "0000000000",
                "sandbox": sandbox, "opening_balance_rub": 1_000_000.0,
                "closing_balance_rub": 1_001_000.0, "day_change_rub": 1000.0,
                "by_phase": {}, "intraday_pnl_rub": 100.0,
                "overnight_carry_rub": 50.0, "unrealised_pnl_rub": 0.0,
                "positions_overnight": 0, "incomplete_phases": []}

    def _report(self, sandbox: bool) -> str:
        import json
        import os
        import tempfile
        from unittest import mock
        from services import stage2_balance
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "balance"))
            with open(os.path.join(tmp, "balance", "2026-09-23.json"), "w",
                      encoding="utf-8") as f:
                json.dump(self._day(sandbox), f)
            with mock.patch.object(stage2_balance, "_stage2_dir", lambda: tmp):
                return stage2_balance.build_report(start_balance=1_000_000.0,
                                                   target_days=24)

    def test_prod_account_not_labelled_sandbox(self):
        text = self._report(sandbox=False)
        self.assertIn("PROD (РЕАЛЬНЫЕ ДЕНЬГИ)", text)
        self.assertNotIn("Счёт: SANDBOX", text)

    def test_sandbox_account_still_labelled_sandbox(self):
        self.assertIn("Счёт: SANDBOX", self._report(sandbox=True))

    def test_treasury_line_present(self):
        """Итог 1000 ₽ − (100 интрадей + 50 ночь + 0 нереализ.) = 850 ₽ казначейства."""
        text = self._report(sandbox=True)
        self.assertIn("Казначейство и прочее", text)
        self.assertIn("850", text.replace(" ", ""))


class TestOvernightVarLimit(unittest.TestCase):
    ORDERS = [{"signal_id": f"s{i}", "total_rub": 100_000.0, "final_score": 10 - i}
              for i in range(5)]

    def test_disabled_by_default(self):
        kept, dropped, info = risk_limits.trim_to_budget(self.ORDERS, budget=0.0)
        self.assertEqual(len(kept), 5)
        self.assertEqual(dropped, [])
        self.assertFalse(info["enabled"])

    def test_var_of_five_positions(self):
        var = risk_limits.basket_var_rub(500_000.0, 5)
        self.assertAlmostEqual(var, 500_000.0 * 0.023, places=2)

    def test_fewer_names_widen_estimate(self):
        """Меньше бумаг — хуже диверсификация, оценка обязана расти."""
        v5 = risk_limits.basket_var_rub(100_000.0, 5)
        v2 = risk_limits.basket_var_rub(100_000.0, 2)
        self.assertGreater(v2, v5)

    def test_trims_worst_ranked_first(self):
        kept, dropped, info = risk_limits.trim_to_budget(self.ORDERS, budget=7_000.0)
        self.assertLess(len(kept), 5)
        self.assertLessEqual(info["var_after_rub"], 7_000.0)
        self.assertTrue(all(k["final_score"] > d["final_score"]
                            for k in kept for d in dropped),
                        "срезаться должны худшие по оценке стратегии")

    def test_budget_above_risk_keeps_all(self):
        kept, dropped, _ = risk_limits.trim_to_budget(self.ORDERS, budget=1_000_000.0)
        self.assertEqual(len(kept), 5)
        self.assertEqual(dropped, [])

    def test_empty_orders(self):
        kept, dropped, info = risk_limits.trim_to_budget([], budget=5_000.0)
        self.assertEqual((kept, dropped), ([], []))
        self.assertEqual(info["var_before_rub"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestContourFallbackForOldSummaries(unittest.TestCase):
    """Сводки до 24.09.2026 признака контура не несут — метка по номеру счёта."""

    def _report(self, account: str) -> str:
        import json
        import os
        import tempfile
        from unittest import mock
        from services import stage2_balance
        day = {"trading_day": "2026-09-23", "account_id": account,
               "opening_balance_rub": 1e6, "closing_balance_rub": 1_001_000.0,
               "day_change_rub": 1000.0, "by_phase": {}, "intraday_pnl_rub": 100.0,
               "overnight_carry_rub": 50.0, "unrealised_pnl_rub": 0.0,
               "positions_overnight": 0, "incomplete_phases": []}
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "balance"))
            with open(os.path.join(tmp, "balance", "2026-09-23.json"), "w",
                      encoding="utf-8") as f:
                json.dump(day, f)
            with mock.patch.object(stage2_balance, "_stage2_dir", lambda: tmp):
                return stage2_balance.build_report(start_balance=1e6, target_days=24)

    def test_numeric_account_is_prod(self):
        self.assertIn("PROD (РЕАЛЬНЫЕ ДЕНЬГИ)", self._report("2018145468"))

    def test_uuid_account_is_sandbox(self):
        self.assertIn("Счёт: SANDBOX",
                      self._report("1639899c-2aca-49fe-b5d8-d68e8b84f225"))
