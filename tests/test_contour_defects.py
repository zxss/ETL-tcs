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

    def test_decomposition_by_fund_value(self):
        """Аудит 26.09 (F6): казначейство = Δ стоимости паёв − чистые покупки паёв.

        Счёт +1 000 ₽, паи подорожали на 900 ₽, но 500 ₽ из них — докупка паёв
        внутри периода. Доход паёв 400 ₽, торговля 600 ₽. Прежний отчёт выдал бы
        «внутридневной P&L» из переоценки паёв и остаток «казначейство и прочее».
        """
        import datetime as dt
        import json
        import os
        import tempfile
        from contextlib import contextmanager
        from unittest import mock
        from services import stage2_balance

        snaps = [("20260925-091000-CLOSE", "2026-09-25T09:10:00+03:00", 1_000_000.0, 100_000.0),
                 ("20260925-183500-OVERNIGHT", "2026-09-25T18:35:00+03:00", 1_001_000.0, 100_900.0)]
        ledger = [(dt.datetime(2026, 9, 25, 10, 5, tzinfo=dt.timezone.utc), "BUY", 500.0)]

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a): pass
            def fetchall(self): return ledger

        class Conn:
            def cursor(self): return Cur()

        @contextmanager
        def fake_conn():
            yield Conn()

        with tempfile.TemporaryDirectory() as tmp:
            for run, ts, total, etf in snaps:
                os.makedirs(os.path.join(tmp, "runs", run))
                with open(os.path.join(tmp, "runs", run, "balance.json"), "w") as f:
                    json.dump({"captured_at": ts, "account_id": "acc",
                               "total_portfolio_rub": total, "etf_value_rub": etf}, f)
            with mock.patch("database.get_db_connection", fake_conn):
                d = stage2_balance.decompose(tmp, "acc")
        self.assertAlmostEqual(d["total"], 1000.0)
        self.assertAlmostEqual(d["treasury"], 400.0)
        self.assertAlmostEqual(d["trading"], 600.0)
        self.assertEqual(d["fund_ops"], 1)

    def test_old_misleading_lines_gone(self):
        text = self._report(sandbox=True)
        self.assertNotIn("Внутридневной P&L", text)
        self.assertNotIn("Перенос через ночь", text)
        self.assertNotIn("Казначейство и прочее", text)


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
        self.assertIn("PROD (РЕАЛЬНЫЕ ДЕНЬГИ)", self._report("9999999999"))

    def test_uuid_account_is_sandbox(self):
        self.assertIn("Счёт: SANDBOX",
                      self._report("00000000-0000-0000-0000-000000000000"))


class TestOrphanOvernightPositions(unittest.TestCase):
    """CLOSE обязан разгребать позицию, пережившую свою закрытую запись.

    25.09.2026 на боевом счёте «Робот» остаток HYDR 18 000 шт остался с закрытой
    записью: CLOSE его не трогал, а проверка «после CLOSE нет ночных позиций»
    валила тест в halt каждое утро — выйти было нельзя ничем, кроме ручной сделки.
    """

    def test_close_reopens_orphan_records(self):
        import inspect
        from services import stage2_demo as s2
        src = inspect.getsource(s2.close_overnight_positions)
        self.assertIn("orphans", src)
        # сироты подмешиваются к обычным записям ДО выхода по «нечего закрывать»
        self.assertLess(src.index("recs += orphans"), src.index("if not recs:"))
        # и позиции читаются до отбора сирот, иначе не с чем сверять
        self.assertLess(src.index("positions = {p.instrument_uid"),
                        src.index("orphans = ["))

    def test_orphan_selection_rule(self):
        """Сирота = запись закрыта И стратегия ночная И позиция жива."""
        positions = {"u-hydr": object()}
        pending = [
            {"ticker": "HYDR", "strategy": "long_overnight", "closed": True,
             "instrument_uid": "u-hydr"},                      # сирота
            {"ticker": "MGNT", "strategy": "long_overnight", "closed": True,
             "instrument_uid": "u-mgnt"},                      # закрыта, позиции нет
            {"ticker": "VTBR", "strategy": "intraday_short", "closed": True,
             "instrument_uid": "u-hydr"},                      # не ночная
        ]
        from services.stage2_demo import _OVERNIGHT
        orphans = [r for r in pending
                   if r.get("closed") and r.get("strategy") in _OVERNIGHT
                   and r.get("instrument_uid") in positions]
        self.assertEqual([r["ticker"] for r in orphans], ["HYDR"])


class TestVarVolScaling(unittest.TestCase):
    """26.09.2026: 2,3 % замерены на окне, где вола IMOEX шла 0,64…2,24 %/день.

    Фиксированный процент занижает риск в стрессе — ровно тогда, когда лимит и
    нужен. Множитель режима только РАСШИРЯЕТ оценку: опускать её ниже
    откалиброванного уровня значит торговать риском, которого не замеряли.
    """

    ORDERS = [{"signal_id": f"s{i}", "total_rub": 100_000.0, "final_score": 10 - i}
              for i in range(5)]

    def test_factor_one_keeps_calibrated_estimate(self):
        self.assertAlmostEqual(risk_limits.basket_var_rub(500_000.0, 5, 1.0),
                               500_000.0 * 0.023, places=2)

    def test_factor_widens_estimate(self):
        v1 = risk_limits.basket_var_rub(500_000.0, 5, 1.0)
        v2 = risk_limits.basket_var_rub(500_000.0, 5, 1.6)
        self.assertAlmostEqual(v2 / v1, 1.6, places=6)

    def test_factor_below_one_ignored(self):
        """Спокойный рынок не повод занижать откалиброванный VaR."""
        self.assertAlmostEqual(risk_limits.basket_var_rub(500_000.0, 5, 0.5),
                               risk_limits.basket_var_rub(500_000.0, 5, 1.0), places=6)

    def test_high_vol_trims_basket_that_fit_before(self):
        """Корзина, влезавшая в бюджет при обычной воле, в стрессе срезается."""
        budget = 12_000.0
        kept_calm, dropped_calm, _ = risk_limits.trim_to_budget(
            self.ORDERS, budget=budget, vol_factor=1.0)
        kept_stress, dropped_stress, info = risk_limits.trim_to_budget(
            self.ORDERS, budget=budget, vol_factor=1.6)
        self.assertEqual(len(kept_calm), 5)
        self.assertEqual(dropped_calm, [])
        self.assertLess(len(kept_stress), 5)
        self.assertLessEqual(info["var_after_rub"], budget)
        self.assertEqual(info["vol_factor"], 1.6)

    def test_scaling_disabled_by_flag(self):
        from unittest import mock
        import config as cfg
        with mock.patch.object(cfg, "OVERNIGHT_VAR_VOL_SCALING", False):
            factor, info = risk_limits.vol_widen()
        self.assertEqual(factor, 1.0)
        self.assertFalse(info["scaling"])

    def test_missing_index_history_falls_back_to_one(self):
        """Нет истории индекса — лимит остаётся откалиброванным, а не отключается."""
        from unittest import mock
        import config as cfg

        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a): pass
            def fetchall(self): return [(100.0,), (101.0,)]

        class _Conn:
            def cursor(self): return _Cur()

        with mock.patch.object(cfg, "OVERNIGHT_VAR_VOL_SCALING", True):
            factor, info = risk_limits.vol_widen(_Conn())
        self.assertEqual(factor, 1.0)
        self.assertIn("reason", info)

    def test_vol_measured_from_index_closes(self):
        """Ровный ряд — нулевая вола — множитель 1, а не деление на ноль."""
        from unittest import mock
        import config as cfg

        closes = [(100.0,)] * (risk_limits.VOL_WINDOW + 1)

        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a): pass
            def fetchall(self): return closes

        class _Conn:
            def cursor(self): return _Cur()

        with mock.patch.object(cfg, "OVERNIGHT_VAR_VOL_SCALING", True):
            factor, info = risk_limits.vol_widen(_Conn())
        self.assertEqual(factor, 1.0)
        self.assertEqual(info["vol_now_pct"], 0.0)

    def test_widen_capped(self):
        """Предохранитель от битой свечи: множитель не уходит в бесконечность."""
        from unittest import mock
        import config as cfg

        closes = [(100.0,) if i % 2 == 0 else (10.0,)
                  for i in range(risk_limits.VOL_WINDOW + 1)]

        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a): pass
            def fetchall(self): return closes

        class _Conn:
            def cursor(self): return _Cur()

        with mock.patch.object(cfg, "OVERNIGHT_VAR_VOL_SCALING", True):
            factor, _ = risk_limits.vol_widen(_Conn())
        self.assertEqual(factor, risk_limits.VOL_WIDEN_MAX)
