"""
Acceptance-тесты Этапа 2 (STAGE2-DEMO-TZ.md §15). Обязательны до первого запуска.

Все десять проверок из ТЗ, без сети и без реального брокера: брокер и тяжёлые
расчёты подменены заглушками, каталог теста — временный.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                     # noqa: E402
from services import stage2_demo as s2            # noqa: E402
from services import stage2_balance as s2b        # noqa: E402



def _code_identifiers(func) -> set[str]:
    """Имена, которые функция РЕАЛЬНО использует: вызовы и импорты.

    Разбор AST, а не поиск по тексту: строка «не вызывает place_limits» в
    докстринге не должна считаться вызовом.
    """
    import ast, inspect, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


# ── Заглушки ─────────────────────────────────────────────────────────────────

class FakePosition:
    def __init__(self, uid, shares=10):
        self.instrument_uid, self.balance_shares = uid, shares

    @property
    def is_open(self):
        return self.balance_shares != 0


class FakeBroker:
    """Считает вызовы, чтобы тесты могли утверждать «заявок не было»."""

    def __init__(self, positions=None):
        self.calls: list[str] = []
        self._positions = positions or []

    def post_limit_order(self, **kw):
        self.calls.append("post_limit_order")
        raise AssertionError("заявка не должна выставляться в этой фазе")

    def get_positions(self, account_id):
        self.calls.append("get_positions")
        return self._positions

    def get_active_orders(self, account_id):
        return []

    def cancel_order(self, *, account_id, order_id):
        self.calls.append("cancel_order")

    def get_order_state(self, *, account_id, order_id):
        raise RuntimeError("нет состояния")

    def _post(self, method, payload=None):
        return {}


class Stage2TestCase(unittest.TestCase):
    """Временный каталог теста + восстановление конфигурации."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stage2-")
        self._saved = {k: getattr(config, k, None)
                       for k in ("STAGE2_DIR", "STAGE2_ENABLED", "STAGE2_TARGET_DAYS",
                                 "STAGE2_HALT_ON_FAIL", "TRADING_MODE", "PROD_ACCOUNT_ID",
                                 "TELEGRAM_ENABLED", "STAGE2_START_DATE")}
        config.STAGE2_DIR = self.tmp
        config.STAGE2_ENABLED = True
        config.TRADING_MODE = "sandbox"
        config.PROD_ACCOUNT_ID = ""
        # На сервере .env включает уведомления и задаёт дату старта: без сброса
        # тесты фаз слали бы настоящие сообщения и пропускались бы до старта.
        config.TELEGRAM_ENABLED = False
        config.STAGE2_START_DATE = ""
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_state(self, **kw):
        st = s2.load_state()
        st.update(kw)
        s2.save_state(st)
        return st


# ── 1-2. Square-off: интрадей закрывается, овернайт переживает ───────────────

class TestSquareOffScope(Stage2TestCase):

    def test_1_square_off_targets_only_intraday(self):
        """После CLEANUP не должно остаться позиций intraday_*."""
        reg = {"uid-A": {"instrument_uid": "uid-A", "ticker": "SBER",
                         "strategy": "intraday_short"},
               "uid-B": {"instrument_uid": "uid-B", "ticker": "GAZP",
                         "strategy": "long_overnight"}}
        broker = FakeBroker([FakePosition("uid-A"), FakePosition("uid-B")])
        with mock.patch.object(s2, "_strategy_by_uid", return_value=reg):
            open_intraday = s2._open_intraday(broker, "acc")
        self.assertEqual(open_intraday, ["SBER"],
                         "в список на закрытие попала не только дневная позиция")

    def test_2_overnight_survives_cleanup(self):
        """Ночная позиция не попадает под square-off."""
        reg = {"uid-B": {"instrument_uid": "uid-B", "ticker": "GAZP",
                         "strategy": "long_overnight"}}
        broker = FakeBroker([FakePosition("uid-B")])
        with mock.patch.object(s2, "_strategy_by_uid", return_value=reg):
            self.assertEqual(s2._open_intraday(broker, "acc"), [])

    def test_2b_unknown_strategy_is_left_alone(self):
        """Позиция без стратегии в реестре не трогается: закрыть по ошибке
        ночную хуже, чем не закрыть дневную."""
        broker = FakeBroker([FakePosition("uid-X")])
        with mock.patch.object(s2, "_strategy_by_uid", return_value={}):
            self.assertEqual(s2._open_intraday(broker, "acc"), [])


# ── 3. PREP не выставляет заявок ─────────────────────────────────────────────

class TestPrepPlacesNothing(Stage2TestCase):

    def test_3_prep_never_calls_broker_post_order(self):
        """Фаза PREP не вызывает и не импортирует ничего, чем ставят заявки."""
        used = _code_identifiers(s2.phase_prep)
        for forbidden in ("place_limits", "sync_portfolio", "post_limit_order",
                          "attach_stops"):
            self.assertNotIn(forbidden, used,
                             f"PREP не должен уметь выставлять заявки: {forbidden}")

    def test_3b_prep_writes_plan_not_orders(self):
        """PREP создаёт plan.json, но не orders.json."""
        import inspect
        src = inspect.getsource(s2.phase_prep)
        self.assertIn('"plan.json"', src)
        self.assertNotIn('"orders.json"', src)


# ── 4. ORDER исполняет план, а не пересчитывает ──────────────────────────────

class TestOrderExecutesPlan(Stage2TestCase):

    def test_4_order_does_not_recompute(self):
        """Фаза ORDER не вызывает пересчёт сигналов — только исполняет план."""
        used = _code_identifiers(s2.phase_order)
        for forbidden in ("compute_orders", "select_top_rows", "build_orders",
                          "run_validation"):
            self.assertNotIn(forbidden, used,
                             f"ORDER обязан исполнять план, а не считать: {forbidden}")

    def test_4c_overnight_may_recompute(self):
        """Контроль: OVERNIGHT пересчитывать ОБЯЗАН — якорь должен быть ценой
        закрытия, иначе вход не соответствует определению стратегии."""
        self.assertIn("compute_orders", _code_identifiers(s2.phase_overnight))

    def test_4b_order_builds_from_plan_prices(self):
        """Цены Order берутся из плана дословно."""
        from tft_forecast.combined import Order
        po = {"ticker": "SBER", "strategy_type": "intraday_short", "direction": "SHORT",
              "anchor_price": 312.45, "entry_price": 313.20, "stop_price": 318.10,
              "tp_price": 309.80, "quantity_lots": 32, "lot_size": 1,
              "total_rub": 10022.40}
        o = s2._order_from_plan(po, Order)
        self.assertEqual(o.entry_price, 313.20)
        self.assertEqual(o.stop_price, 318.10)
        self.assertEqual(o.tp_price, 309.80)
        self.assertEqual(o.quantity_lots, 32)


# ── 5. Отказ при смене конфигурации или датасета ─────────────────────────────

class TestPlanValidation(Stage2TestCase):

    DAY = dt.date(2026, 9, 10)

    def _plan(self, **over):
        p = {"trading_day": self.DAY.isoformat(),
             "config_hash": s2.config_hash(),
             "dataset": {"dataset_hash": "sha256:aaa"}}
        p.update(over)
        return p

    def test_5_config_change_blocks_order(self):
        plan = self._plan(config_hash="sha256:другой")
        errs = s2.validate_plan(plan, self.DAY, {"dataset_hash": "sha256:aaa"})
        self.assertTrue(any("config_hash" in e for e in errs))

    def test_5b_dataset_drift_blocks_when_not_allowed(self):
        saved = getattr(config, "STAGE2_ALLOW_DATASET_DRIFT", False)
        config.STAGE2_ALLOW_DATASET_DRIFT = False
        self.addCleanup(setattr, config, "STAGE2_ALLOW_DATASET_DRIFT", saved)
        errs = s2.validate_plan(self._plan(), self.DAY, {"dataset_hash": "sha256:bbb"})
        self.assertTrue(any("dataset_hash" in e for e in errs))

    def test_5c_dataset_drift_allowed_by_flag(self):
        saved = getattr(config, "STAGE2_ALLOW_DATASET_DRIFT", False)
        config.STAGE2_ALLOW_DATASET_DRIFT = True
        self.addCleanup(setattr, config, "STAGE2_ALLOW_DATASET_DRIFT", saved)
        errs = s2.validate_plan(self._plan(), self.DAY, {"dataset_hash": "sha256:bbb"})
        self.assertEqual(errs, [])

    def test_5d_missing_plan_blocks(self):
        self.assertTrue(s2.validate_plan(None, self.DAY, {"dataset_hash": "x"}))

    def test_5e_plan_from_another_day_blocks(self):
        plan = self._plan(trading_day="2026-09-09")
        errs = s2.validate_plan(plan, self.DAY, {"dataset_hash": "sha256:aaa"})
        self.assertTrue(any("другого дня" in e for e in errs))

    def test_5f_valid_plan_passes(self):
        self.assertEqual(
            s2.validate_plan(self._plan(), self.DAY, {"dataset_hash": "sha256:aaa"}), [])


# ── 6. Каталоги запусков не перезаписываются ─────────────────────────────────

class TestRunDirsAreImmutable(Stage2TestCase):

    def test_6_duplicate_run_id_raises(self):
        rid = s2.new_run_id("PREP")
        s2.make_run_dir(rid)
        with self.assertRaises(FileExistsError,
                               msg="повторный RUN_ID обязан быть ошибкой, а не перезаписью"):
            s2.make_run_dir(rid)

    def test_6b_run_id_format(self):
        rid = s2.new_run_id("ORDER", dt.datetime(2026, 9, 10, 10, 5, 0, tzinfo=s2.MSK))
        self.assertEqual(rid, "20260910-100500-ORDER")


# ── 7. Боевой контур недостижим ──────────────────────────────────────────────

class TestProdUnreachable(Stage2TestCase):

    def test_7_preflight_rejects_prod_flag(self):
        res = s2.PhaseResult("ORDER", "rid", self.tmp)
        s2.preflight(res, env="SANDBOX", prod_flag=True)
        self.assertTrue(any("--prod" in e for e in res.errors))
        self.assertEqual(res.verdict, s2.FAIL)

    def test_7b_preflight_rejects_prod_env(self):
        res = s2.PhaseResult("ORDER", "rid", self.tmp)
        s2.preflight(res, env="PROD", prod_flag=False)
        self.assertTrue(res.errors)

    def test_7c_preflight_rejects_filled_prod_account(self):
        config.PROD_ACCOUNT_ID = "2018145468"
        res = s2.PhaseResult("ORDER", "rid", self.tmp)
        s2.preflight(res, env="SANDBOX", prod_flag=False)
        self.assertTrue(any("PROD_ACCOUNT_ID" in e for e in res.errors))

    def test_7d_clean_sandbox_passes(self):
        res = s2.PhaseResult("ORDER", "rid", self.tmp)
        s2.preflight(res, env="SANDBOX", prod_flag=False)
        self.assertEqual(res.errors, [])
        self.assertEqual(res.critical_passed, 4)


# ── 8. Счётчик считает только торговые дни ───────────────────────────────────

class TestTradingDayCounter(Stage2TestCase):

    def test_8_saturday_is_skipped(self):
        """Суббота не увеличивает счётчик: фаза выходит с кодом 0 и skipped."""
        sat = dt.datetime(2026, 9, 12, 9, 45, tzinfo=s2.MSK)
        self.assertEqual(sat.weekday(), 5)
        self.write_state(completed_trading_days=0)
        with mock.patch.object(s2.trading_cal, "is_trading_day", return_value=False):
            code = s2.phase_prep(now=sat)
        self.assertEqual(code, 0)
        self.assertEqual(s2.load_state()["completed_trading_days"], 0)

    def test_8b_skip_writes_reason(self):
        sat = dt.datetime(2026, 9, 12, 9, 45, tzinfo=s2.MSK)
        with mock.patch.object(s2.trading_cal, "is_trading_day", return_value=False):
            s2.phase_prep(now=sat)
        metas = []
        for d in os.listdir(os.path.join(self.tmp, "runs")):
            m = s2._read_json(os.path.join(self.tmp, "runs", d, "run_meta.json"), {})
            metas.append(m.get("skipped"))
        self.assertIn("not_a_trading_day", metas)


# ── 9. Автостоп на 15-м дне ──────────────────────────────────────────────────

class TestAutoStop(Stage2TestCase):

    def test_9_all_phases_noop_when_finished(self):
        self.write_state(completed_trading_days=15, target_trading_days=15,
                         status="running")
        now = dt.datetime(2026, 9, 10, 9, 45, tzinfo=s2.MSK)
        with mock.patch.object(s2.trading_cal, "is_trading_day", return_value=True):
            for fn in (s2.phase_prep, s2.phase_order, s2.phase_cleanup, s2.phase_overnight):
                self.assertEqual(fn(now=now), 0, f"{fn.__name__} обязана быть no-op")

    def test_9b_halt_blocks_phases(self):
        self.write_state(status="halted", halted_reason="тестовая остановка")
        now = dt.datetime(2026, 9, 10, 10, 5, tzinfo=s2.MSK)
        with mock.patch.object(s2.trading_cal, "is_trading_day", return_value=True):
            self.assertEqual(s2.phase_order(now=now), 0)

    def test_9c_resume_clears_halt(self):
        self.write_state(status="halted", halted_reason="тестовая остановка")
        s2.cmd_resume()
        st = s2.load_state()
        self.assertEqual(st["status"], "running")
        self.assertIsNone(st["halted_reason"])


# ── 10. Баланс на всех четырёх фазах ─────────────────────────────────────────

class TestBalanceAcrossPhases(Stage2TestCase):

    def _make_day(self, day, values):
        runs = {}
        for ph, val in values.items():
            d = os.path.join(self.tmp, "runs", f"{day.strftime('%Y%m%d')}-000000-{ph}")
            os.makedirs(d, exist_ok=True)
            s2._write_json(os.path.join(d, "balance.json"), {
                "phase": ph, "total_portfolio_rub": val, "open_positions": 2,
                "shares_value_rub": 19842.0, "unrealised_pnl_rub": 73.2,
                "account_id": "1639899c"})
            runs[ph] = d
        return runs

    def test_10_all_phases_aggregated(self):
        day = dt.date(2026, 9, 10)
        runs = self._make_day(day, {"PREP": 100000.0, "CLOSE": 100000.0, "ORDER": 99994.1,
                                    "CLEANUP": 99880.2, "OVERNIGHT": 99871.4})
        s = s2b.collect_day(day, runs, prev_closing=100000.0)
        self.assertEqual(set(s["by_phase"]), set(s2b.PHASES))
        self.assertEqual(s["incomplete_phases"], [])
        self.assertAlmostEqual(s["day_change_rub"], -128.6, places=2)
        self.assertAlmostEqual(s["intraday_pnl_rub"], -113.9, places=2)
        self.assertAlmostEqual(s["overnight_carry_rub"], -8.8, places=2)
        self.assertEqual(s["account_id"], "1639899c")

    def test_10b_missing_phase_is_flagged(self):
        day = dt.date(2026, 9, 11)
        runs = self._make_day(day, {"PREP": 100.0, "ORDER": 99.0})
        s = s2b.collect_day(day, runs)
        self.assertEqual(s["incomplete_phases"], ["CLOSE", "CLEANUP", "OVERNIGHT"])

    def test_10c_report_has_no_performance_metrics(self):
        """§1: отчёт намеренно не содержит Sharpe, CAGR и Profit Factor."""
        day = dt.date(2026, 9, 10)
        runs = self._make_day(day, {"PREP": 100000.0, "ORDER": 99994.1,
                                    "CLEANUP": 99880.2, "OVERNIGHT": 99871.4})
        s2b.save_day(s2b.collect_day(day, runs, prev_closing=100000.0))
        txt = s2b.build_report()
        for forbidden in ("Sharpe", "CAGR", "Profit Factor"):
            self.assertNotIn(forbidden + ":", txt)
        self.assertIn("ОТЧЁТ ПО ТЕСТОВОМУ БАЛАНСУ", txt)
        self.assertIn("2026-09-10", txt)


# ── Воспроизводимость хешей ──────────────────────────────────────────────────

class TestHashes(Stage2TestCase):

    def test_config_hash_stable(self):
        self.assertEqual(s2.config_hash(), s2.config_hash())

    def test_config_hash_changes_with_config(self):
        before = s2.config_hash()
        saved = config.BEST_TRADES_TOP_N
        config.BEST_TRADES_TOP_N = saved + 1
        self.addCleanup(setattr, config, "BEST_TRADES_TOP_N", saved)
        self.assertNotEqual(before, s2.config_hash())

    def test_config_snapshot_covers_trading_params(self):
        snap = s2.config_snapshot()
        for key in ("TRADING_MODE", "TRADING_STRATEGIES", "LIMIT_ENTRY_FRACTION",
                    "VALIDATION_COST_RT", "INTRADAY_SQUARE_OFF_TIME"):
            self.assertIn(key, snap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
