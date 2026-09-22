"""
Согласованность контура: песочница против боевого счёта (турнир, 23.09.2026).

Правка preflight открывает боевой путь, поэтому тесты держат две границы:
изоляция песочницы не ослабла, а боевой режим включается только при полном
совпадении всех признаков.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import stage2_demo as s2                   # noqa: E402

PROD_ACC = "2018145468"


def result():
    return s2.PhaseResult("prep", "run", "/tmp/run")


def cfg(**kw):
    """Подменяет config-атрибуты, которые читает контур."""
    base = {"TRADING_MODE": "sandbox", "PROD_ACCOUNT_ID": "",
            "BEST_TRADES_POSITION_RUB": 20000.0, "PROD_MAX_POSITION_RUB": 25000.0}
    base.update(kw)
    return mock.patch.multiple(s2.config, **base)


def env(unattended: str = ""):
    return mock.patch.dict(os.environ, {"ALLOW_UNATTENDED_PROD": unattended}, clear=False)


class TestContourOk(unittest.TestCase):

    def test_sandbox(self):
        with cfg():
            self.assertIsNone(s2.contour_ok("SANDBOX"))
        with cfg(TRADING_MODE="prod"):
            self.assertIn("TRADING_MODE=prod", s2.contour_ok("SANDBOX"))

    def test_prod_requires_all_signs(self):
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC), env("1"):
            self.assertIsNone(s2.contour_ok("PROD"))
        with cfg(TRADING_MODE="sandbox", PROD_ACCOUNT_ID=PROD_ACC), env("1"):
            self.assertIn("TRADING_MODE", s2.contour_ok("PROD"))
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=""), env("1"):
            self.assertIn("PROD_ACCOUNT_ID", s2.contour_ok("PROD"))
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC), env(""):
            self.assertIn("ALLOW_UNATTENDED_PROD", s2.contour_ok("PROD"))

    def test_unknown_env(self):
        with cfg():
            self.assertIn("неизвестный контур", s2.contour_ok("STAGE"))


class TestSandboxIsolation(unittest.TestCase):
    """Без --prod поведение прежнее: боевые параметры валят фазу."""

    def test_clean_sandbox_passes(self):
        r = result()
        with cfg():
            s2.preflight(r, env="SANDBOX", prod_flag=False, account_id="sandbox-1")
        self.assertEqual(r.errors, [])

    def test_prod_mode_without_flag_fails(self):
        r = result()
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC):
            s2.preflight(r, env="SANDBOX", prod_flag=False, account_id="sandbox-1")
        self.assertEqual(len(r.errors), 2)                # TRADING_MODE и PROD_ACCOUNT_ID

    def test_prod_contour_without_flag_fails(self):
        r = result()
        with cfg():
            s2.preflight(r, env="PROD", prod_flag=False, account_id=PROD_ACC)
        self.assertTrue(any("не SANDBOX" in e for e in r.errors))


class TestProdPath(unittest.TestCase):

    def prod(self, r, *, account_id=PROD_ACC, unattended="1", dir_name="stage2-tournament", **kw):
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC, **kw), env(unattended), \
                mock.patch.object(s2, "base_dir", return_value=f"audit/{dir_name}"):
            s2.preflight(r, env="PROD", prod_flag=True, account_id=account_id)
        return r

    def test_consistent_prod_passes(self):
        r = self.prod(result())
        self.assertEqual(r.errors, [])
        self.assertEqual(r.critical_passed, 7)

    def test_account_mismatch_fails(self):
        r = self.prod(result(), account_id="2001033006")
        self.assertTrue(any("PROD_ACCOUNT_ID" in e for e in r.errors))

    def test_unattended_required(self):
        r = self.prod(result(), unattended="")
        self.assertTrue(any("ALLOW_UNATTENDED_PROD" in e for e in r.errors))

    def test_sandbox_dir_rejected(self):
        r = self.prod(result(), dir_name="stage2-demo")
        self.assertTrue(any("STAGE2_DIR" in e for e in r.errors))

    def test_position_cap(self):
        r = self.prod(result(), BEST_TRADES_POSITION_RUB=200000.0)
        self.assertTrue(any("выше предела" in e for e in r.errors))
        ok = self.prod(result(), BEST_TRADES_POSITION_RUB=25000.0)
        self.assertEqual(ok.errors, [])

    def test_flag_without_prod_contour_fails(self):
        r = result()
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC), env("1"):
            s2.preflight(r, env="SANDBOX", prod_flag=True, account_id=PROD_ACC)
        self.assertTrue(any("контур SANDBOX" in e for e in r.errors))


class TestOrderGate(unittest.TestCase):

    def order(self):
        return {"strategy_type": "intraday_short", "direction": "LONG", "ticker": "SBER",
                "quantity_lots": 1, "total_rub": 1000.0, "signal_id": "s1"}

    def test_prod_order_allowed_only_on_consistent_contour(self):
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC), env("1"), \
                mock.patch("tft_forecast.combined.trading_strategies",
                           return_value={"intraday_short"}), \
                mock.patch("tft_forecast.combined.non_shortable_tickers", return_value=set()):
            self.assertIsNone(s2.check_order_allowed(self.order(), env="PROD", used_ids=set()))
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC), env(""):
            self.assertIn("ALLOW_UNATTENDED_PROD",
                          s2.check_order_allowed(self.order(), env="PROD", used_ids=set()))

    def test_sandbox_with_prod_mode_refused(self):
        with cfg(TRADING_MODE="prod", PROD_ACCOUNT_ID=PROD_ACC):
            self.assertIn("TRADING_MODE=prod",
                          s2.check_order_allowed(self.order(), env="SANDBOX", used_ids=set()))


if __name__ == "__main__":
    unittest.main()
