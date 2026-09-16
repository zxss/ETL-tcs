"""
Итоговая боевая конфигурация Этапа 2 (Paper Trading / Sandbox).

Пин всех канонических значений и двух новых предохранителей. Смысл файла —
поймать молчаливый сдвиг конфигурации: каждое значение здесь получено
измерением на walk-forward реплее, и менять его следует осознанно.
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

try:
    from tests.conftest import make_dashboard_row
except ImportError:                                        # pragma: no cover
    from conftest import make_dashboard_row

from tft_forecast.combined import (  # noqa: E402
    apply_strategy_specialisation, non_shortable_tickers,
)

_DIR = {"intraday_long": "LONG", "intraday_short": "SHORT",
        "long_overnight": "LONG"}


def _cfg():
    import config
    return importlib.reload(config)


class TestSafetyGuards(unittest.TestCase):
    """Предохранители: шорт-сквиз и перегретая волатильность."""

    @staticmethod
    def row(strategy, **over):
        # market_atr_pctl=70 > 50 и ret1<0 → импульс продавцов подтверждён,
        # чтобы каждый тест проверял ИМЕННО свой предохранитель, а не
        # спотыкался о фильтр импульса (порог там строгий: > 50).
        base = dict(strategy=strategy, direction=_DIR[strategy],
                    ret1=-1.0, market_atr_pctl=70.0, exp_pnl=1.0, cost_rt=0.13,
                    index_above_ema50=False)
        base.update(over)
        return make_dashboard_row(**base)

    def keep(self, rows, **kw):
        kw.setdefault("verbose", False)
        return apply_strategy_specialisation(rows, **kw)

    # ── предохранитель от шорт-сквиза ───────────────────────────────────────

    def test_short_blocked_when_index_above_ema50(self):
        r = self.row("intraday_short", index_above_ema50=True)
        self.assertEqual(self.keep([r], block_short_uptrend=True), [])

    def test_short_allowed_when_index_below_ema50(self):
        r = self.row("intraday_short", index_above_ema50=False)
        self.assertEqual(len(self.keep([r], block_short_uptrend=True)), 1)

    def test_squeeze_guard_can_be_disabled(self):
        r = self.row("intraday_short", index_above_ema50=True)
        self.assertEqual(len(self.keep([r], block_short_uptrend=False)), 1)

    def test_squeeze_guard_ignores_missing_data(self):
        """Признак недоступен — не отбрасываем: незнание не есть отказ."""
        r = self.row("intraday_short", index_above_ema50=None)
        self.assertEqual(len(self.keep([r], block_short_uptrend=True)), 1)

    def test_squeeze_guard_does_not_touch_overnight(self):
        r = self.row("long_overnight", index_above_ema50=True)
        self.assertEqual(len(self.keep([r], block_short_uptrend=True)), 1)

    # ── потолок волатильности для овернайта ─────────────────────────────────

    def test_overnight_blocked_on_hot_market(self):
        r = self.row("long_overnight", market_atr_pctl=85.0)
        self.assertEqual(self.keep([r], overnight_max_market_atr=70.0), [])

    def test_overnight_allowed_on_calm_market(self):
        r = self.row("long_overnight", market_atr_pctl=40.0)
        self.assertEqual(len(self.keep([r], overnight_max_market_atr=70.0)), 1)

    def test_overnight_threshold_is_strict(self):
        exactly = self.row("long_overnight", market_atr_pctl=70.0)
        above = self.row("long_overnight", market_atr_pctl=70.01)
        self.assertEqual(len(self.keep([exactly], overnight_max_market_atr=70.0)), 1)
        self.assertEqual(self.keep([above], overnight_max_market_atr=70.0), [])

    def test_hot_market_guard_does_not_touch_short(self):
        r = self.row("intraday_short", market_atr_pctl=95.0, ret1=-1.0)
        self.assertEqual(len(self.keep([r], overnight_max_market_atr=70.0)), 1)

    def test_hot_market_guard_ignores_missing_data(self):
        r = self.row("long_overnight", market_atr_pctl=None)
        self.assertEqual(len(self.keep([r], overnight_max_market_atr=70.0)), 1)


class TestCanonicalValues(unittest.TestCase):
    """Каждое значение получено измерением — менять осознанно."""

    def test_trading_mode_is_sandbox(self):
        self.assertEqual(_cfg().TRADING_MODE, "sandbox")

    def test_strategy_menu(self):
        c = _cfg()
        self.assertEqual(set(c.TRADING_STRATEGIES),
                         {"intraday_short", "long_overnight"})
        self.assertNotIn("intraday_long", c.TRADING_STRATEGIES)
        # валидатор продолжает считать всю тройку — мониторинг смены режима
        self.assertIn("intraday_long", c.VALIDATION_STRATS)

    def test_scoring_block(self):
        c = _cfg()
        self.assertEqual(c.SCORE_MODE, "heuristic")
        self.assertFalse(c.APPLY_RISK_PENALTIES)
        self.assertTrue(c.SELLER_MOMENTUM_SHORT_ENABLED)
        self.assertEqual(c.SHORT_IMOEX_MAX_TREND, "EMA50")
        self.assertAlmostEqual(c.OVERNIGHT_MAX_MARKET_ATR_PCTL, 70.0)
        self.assertAlmostEqual(c.OVERNIGHT_MIN_EDGE_X_COST, 0.5)

    def test_execution_block(self):
        c = _cfg()
        self.assertAlmostEqual(c.LIMIT_ENTRY_FRACTION, 0.2)
        self.assertAlmostEqual(c.LIMIT_TP_FRACTION, 0.5)
        self.assertAlmostEqual(c.ORDER_FILL_WAIT_SEC, 60.0)
        self.assertTrue(c.INTRADAY_SQUARE_OFF_ENABLED)
        # 18:20, а не 18:35: фаза OVERNIGHT Этапа 2 выставляет ночные заявки
        # в 18:35, и cleanup должен освободить окно до неё (STAGE2-DEMO-TZ §13).
        self.assertEqual(c.INTRADAY_SQUARE_OFF_TIME, "18:20")
        self.assertLess(c.INTRADAY_SQUARE_OFF_TIME, c.STAGE2_OVERNIGHT_TIME,
                        "cleanup обязан идти раньше выставления овернайта")

    def test_sizing_block(self):
        c = _cfg()
        self.assertEqual(c.FIXED_POSITION_OVERFLOW_MODE, "skip")
        self.assertEqual(c.BEST_TRADES_TOP_N, 5)
        self.assertAlmostEqual(c.BEST_TRADES_POSITION_RUB, 100_000.0)   # r4: депозит 5 млн ₽

    def test_non_shortable_blacklist(self):
        self.assertTrue({"AKRN", "CBOM", "MVID"} <= non_shortable_tickers())

    def test_tls_verify_on_by_default(self):
        self.assertEqual(_cfg().INVEST_TLS_VERIFY, 1)

    def test_legacy_momentum_alias(self):
        c = _cfg()
        self.assertEqual(c.INTRADAY_SHORT_REQUIRE_MOMENTUM,
                         c.SELLER_MOMENTUM_SHORT_ENABLED)

    def test_overflow_mode_rejects_force_min_lot(self):
        """Округление вверх до лота обходит риск-сайзинг — отказ на старте."""
        os.environ["FIXED_POSITION_OVERFLOW_MODE"] = "force_min_1_lot"
        try:
            with self.assertRaises(ValueError):
                _cfg()
        finally:
            os.environ.pop("FIXED_POSITION_OVERFLOW_MODE", None)
            _cfg()


class TestTradingModeGate(unittest.TestCase):
    """TRADING_MODE — второй независимый ключ к боевому контуру."""

    def setUp(self):
        import services.place_orders as po
        self.po = po

    def test_sandbox_mode_rejects_prod_flag(self):
        with mock.patch.object(self.po.config, "TRADING_MODE", "sandbox"):
            with self.assertRaises(RuntimeError):
                self.po._resolve_env(True)
            self.assertFalse(self.po._resolve_env(False))

    def test_prod_mode_still_requires_flag(self):
        """Переменная не включает боевой контур сама по себе."""
        with mock.patch.object(self.po.config, "TRADING_MODE", "prod"):
            self.assertTrue(self.po._resolve_env(True))
            self.assertFalse(self.po._resolve_env(False))

    def test_unknown_mode_is_rejected(self):
        with mock.patch.object(self.po.config, "TRADING_MODE", "живьём"):
            with self.assertRaises(RuntimeError):
                self.po._resolve_env(False)

    def test_unattended_prod_still_guarded(self):
        with self.assertRaises(RuntimeError):
            self.po._guard_unattended_prod(True, True)


class TestEnvExampleMatchesCode(unittest.TestCase):
    """.env.example должен описывать РЕАЛЬНЫЕ дефолты, а не пожелания."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, ".env.example")
        cls.env = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                # inline-комментарий: python-dotenv срезает " #" у незакавыченных
                # значений — парсер теста обязан вести себя так же, иначе
                # осмысленный комментарий в .env.example ломает сверку.
                v = v.split(" #", 1)[0]
                cls.env[k.strip()] = v.strip().strip('"').strip("'")

    def test_declared_defaults_match_config(self):
        c = _cfg()
        checks = {
            "TRADING_MODE": c.TRADING_MODE,
            "SCORE_MODE": c.SCORE_MODE,
            "SHORT_IMOEX_MAX_TREND": c.SHORT_IMOEX_MAX_TREND,
            "INTRADAY_SQUARE_OFF_TIME": c.INTRADAY_SQUARE_OFF_TIME,
            "FIXED_POSITION_OVERFLOW_MODE": c.FIXED_POSITION_OVERFLOW_MODE,
            "BEST_TRADES_TOP_N": str(c.BEST_TRADES_TOP_N),
            "OVERNIGHT_MAX_MARKET_ATR_PCTL": str(int(c.OVERNIGHT_MAX_MARKET_ATR_PCTL)),
            "APPLY_RISK_PENALTIES": "0" if not c.APPLY_RISK_PENALTIES else "1",
            "SELLER_MOMENTUM_SHORT_ENABLED": "1" if c.SELLER_MOMENTUM_SHORT_ENABLED else "0",
        }
        for key, expected in checks.items():
            self.assertIn(key, self.env, f"{key} отсутствует в .env.example")
            self.assertEqual(self.env[key], str(expected),
                             f"{key}: .env.example разошёлся с config.py")

    def test_strategy_lists_match(self):
        c = _cfg()
        self.assertEqual(set(self.env["TRADING_STRATEGIES"].split()),
                         set(c.TRADING_STRATEGIES))
        self.assertEqual(set(self.env["VALIDATION_STRATS"].split()),
                         set(c.VALIDATION_STRATS))

    def test_no_real_token_committed(self):
        """В шаблоне не должно быть настоящего токена."""
        tok = self.env.get("INVEST_TOKEN", "")
        self.assertTrue(tok.startswith("t.your_"), "в .env.example попал живой токен")

    def test_prod_account_id_is_commented_out(self):
        self.assertNotIn("PROD_ACCOUNT_ID", self.env,
                         "PROD_ACCOUNT_ID должен оставаться закомментированным")


if __name__ == "__main__":
    unittest.main()
