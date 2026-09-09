"""
Sprint 4.1 — юнит-тесты критических узлов исполнения.

Только чистые функции: ни сети, ни БД, ни моков брокера. Проверяются четыре
узла, где ошибка стоит денег:

  1. лотность и квантование        — services.place_orders._api_quantity
  2. цены входа / стопа / тейка    — combined._limit_entry_price, _take_profit_price
  3. риск-паритет                  — combined._risk_parity_alloc
  4. FinalScore и риск-штрафы      — combined._score_row

Запуск:
    python3 -m unittest tests.test_risk_and_orders -v
    python3 -m pytest tests/test_risk_and_orders.py -q
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tests.conftest import (make_dashboard_row, make_instrument, make_order,
                                make_geom)
except ImportError:  # запуск изнутри каталога tests/
    from conftest import (make_dashboard_row, make_instrument, make_order,
                          make_geom)

from services.place_orders import _api_quantity, _short_blocked      # noqa: E402
from tft_forecast.combined import (                                   # noqa: E402
    _lot_size, _limit_entry_price, _take_profit_price,
    _risk_parity_alloc, _score_row, build_orders, select_top_rows,
    non_shortable_tickers, _drop_blocked_shorts, _DEFAULT_NON_SHORTABLE,
    _apply_penalty, is_rejected, warn_unvalidated,
    apply_strategy_specialisation, trading_strategies,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Лотность и квантование
# ═══════════════════════════════════════════════════════════════════════════

class TestLotSizes(unittest.TestCase):
    """Справочник лотности: нестандартные лоты не должны потеряться."""

    def test_known_non_standard_lots(self):
        self.assertEqual(_lot_size("GMKN"), (10, True))     # сплит 1:100, апр 2024
        self.assertEqual(_lot_size("OGKB"), (1000, True))
        self.assertEqual(_lot_size("VTBR"), (10000, True))
        self.assertEqual(_lot_size("LKOH"), (1, True))

    def test_case_insensitive(self):
        self.assertEqual(_lot_size("gmkn"), _lot_size("GMKN"))

    def test_unknown_ticker_defaults_to_one_and_is_flagged(self):
        lot, known = _lot_size("ZZZZ")
        self.assertEqual(lot, 1)
        self.assertFalse(known, "неизвестный лот обязан помечаться как непроверенный")


class TestApiQuantity(unittest.TestCase):
    """Пересчёт лотов дашборда в лоты API PostOrder."""

    def test_gmkn_exact_division(self):
        """GMKN: лот дашборда 10 = лот API 10 → 3 лота остаются 3 лотами."""
        q, shares, warns = _api_quantity(make_order(quantity_lots=3, lot_size=10),
                                         make_instrument(lot=10, ticker="GMKN"))
        self.assertEqual((q, shares), (3, 30))
        self.assertEqual(warns, [])

    def test_ogkb_large_lot(self):
        """OGKB: лот 1000, 2 лота дашборда = 2000 акций = 2 лота API."""
        q, shares, warns = _api_quantity(make_order(quantity_lots=2, lot_size=1000),
                                         make_instrument(lot=1000, ticker="OGKB"))
        self.assertEqual((q, shares), (2, 2000))
        self.assertEqual(warns, [])

    def test_dashboard_lot_smaller_than_api_lot_rounds_down(self):
        """25 акций при API-лоте 10 → 2 лота (20 акций), с предупреждением."""
        q, shares, warns = _api_quantity(make_order(quantity_lots=25, lot_size=1),
                                         make_instrument(lot=10))
        self.assertEqual((q, shares), (2, 25))
        self.assertTrue(any("округляем вниз" in w for w in warns))

    # ── граничные случаи ────────────────────────────────────────────────────

    def test_shares_below_one_api_lot_returns_zero(self):
        """Акций меньше одного API-лота → 0, а не «докупить до лота».

        Раньше здесь стоял max(1, shares // inst.lot): при 1 акции и лоте 10
        функция возвращала 1 лот, то есть покупала В 10 РАЗ больше, чем
        посчитала модель, и обходила проверку `if api_q <= 0` у вызывающего.
        """
        q, shares, warns = _api_quantity(make_order(quantity_lots=1, lot_size=1),
                                         make_instrument(lot=10))
        self.assertEqual(q, 0, "размер меньше лота обязан давать 0")
        self.assertEqual(shares, 1)
        self.assertTrue(any("меньше API-лота" in w for w in warns))

    def test_zero_lots_returns_zero(self):
        q, shares, warns = _api_quantity(make_order(quantity_lots=0, lot_size=10),
                                         make_instrument(lot=10))
        self.assertEqual((q, shares), (0, 0))
        self.assertIn("нулевое число акций", warns)

    def test_none_lots_returns_zero(self):
        q, shares, _ = _api_quantity(make_order(quantity_lots=None, lot_size=10),
                                     make_instrument(lot=10))
        self.assertEqual((q, shares), (0, 0))

    def test_quantity_is_never_negative(self):
        """Ни одна комбинация не должна давать отрицательный размер."""
        for lots in (None, 0, 1, 3, 17):
            for dash_lot in (1, 10, 100, 1000):
                for api_lot in (1, 10, 1000, 10000):
                    q, shares, _ = _api_quantity(
                        make_order(quantity_lots=lots, lot_size=dash_lot),
                        make_instrument(lot=api_lot))
                    self.assertGreaterEqual(q, 0, f"{lots}×{dash_lot} @ API {api_lot}")
                    self.assertGreaterEqual(shares, 0)

    def test_never_buys_more_shares_than_computed(self):
        """Инвариант сайзинга: отправленный объём ≤ посчитанного моделью."""
        for lots in (1, 2, 7):
            for dash_lot in (1, 10, 100):
                for api_lot in (1, 10, 100, 1000):
                    o = make_order(quantity_lots=lots, lot_size=dash_lot)
                    q, shares, _ = _api_quantity(o, make_instrument(lot=api_lot))
                    self.assertLessEqual(
                        q * api_lot, shares,
                        f"{lots}×{dash_lot}: отправили {q * api_lot} акций "
                        f"вместо {shares}")

    def test_unknown_lot_is_warned(self):
        _, _, warns = _api_quantity(make_order(quantity_lots=5, lot_size=1, lot_known=False),
                                    make_instrument(lot=1))
        self.assertTrue(any("не подтверждён" in w for w in warns))


class TestBuildOrdersSizing(unittest.TestCase):
    """Сайзинг в build_orders: фикс-размер против риск-паритета."""

    @staticmethod
    def _row(**over):
        r = make_dashboard_row(ticker="GMKN", anchor_price=130.0,
                               f_low=128.0, f_high=134.0, down=-2.0)
        r.update(over)
        return r

    def test_fixed_position_skips_when_lot_exceeds_limit(self):
        """Лот дороже лимита позиции → бумага пропускается, а не доливается.

        GMKN: 1 лот ≈ 1296 ₽ при лимите 1000 ₽. Прежний max(1, ...) отправлял
        заявку на 1296 ₽ — незапланированное плечо/овердрафт и перекос
        диверсификации на дорогих лотах.
        """
        o = build_orders([self._row()], position_rub=1000.0, entry_frac=0.2)[0]
        self.assertIsNone(o.quantity_lots)
        self.assertIsNone(o.total_rub)
        self.assertFalse(o.is_placeable)

    def test_risk_parity_skips_when_budget_below_one_lot(self):
        """В режиме бюджета лот НЕ форсируется: не хватило — заявки нет."""
        o = build_orders([self._row()], position_rub=1000.0,
                         entry_frac=0.2, budget_rub=500.0)[0]
        self.assertIsNone(o.quantity_lots)
        self.assertIsNone(o.total_rub)
        self.assertFalse(o.is_placeable)

    def test_both_modes_share_the_same_contract(self):
        """Контракт одинаков: не хватает на лот — None в обоих режимах,
        хватает — обе ветки считают лоты округлением ВНИЗ."""
        row = self._row()
        cheap_fixed = build_orders([row], position_rub=500.0, entry_frac=0.2)[0]
        cheap_budget = build_orders([row], position_rub=500.0,
                                    entry_frac=0.2, budget_rub=500.0)[0]
        self.assertIsNone(cheap_fixed.quantity_lots)
        self.assertIsNone(cheap_budget.quantity_lots)
        self.assertEqual(cheap_fixed.is_placeable, cheap_budget.is_placeable)

        rich_fixed = build_orders([row], position_rub=50_000.0, entry_frac=0.2)[0]
        rich_budget = build_orders([row], position_rub=50_000.0,
                                   entry_frac=0.2, budget_rub=50_000.0)[0]
        self.assertEqual(rich_fixed.quantity_lots, rich_budget.quantity_lots)
        self.assertTrue(rich_fixed.is_placeable and rich_budget.is_placeable)

    def test_fixed_position_never_exceeds_limit(self):
        """Инвариант: сумма позиции не превышает заданный лимит ни на одной
        лотности — именно это ломал прежний max(1, ...)."""
        cases = [("GMKN", 130.0), ("OGKB", 0.52), ("LKOH", 5000.0),
                 ("VTBR", 0.021), ("MSNG", 3.1)]
        for tk, price in cases:
            for limit in (500.0, 1000.0, 5000.0, 50_000.0):
                o = build_orders([self._row(ticker=tk, anchor_price=price,
                                            f_low=price * 0.98, f_high=price * 1.03)],
                                 position_rub=limit, entry_frac=0.2)[0]
                if o.total_rub is not None:
                    self.assertLessEqual(
                        o.total_rub, limit,
                        f"{tk} @ {price}: позиция {o.total_rub:.2f} ₽ > лимита {limit} ₽")

    def test_lot_that_fits_is_still_placed(self):
        """Регрессия наоборот: когда лот влезает, заявка по-прежнему считается."""
        o = build_orders([self._row()], position_rub=5000.0, entry_frac=0.2)[0]
        self.assertEqual(o.quantity_lots, 3)          # 5000 / (129.60 × 10)
        self.assertLessEqual(o.total_rub, 5000.0)
        self.assertTrue(o.is_placeable)

    def test_unavailable_ticker_gets_no_size(self):
        os.environ["TINKOFF_UNAVAILABLE_TICKERS"] = "GMKN"
        self.addCleanup(os.environ.pop, "TINKOFF_UNAVAILABLE_TICKERS", None)
        o = build_orders([self._row()], position_rub=100_000.0, entry_frac=0.2)[0]
        self.assertTrue(o.unavailable)
        self.assertIsNone(o.quantity_lots)
        self.assertFalse(o.is_placeable)


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Нешортабельные бумаги
# ═══════════════════════════════════════════════════════════════════════════

class TestNonShortableTickers(unittest.TestCase):
    """Шорт по бумагам без маржиналки не должен доходить до брокера.

    Базовый список сверен с InstrumentsService/ShareBy по всем 46 тикерам
    config.TICKERS: shortEnabledFlag=false ровно у AKRN, CBOM, MVID.
    """

    def setUp(self):
        os.environ.pop("NON_SHORTABLE_TICKERS", None)
        self.addCleanup(os.environ.pop, "NON_SHORTABLE_TICKERS", None)

    def test_default_blacklist(self):
        self.assertEqual(_DEFAULT_NON_SHORTABLE, {"AKRN", "CBOM", "MVID"})
        self.assertTrue({"AKRN", "CBOM", "MVID"} <= non_shortable_tickers())

    def test_env_extends_blacklist(self):
        os.environ["NON_SHORTABLE_TICKERS"] = "zzzz yyyy"
        got = non_shortable_tickers()
        self.assertIn("ZZZZ", got, "расширение из .env должно приводиться к верхнему регистру")
        self.assertIn("YYYY", got)
        self.assertTrue(_DEFAULT_NON_SHORTABLE <= got, "базовый список не должен теряться")

    # ── фильтр строк дашборда ───────────────────────────────────────────────

    @staticmethod
    def _rows():
        return [
            make_dashboard_row(ticker="AKRN", strategy="intraday_short",
                               direction="SHORT", exp_pnl=1.0),
            make_dashboard_row(ticker="AKRN", strategy="intraday_long",
                               direction="LONG", exp_pnl=0.1),
            make_dashboard_row(ticker="SBER", strategy="intraday_short",
                               direction="SHORT", exp_pnl=0.5),
        ]

    def test_blocked_short_is_dropped(self):
        kept = _drop_blocked_shorts(self._rows())
        self.assertNotIn(("AKRN", "SHORT"), [(r["ticker"], r["direction"]) for r in kept])

    def test_long_on_blocked_ticker_survives(self):
        """Бумага не выпадает целиком — LONG-кандидат по ней остаётся."""
        kept = _drop_blocked_shorts(self._rows())
        self.assertIn(("AKRN", "LONG"), [(r["ticker"], r["direction"]) for r in kept])

    def test_short_on_allowed_ticker_survives(self):
        kept = _drop_blocked_shorts(self._rows())
        self.assertIn(("SBER", "SHORT"), [(r["ticker"], r["direction"]) for r in kept])

    def test_filter_logs_skip(self):
        with self.assertLogs("tft.combined", level="INFO") as cm:
            _drop_blocked_shorts(self._rows())
        self.assertTrue(any("[SKIP SHORT] AKRN" in line for line in cm.output))

    def test_select_top_rows_excludes_blocked_short(self):
        """Сквозная проверка: сигнал не доходит до отбора топ-N.

        У AKRN шорт выгоднее лонга, поэтому без фильтра победил бы именно он.
        """
        forecasts = {tk: {"anchor_price": 100.0, "ForecastLow": 98.0,
                          "ForecastHigh": 104.0, "RangePct": 6.0,
                          "CoverageProb": 0.8, "LiqScore": 50,
                          "directional": {
                              "intraday_short": {"ExpPnL": 1.0, "ProbProfit": 0.8,
                                                 "Downside": -2.0, "Upside": 3.0},
                              "intraday_long": {"ExpPnL": 0.1, "ProbProfit": 0.55,
                                                "Downside": -2.0, "Upside": 3.0}}}
                     for tk in ("AKRN", "SBER")}
        top = select_top_rows(None, forecasts, ["AKRN", "SBER"],
                              ["intraday_short", "intraday_long"], top_n=10)
        picked = {(r["ticker"], r["direction"]) for r in top}
        self.assertNotIn(("AKRN", "SHORT"), picked)
        self.assertIn(("SBER", "SHORT"), picked, "шортабельная бумага не должна страдать")

    # ── вторая линия защиты: build_orders ───────────────────────────────────

    def test_build_orders_marks_blocked_short_unavailable(self):
        row = make_dashboard_row(ticker="MVID", strategy="intraday_short",
                                 direction="SHORT")
        o = build_orders([row], position_rub=50_000.0, entry_frac=0.2)[0]
        self.assertTrue(o.unavailable)
        self.assertIsNone(o.quantity_lots)
        self.assertFalse(o.is_placeable)

    def test_build_orders_allows_long_on_blocked_ticker(self):
        row = make_dashboard_row(ticker="MVID", strategy="intraday_long",
                                 direction="LONG")
        o = build_orders([row], position_rub=50_000.0, entry_frac=0.2)[0]
        self.assertFalse(o.unavailable)
        self.assertTrue(o.is_placeable)

    # ── гард по живому флагу брокера ────────────────────────────────────────

    def test_live_flag_blocks_short_entry(self):
        o = make_order(direction="SHORT")
        self.assertTrue(_short_blocked(o, self._instrument(short_enabled=False)))
        self.assertFalse(_short_blocked(o, self._instrument(short_enabled=True)))

    def test_live_flag_does_not_block_long(self):
        """Выход из лонга — тоже SELL, но маржи не требует: блокировать нельзя."""
        o = make_order(direction="LONG")
        self.assertFalse(_short_blocked(o, self._instrument(short_enabled=False)))

    @staticmethod
    def _instrument(short_enabled: bool):
        from dataclasses import replace
        return replace(make_instrument(lot=1), short_enabled=short_enabled)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Цены входа, стопа и тейк-профита
# ═══════════════════════════════════════════════════════════════════════════

class TestLimitEntryPrice(unittest.TestCase):
    """Вход — лимитка внутри квантильного коридора по LIMIT_ENTRY_FRACTION."""

    def test_long_moves_down_toward_f_low(self):
        price, note = _limit_entry_price("LONG", 100.0, 96.0, 104.0, 0.25)
        self.assertAlmostEqual(price, 99.0)          # 100 − 0.25·(100−96)
        self.assertIn("F.Low", note)

    def test_short_moves_up_toward_f_high(self):
        price, note = _limit_entry_price("SHORT", 100.0, 96.0, 104.0, 0.25)
        self.assertAlmostEqual(price, 101.0)         # 100 + 0.25·(104−100)
        self.assertIn("F.High", note)

    def test_frac_zero_is_spot(self):
        for direction in ("LONG", "SHORT"):
            price, _ = _limit_entry_price(direction, 100.0, 96.0, 104.0, 0.0)
            self.assertAlmostEqual(price, 100.0)

    def test_frac_one_hits_corridor_edge(self):
        self.assertAlmostEqual(_limit_entry_price("LONG", 100.0, 96.0, 104.0, 1.0)[0], 96.0)
        self.assertAlmostEqual(_limit_entry_price("SHORT", 100.0, 96.0, 104.0, 1.0)[0], 104.0)

    def test_entry_always_inside_corridor(self):
        for frac in (0.0, 0.2, 0.5, 0.8, 1.0):
            for direction in ("LONG", "SHORT"):
                price, _ = _limit_entry_price(direction, 100.0, 96.0, 104.0, frac)
                self.assertGreaterEqual(price, 96.0)
                self.assertLessEqual(price, 104.0)

    def test_long_entry_never_worse_than_spot(self):
        """LONG покупает не дороже спота, SHORT продаёт не дешевле."""
        long_p, _ = _limit_entry_price("LONG", 100.0, 96.0, 104.0, 0.3)
        short_p, _ = _limit_entry_price("SHORT", 100.0, 96.0, 104.0, 0.3)
        self.assertLessEqual(long_p, 100.0)
        self.assertGreaterEqual(short_p, 100.0)

    def test_degenerate_corridor_falls_back_to_anchor(self):
        """Сторона коридора «не лучше» спота → вход по споту, без вывернутой цены."""
        self.assertEqual(_limit_entry_price("LONG", 100.0, 101.0, 104.0, 0.8),
                         (100.0, "anchor"))
        self.assertEqual(_limit_entry_price("SHORT", 100.0, 96.0, 99.0, 0.8),
                         (100.0, "anchor"))

    def test_missing_anchor_returns_none(self):
        self.assertEqual(_limit_entry_price("LONG", None, 96.0, 104.0, 0.5)[0], None)
        self.assertEqual(_limit_entry_price("LONG", 0.0, 96.0, 104.0, 0.5)[0], None)


class TestTakeProfitPrice(unittest.TestCase):
    """Тейк — доля пути от входа к ПРОТИВОПОЛОЖНОЙ границе коридора."""

    def test_long_targets_f_high(self):
        self.assertAlmostEqual(_take_profit_price("LONG", 99.0, 96.0, 104.0, 1.0), 104.0)
        self.assertAlmostEqual(_take_profit_price("LONG", 99.0, 96.0, 104.0, 0.5), 101.5)

    def test_short_targets_f_low(self):
        self.assertAlmostEqual(_take_profit_price("SHORT", 101.0, 96.0, 104.0, 1.0), 96.0)
        self.assertAlmostEqual(_take_profit_price("SHORT", 101.0, 96.0, 104.0, 0.5), 98.5)

    def test_take_profit_is_profitable_side(self):
        long_tp = _take_profit_price("LONG", 99.0, 96.0, 104.0, 0.8)
        short_tp = _take_profit_price("SHORT", 101.0, 96.0, 104.0, 0.8)
        self.assertGreater(long_tp, 99.0, "тейк LONG обязан быть выше входа")
        self.assertLess(short_tp, 101.0, "тейк SHORT обязан быть ниже входа")

    def test_none_when_corridor_not_profitable(self):
        self.assertIsNone(_take_profit_price("LONG", 105.0, 96.0, 104.0, 1.0))
        self.assertIsNone(_take_profit_price("SHORT", 95.0, 96.0, 104.0, 1.0))
        self.assertIsNone(_take_profit_price("LONG", None, 96.0, 104.0, 1.0))


class TestTakeProfitDefault(unittest.TestCase):
    """Дефолт LIMIT_TP_FRACTION = 0.5 и его смысл.

    Основание для значения — моделирование исполнения на 5-минутном пути цены
    (2 577 сигналов): при 1.0 цель достигалась в 1,9% сделок, при 0.5 — в 11,2%.
    См. AUDIT-PROFITABILITY-REPORT.md, раздел 6.
    """

    def test_config_default_is_half(self):
        import importlib
        import config
        importlib.reload(config)
        self.assertAlmostEqual(config.LIMIT_TP_FRACTION, 0.5)

    def test_build_orders_uses_config_not_hardcoded_one(self):
        """tp_frac не передан → берётся из конфигурации, а не 1.0 из сигнатуры.

        Регрессия: раньше в build_orders и compute_orders стояла захардкоженная
        1.0, которая молча побеждала LIMIT_TP_FRACTION при вызове без аргумента.
        """
        import config
        row = make_dashboard_row(anchor_price=100.0, f_low=98.0, f_high=104.0)
        o = build_orders([row], position_rub=100_000.0, entry_frac=0.2)[0]

        entry = o.entry_price
        expected = _take_profit_price("LONG", entry, 98.0, 104.0,
                                      config.LIMIT_TP_FRACTION)
        self.assertAlmostEqual(o.tp_price, expected)
        far_edge = _take_profit_price("LONG", entry, 98.0, 104.0, 1.0)
        self.assertLess(o.tp_price, far_edge,
                        "дефолтный тейк обязан быть ближе входа, чем дальняя граница")

    def test_explicit_argument_still_wins(self):
        row = make_dashboard_row(anchor_price=100.0, f_low=98.0, f_high=104.0)
        o = build_orders([row], position_rub=100_000.0, entry_frac=0.2,
                         tp_frac=1.0)[0]
        self.assertAlmostEqual(o.tp_price, 104.0)

    def test_target_is_midpoint_between_entry_and_far_edge(self):
        """0.5 — это середина отрезка ВХОД→ДАЛЬНЯЯ ГРАНИЦА, а не медиана прогноза.

        Различие существенное и легко теряется при чтении конфигурации.
        Дальняя граница коридора — это q0.9 (для LONG) и q0.1 (для SHORT), а
        не q0.5, поэтому tp_frac=0.5 ставит цель ВЫШЕ медианы прогноза,
        примерно на 40% пути от якоря к верхней границе.
        """
        entry, f_low, f_high = 99.0, 96.0, 104.0
        tp = _take_profit_price("LONG", entry, f_low, f_high, 0.5)
        self.assertAlmostEqual(tp, (entry + f_high) / 2.0)

        anchor = 100.0                      # медиана прогноза ≈ якорь
        self.assertGreater(tp, anchor,
                           "0.5 ставит цель выше медианы прогноза, а не на неё")

    def test_closer_target_is_monotonically_easier_to_reach(self):
        """Чем меньше tp_frac, тем ближе цель ко входу — и тем достижимее.

        Именно это свойство и есть причина смены дефолта: при 1.0 механизм
        take-profit практически не участвовал в сделке.
        """
        entry, f_low, f_high = 99.0, 96.0, 104.0
        longs = [_take_profit_price("LONG", entry, f_low, f_high, f)
                 for f in (0.2, 0.4, 0.5, 0.8, 1.0)]
        self.assertEqual(longs, sorted(longs), "цель LONG должна расти вместе с frac")
        for tp in longs:
            self.assertGreater(tp, entry)

        shorts = [_take_profit_price("SHORT", 101.0, f_low, f_high, f)
                  for f in (0.2, 0.4, 0.5, 0.8, 1.0)]
        self.assertEqual(shorts, sorted(shorts, reverse=True),
                         "цель SHORT должна опускаться вместе с ростом frac")
        for tp in shorts:
            self.assertLess(tp, 101.0)

    def test_default_tp_is_reachable_within_forecast_corridor(self):
        """Дефолтная цель лежит строго внутри прогнозного коридора."""
        import config
        row = make_dashboard_row(anchor_price=100.0, f_low=98.0, f_high=104.0)
        o = build_orders([row], position_rub=100_000.0, entry_frac=0.2)[0]
        self.assertGreater(o.tp_price, o.entry_price)
        self.assertLess(o.tp_price, 104.0)
        self.assertGreater(config.LIMIT_TP_FRACTION, 0.0)
        self.assertLessEqual(config.LIMIT_TP_FRACTION, 1.0)


class TestStopGeometry(unittest.TestCase):
    """Стоп ставится от ВХОДА (не от спота) и в правильную сторону."""

    def test_long_stop_below_entry_short_above(self):
        rows = [make_dashboard_row(ticker="LKOH", direction="LONG",
                                   strategy="intraday_long", down=-2.0),
                make_dashboard_row(ticker="LKOH", direction="SHORT",
                                   strategy="intraday_short", down=-2.0)]
        long_o, short_o = build_orders(rows, position_rub=100_000.0, entry_frac=0.2)
        self.assertLess(long_o.stop_price, long_o.entry_price)
        self.assertGreater(short_o.stop_price, short_o.entry_price)
        for o in (long_o, short_o):
            self.assertAlmostEqual(o.stop_pct, 2.0)
            self.assertAlmostEqual(abs(o.stop_price - o.entry_price) / o.entry_price * 100,
                                   2.0, places=6)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Риск-паритет
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskParityAllocation(unittest.TestCase):
    """Бюджет делится ∝ 1/стоп%, чтобы рублёвый риск был одинаков."""

    def test_equal_ruble_risk_across_positions(self):
        geoms = [make_geom(1.0), make_geom(2.0), make_geom(4.0)]
        alloc = _risk_parity_alloc(geoms, 100_000.0)
        risks = [a * g["stop_pct"] / 100.0 for a, g in zip(alloc, geoms)]
        for r in risks[1:]:
            self.assertAlmostEqual(r, risks[0], places=6,
                                   msg=f"риск неодинаков: {risks}")

    def test_weights_inverse_to_stop(self):
        """Стоп вдвое шире → денег вдвое меньше."""
        alloc = _risk_parity_alloc([make_geom(1.0), make_geom(2.0)], 30_000.0)
        self.assertAlmostEqual(alloc[0], 20_000.0)
        self.assertAlmostEqual(alloc[1], 10_000.0)

    def test_budget_fully_distributed(self):
        geoms = [make_geom(1.5), make_geom(2.5), make_geom(3.0)]
        alloc = _risk_parity_alloc(geoms, 50_000.0)
        self.assertAlmostEqual(sum(alloc), 50_000.0, places=6)

    def test_liquidity_cap_applies(self):
        """MaxPos ограничивает аллокацию сверху."""
        geoms = [make_geom(1.0, max_pos=5_000.0), make_geom(1.0)]
        alloc = _risk_parity_alloc(geoms, 100_000.0)
        self.assertAlmostEqual(alloc[0], 5_000.0)
        self.assertAlmostEqual(alloc[1], 50_000.0)

    def test_non_sizable_gets_nothing(self):
        geoms = [make_geom(2.0), make_geom(2.0, sizable=False)]
        alloc = _risk_parity_alloc(geoms, 10_000.0)
        self.assertAlmostEqual(alloc[0], 10_000.0)
        self.assertEqual(alloc[1], 0.0)

    def test_all_non_sizable_returns_zeros(self):
        alloc = _risk_parity_alloc([make_geom(2.0, sizable=False)] * 3, 10_000.0)
        self.assertEqual(alloc, [0.0, 0.0, 0.0])

    def test_allocations_are_never_negative(self):
        alloc = _risk_parity_alloc(
            [make_geom(0.1), make_geom(50.0), make_geom(2.0, sizable=False)], 1_000.0)
        for a in alloc:
            self.assertGreaterEqual(a, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FinalScore и риск-штрафы
# ═══════════════════════════════════════════════════════════════════════════

class TestRawAlphaScore(unittest.TestCase):
    """Режим сырой альфы (USE_RAW_ALPHA_SCORE=1, дефолт).

    Рейтинг = ExpPnL модели, скорректированный риск-штрафами. Основание —
    квант-аудит: exp_pnl в одиночку даёт Rank IC +0.0544 (t 2.99), а прежняя
    смесь из семи компонент — +0.0262 (t 1.30), то есть белый шум.
    """

    def score(self, strict=False, hard_exclude=True, apply_penalties=True, **over):
        # Штрафы включаются ЯВНО: этот класс проверяет их механику, а
        # канонический дефолт config.APPLY_RISK_PENALTIES теперь 0.
        return _score_row(make_dashboard_row(**over), strict,
                          raw_alpha=True, hard_exclude=hard_exclude,
                          apply_penalties=apply_penalties)

    def test_score_equals_exp_pnl_when_no_risk(self):
        """Без риск-условий рейтинг равен ExpPnL — без примесей."""
        final, allowed, flags = self.score(exp_pnl=0.42)
        self.assertAlmostEqual(final, 0.42, places=9)
        self.assertTrue(allowed)
        self.assertEqual(flags, [])

    def test_ranking_follows_exp_pnl(self):
        """Порядок кандидатов определяется только ExpPnL."""
        vals = [-1.0, -0.2, 0.0, 0.3, 1.5]
        scores = [self.score(exp_pnl=v)[0] for v in vals]
        self.assertEqual(scores, sorted(scores))

    def test_liquidity_and_prob_do_not_move_ranking(self):
        """Шумовые компоненты больше не влияют на рейтинг."""
        base, _, _ = self.score(exp_pnl=0.3)
        for over in ({"liq_score": 0}, {"liq_score": 100},
                     {"prob_profit": 0.0}, {"prob_profit": 1.0}):
            final, _, _ = self.score(exp_pnl=0.3, **over)
            self.assertAlmostEqual(final, base, places=9)

    def test_rs_no_longer_ranks_but_still_filters_shorts(self):
        """rs убран из рейтинга (IC -0.002), но остался риск-фильтром шорта."""
        long_hi, _, flags = self.score(exp_pnl=0.3, rs=6.0)
        long_lo, _, _ = self.score(exp_pnl=0.3, rs=-6.0)
        self.assertAlmostEqual(long_hi, long_lo, places=9)
        self.assertEqual(flags, [])

        short, _, sflags = self.score(exp_pnl=0.3, direction="SHORT",
                                      strategy="intraday_short", rs=6.0)
        self.assertIn("High Risk Short", sflags)
        self.assertAlmostEqual(short, 0.3 * 0.75, places=9)

    def test_rs_removed_from_long_overnight(self):
        """На long_overnight rs прямо вредил (IC -0.0371, t -4.30) — его нет."""
        for rs in (-8.0, 0.0, 8.0):
            final, _, _ = self.score(exp_pnl=0.25, strategy="long_overnight",
                                     direction="LONG", rs=rs)
            self.assertAlmostEqual(final, 0.25, places=9)

    def test_regime_no_longer_ranks_within_direction(self):
        """regime_score убран: NEUTRAL и BULL для LONG дают один рейтинг.

        regime как КОМПОНЕНТ неотличим от нуля (IC +0.0088, t +0.26): в день он
        принимает всего два значения — одно на все LONG, другое на все SHORT.
        """
        neutral, _, _ = self.score(exp_pnl=0.3, regime="NEUTRAL")
        bull, _, _ = self.score(exp_pnl=0.3, regime="BULL")
        self.assertAlmostEqual(neutral, bull, places=9)

    def test_regime_penalty_survives_as_direction_tilt(self):
        """Штраф за контртренд остаётся: он наклоняет LONG против SHORT."""
        final, _, _ = self.score(exp_pnl=0.3, regime="BEAR")
        self.assertAlmostEqual(final, 0.3 * 0.70, places=9)

    # ── знак-безопасность штрафов ───────────────────────────────────────────

    def test_penalty_never_improves_a_negative_score(self):
        """Ключевая ловушка: -0.5 x 0.70 = -0.35 подняло бы плохой сигнал."""
        clean, _, _ = self.score(exp_pnl=-0.5)
        penalised, _, _ = self.score(exp_pnl=-0.5, regime="BEAR")
        self.assertLess(penalised, clean,
                        "штраф обязан ухудшать рейтинг и на отрицательной стороне")
        self.assertAlmostEqual(penalised, -0.5 / 0.70, places=9)

    def test_penalty_direction_is_consistent_across_sign(self):
        for exp in (-2.0, -0.1, 0.0, 0.1, 2.0):
            clean, _, _ = self.score(exp_pnl=exp)
            pen, _, _ = self.score(exp_pnl=exp, regime="BEAR")
            self.assertLessEqual(pen, clean, f"ExpPnL={exp}")

    def test_apply_penalty_helper(self):
        self.assertAlmostEqual(_apply_penalty(1.0, 0.7), 0.7)
        self.assertAlmostEqual(_apply_penalty(-1.0, 0.7), -1.0 / 0.7)
        self.assertAlmostEqual(_apply_penalty(0.0, 0.7), 0.0)
        self.assertAlmostEqual(_apply_penalty(5.0, 1.0), 5.0)

    # ── жёсткое отсечение тяжёлых рисков ────────────────────────────────────

    def test_volume_climax_excludes_when_hard(self):
        _, allowed, flags = self.score(exp_pnl=1.0, vol_spike=5.0)
        self.assertFalse(allowed)
        self.assertIn("⚠ Volume Climax", flags)

    def test_volume_climax_only_penalises_when_soft(self):
        final, allowed, _ = self.score(exp_pnl=1.0, vol_spike=5.0, hard_exclude=False)
        self.assertTrue(allowed)
        self.assertAlmostEqual(final, 0.70, places=9)

    def test_volume_climax_threshold_is_strict(self):
        _, allowed_at, _ = self.score(exp_pnl=1.0, vol_spike=4.0)
        _, allowed_above, _ = self.score(exp_pnl=1.0, vol_spike=4.01)
        self.assertTrue(allowed_at)
        self.assertFalse(allowed_above)

    def test_gap_risk_excludes_overnight_only(self):
        _, allowed, flags = self.score(exp_pnl=1.0, strategy="long_overnight",
                                       gap_down_prob=0.55)
        self.assertFalse(allowed)
        self.assertIn("⚠ High Overnight Risk", flags)

        _, allowed_intraday, iflags = self.score(exp_pnl=1.0,
                                                 strategy="intraday_long",
                                                 gap_down_prob=0.9)
        self.assertTrue(allowed_intraday)
        self.assertEqual(iflags, [])

    def test_strict_market_filter_still_blocks(self):
        _, allowed, _ = self.score(exp_pnl=1.0, strict=True, regime="BEAR")
        self.assertFalse(allowed)

    def test_missing_exp_pnl_scores_zero(self):
        final, _, _ = self.score(exp_pnl=None)
        self.assertEqual(final, 0.0)


class TestHeuristicScoreReweighted(unittest.TestCase):
    """Режим USE_RAW_ALPHA_SCORE=0: веса exp 0.70 / prob 0.20 / liq 0.10."""

    def score(self, strict=False, **over):
        return _score_row(make_dashboard_row(**over), strict,
                          raw_alpha=False, hard_exclude=False)

    NEUTRAL = 0.70 * 0.5 + 0.20 * 0.5 + 0.10 * 0.5   # = 0.5

    def test_neutral_row_baseline(self):
        final, allowed, flags = self.score()
        self.assertAlmostEqual(final, self.NEUTRAL, places=9)
        self.assertTrue(allowed)
        self.assertEqual(flags, [])

    def test_weights_sum_to_one(self):
        """Максимум по всем трём компонентам даёт ровно 1.0 — веса нормированы."""
        final, _, _ = self.score(exp_pnl=1.0, prob_profit=1.0, liq_score=100)
        self.assertAlmostEqual(final, 1.0, places=9)

    def test_exp_dominates(self):
        """Вес exp 0.70 больше суммы остальных."""
        exp_only, _, _ = self.score(exp_pnl=1.0, prob_profit=0.5, liq_score=50)
        rest_only, _, _ = self.score(exp_pnl=0.0, prob_profit=1.0, liq_score=100)
        self.assertGreater(exp_only, rest_only)

    def test_score_stays_in_unit_range(self):
        for over in ({}, {"exp_pnl": 5.0, "prob_profit": 1.0, "liq_score": 100},
                     {"exp_pnl": -5.0, "prob_profit": 0.0, "liq_score": 0,
                      "vol_spike": 9.0}):
            final, _, _ = self.score(**over)
            self.assertGreaterEqual(final, 0.0)
            self.assertLessEqual(final, 1.0)

    def test_no_rs_no_regime_no_vol_in_formula(self):
        base, _, _ = self.score()
        for over in ({"rs": 9.0}, {"rs": -9.0}, {"regime": "BULL"},
                     {"vol_spike": 1.9}, {"vol_spike": 0.5}):
            final, _, _ = self.score(**over)
            self.assertAlmostEqual(final, base, places=9,
                                   msg=f"{over} не должен влиять на рейтинг")


class TestTradeScoreMode(unittest.TestCase):
    """Режим trade_score: ExpPnL, нормированный на волатильность."""

    def score(self, **over):
        return _score_row(make_dashboard_row(**over), False, mode="trade_score",
                          apply_penalties=False, hard_exclude=False)

    def test_normalises_by_atr(self):
        """+0.8% при ATR 1% должно стоять выше +1.2% при ATR 3%."""
        calm, _, _ = self.score(exp_pnl=0.8, atr_pct=1.0)
        wild, _, _ = self.score(exp_pnl=1.2, atr_pct=3.0)
        self.assertGreater(calm, wild)
        self.assertAlmostEqual(calm, 0.8, places=9)
        self.assertAlmostEqual(wild, 0.4, places=9)

    def test_costs_are_not_subtracted_twice(self):
        """ExpPnL приходит уже нетто round-trip — вычитать издержки нельзя.

        Регрессия на формулу (exp_pnl - CostRT)/ATR%: она вычла бы издержки
        второй раз, см. directional.strategy_pnl (exp_net = med - cost_rt).
        """
        final, _, _ = self.score(exp_pnl=0.5, atr_pct=1.0)
        self.assertAlmostEqual(final, 0.5, places=9)

    def test_falls_back_to_range_when_atr_missing(self):
        """Без ATR% знаменателем становится ширина коридора, делённая на 4."""
        final, _, _ = self.score(exp_pnl=1.0, atr_pct=None, range_pct=8.0)
        self.assertAlmostEqual(final, 1.0 / 2.0, places=9)

    def test_denominator_floor_prevents_blowup(self):
        """Околонулевой ATR не должен давать бесконечный рейтинг."""
        final, _, _ = self.score(exp_pnl=1.0, atr_pct=0.0001)
        self.assertAlmostEqual(final, 1.0 / 0.2, places=9)

    def test_sign_is_preserved(self):
        neg, _, _ = self.score(exp_pnl=-1.0, atr_pct=2.0)
        self.assertLess(neg, 0.0)

    def test_missing_exp_pnl_scores_zero(self):
        final, _, _ = self.score(exp_pnl=None, atr_pct=1.0)
        self.assertEqual(final, 0.0)


class TestScoreModeResolution(unittest.TestCase):
    """Выбор режима и обратная совместимость со старым булевым флагом."""

    def row(self, **over):
        base = dict(exp_pnl=0.4, prob_profit=0.5, liq_score=50, atr_pct=1.0)
        base.update(over)
        return make_dashboard_row(**base)

    def test_explicit_mode_wins(self):
        raw, _, _ = _score_row(self.row(), False, mode="raw_alpha",
                               apply_penalties=False)
        self.assertAlmostEqual(raw, 0.4, places=9)

    def test_legacy_true_maps_to_raw_alpha(self):
        legacy, _, _ = _score_row(self.row(), False, raw_alpha=True,
                                  apply_penalties=False)
        explicit, _, _ = _score_row(self.row(), False, mode="raw_alpha",
                                    apply_penalties=False)
        self.assertAlmostEqual(legacy, explicit, places=9)

    def test_legacy_false_maps_to_heuristic(self):
        legacy, _, _ = _score_row(self.row(), False, raw_alpha=False,
                                  apply_penalties=False)
        explicit, _, _ = _score_row(self.row(), False, mode="heuristic",
                                    apply_penalties=False)
        self.assertAlmostEqual(legacy, explicit, places=9)

    def test_unknown_mode_falls_back_to_heuristic(self):
        bad, _, _ = _score_row(self.row(), False, mode="нет-такого",
                               apply_penalties=False)
        good, _, _ = _score_row(self.row(), False, mode="heuristic",
                                apply_penalties=False)
        self.assertAlmostEqual(bad, good, places=9)

    def test_config_default_is_heuristic(self):
        import importlib
        import config
        importlib.reload(config)
        self.assertEqual(config.SCORE_MODE, "heuristic")
        self.assertFalse(config.USE_RAW_ALPHA_SCORE)

    def test_modes_actually_differ(self):
        r = self.row(exp_pnl=1.2, atr_pct=3.0)
        scores = {m: _score_row(r, False, mode=m, apply_penalties=False)[0]
                  for m in ("heuristic", "raw_alpha", "trade_score")}
        self.assertEqual(len(set(round(v, 9) for v in scores.values())), 3)


class TestCanonicalDefaults(unittest.TestCase):
    """Канонические дефолты по итогам Спринта 2 — пин, чтобы их не сдвинули молча."""

    @staticmethod
    def cfg():
        import importlib
        import config
        return importlib.reload(config)

    def test_score_mode(self):
        """Победитель теста на устойчивость: первый в ОБЕИХ половинах выборки."""
        self.assertEqual(self.cfg().SCORE_MODE, "heuristic")

    def test_risk_penalties_disabled(self):
        """Штрафы вредят во всех режимах: альфа эвристики -1.2% -> +2.1%."""
        self.assertFalse(self.cfg().APPLY_RISK_PENALTIES)

    def test_take_profit_fraction(self):
        self.assertAlmostEqual(self.cfg().LIMIT_TP_FRACTION, 0.5)

    def test_intraday_square_off_enabled(self):
        self.assertTrue(self.cfg().INTRADAY_SQUARE_OFF_ENABLED)

    def test_square_off_time_is_before_closing_auction(self):
        """Аукцион закрытия основной сессии 18:40-18:50 — выходить нужно раньше."""
        import datetime as dt
        t = dt.datetime.strptime(self.cfg().INTRADAY_SQUARE_OFF_TIME, "%H:%M").time()
        self.assertLess(t, dt.time(18, 40))
        self.assertGreater(t, dt.time(10, 0))

    def test_penalties_off_means_no_penalty_applied(self):
        """Сквозная проверка: при дефолтах штраф действительно не применяется."""
        r = make_dashboard_row(exp_pnl=0.5, direction="SHORT",
                               strategy="intraday_short", rs=6.0, regime="BULL",
                               vol_spike=5.0)
        clean, allowed, flags = _score_row(r, False, mode="raw_alpha",
                                           apply_penalties=self.cfg().APPLY_RISK_PENALTIES,
                                           hard_exclude=False)
        self.assertAlmostEqual(clean, 0.5, places=9)
        # флаги остаются — они информируют, но больше не наказывают
        self.assertIn("High Risk Short", flags)
        self.assertIn("⚠ Volume Climax", flags)



class TestPenaltyToggle(unittest.TestCase):
    """APPLY_RISK_PENALTIES отключает мультипликативные штрафы."""

    def test_penalties_off_leaves_score_clean(self):
        r = make_dashboard_row(exp_pnl=0.5, direction="SHORT",
                               strategy="intraday_short", rs=6.0, regime="BULL")
        on, _, flags_on = _score_row(r, False, mode="raw_alpha",
                                     apply_penalties=True, hard_exclude=False)
        off, _, flags_off = _score_row(r, False, mode="raw_alpha",
                                       apply_penalties=False, hard_exclude=False)
        self.assertAlmostEqual(off, 0.5, places=9)
        self.assertLess(on, off)
        # флаги остаются в обоих случаях — они информируют, а не наказывают
        self.assertIn("High Risk Short", flags_on)
        self.assertIn("High Risk Short", flags_off)

    def test_hard_exclusion_independent_of_penalty_toggle(self):
        r = make_dashboard_row(exp_pnl=1.0, vol_spike=5.0)
        _, allowed, _ = _score_row(r, False, mode="raw_alpha",
                                   apply_penalties=False, hard_exclude=True)
        self.assertFalse(allowed)


class TestStrategySpecialisation(unittest.TestCase):
    """Путь А: торгуем только там, где измерено преимущество."""

    @staticmethod
    def row(strategy, **over):
        base = dict(strategy=strategy, direction=_DIR[strategy],
                    ret1=-1.0, market_atr_pctl=70.0, exp_pnl=1.0, cost_rt=0.13)
        base.update(over)
        return make_dashboard_row(**base)

    def keep(self, rows, **kw):
        kw.setdefault("verbose", False)
        return apply_strategy_specialisation(rows, **kw)

    # ── 1. меню стратегий ───────────────────────────────────────────────────
    def test_intraday_long_is_removed(self):
        """Самая убыточная стратегия (средняя -0.2393%, t -17.26) не торгуется."""
        rows = [self.row("intraday_long"), self.row("intraday_short"),
                self.row("long_overnight")]
        kept = self.keep(rows)
        self.assertNotIn("intraday_long", {r["strategy"] for r in kept})
        self.assertEqual(len(kept), 2)

    def test_allowed_set_is_configurable(self):
        rows = [self.row("intraday_long"), self.row("intraday_short")]
        kept = self.keep(rows, allowed={"intraday_long"})
        self.assertEqual([r["strategy"] for r in kept], ["intraday_long"])

    # ── 2. импульс продавцов для intraday_short ─────────────────────────────
    def test_short_kept_on_seller_momentum(self):
        r = self.row("intraday_short", ret1=-1.5, market_atr_pctl=70.0)
        self.assertEqual(len(self.keep([r])), 1)

    def test_short_dropped_on_calm_day(self):
        """Вчера рост — импульса продавцов нет."""
        r = self.row("intraday_short", ret1=+1.5, market_atr_pctl=70.0)
        self.assertEqual(self.keep([r]), [])

    def test_short_dropped_on_low_market_vol(self):
        r = self.row("intraday_short", ret1=-1.5, market_atr_pctl=30.0)
        self.assertEqual(self.keep([r]), [])

    def test_short_kept_when_data_missing(self):
        """Нет данных — не отбрасываем: отсутствие признака не есть отказ."""
        for over in ({"ret1": None}, {"market_atr_pctl": None}):
            r = self.row("intraday_short", **over)
            self.assertEqual(len(self.keep([r])), 1, over)

    def test_momentum_filter_can_be_disabled(self):
        r = self.row("intraday_short", ret1=+1.5, market_atr_pctl=30.0)
        self.assertEqual(len(self.keep([r], require_momentum=False)), 1)

    def test_momentum_filter_does_not_touch_overnight(self):
        """Фильтр импульса — только для intraday_short."""
        r = self.row("long_overnight", ret1=+2.0, market_atr_pctl=10.0)
        self.assertEqual(len(self.keep([r])), 1)

    # ── 3. порог преимущества для long_overnight ────────────────────────────
    def test_overnight_kept_above_threshold(self):
        r = self.row("long_overnight", exp_pnl=0.20, cost_rt=0.13)
        self.assertEqual(len(self.keep([r], overnight_k=0.5)), 1)   # 0.20 > 0.065

    def test_overnight_dropped_below_threshold(self):
        r = self.row("long_overnight", exp_pnl=0.05, cost_rt=0.13)
        self.assertEqual(self.keep([r], overnight_k=0.5), [])        # 0.05 < 0.065

    def test_overnight_threshold_zero_disables(self):
        r = self.row("long_overnight", exp_pnl=-1.0, cost_rt=0.13)
        self.assertEqual(len(self.keep([r], overnight_k=0.0)), 1)

    def test_overnight_threshold_does_not_touch_short(self):
        r = self.row("intraday_short", exp_pnl=-5.0, ret1=-1.0, market_atr_pctl=70.0)
        self.assertEqual(len(self.keep([r], overnight_k=10.0)), 1)

    def test_overnight_falls_back_to_config_cost(self):
        """Без cost_rt в строке берётся config.TFT_COST_RT, а не ноль."""
        r = self.row("long_overnight", exp_pnl=0.01, cost_rt=None)
        self.assertEqual(self.keep([r], overnight_k=1.0), [])

    # ── интеграция ──────────────────────────────────────────────────────────
    def test_config_defaults_wired(self):
        import importlib
        import config
        importlib.reload(config)
        self.assertEqual(set(config.TRADING_STRATEGIES),
                         {"long_overnight", "intraday_short"})
        self.assertNotIn("intraday_long", config.TRADING_STRATEGIES)
        # валидация продолжает считать все три — иначе перестанем видеть
        # что происходит с исключённой стратегией
        self.assertIn("intraday_long", config.VALIDATION_STRATS)

    def test_select_top_rows_applies_specialisation(self):
        rows = _select_specialised(["intraday_long", "intraday_short"])
        self.assertNotIn("intraday_long", {r["strategy"] for r in rows})

    def test_specialise_can_be_turned_off(self):
        rows = _select_specialised(["intraday_long", "intraday_short"],
                                   specialise=False)
        self.assertIn("intraday_long", {r["strategy"] for r in rows})


_DIR = {"intraday_long": "LONG", "intraday_short": "SHORT",
        "long_overnight": "LONG"}


def _select_specialised(strats, specialise=True):
    """select_top_rows на синтетическом прогнозе по заданным стратегиям."""
    tk = "AAA"
    val_rows = [{"ticker": tk, "strategy": st, "verdict": None, "white_rc_p": 0.5,
                 "spa_p": 0.5, "pbo": 0.1, "fdr_pass": True, "ruin30": 0.1,
                 "lb_struct": True} for st in strats]
    forecasts = {tk: {
        "ForecastLow": 98.0, "ForecastHigh": 104.0, "RangePct": 6.0,
        "CoverageProb": 0.8, "anchor_price": 100.0, "LiqScore": 50,
        "Regime": "NEUTRAL", "RS": 0.0, "VolSpike": 1.0, "ATRpctl": 50,
        "GapDownProb": 0.1, "Ret1": -1.0, "MarketATRpctl": 70.0, "CostRT": 0.13,
        "directional": {st: {"ExpPnL": 1.0, "ProbProfit": 0.6,
                             "Downside": -2.0, "Upside": 3.0} for st in strats}}}
    return select_top_rows(val_rows, forecasts, [tk], strats, top_n=10,
                           strict=False, show_all=True, specialise=specialise)


class TestValidationGate(unittest.TestCase):
    """Гейт по вердикту контура валидации (задача 2.2)."""

    @staticmethod
    def _rows(verdicts):
        return [make_dashboard_row(ticker=f"T{i}", verdict=v, exp_pnl=1.0 - i * 0.1)
                for i, v in enumerate(verdicts)]

    def test_is_rejected(self):
        self.assertTrue(is_rejected({"verdict": "REJECTED"}))
        self.assertFalse(is_rejected({"verdict": "CANDIDATE EDGE"}))
        self.assertFalse(is_rejected({"verdict": None}))
        self.assertFalse(is_rejected({}))

    def test_warn_counts_rejected(self):
        rows = self._rows(["REJECTED", "WEAK / INCONCLUSIVE", "REJECTED"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = warn_unvalidated(rows)
        self.assertEqual(n, 2)
        self.assertIn("REJECTED", buf.getvalue())

    def test_warn_silent_when_all_validated(self):
        rows = self._rows(["CANDIDATE EDGE", "WEAK / INCONCLUSIVE"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = warn_unvalidated(rows)
        self.assertEqual(n, 0)
        self.assertEqual(buf.getvalue(), "")

    def test_warn_mentions_prod(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            warn_unvalidated(self._rows(["REJECTED"]), env="PROD")
        self.assertIn("БОЕВОЙ", buf.getvalue())

    def test_gate_off_keeps_rejected(self):
        """Дефолт: вердикт не блокирует — прежнее поведение сохранено."""
        rows = _select(["REJECTED", "REJECTED"], gate=False)
        self.assertEqual(len(rows), 2)

    def test_gate_on_drops_rejected(self):
        rows = _select(["REJECTED", "CANDIDATE EDGE"], gate=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "CANDIDATE EDGE")

    def test_gate_on_can_empty_the_book(self):
        """138 из 138 комбинаций REJECTED — гейт останавливает торговлю целиком.
        Это ожидаемое поведение, а не сбой."""
        self.assertEqual(_select(["REJECTED"] * 5, gate=True), [])


def _select(verdicts, gate: bool):
    """Прогоняет select_top_rows на синтетическом прогнозе с заданными вердиктами."""
    tickers = [f"T{i}" for i in range(len(verdicts))]
    val_rows = [{"ticker": tk, "strategy": "long_overnight", "verdict": v,
                 "white_rc_p": 0.5, "spa_p": 0.5, "pbo": 0.1, "fdr_pass": True,
                 "ruin30": 0.1, "lb_struct": True}
                for tk, v in zip(tickers, verdicts)]
    forecasts = {
        tk: {"ForecastLow": 98.0, "ForecastHigh": 104.0, "RangePct": 6.0,
             "CoverageProb": 0.8, "anchor_price": 100.0, "LiqScore": 50,
             "Regime": "NEUTRAL", "RS": 0.0, "VolSpike": 1.0, "ATRpctl": 50,
             "GapDownProb": 0.1,
             "directional": {"long_overnight": {
                 "ExpPnL": 1.0 - i * 0.1, "ProbProfit": 0.5,
                 "Downside": -2.0, "Upside": 3.0}}}
        for i, tk in enumerate(tickers)
    }
    return select_top_rows(val_rows, forecasts, tickers, ["long_overnight"],
                           top_n=10, strict=False, validation_gate=gate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
