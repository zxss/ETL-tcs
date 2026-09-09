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

class TestFinalScore(unittest.TestCase):
    """Штрафы и флаги _score_row.

    NEUTRAL_SCORE — «золотое» значение для нейтральной строки: пин на формулу,
    ловит непреднамеренное изменение весов.
    """

    NEUTRAL_SCORE = 0.505

    def score(self, strict=False, **over):
        return _score_row(make_dashboard_row(**over), strict)

    def test_neutral_row_baseline(self):
        final, allowed, flags = self.score()
        self.assertAlmostEqual(final, self.NEUTRAL_SCORE, places=6)
        self.assertTrue(allowed)
        self.assertEqual(flags, [])

    def test_score_in_unit_range(self):
        for over in ({}, {"exp_pnl": 5.0, "prob_profit": 1.0, "liq_score": 100},
                     {"exp_pnl": -5.0, "prob_profit": 0.0, "liq_score": 0,
                      "regime": "BEAR", "vol_spike": 9.0}):
            final, _, _ = self.score(**over)
            self.assertGreaterEqual(final, 0.0)
            self.assertLessEqual(final, 1.0)

    # ── штраф контртренда к IMOEX ───────────────────────────────────────────

    def test_long_in_bear_is_penalised(self):
        """LONG против медвежьего рынка: regime_score 0.5→0.2, затем ×0.70."""
        final, _, _ = self.score(regime="BEAR")
        expected = (self.NEUTRAL_SCORE - 0.10 * (0.5 - 0.2)) * 0.70
        self.assertAlmostEqual(final, expected, places=6)
        self.assertLess(final, self.NEUTRAL_SCORE)

    def test_short_in_bull_is_penalised(self):
        final, _, _ = self.score(direction="SHORT", strategy="intraday_short",
                                 regime="BULL")
        expected = (self.NEUTRAL_SCORE - 0.10 * (0.5 - 0.2)) * 0.70
        self.assertAlmostEqual(final, expected, places=6)

    def test_trend_following_is_not_penalised(self):
        """LONG в BULL — по тренду: только рост суб-скора, без множителя."""
        final, _, _ = self.score(regime="BULL")
        self.assertAlmostEqual(final, self.NEUTRAL_SCORE + 0.10 * 0.5, places=6)

    def test_strict_filter_blocks_counter_trend(self):
        _, allowed, _ = self.score(strict=True, regime="BEAR")
        self.assertFalse(allowed)
        _, allowed, _ = self.score(strict=True, direction="SHORT",
                                   strategy="intraday_short", regime="BULL")
        self.assertFalse(allowed)

    def test_strict_filter_allows_trend_following(self):
        _, allowed, _ = self.score(strict=True, regime="BULL")
        self.assertTrue(allowed)

    # ── High Risk Short ─────────────────────────────────────────────────────

    def test_high_risk_short_flag_and_penalty(self):
        """SHORT по сильной бумаге (RS > +5%): rs_score→0, затем ×0.75."""
        final, _, flags = self.score(direction="SHORT", strategy="intraday_short",
                                     rs=6.0)
        self.assertIn("High Risk Short", flags)
        expected = (self.NEUTRAL_SCORE - 0.10 * 0.5) * 0.75
        self.assertAlmostEqual(final, expected, places=6)

    def test_high_risk_short_threshold(self):
        """Порог строгий: ровно +5.0% ещё не High Risk Short."""
        _, _, flags = self.score(direction="SHORT", strategy="intraday_short", rs=5.0)
        self.assertNotIn("High Risk Short", flags)
        _, _, flags = self.score(direction="SHORT", strategy="intraday_short", rs=5.01)
        self.assertIn("High Risk Short", flags)

    def test_long_with_high_rs_is_not_penalised(self):
        """Тот же RS для LONG — это сила по тренду, а не риск."""
        _, _, flags = self.score(rs=6.0)
        self.assertNotIn("High Risk Short", flags)

    # ── Volume Climax ───────────────────────────────────────────────────────

    def test_volume_climax_flag_threshold(self):
        self.assertNotIn("⚠ Volume Climax", self.score(vol_spike=2.5)[2])
        self.assertIn("⚠ Volume Climax", self.score(vol_spike=2.51)[2])

    def test_volume_climax_penalty_only_above_four(self):
        """Порог 4.0x строгий. За ним меняются СРАЗУ две вещи: корзина объёма
        (0.4 → 0.2) и множитель ×0.70 — поэтому проверяем абсолютные значения,
        а не отношение."""
        at_4, _, _ = self.score(vol_spike=4.0)
        above_4, _, _ = self.score(vol_spike=4.01)
        self.assertAlmostEqual(at_4, self.NEUTRAL_SCORE - 0.05 * (0.7 - 0.4), places=6)
        self.assertAlmostEqual(
            above_4, (self.NEUTRAL_SCORE - 0.05 * (0.7 - 0.2)) * 0.70, places=6)
        self.assertLess(above_4, at_4)

    def test_volume_climax_absolute_value(self):
        final, _, flags = self.score(vol_spike=5.0)
        expected = (self.NEUTRAL_SCORE - 0.05 * (0.7 - 0.2)) * 0.70
        self.assertAlmostEqual(final, expected, places=6)
        self.assertIn("⚠ Volume Climax", flags)

    def test_low_volume_lowers_score_without_flag(self):
        final, _, flags = self.score(vol_spike=0.5)
        self.assertLess(final, self.NEUTRAL_SCORE)
        self.assertEqual(flags, [])

    # ── Overnight gap risk ──────────────────────────────────────────────────

    def test_overnight_gap_penalty_is_pure_multiplier(self):
        """gap_down_prob не входит в суб-скоры, поэтому 0.45 → 0.55 даёт
        ровно ×0.70."""
        low, _, flags_low = self.score(strategy="long_overnight", gap_down_prob=0.45)
        high, _, flags_high = self.score(strategy="long_overnight", gap_down_prob=0.55)
        self.assertIn("⚠ High Overnight Risk", flags_low)
        self.assertIn("⚠ High Overnight Risk", flags_high)
        self.assertAlmostEqual(high / low, 0.70, places=6)

    def test_gap_risk_ignored_for_intraday(self):
        """GapRisk относится только к long_overnight."""
        final, _, flags = self.score(strategy="intraday_long", gap_down_prob=0.9)
        self.assertEqual(flags, [])
        self.assertAlmostEqual(final, self.NEUTRAL_SCORE, places=6)

    # ── накопление штрафов ──────────────────────────────────────────────────

    def test_penalties_compound(self):
        """Несколько рисков сразу перемножаются, а не берётся худший."""
        final, _, flags = self.score(direction="SHORT", strategy="intraday_short",
                                     regime="BULL", rs=6.0, vol_spike=5.0)
        self.assertIn("High Risk Short", flags)
        self.assertIn("⚠ Volume Climax", flags)
        base = (self.NEUTRAL_SCORE
                - 0.10 * (0.5 - 0.2)      # regime_score BULL для SHORT
                - 0.10 * 0.5              # rs_score → 0
                - 0.05 * (0.7 - 0.2))     # vol_score → 0.2
        self.assertAlmostEqual(final, base * 0.70 * 0.75 * 0.70, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
