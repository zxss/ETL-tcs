"""
Дивидендные отсечки: календарь, блокировка ночного входа, поправка обучения.

Свечи T-Invest не скорректированы на дивиденды, и до 26.09.2026 контур об этом
не знал вообще: во всём services/ не было ни одного упоминания отсечки.

1. Вход в ночь вечером last_buy_date: дивиденд получен, но утренний гэп съеден
   целиком, а выплата придёт через недели и за вычетом НДФЛ — ожидание ≈ −13 %
   от дивиденда. По универсу робота с 21.05.2024 таких ночей 34, медианный гэп
   5,21 %, максимум 20,4 % (SNGSP 17.07.2024) — против MIN_STOP_PCT = 1 %.
2. Признак overnight и цель next_overnight в дни отсечек показывали не
   доходность, а дивиденд с обратным знаком: модель училась, что
   высокодивидендные бумаги проваливаются за ночь без причины.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import config
from services.load_dividends import parse_dividends, _quotation
from tft_forecast.features import _build_features, _div_gap_series


def _q(units: int, nano: int = 0) -> dict:
    return {"units": units, "nano": nano, "currency": "rub"}


class TestParseDividends(unittest.TestCase):
    def test_quotation_units_and_nano(self):
        self.assertAlmostEqual(_quotation(_q(12, 500_000_000)), 12.5)
        self.assertEqual(_quotation(None), 0.0)

    def test_basic_payment_parsed(self):
        (row,) = parse_dividends("SBER", {"dividends": [{
            "dividendNet": _q(34, 840_000_000), "lastBuyDate": "2024-07-10T00:00:00Z",
            "recordDate": "2024-07-11T00:00:00Z", "paymentDate": "2024-08-01T00:00:00Z",
            "closePrice": _q(330)}]})
        self.assertEqual(row["ticker"], "SBER")
        self.assertEqual(row["last_buy_date"], "2024-07-10")
        self.assertAlmostEqual(row["dividend_net"], 34.84)
        self.assertEqual(row["currency"], "rub")

    def test_zero_amount_dropped(self):
        """Нулевая выплата гэпа не создаёт — в календаре ей места нет."""
        self.assertEqual(parse_dividends("X", {"dividends": [
            {"dividendNet": _q(0), "lastBuyDate": "2024-07-10T00:00:00Z"}]}), [])

    def test_missing_last_buy_date_dropped(self):
        """Без lastBuyDate неизвестно, к какому дню относится гэп."""
        self.assertEqual(parse_dividends("X", {"dividends": [
            {"dividendNet": _q(5)}]}), [])

    def test_1970_placeholder_dropped(self):
        """1970 в API — «дата не заполнена», а не отсечка времён СССР."""
        self.assertEqual(parse_dividends("X", {"dividends": [
            {"dividendNet": _q(5), "lastBuyDate": "1970-01-01T00:00:00Z"}]}), [])
        (row,) = parse_dividends("X", {"dividends": [
            {"dividendNet": _q(5), "lastBuyDate": "2025-05-05T00:00:00Z",
             "recordDate": "1970-01-01T00:00:00Z"}]})
        self.assertIsNone(row["record_date"])

    def test_empty_payload(self):
        self.assertEqual(parse_dividends("X", {}), [])
        self.assertEqual(parse_dividends("X", {"dividends": None}), [])


class TestDivGapSeries(unittest.TestCase):
    """Гэп ложится на открытие ПЕРВОГО дня ряда строго после last_buy_date."""

    @staticmethod
    def _frame(n: int = 6, start: str = "2025-06-02") -> pd.DataFrame:
        days = pd.bdate_range(start, periods=n)
        return pd.DataFrame({"date": days, "open": 100.0, "high": 101.0,
                             "low": 99.0, "close": 100.0, "volume": 1000.0})

    def test_gap_on_next_trading_day(self):
        df = self._frame()                      # 02,03,04,05,06,09 июня
        div = _div_gap_series(df, [{"last_buy_date": dt.date(2025, 6, 4),
                                    "dividend_net": 5.0}])
        self.assertEqual(div.iloc[3], 5.0, "гэп обязан попасть на 05.06, не на 04.06")
        self.assertEqual(div.drop(index=3).sum(), 0.0)

    def test_weekend_skipped(self):
        """Отсечка в пятницу — гэп в понедельник, а не в субботу."""
        df = self._frame()                      # ...05(чт),06(пт),09(пн)
        div = _div_gap_series(df, [{"last_buy_date": dt.date(2025, 6, 6),
                                    "dividend_net": 3.0}])
        self.assertEqual(div.iloc[5], 3.0)
        self.assertEqual(df["date"].iloc[5].date(), dt.date(2025, 6, 9))

    def test_payment_after_series_dropped(self):
        div = _div_gap_series(self._frame(), [{"last_buy_date": dt.date(2030, 1, 1),
                                              "dividend_net": 9.0}])
        self.assertEqual(div.sum(), 0.0)

    def test_payment_before_series_dropped(self):
        """Гэп до начала ряда попал бы в строку 0, где нет prev_close."""
        div = _div_gap_series(self._frame(), [{"last_buy_date": dt.date(2020, 1, 1),
                                              "dividend_net": 9.0}])
        self.assertEqual(div.sum(), 0.0)

    def test_two_payments_same_day_sum(self):
        pays = [{"last_buy_date": dt.date(2025, 6, 4), "dividend_net": 2.0},
                {"last_buy_date": dt.date(2025, 6, 4), "dividend_net": 3.0}]
        self.assertEqual(_div_gap_series(self._frame(), pays).iloc[3], 5.0)

    def test_no_payments(self):
        self.assertEqual(_div_gap_series(self._frame(), []).sum(), 0.0)


class TestFeatureAdjustment(unittest.TestCase):
    """Гэп отсечки не должен попадать в признаки и цели как доходность."""

    DIV = 5.0
    EX_ROW = 3          # день гэпа в ряду

    def _frame(self) -> pd.DataFrame:
        """Цена ровно 100 каждый день; в день гэпа все цены на дивиденд ниже."""
        days = pd.bdate_range("2025-06-02", periods=8)
        o = np.full(8, 100.0); h = np.full(8, 100.0)
        lo = np.full(8, 100.0); c = np.full(8, 100.0)
        for col in (o, h, lo, c):
            col[self.EX_ROW:] -= self.DIV      # после отсечки цена ниже навсегда
        return pd.DataFrame({"date": days, "open": o, "high": h, "low": lo,
                             "close": c, "volume": np.full(8, 1000.0)})

    @property
    def _pays(self):
        # last_buy_date = предыдущий торговый день, гэп → EX_ROW
        return [{"last_buy_date": dt.date(2025, 6, 4), "dividend_net": self.DIV}]

    def test_overnight_false_gap_removed(self):
        raw = self._frame()
        bad = _build_features(raw, [])["overnight"].iloc[self.EX_ROW]
        good = _build_features(raw, self._pays)["overnight"].iloc[self.EX_ROW]
        self.assertAlmostEqual(bad, -5.0, places=6, msg="без поправки виден ложный обвал")
        self.assertAlmostEqual(good, 0.0, places=6, msg="полная доходность ночи = 0")

    def test_intraday_untouched(self):
        """close/open — обе цены уже после гэпа, дивиденд их не задевает."""
        raw = self._frame()
        a = _build_features(raw, [])["intraday"]
        b = _build_features(raw, self._pays)["intraday"]
        pd.testing.assert_series_equal(a, b)

    def test_log_ret_is_total_return(self):
        d = _build_features(self._frame(), self._pays)
        self.assertAlmostEqual(d["log_ret"].iloc[self.EX_ROW], 0.0, places=6)

    def test_atr_not_spiked_by_dividend(self):
        """Истинный диапазон считал гэп движением цены и задирал ATR."""
        raw = self._frame()
        bad = _build_features(raw, [])["atr_pct"].iloc[self.EX_ROW:].max()
        good = _build_features(raw, self._pays)["atr_pct"].iloc[self.EX_ROW:].max()
        self.assertGreater(bad, good)
        self.assertAlmostEqual(good, 0.0, places=6)

    def test_next_overnight_target_adjusted(self):
        """Цель предыдущего дня смотрит на завтрашнее открытие — тоже с гэпом."""
        raw = self._frame()
        prev = self.EX_ROW - 1
        bad = _build_features(raw, [])["next_overnight"].iloc[prev]
        good = _build_features(raw, self._pays)["next_overnight"].iloc[prev]
        self.assertAlmostEqual(bad, -5.0, places=6)
        self.assertAlmostEqual(good, 0.0, places=6)

    def test_next_total_target_adjusted(self):
        raw = self._frame()
        prev = self.EX_ROW - 1
        self.assertAlmostEqual(
            _build_features(raw, self._pays)["next_total"].iloc[prev], 0.0, places=6)

    def test_div_gap_column_exposed_for_audit(self):
        d = _build_features(self._frame(), self._pays)
        self.assertEqual(d["div_gap"].iloc[self.EX_ROW], self.DIV)
        self.assertEqual(d["div_gap"].sum(), self.DIV)

    def test_no_payments_matches_old_behaviour(self):
        """Пустой календарь — ряд считается ровно как до правки."""
        raw = self._frame()
        d = _build_features(raw, [])
        self.assertAlmostEqual(d["overnight"].iloc[self.EX_ROW], -5.0, places=6)
        self.assertEqual(d["div_gap"].sum(), 0.0)


class TestOvernightEntryBlocked(unittest.TestCase):
    """Вечер last_buy_date закрыт для ночного входа, внутридневные — нет."""

    EX_DIV = {"SBER": {"last_buy_date": dt.date(2025, 7, 10), "dividend_net": 34.84,
                       "close_price": 330.0, "gap_pct": 10.56}}

    @staticmethod
    def _po(ticker: str, strategy: str) -> dict:
        return {"ticker": ticker, "strategy_type": strategy, "direction": "LONG",
                "quantity_lots": 1, "total_rub": 10_000.0,
                "signal_id": f"20250710-{ticker}-{strategy}"}

    def _check(self, po: dict, ex_div=None):
        from services import stage2_demo as s2
        with mock.patch("tft_forecast.combined.trading_strategies",
                        return_value=("long_overnight", "intraday_long")), \
             mock.patch("tft_forecast.combined.non_shortable_tickers",
                        return_value=frozenset()), \
             mock.patch.object(s2, "contour_ok", return_value=None):
            return s2.check_order_allowed(po, env="SANDBOX", used_ids=set(),
                                          ex_div=ex_div)

    def test_overnight_on_last_buy_date_rejected(self):
        reason = self._check(self._po("SBER", "long_overnight"), self.EX_DIV)
        self.assertIsNotNone(reason)
        self.assertIn("отсечка", reason)

    def test_other_ticker_allowed(self):
        self.assertIsNone(self._check(self._po("LKOH", "long_overnight"), self.EX_DIV))

    def test_intraday_not_blocked(self):
        """Внутридневная считает close/open — обе цены после гэпа."""
        self.assertIsNone(self._check(self._po("SBER", "intraday_long"), self.EX_DIV))

    def test_no_calendar_no_block(self):
        """Без календаря контур обязан работать как прежде."""
        self.assertIsNone(self._check(self._po("SBER", "long_overnight"), {}))
        self.assertIsNone(self._check(self._po("SBER", "long_overnight"), None))

    def test_flag_off_returns_empty(self):
        from services import dividends as div_svc
        with mock.patch.object(config, "OVERNIGHT_SKIP_EX_DIV", False):
            self.assertEqual(div_svc.ex_div_block(dt.date(2025, 7, 10)), {})

    def test_db_failure_does_not_block_trading(self):
        """Календарь недоступен — торгуем как прежде, но с предупреждением."""
        from services import dividends as div_svc
        with mock.patch.object(config, "OVERNIGHT_SKIP_EX_DIV", True), \
             mock.patch("database.get_dividends_on", side_effect=RuntimeError("нет БД")):
            self.assertEqual(div_svc.ex_div_block(dt.date(2025, 7, 10)), {})


class TestDescribe(unittest.TestCase):
    def test_describe_mentions_gap(self):
        from services import dividends as div_svc
        text = div_svc.describe("SBER", {"last_buy_date": dt.date(2025, 7, 10),
                                         "dividend_net": 34.84, "gap_pct": 10.56})
        self.assertIn("SBER", text)
        self.assertIn("10.56", text)


if __name__ == "__main__":
    unittest.main()
