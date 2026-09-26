"""
Безрисковая планка: ключевая ставка ЦБ как знаменатель отчёта.

Итог контура сам по себе ничего не доказывает: свободные деньги лежат в фонде
денежного рынка, фонд идёт за ставкой ЦБ. При ставке 14 % «плюс 0,03 % за день»
— это ровно то, что дала бы парковка без всякого риска. До 26.09.2026 отчёт
сравнивал результат с нулём.

Макро здесь именно знаменатель, а не признак модели: макро как предиктор
доходностей закрыто и литературой (Welch–Goyal), и нашими тестами.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from services.load_macro import parse_key_rate
from services.stage2_balance import _benchmark_block

# Форма ответа SOAP-сервиса ЦБ, сокращённая до двух записей. Записи KR лежат в
# diffgram без пространства имён — разбор обязан находить их в любой обёртке.
REAL_XML = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
<soap:Body><KeyRateResponse xmlns="http://web.cbr.ru/"><KeyRateResult>
<diffgr:diffgram xmlns:msdata="urn:schemas-microsoft-com:xml-msdata"
 xmlns:diffgr="urn:schemas-microsoft-com:xml-diffgram-v1">
<KeyRate xmlns="">
<KR diffgr:id="KR1" msdata:rowOrder="0"><DT>2026-09-25T00:00:00+03:00</DT><Rate>14.00</Rate></KR>
<KR diffgr:id="KR2" msdata:rowOrder="1"><DT>2026-09-24T00:00:00+03:00</DT><Rate>14.00</Rate></KR>
</KeyRate></diffgr:diffgram></KeyRateResult></KeyRateResponse></soap:Body></soap:Envelope>"""


class TestParseKeyRate(unittest.TestCase):
    def test_real_response_shape(self):
        rows = parse_key_rate(REAL_XML)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["date"], "2026-09-25")
        self.assertAlmostEqual(rows[0]["key_rate_pct"], 14.0)
        self.assertEqual(rows[0]["source"], "cbr")

    def test_empty_dataset(self):
        self.assertEqual(parse_key_rate(
            '<?xml version="1.0"?><root><KeyRate xmlns=""/></root>'), [])

    def test_comma_decimal_accepted(self):
        (row,) = parse_key_rate(
            '<root><KR><DT>2026-01-01T00:00:00</DT><Rate>16,50</Rate></KR></root>')
        self.assertAlmostEqual(row["key_rate_pct"], 16.5)

    def test_zero_and_broken_rate_skipped(self):
        self.assertEqual(parse_key_rate(
            '<root><KR><DT>2026-01-01T00:00:00</DT><Rate>0</Rate></KR>'
            '<KR><DT>2026-01-02T00:00:00</DT><Rate>нет</Rate></KR></root>'), [])

    def test_record_without_date_skipped(self):
        self.assertEqual(parse_key_rate('<root><KR><Rate>14.0</Rate></KR></root>'), [])

    def test_malformed_xml_raises(self):
        with self.assertRaises(RuntimeError):
            parse_key_rate("это не XML")


class TestBenchmarkBlock(unittest.TestCase):
    """Планка считается по календарным дням: деньги в фонде работают и в выходные."""

    SUMM = [{"trading_day": "2026-09-01"}, {"trading_day": "2026-09-30"}]
    BASE = 1_000_000.0

    def _block(self, total: float, rate: float | None = 14.0):
        with mock.patch("database.get_key_rate_avg", return_value=rate), \
             mock.patch("database.get_key_rate",
                        return_value=(dt.date(2026, 9, 12), rate)):
            return "\n".join(_benchmark_block(self.SUMM, self.BASE, total))

    def test_risk_free_amount(self):
        """30 дней под 14 % на 1 млн ₽ = 1 000 000 × 0,14 × 30/365 = 11 506,85 ₽."""
        text = self._block(20_000.0).replace(" ", "")
        self.assertIn("11506,85", text)
        self.assertIn("(1,151 %)", text)

    def test_verdict_above_rate(self):
        self.assertIn("ВЫШЕ ставки", self._block(20_000.0))

    def test_verdict_below_rate(self):
        """Плюс 5 000 ₽ при ставке 11 507 ₽ — это проигрыш парковке."""
        self.assertIn("НИЖЕ ставки", self._block(5_000.0))

    def test_rate_shown(self):
        text = self._block(1.0)
        self.assertIn("Ключевая ставка ЦБ", text)
        self.assertIn("14,00", text)
        self.assertIn("2026-09-12", text)

    def test_alpha_is_result_minus_rate(self):
        """Сверх ставки = результат − безрисковая планка, без округлений в пользу."""
        text = self._block(20_000.0).replace(" ", "")
        self.assertIn("8493,15", text)

    def test_no_rate_no_block(self):
        """Без таблицы ставки планку выдумывать нельзя — блока просто нет."""
        self.assertEqual(self._block(20_000.0, rate=None), "")

    def test_db_failure_no_block(self):
        with mock.patch("database.get_key_rate_avg", side_effect=RuntimeError("нет БД")):
            self.assertEqual(_benchmark_block(self.SUMM, self.BASE, 1.0), [])

    def test_no_summaries(self):
        self.assertEqual(_benchmark_block([], self.BASE, 1.0), [])
        self.assertEqual(_benchmark_block(self.SUMM, self.BASE, None), [])

    def test_bad_date_no_crash(self):
        bad = [{"trading_day": "не дата"}, {"trading_day": "не дата"}]
        self.assertEqual(_benchmark_block(bad, self.BASE, 1.0), [])


if __name__ == "__main__":
    unittest.main()
