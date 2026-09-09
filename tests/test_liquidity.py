"""
Тесты слоя ликвидности: рублёвый оборот считается с учётом размера лота.

Регрессия, которую они закрывают: market_data.volume приходит В ЛОТАХ, а
_max_pos_for считал оборот как close × volume, занижая его ровно в размер лота
(для TGKA — в 100 000 раз). Это резало Max Pos ₽ и искажало балл ликвидности
Liq, который весит 15% в FinalScore.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INVEST_TOKEN", "test-token")

from tft_forecast.liquidity import _max_pos_for  # noqa: E402


def rows(n=60, close=100.0, volume=1000.0, drift=0.01):
    """Свежие→старые (как отдаёт SQL с ORDER BY date DESC), цена слегка гуляет."""
    out = []
    for i in range(n):
        c = close * (1.0 + drift * ((-1) ** i))
        out.append((c, volume))
    return out


class LotSizeInDollarVolume(unittest.TestCase):

    def test_adv_scales_linearly_with_lot(self):
        """Оборот пропорционален размеру лота: lot=100 → ADV в 100 раз больше."""
        r = rows()
        a = _max_pos_for(r, lot=1)
        b = _max_pos_for(r, lot=100)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["adv_rub"] / a["adv_rub"], 100.0, places=6)

    def test_amihud_falls_as_lot_grows(self):
        """ILLIQ = |ret| / оборот: больше оборот → меньше неликвидность."""
        r = rows()
        a = _max_pos_for(r, lot=1)
        b = _max_pos_for(r, lot=1000)
        self.assertAlmostEqual(a["illiq"] / b["illiq"], 1000.0, places=3)

    def test_max_pos_scales_with_lot(self):
        """Max Pos ₽ растёт вместе с реальным оборотом."""
        r = rows()
        a = _max_pos_for(r, lot=1)
        b = _max_pos_for(r, lot=10_000)
        self.assertGreater(b["max_pos"], a["max_pos"])

    def test_tgka_scale_regression(self):
        """TGKA: лот 100 000, цена копеечная — прежняя формула давала ~0 оборота.

        При close=0.006 и 500 000 лотах в день реальный оборот равен
        0.006 × 500000 × 100000 = 300 млн ₽, а не 3 000 ₽.
        """
        r = [(0.006, 500_000.0)] * 30 + [(0.0061, 500_000.0)] * 30
        m = _max_pos_for(r, lot=100_000)
        self.assertIsNotNone(m)
        self.assertGreater(m["adv_rub"], 100_000_000)
        old = _max_pos_for(r, lot=1)
        self.assertAlmostEqual(m["adv_rub"] / old["adv_rub"], 100_000.0, places=3)

    def test_lot_is_reported_back(self):
        """Использованный лот возвращается — чтобы результат можно было проверить."""
        m = _max_pos_for(rows(), lot=10)
        self.assertEqual(m["lot"], 10)


class GuardsAgainstUnknownLot(unittest.TestCase):

    def test_zero_lot_returns_none(self):
        """Лот 0 — это не «лот 1», а отсутствие данных."""
        self.assertIsNone(_max_pos_for(rows(), lot=0))

    def test_negative_lot_returns_none(self):
        self.assertIsNone(_max_pos_for(rows(), lot=-10))

    def test_none_lot_returns_none(self):
        self.assertIsNone(_max_pos_for(rows(), lot=None))

    def test_too_few_rows_returns_none(self):
        self.assertIsNone(_max_pos_for(rows(n=19), lot=1))

    def test_zero_volume_returns_none(self):
        self.assertIsNone(_max_pos_for([(100.0, 0.0)] * 60, lot=10))


class ComputeSkipsUnknownLots(unittest.TestCase):
    """compute() не должен подставлять lot=1 для тикеров вне кэша."""

    def test_missing_ticker_is_skipped_not_defaulted(self):
        from tft_forecast import liquidity

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **kw):
                self._rows = rows()

            def fetchall(self):
                return self._rows

        class FakeConn:
            def cursor(self):
                return FakeCursor()

        out = liquidity.compute(FakeConn(), ["SBER", "TGKA"], lots={"SBER": 1})
        self.assertIn("SBER", out)
        self.assertNotIn("TGKA", out, "тикер с неизвестным лотом должен быть пропущен, "
                                      "а не посчитан с lot=1")

    def test_empty_cache_returns_empty(self):
        from tft_forecast import liquidity

        class FakeConn:
            def cursor(self):
                raise AssertionError("к БД обращаться не должны при пустом кэше")

        self.assertEqual(liquidity.compute(FakeConn(), ["SBER"], lots={}), {})


if __name__ == "__main__":
    unittest.main()
