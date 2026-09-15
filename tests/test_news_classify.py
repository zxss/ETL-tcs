"""
Классификатор постов markettwits (research/news_classify.py). Без сети и БД.
Примеры — синтетические, в духе канала; реальные тексты в репозиторий не идут.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import news_classify as nc                 # noqa: E402

C = nc.Classifier()


def one(text, ticker):
    return C.classify(text, ticker)


class TestCategory(unittest.TestCase):

    def test_dividend_refusal_is_strongly_negative(self):
        r = one("❌🇷🇺#AAAA #дивиденд\nСД ЭН+ ГРУП ДИВИДЕНДЫ 2024Г\nНЕ ВЫПЛАЧИВАТЬ", "ENPG")
        self.assertEqual((r["category"], r["sentiment"]), ("DIVIDEND", -1.0))

    def test_dividend_recommendation_positive(self):
        r = one("❗️🇷🇺#PLZL #дивиденд\nСД ПОЛЮС: ДИВИДЕНДЫ 2024Г = 730 РУБ/АКЦ\nотсечка - 25 апреля", "PLZL")
        self.assertEqual(r["category"], "DIVIDEND")
        self.assertGreater(r["sentiment"], 0)
        self.assertFalse(r["price_report"])                      # «= 730 РУБ» — не отчёт о цене

    def test_headline_decides_primary_category(self):
        r = one("🇷🇺#GAZP #отчетность\nЧИСТАЯ ПРИБЫЛЬ ГАЗПРОМА ПО МСФО СОСТАВИЛА 1,04 ТРЛН РУБ. "
                "ПРОТИВ 296,2 МЛРД РУБ. ГОДОМ РАНЕЕ\nДИВИДЕНДНАЯ БАЗА ВЫРОСЛА", "GAZP")
        self.assertEqual((r["category"], r["category2"]), ("FINANCIAL", "DIVIDEND"))
        self.assertEqual(r["sentiment"], 1.0)                    # рост в 3,5 раза

    def test_noisy_channel_hashtag_does_not_decide(self):
        r = one("🇷🇺#ALRS #дивиденд\nАлроса перечислила в бюджет республики 631 млрд рублей налогов", "ALRS")
        self.assertNotEqual(r["category"], "DIVIDEND")

    def test_sanctions(self):
        r = one("⚠️🇷🇺#санкции #NVTK\nДОЧКИ НОВАТЭКА ВКЛЮЧЕНЫ В БРИТАНСКИЙ САНКЦИОННЫЙ СПИСОК", "NVTK")
        self.assertEqual((r["category"], r["sentiment"]), ("SANCTIONS_MACRO", -1.0))

    def test_buyback_positive_spo_negative(self):
        self.assertGreater(one("🇷🇺#YDEX\nСД рассмотрит обратный выкуп акций", "YDEX")["sentiment"], 0)
        self.assertLess(one("🇷🇺#ETLN\nЭталон проводит SPO по 46 рублей", "ETLN")["sentiment"], 0)


class TestNumbers(unittest.TestCase):

    def test_compare(self):
        self.assertEqual(nc.Classifier.compare_numbers("ПРИБЫЛЬ +30.23 МЛРД РУБ ПРОТИВ ПРИБЫЛИ +18.67 МЛРД"), 1.0)
        self.assertEqual(nc.Classifier.compare_numbers("46,5 МЛРД РУБ. ПРОТИВ 47,2 МЛРД РУБ."), 0.0)
        self.assertEqual(nc.Classifier.compare_numbers("142,7 МЛРД РУБ. ПРОТИВ 120 МЛРД"), 0.5)
        self.assertEqual(nc.Classifier.compare_numbers("прибыль 10 млрд против убытка 5 млрд"), 1.0)
        self.assertEqual(nc.Classifier.compare_numbers("без сравнения"), 0.0)


class TestPriceReportsAndDigests(unittest.TestCase):

    def test_multiline_and_level_price_reports(self):
        self.assertTrue(one("после отчетностей:\n🇷🇺#BSPB = +2.5%\n🇷🇺#NVTK = 0%", "BSPB")["price_report"])
        self.assertTrue(one("⚠️🇷🇺#GAZP = мин за 5 мес", "GAZP")["price_report"])
        self.assertTrue(one("💥🇷🇺#GAZP > 138", "GAZP")["price_report"])
        self.assertTrue(one("❗️🇷🇺#VKCO в моменте", "VKCO")["price_report"])
        r = one("❗️🇷🇺#VKCO в моменте", "VKCO")
        self.assertEqual((r["category"], r["sentiment"], r["relation"]), ("OTHER", 0.0, "упоминание"))

    def test_digest_uses_only_ticker_line(self):
        text = ("🗓КАЛЕНДАРЬ НА СЕГОДНЯ — 2024.05.28\n🇺🇸США - индекс доверия\n"
                "🇷🇺#FEES Россети - сд решит по дивидендам 2023г\n🇷🇺#SBER санкции против кого-то")
        r = one(text, "FEES")
        self.assertTrue(r["digest"])
        self.assertEqual((r["category"], r["relation"]), ("DIVIDEND", "объект"))


class TestRelation(unittest.TestCase):

    def test_omonym(self):
        r = one("❗️🇷🇺#AMEZ\nАО \"УРАЛ-ВК\" НАПРАВИЛО ОБЯЗАТЕЛЬНОЕ ПРЕДЛОЖЕНИЕ", "VKCO")
        self.assertEqual(r["relation"], "омоним")

    def test_sources(self):
        self.assertEqual(one("🇷🇺#дкп #россия\nВТБ допускает переход к жёсткой ДКП", "VTBR")["relation"], "источник")
        self.assertEqual(one("🇷🇺#депозиты\nПо оценке ВТБ, сбережения выросли", "VTBR")["relation"], "источник")
        self.assertEqual(one("🇷🇺#ipo\nСбер: не менее пяти IPO ожидается в 2026 году", "SBER")["relation"], "источник")
        self.assertEqual(one("🛢#газ\nПотребление газа выросло\n— Газпром", "GAZP")["relation"], "источник")

    def test_company_action_is_object(self):
        self.assertEqual(one("⚠️🇷🇺#ипотека\n\"Сбер\" повышает ставки по ипотеке", "SBER")["relation"], "объект")

    def test_other_company_post_is_mention(self):
        r = one("💥🇷🇺#MTLR\nАкционеры одобрили поручительство по сделкам с ВТБ", "VTBR")
        self.assertEqual(r["relation"], "упоминание")

    def test_hashtag_outside_head_with_many_tickers(self):
        r = one("⚠️🇷🇺#ecommerce #россия\nМаркетплейсы, включая «Мегамаркет» (#SBER) и «Яндекс Маркет» (#YNDX)",
                "SBER")
        self.assertEqual(r["relation"], "упоминание")


class TestEval(unittest.TestCase):

    def test_evaluate_and_wilson(self):
        texts = {1: "❌🇷🇺#MGNT #дивиденд\nСД МАГНИТ: ДИВИДЕНДЫ\nНЕ ВЫПЛАЧИВАТЬ"}
        labels = [{"message_id": "1", "ticker": "MGNT", "category": "DIVIDEND", "category2": "",
                   "sentiment": "-1", "relation": "объект"}]
        res = nc.evaluate(texts, labels, C)
        self.assertEqual((res["relation_accuracy"], res["category_accuracy_on_objects"],
                          res["sentiment_sign_accuracy"]), (1.0, 1.0, 1.0))
        lo, hi = nc.wilson(143, 150)
        self.assertTrue(0.90 < lo < 0.954 < hi)
        self.assertEqual(nc.snap(0.3), 0.5)


if __name__ == "__main__":
    unittest.main()
