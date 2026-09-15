"""
Классификатор постов markettwits (research/news_classify.py, словари v2). Без
сети и БД. Примеры — синтетические, в духе канала; реальные тексты в репо не идут.
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


class TestDividends(unittest.TestCase):

    def test_refusal_is_announce_strongly_negative(self):
        r = one("❌🇷🇺#ENPG #дивиденд\nСД ЭН+ ГРУП ДИВИДЕНДЫ 2024Г\nНЕ ВЫПЛАЧИВАТЬ", "ENPG")
        self.assertEqual((r["category"], r["sentiment"]), ("DIVIDEND_ANNOUNCE", -1.0))

    def test_recommendation_with_record_date_is_announce(self):
        r = one("❗️🇷🇺#PLZL #дивиденд\nСД ПОЛЮС: ДИВИДЕНДЫ 2024Г = 730 РУБ/АКЦ\nотсечка - 25 апреля", "PLZL")
        self.assertEqual(r["category"], "DIVIDEND_ANNOUNCE")
        self.assertGreater(r["sentiment"], 0)
        self.assertFalse(r["price_report"])

    def test_calendar_is_neutral(self):
        for text, tk in (("❗️🇷🇺#ROSN #дивиденд\n17 ноября - РОСНЕФТЬ - сд по дивидендам 9м 2025г", "ROSN"),
                         ("🇷🇺#MSNG #дивиденд\n24 июня - Мосэнерго - ГОСА по дивидендам 2024", "MSNG"),
                         ("🇷🇺#SBER\nСбер: последний день с дивидендами — сегодня, отсечка завтра", "SBER")):
            r = one(text, tk)
            self.assertEqual((r["category"], r["sentiment"]), ("DIVIDEND_CALENDAR", 0.0), text)

    def test_failed_meeting_is_announce(self):
        r = one("❗️🇷🇺#TATN #дивиденд\nВОСА ТАТНЕФТИ ПО ДИВИДЕНДАМ НЕ СОСТОЯЛОСЬ ИЗ-ЗА ОТСУТСТВИЯ КВОРУМА", "TATN")
        self.assertEqual(r["category"], "DIVIDEND_ANNOUNCE")
        self.assertLess(r["sentiment"], 0)

    def test_headline_decides_primary_category(self):
        r = one("🇷🇺#GAZP #отчетность\nЧИСТАЯ ПРИБЫЛЬ ГАЗПРОМА ПО МСФО СОСТАВИЛА 1,04 ТРЛН РУБ. "
                "ПРОТИВ 296,2 МЛРД РУБ. ГОДОМ РАНЕЕ\nДИВИДЕНДНАЯ БАЗА ВЫРОСЛА", "GAZP")
        self.assertEqual((r["category"], r["category2"]), ("FINANCIAL", "DIVIDEND_ANNOUNCE"))
        self.assertEqual(r["sentiment"], 1.0)

    def test_noisy_channel_hashtag_does_not_decide(self):
        r = one("🇷🇺#ALRS #дивиденд\nАлроса перечислила в бюджет республики 631 млрд рублей налогов", "ALRS")
        self.assertFalse(r["category"].startswith("DIVIDEND"))


class TestOtherCategories(unittest.TestCase):

    def test_sanctions(self):
        r = one("⚠️🇷🇺#санкции #NVTK\nДОЧКИ НОВАТЭКА ВКЛЮЧЕНЫ В БРИТАНСКИЙ САНКЦИОННЫЙ СПИСОК", "NVTK")
        self.assertEqual((r["category"], r["sentiment"]), ("SANCTIONS_MACRO", -1.0))

    def test_buyback_positive_spo_negative(self):
        self.assertGreater(one("🇷🇺#YDEX\nСД рассмотрит обратный выкуп акций", "YDEX")["sentiment"], 0)
        self.assertLess(one("🇷🇺#ETLN\nЭталон проводит SPO по 46 рублей", "ETLN")["sentiment"], 0)

    def test_compare_numbers(self):
        f = nc.Classifier.compare_numbers
        self.assertEqual(f("ПРИБЫЛЬ +30.23 МЛРД РУБ ПРОТИВ ПРИБЫЛИ +18.67 МЛРД"), 1.0)
        self.assertEqual(f("46,5 МЛРД РУБ. ПРОТИВ 47,2 МЛРД РУБ."), 0.0)
        self.assertEqual(f("142,7 МЛРД РУБ. ПРОТИВ 120 МЛРД"), 0.5)
        self.assertEqual(f("прибыль 10 млрд против убытка 5 млрд"), 1.0)


class TestFeedHygiene(unittest.TestCase):

    def test_video_and_ads_are_spam(self):
        self.assertTrue(C.is_spam("📌 Влияние на рынок. СМОТРЕТЬ ВИДЕО: https://youtu.be/abc"))
        self.assertTrue(C.is_spam("Загрузил видео на https://rutube.ru/video/x"))
        self.assertTrue(C.is_spam("Реклама. ООО Ромашка, erid: 2VtzqwX"))
        self.assertFalse(C.is_spam("❗️🇷🇺#SBER\nСбер отчитался по РСБУ"))

    def test_digest_by_marker_or_many_tickers(self):
        self.assertTrue(C.is_digest_post("🗓КАЛЕНДАРЬ НА СЕГОДНЯ — 2025.04.01\n#AFLT дивиденды"))
        many = "Лидеры: #SBER #GAZP #LKOH #ROSN #NVTK #TATN"
        self.assertTrue(C.is_digest_post(many))
        self.assertFalse(C.is_digest_post("🇷🇺#SBER #GAZP\nСбер и Газпром подписали соглашение"))

    def test_price_reports(self):
        self.assertTrue(one("после отчетностей:\n🇷🇺#BSPB = +2.5%\n🇷🇺#NVTK = 0%", "BSPB")["price_report"])
        self.assertTrue(one("⚠️🇷🇺#GAZP = мин за 5 мес", "GAZP")["price_report"])
        self.assertTrue(one("💥🇷🇺#GAZP > 138", "GAZP")["price_report"])
        r = one("❗️🇷🇺#VKCO в моменте", "VKCO")
        self.assertEqual((r["category"], r["sentiment"], r["relation"]), ("OTHER", 0.0, "упоминание"))


class TestTickerT(unittest.TestCase):

    def test_tcsg_hashtag_and_names(self):
        self.assertIn("T", C.sources("🇷🇺#TCSG #отчетность\nТКС Холдинг: прибыль выросла"))
        self.assertIn("T", C.sources("🇷🇺#T\nТ-Технологии объявили байбек"))
        self.assertIn("T", C.sources("Тинькофф запустил новый продукт"))
        self.assertNotIn("T", C.sources("🇷🇺#TATN\nТатнефть отчиталась"))

    def test_t_event_is_object(self):
        r = one("🇷🇺#TCSG #отчетность\nЧИСТАЯ ПРИБЫЛЬ ТКС ПО МСФО 20 МЛРД РУБ ПРОТИВ 10 МЛРД", "T")
        self.assertEqual((r["category"], r["relation"]), ("FINANCIAL", "объект"))
        self.assertGreater(r["sentiment"], 0)


class TestRelation(unittest.TestCase):

    def test_omonym(self):
        self.assertEqual(one("❗️🇷🇺#AMEZ\nАО \"УРАЛ-ВК\" НАПРАВИЛО ПРЕДЛОЖЕНИЕ", "VKCO")["relation"], "омоним")

    def test_sources(self):
        self.assertEqual(one("🇷🇺#дкп #россия\nВТБ допускает переход к жёсткой ДКП", "VTBR")["relation"], "источник")
        self.assertEqual(one("🇷🇺#депозиты\nПо оценке ВТБ, сбережения выросли", "VTBR")["relation"], "источник")
        self.assertEqual(one("🇷🇺#ipo\nСбер: не менее пяти IPO ожидается в 2026 году", "SBER")["relation"], "источник")
        self.assertEqual(one("🛢#газ\nПотребление газа выросло\n— Газпром", "GAZP")["relation"], "источник")

    def test_company_action_is_object(self):
        self.assertEqual(one("⚠️🇷🇺#ипотека\n\"Сбер\" повышает ставки по ипотеке", "SBER")["relation"], "объект")

    def test_other_company_post_is_mention(self):
        self.assertEqual(one("💥🇷🇺#MTLR\nАкционеры одобрили поручительство по сделкам с ВТБ", "VTBR")["relation"],
                         "упоминание")


class TestEval(unittest.TestCase):

    def test_v1_label_dividend_matches_both_v2_kinds(self):
        texts = {1: "❌🇷🇺#MGNT #дивиденд\nСД МАГНИТ: ДИВИДЕНДЫ\nНЕ ВЫПЛАЧИВАТЬ",
                 2: "🇷🇺#ROSN\n17 ноября - РОСНЕФТЬ - сд по дивидендам"}
        labels = [{"message_id": "1", "ticker": "MGNT", "category": "DIVIDEND", "category2": "",
                   "sentiment": "-1", "relation": "объект"},
                  {"message_id": "2", "ticker": "ROSN", "category": "DIVIDEND", "category2": "",
                   "sentiment": "0", "relation": "объект"}]
        res = nc.evaluate(texts, labels, C)
        self.assertEqual((res["relation_accuracy"], res["category_accuracy_on_objects"]), (1.0, 1.0))
        lo, hi = nc.wilson(143, 150)
        self.assertTrue(0.90 < lo < 0.954 < hi)


if __name__ == "__main__":
    unittest.main()
