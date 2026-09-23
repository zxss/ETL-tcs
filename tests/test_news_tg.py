"""
Тесты сбора новостей из Telegram (services/news_tg.py).

Сеть не трогается: разбор проверяется на замороженной странице t.me/s/markettwits
(tests/fixtures), загрузка — на заглушке. Фикстура — реальная разметка Telegram,
а не выдуманная: подделка проверяла бы парсер против моих же представлений.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import news_tg                              # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "tme_markettwits.html")


def _fixture() -> str:
    with open(FIXTURE, encoding="utf-8") as f:
        return f.read()


class TestParsePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posts = news_tg.parse_page(_fixture())

    def test_finds_every_post(self):
        self.assertEqual(len(self.posts), 4)

    def test_channel_and_id_from_data_post(self):
        self.assertTrue(all(p["channel"] == "markettwits" for p in self.posts))
        ids = [p["message_id"] for p in self.posts]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_every_post_has_time_and_text(self):
        for p in self.posts:
            self.assertIsNotNone(p["posted_at"], p["message_id"])
            self.assertTrue(p["text"], p["message_id"])

    def test_timestamp_is_iso_with_timezone(self):
        import datetime as dt
        d = dt.datetime.fromisoformat(self.posts[0]["posted_at"])
        self.assertIsNotNone(d.tzinfo)

    def test_line_breaks_are_preserved(self):
        """В постах markettwits перенос строки отделяет заголовок от тела —
        схлопнув его, мы потеряем структуру сообщения."""
        self.assertTrue(any("\n" in (p["text"] or "") for p in self.posts))

    def test_no_html_tags_leak_into_text(self):
        for p in self.posts:
            self.assertNotIn("<", p["text"])
            self.assertNotIn("&nbsp;", p["text"])

    def test_hashtags_survive(self):
        """Хештеги канала (#нефть, #сша) — основной способ фильтровать поток,
        и они не должны теряться вместе с разметкой ссылок."""
        self.assertTrue(any("#" in (p["text"] or "") for p in self.posts))

    def test_external_links_collected_telegram_links_skipped(self):
        links = [l for p in self.posts for l in p["links"]]
        self.assertTrue(links, "внешние ссылки не собраны")
        self.assertFalse([l for l in links if "//t.me/" in l],
                         "ссылки внутрь Telegram — это упоминания, не источники")

    def test_media_flag_detected(self):
        self.assertTrue(any(p["has_media"] for p in self.posts))

    def test_views_parsed_as_numbers(self):
        for p in self.posts:
            self.assertIsInstance(p["views"], int)
            self.assertGreater(p["views"], 0)


class TestViewsFormat(unittest.TestCase):
    def test_telegram_abbreviations(self):
        """«30.8K» обязано стать 30800: иначе сортировка по популярности
        сравнивает строки, и 9 оказывается больше 30.8K."""
        self.assertEqual(news_tg._views("30.8K"), 30800)
        self.assertEqual(news_tg._views("1.2M"), 1_200_000)
        self.assertEqual(news_tg._views("847"), 847)
        self.assertEqual(news_tg._views("30,8K"), 30800)

    def test_garbage_returns_none(self):
        self.assertIsNone(news_tg._views("вчера"))
        self.assertIsNone(news_tg._views(""))


class TestCleanText(unittest.TestCase):
    def test_entities_are_unescaped(self):
        self.assertEqual(news_tg._clean("Русал &amp; ГМК &lt;тест&gt;"),
                         "Русал & ГМК <тест>")

    def test_blank_lines_collapsed_but_breaks_kept(self):
        self.assertEqual(news_tg._clean("а\n\n\n б  в \n\nг"), "а\nб в\nг")

    def test_empty_becomes_none(self):
        self.assertIsNone(news_tg._clean("   \n  \n "))


class TestMalformedInput(unittest.TestCase):
    """Разметка Telegram однажды поменяется. Важно, чтобы парсер тогда вернул
    меньше данных, а не упал — сбор не должен ронять крон."""

    def test_empty_page(self):
        self.assertEqual(news_tg.parse_page(""), [])

    def test_page_without_posts(self):
        self.assertEqual(news_tg.parse_page("<html><body><p>Ничего</p></body></html>"), [])

    def test_unclosed_tags_do_not_raise(self):
        broken = _fixture()[: len(_fixture()) // 2]
        news_tg.parse_page(broken)          # не должно бросить

    def test_post_without_time_is_dropped(self):
        """Без времени публикации пост бесполезен: его нельзя ни отсортировать,
        ни сопоставить со свечой."""
        html = ('<div class="tgme_widget_message" data-post="ch/1">'
                '<div class="tgme_widget_message_text">текст</div></div>')
        self.assertEqual(news_tg.parse_page(html), [])


class TestCollectPagination(unittest.TestCase):
    """Обход назад: одна страница при штатном опросе, добор — только при дыре."""

    def setUp(self):
        self.conn = mock.MagicMock()

    def _run(self, pages, known, max_pages=3):
        with mock.patch.object(news_tg, "fetch_page",
                               side_effect=[(p, 200) for p in pages]) as fp, \
             mock.patch.object(news_tg, "last_message_id", return_value=known), \
             mock.patch.object(news_tg, "save_posts",
                               side_effect=lambda c, ps: (len(ps), 0)), \
             mock.patch("time.sleep"):
            stat = news_tg.collect(self.conn, "ch", max_pages=max_pages)
        return stat, fp

    def _page(self, ids):
        return "".join(
            f'<div class="tgme_widget_message" data-post="ch/{i}">'
            f'<div class="tgme_widget_message_text">пост {i}</div>'
            f'<time datetime="2026-09-10T10:0{i % 10}:00+00:00"></time></div>'
            for i in ids)

    def test_stops_when_it_meets_known_id(self):
        stat, fp = self._run([self._page([100, 101, 102])], known=100)
        self.assertEqual(fp.call_count, 1)
        self.assertEqual(stat["pages"], 1)
        self.assertTrue(stat["ok"])

    def test_walks_back_over_a_gap(self):
        """Дыра закрывается ровно тогда, когда страница дотянулась до
        известного id: known=100 и oldest=100 на второй странице — смыкание,
        третий запрос уже тянул бы то, что в базе есть."""
        stat, fp = self._run([self._page([200, 201]), self._page([100, 101]),
                              self._page([90, 91])], known=100)
        self.assertEqual(fp.call_count, 2)
        self.assertEqual(stat["seen"], 4)
        self.assertTrue(stat["ok"])

    def test_max_pages_caps_empty_table(self):
        """Пустая таблица не должна превращаться в выкачивание всего архива."""
        pages = [self._page([300 - 10 * i, 301 - 10 * i]) for i in range(5)]
        stat, fp = self._run(pages, known=None, max_pages=2)
        self.assertEqual(fp.call_count, 2)

    def test_network_error_is_recorded_not_raised(self):
        with mock.patch.object(news_tg, "fetch_page", side_effect=OSError("нет сети")), \
             mock.patch.object(news_tg, "last_message_id", return_value=None):
            stat = news_tg.collect(self.conn, "ch")
        self.assertFalse(stat["ok"])
        self.assertIn("нет сети", stat["error"])

    def test_http_error_is_recorded(self):
        import urllib.error
        err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        with mock.patch.object(news_tg, "fetch_page", side_effect=err), \
             mock.patch.object(news_tg, "last_message_id", return_value=None):
            stat = news_tg.collect(self.conn, "ch")
        self.assertFalse(stat["ok"])
        self.assertEqual(stat["http_status"], 429)


class TestChannelsSetting(unittest.TestCase):
    def test_parses_separators_and_strips_at(self):
        import config
        for raw, want in (("markettwits", ["markettwits"]),
                          ("@a @b", ["a", "b"]),
                          ("a, b,c", ["a", "b", "c"]),
                          ("", [])):
            with mock.patch.object(config, "NEWS_TG_CHANNELS", raw):
                self.assertEqual(news_tg.channels(), want)


if __name__ == "__main__":
    unittest.main()
