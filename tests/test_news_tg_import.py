"""
Импорт выгрузки Telegram Desktop (services/news_tg_import.py). Без сети и БД.

Разметка в фикстуре взята из реальной выгрузки markettwits (14.09).
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import news_tg as m                          # noqa: E402
from services import news_tg_import as imp               # noqa: E402

MSK = ZoneInfo("Europe/Moscow")

PAGE = """<!DOCTYPE html><html><body><div class="page_body chat_page"><div class="history">
<div class="message service" id="message-1">
<div class="body details">
9 ноября 2017
</div>
</div>
<div class="message service" id="message1">
<div class="body details">
Канал создан
</div>
</div>
<div class="message default clearfix" id="message5">
<div class="pull_left userpic_wrap">
<img class="userpic" style="width: 42px; height: 42px" src="photos/author_1.jpg"/>
</div>
<div class="body">
<div class="pull_right date details" title="9 ноября 2017, 15:54:55">15:54</div>
<div class="from_name">
MarketTwits
</div>
<div class="text">
&quot;Русская Аквакультура&quot; объявила о SPO
</div>
</div>
</div>
<div class="message default clearfix joined" id="message61899">
<div class="body">
<div class="pull_right date details" title="20 декабря 2019, 12:00:34">edited 12:00</div>
<div class="reply_to details">In reply to <a href="#go_to_message61842" onclick="return GoToMessage(61842)">this message</a></div>
<div class="text">
🇬🇧<a onclick="return ShowHashtag('BOE')" href="#">#BOE</a> <a onclick="return ShowHashtag('ЦБ')" href="#">#ЦБ</a><br>Новый глава Банка Англии <strong>- Эндрю Бейли - </strong>официальное назначение<br><br>срок = 8 лет
</div>
</div>
</div>
<div class="message default clearfix joined" id="message167">
<div class="body">
<div class="pull_right date details" title="14 ноября 2017, 17:43:59">edited 17:43</div>
<div class="media_wrap clearfix">
<div class="media clearfix pull_left media_photo">
<div class="fill pull_left"></div>
<div class="body">
<div class="title bold">Photo</div>
<div class="status details">Not included, change data exporting settings to download.</div>
</div>
</div>
</div>
<div class="text">
апетит к риску превысил уровни 2000(<a href="http://Dot.com">Dot.com</a>) — <a href="https://t.me/markettwits/1">тут</a>
</div>
</div>
</div>
<div class="message default clearfix joined" id="message168">
<div class="body">
<div class="pull_right date details" title="12.10.2020 08:03:48 UTC+03:00">08:03</div>
<div class="text">
WSJ FRONT PAGE TODAY
</div>
</div>
</div>
</div></div></body></html>"""


def parse(html=PAGE):
    p = imp.ExportParser("markettwits", MSK)
    p.feed(html)
    p.close()
    return {x["message_id"]: x for x in p.posts}, p


class TestParser(unittest.TestCase):

    def test_posts_parsed_and_service_skipped(self):
        posts, _ = parse()
        self.assertEqual(sorted(posts), [5, 167, 168, 61899])

    def test_br_does_not_glue_posts(self):
        """<br> без «/» не сдвигает глубину: пост после многострочного — отдельный."""
        posts, _ = parse()
        self.assertIn(167, posts)
        self.assertNotIn("Dot.com", posts[61899]["text"])

    def test_text_format_matches_collector(self):
        """Переносы строк — как у сборщика t.me: хештеги / тело."""
        posts, _ = parse()
        self.assertEqual(posts[61899]["text"],
                         "🇬🇧#BOE #ЦБ\nНовый глава Банка Англии - Эндрю Бейли - "
                         "официальное назначение\nсрок = 8 лет")
        self.assertEqual(posts[5]["text"], '"Русская Аквакультура" объявила о SPO')

    def test_links_media_and_hashtags(self):
        posts, _ = parse()
        self.assertEqual(posts[167]["links"], ["http://Dot.com"])     # без t.me
        self.assertTrue(posts[167]["has_media"])
        self.assertEqual(posts[61899]["links"], [])                   # хештеги href="#"
        self.assertFalse(posts[61899]["has_media"])

    def test_time_is_msk_and_matches_tme(self):
        """12.10.2020 08:03:48 в выгрузке = 05:03:48 UTC в t.me (сверено 14.09)."""
        posts, _ = parse()
        utc = posts[168]["posted_at"].astimezone(dt.timezone.utc)
        self.assertEqual(utc, dt.datetime(2020, 10, 12, 5, 3, 48, tzinfo=dt.timezone.utc))
        self.assertEqual(posts[5]["posted_at"].astimezone(dt.timezone.utc),
                         dt.datetime(2017, 11, 9, 12, 54, 55, tzinfo=dt.timezone.utc))

    def test_edited_label_does_not_break_date(self):
        posts, _ = parse()
        self.assertEqual(posts[61899]["posted_at"],
                         dt.datetime(2019, 12, 20, 12, 0, 34, tzinfo=MSK))


class TestDates(unittest.TestCase):

    def test_russian_months(self):
        for i, mon in enumerate(("января", "февраля", "марта", "апреля", "мая", "июня",
                                 "июля", "августа", "сентября", "октября", "ноября",
                                 "декабря"), start=1):
            d = imp.parse_export_date(f"1 {mon} 2020, 10:00:00", MSK)
            self.assertEqual((d.month, d.hour), (i, 10))

    def test_explicit_offset_wins_over_tz(self):
        d = imp.parse_export_date("12.10.2020 08:03:48 UTC+05:00", MSK)
        self.assertEqual(d.utcoffset(), dt.timedelta(hours=5))

    def test_garbage_is_none(self):
        self.assertIsNone(imp.parse_export_date("вчера", MSK))


class TestFilesAndSql(unittest.TestCase):

    def test_files_in_number_order(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("messages.html", "messages10.html", "messages2.html"):
                open(os.path.join(d, n), "w").close()
            self.assertEqual([os.path.basename(f) for f in imp.export_files(d)],
                             ["messages.html", "messages2.html", "messages10.html"])

    def test_history_never_overwrites_collector_rows(self):
        sql = m.INSERT_TG_POST_HISTORY_SQL
        self.assertIn("ON CONFLICT (channel, message_id) DO NOTHING", sql)
        self.assertIn("VALUES %s", sql)
        self.assertIn("RETURNING", sql)

    def test_row_has_no_views(self):
        posts, _ = parse()
        row = imp._row(posts[167])
        self.assertEqual(row[:2], ("markettwits", 167))
        self.assertIsNone(row[4])
        self.assertEqual(row[5], ["http://Dot.com"])
        self.assertIsNone(imp._row(posts[5])[5])        # пустой список → NULL


CSV = ('id,date,sender,text,has_media,media_path\n'
       '"384392","2026-09-10T20:51:30.000Z","MarketTwits","❗️🛢#нефть #логистика \n'
       'ETF на ставки фрахта (#BWET) — https://example.com/a. и https://t.me/markettwits/1",'
       '"true",""\n'
       '"384391","2026-09-10T20:43:51.000Z","MarketTwits","","true",""\n'
       '"oops","2026-09-10T20:00:00.000Z","MarketTwits","битая строка","false",""\n'
       '"178855","2022-02-16 06:23:54","MarketTwits","КАЛЕНДАРЬ","false",""\n')


class TestCsv(unittest.TestCase):

    def parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "messages.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write(CSV)
            self.assertEqual(imp.export_sources(p), [p])
            return imp.parse_csv(p, "markettwits", MSK)

    def test_rows_parsed_bad_row_skipped(self):
        posts, skipped = self.parse()
        self.assertEqual([p["message_id"] for p in posts], [384392, 384391, 178855])
        self.assertEqual(skipped, 1)

    def test_utc_z_and_naive_tz(self):
        posts, _ = self.parse()
        by = {p["message_id"]: p for p in posts}
        self.assertEqual(by[384392]["posted_at"],
                         dt.datetime(2026, 9, 10, 20, 51, 30, tzinfo=dt.timezone.utc))
        # без пояса — берётся --tz (MSK): 06:23:54 MSK = 03:23:54 UTC
        self.assertEqual(by[178855]["posted_at"].astimezone(dt.timezone.utc).hour, 3)

    def test_text_links_media_like_collector(self):
        posts, _ = self.parse()
        by = {p["message_id"]: p for p in posts}
        self.assertTrue(by[384392]["text"].startswith("❗️🛢#нефть #логистика\nETF"))
        self.assertEqual(by[384392]["links"], ["https://example.com/a"])   # без t.me и точки
        self.assertTrue(by[384392]["has_media"])
        self.assertIsNone(by[384391]["text"])                              # пустой → NULL
        self.assertFalse(by[178855]["has_media"])


if __name__ == "__main__":
    unittest.main()
