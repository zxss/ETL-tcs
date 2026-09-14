"""
Импорт истории Telegram-канала из выгрузки Telegram Desktop (HTML) в news.tg_posts.

Сборщик (services/news_tg.py) читает веб-превью t.me/s страницами по 20 постов
и архив сознательно не выкачивает. Историю за годы даёт «Экспорт истории чата»
Telegram Desktop — каталог с файлами messages.html, messages2.html, …

Сверено 14.09 на markettwits:
  • id сообщения в выгрузке = message_id поста в канале: посты 49 999–50 001 и
    102 362 совпали с t.me и по тексту, и по времени;
  • время в title — местное время компьютера, где делалась выгрузка, без пояса:
    «12 октября 2020, 08:03:48» в выгрузке = 05:03:48 UTC в t.me. Пояс задаётся
    --tz (по умолчанию Europe/Moscow). Если в title есть «UTC+03:00», берётся он.

Служебные сообщения («Канал создан», «Фото обновлено») не импортируются: это не
посты. Существующие строки не трогаются (ON CONFLICT DO NOTHING): свежие посты —
зона сборщика, у него просмотры и правки, которых в выгрузке нет. Повторный
запуск безопасен — догружает только недостающее.

ГРАНИЦА КОНТУРА та же, что у сборщика: это данные, в скоринг и заявки они не идут.

Запуск:
    python3 -m services.news_tg_import ~/Downloads/ChatExport_MarketTwits --channel markettwits
    python3 -m services.news_tg_import DIR --channel markettwits --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
import os
import re
import sys
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.news_tg import _clean                      # noqa: E402

log = logging.getLogger("news.tg_import")

# Теги без закрывающей пары. В выгрузке <br> пишется без «/», и html.parser
# отдаёт его как открывающий: считать по нему глубину — значит сдвинуть
# счётчик и склеить следующий пост с текущим.
_VOID = {"br", "img", "hr", "meta", "link", "input", "wbr", "source", "area", "col"}

_MONTHS = {m: i for i, m in enumerate(
    ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
     "сентября", "октября", "ноября", "декабря"), start=1)}
_RU_DATE = re.compile(r"(\d{1,2}) (\w+) (\d{4}), (\d{1,2}):(\d{2}):(\d{2})")
_NUM_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4}) (\d{1,2}):(\d{2}):(\d{2})")
_UTC_OFF = re.compile(r"UTC([+-])(\d{1,2}):?(\d{2})")


def parse_export_date(raw: str, tz: dt.tzinfo) -> dt.datetime | None:
    """«9 ноября 2017, 15:54:55» или «09.11.2017 15:54:55 UTC+03:00» → aware datetime."""
    raw = (raw or "").strip()
    m = _RU_DATE.search(raw)
    if m and m.group(2).lower() in _MONTHS:
        d, mon, y, hh, mm, ss = (int(m.group(1)), _MONTHS[m.group(2).lower()],
                                 int(m.group(3)), int(m.group(4)), int(m.group(5)),
                                 int(m.group(6)))
    else:
        m = _NUM_DATE.search(raw)
        if not m:
            return None
        d, mon, y, hh, mm, ss = (int(x) for x in m.groups())
    off = _UTC_OFF.search(raw)
    if off:
        sign = 1 if off.group(1) == "+" else -1
        tz = dt.timezone(sign * dt.timedelta(hours=int(off.group(2)), minutes=int(off.group(3))))
    return dt.datetime(y, mon, d, hh, mm, ss, tzinfo=tz)


def _classes(attrs: dict) -> set[str]:
    return set((attrs.get("class") or "").split())


class ExportParser(HTMLParser):
    """Посты из файла messages*.html выгрузки Telegram Desktop.

    Пост — div.message.default с id="message<N>"; внутри div.date[title] со
    временем, div.text с текстом (переносы — <br>), div.media_wrap при вложении.
    «Склеенные» посты (class … joined) устроены так же, только без автора.
    """

    def __init__(self, channel: str, tz: dt.tzinfo):
        super().__init__(convert_charrefs=True)
        self.channel, self.tz = channel, tz
        self.posts: list[dict] = []
        self.skipped_no_date = 0
        self._post: dict | None = None
        self._depth = 0
        self._post_depth = 0
        self._text_depth: int | None = None
        self._buf: list[str] = []

    def _void(self, tag: str, attrs: dict) -> None:
        if tag == "br" and self._post is not None and self._text_depth is not None:
            self._buf.append("\n")

    def handle_starttag(self, tag, attrs_list):
        attrs = dict(attrs_list)
        if tag in _VOID:
            self._void(tag, attrs)
            return
        self._depth += 1
        cls = _classes(attrs)

        if self._post is None:
            mid = (attrs.get("id") or "")
            if tag == "div" and {"message", "default"} <= cls and mid.startswith("message"):
                num = mid[len("message"):]
                if num.isdigit():
                    self._post = {"channel": self.channel, "message_id": int(num),
                                  "posted_at": None, "text": None, "views": None,
                                  "links": [], "has_media": False}
                    self._post_depth = self._depth
            return

        if self._text_depth is None and tag == "div" and cls == {"text"}:
            self._text_depth = self._depth
            self._buf = []
        elif tag == "div" and {"date", "details"} <= cls and attrs.get("title") \
                and self._post["posted_at"] is None:
            self._post["posted_at"] = parse_export_date(attrs["title"], self.tz)
        elif "media_wrap" in cls:
            self._post["has_media"] = True
        elif tag == "a" and self._text_depth is not None:
            href = attrs.get("href") or ""
            # Хештеги в выгрузке — href="#", упоминания — ссылки на t.me:
            # это не источники, как и в сборщике.
            if href.startswith("http") and "//t.me/" not in href:
                self._post["links"].append(href)

    def handle_startendtag(self, tag, attrs_list):
        self._void(tag, dict(attrs_list))

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if self._post is not None:
            if self._text_depth is not None and self._depth == self._text_depth:
                self._post["text"] = _clean("".join(self._buf))
                self._text_depth = None
            elif self._depth == self._post_depth:
                self._close_post()
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._post is not None and self._text_depth is not None:
            self._buf.append(data)

    def _close_post(self) -> None:
        if self._post is not None:
            if self._post["posted_at"] is not None:
                self.posts.append(self._post)
            else:
                self.skipped_no_date += 1
        self._post = None

    def close(self):
        super().close()
        self._close_post()


def parse_file(path: str, channel: str, tz: dt.tzinfo) -> tuple[list[dict], int]:
    p = ExportParser(channel, tz)
    with open(path, encoding="utf-8") as f:
        p.feed(f.read())
    p.close()
    return p.posts, p.skipped_no_date


def export_files(directory: str) -> list[str]:
    """messages.html, messages2.html, … в порядке номеров."""
    def num(path: str) -> int:
        m = re.search(r"messages(\d*)\.html$", path)
        return int(m.group(1) or 1) if m else 0
    return sorted(glob.glob(os.path.join(directory, "messages*.html")), key=num)


def _row(p: dict) -> tuple:
    return (p["channel"], p["message_id"], p["posted_at"], p.get("text"), None,
            p.get("links") or None, bool(p.get("has_media")))


def save_history(conn, posts: list[dict], *, page_size: int = 1000) -> int:
    """Пишет пачкой, существующие строки не трогает. Возвращает число вставленных."""
    from psycopg2.extras import execute_values
    from models.news_tg import INSERT_TG_POST_HISTORY_SQL
    if not posts:
        return 0
    with conn.cursor() as cur:
        inserted = execute_values(cur, INSERT_TG_POST_HISTORY_SQL,
                                  [_row(p) for p in posts], page_size=page_size,
                                  fetch=True)
    conn.commit()
    return len(inserted)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Импорт выгрузки Telegram Desktop в news.tg_posts")
    ap.add_argument("directory", help="каталог выгрузки с messages*.html")
    ap.add_argument("--channel", required=True, help="имя канала, как в t.me (markettwits)")
    ap.add_argument("--tz", default="Europe/Moscow",
                    help="пояс времени выгрузки, если в title его нет (по умолчанию MSK)")
    ap.add_argument("--dry-run", action="store_true", help="только разобрать, в БД не писать")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = export_files(os.path.expanduser(a.directory))
    if not files:
        log.error("в %s нет файлов messages*.html", a.directory)
        return 1
    tz = ZoneInfo(a.tz)
    channel = a.channel.strip().lstrip("@")

    conn = None
    if not a.dry_run:
        import database
        from services.news_tg import ensure_schema
        conn = database.get_connection()
        ensure_schema(conn)

    total = new = no_date = 0
    lo = hi = None
    try:
        for path in files:
            posts, skipped = parse_file(path, channel, tz)
            total += len(posts)
            no_date += skipped
            if posts:
                ids = [p["message_id"] for p in posts]
                lo = min(ids + ([lo] if lo is not None else []))
                hi = max(ids + ([hi] if hi is not None else []))
            n = 0 if a.dry_run else save_history(conn, posts)
            new += n
            log.info("%s: постов %d%s", os.path.basename(path), len(posts),
                     "" if a.dry_run else f", новых {n}")
    finally:
        if conn is not None:
            conn.close()
            import database
            database.close_pool()

    log.info("итого: файлов %d, постов %d (id %s–%s), %s, без даты пропущено %d",
             len(files), total, lo, hi,
             "проверка без записи" if a.dry_run else f"вставлено {new}, уже были {total - new}",
             no_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
