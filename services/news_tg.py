"""
Сбор постов публичных Telegram-каналов через веб-превью t.me/s/<канал>.

Почему веб-превью, а не Telegram API: у публичного канала превью отдаётся без
авторизации вообще. Bot API читать чужой канал не может (нужны права админа),
MTProto требует api_id/api_hash и вход по номеру телефона — прошлый новостной
контур на этом и встал (см. contrib/experimental_news). Здесь не нужно ничего.

ГРАНИЦА КОНТУРА. Это сбор данных, и только. Ни одна строка отсюда не попадает
в скоринг, отбор бумаг или постановку заявок. Решение вынести новости из
торгового пайплайна принято по итогам аудита и здесь не пересматривается.

Разбор — html.parser из стандартной библиотеки: BeautifulSoup ради одной
страницы не стоит новой зависимости в проекте с закреплёнными версиями, а
регулярками HTML не разбирают.

Запуск:
    python3 -m services.news_tg              # опросить каналы из NEWS_TG_CHANNELS
    python3 -m services.news_tg --init       # создать схему news и выйти
    python3 -m services.news_tg --channel markettwits --pages 5
    python3 -m services.news_tg --stat       # что уже собрано
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import html
import logging
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                            # noqa: E402
import database                                          # noqa: E402
from models.news_tg import (ALL_NEWS_DDL,                # noqa: E402
                            UPSERT_TG_POST_SQL)

log = logging.getLogger("news.tg")

_BASE = "https://t.me/s"
_LOCK = "/tmp/etl-tcs-news-tg.lock"


# ── Разбор страницы ──────────────────────────────────────────────────────────

def _classes(attrs: dict) -> set[str]:
    return set((attrs.get("class") or "").split())


class ChannelPageParser(HTMLParser):
    """Достаёт посты со страницы t.me/s/<канал>.

    Разметка Telegram: каждый пост — div.tgme_widget_message с атрибутом
    data-post="<канал>/<id>"; внутри — div.tgme_widget_message_text с текстом,
    time[datetime] с временем публикации и span.tgme_widget_message_views.

    Вложенность отслеживается счётчиком глубины, а не поиском закрывающего тега:
    внутри текста поста бывают свои div/span, и наивный поиск </div> обрезал бы
    пост на первом вложенном блоке.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.posts: list[dict] = []
        self._post: dict | None = None
        self._depth = 0
        self._post_depth = 0
        self._text_depth: int | None = None
        self._buf: list[str] = []

    # -- служебное ---------------------------------------------------------
    def _start_post(self, attrs: dict) -> None:
        raw = attrs.get("data-post", "")
        channel, _, mid = raw.partition("/")
        if not mid.isdigit():
            return
        self._post = {"channel": channel, "message_id": int(mid), "posted_at": None,
                      "text": None, "views": None, "links": [], "has_media": False}
        self._post_depth = self._depth

    def _close_post(self) -> None:
        if self._post and self._post.get("posted_at"):
            self.posts.append(self._post)
        self._post = None

    # -- HTMLParser --------------------------------------------------------
    def handle_starttag(self, tag, attrs_list):
        attrs = dict(attrs_list)
        self._depth += 1
        cls = _classes(attrs)

        if self._post is None:
            if "tgme_widget_message" in cls and attrs.get("data-post"):
                self._start_post(attrs)
            return

        if self._text_depth is None and "tgme_widget_message_text" in cls:
            self._text_depth = self._depth
            self._buf = []
            return

        if tag == "time" and attrs.get("datetime") and not self._post["posted_at"]:
            self._post["posted_at"] = attrs["datetime"]
        elif tag == "a" and self._text_depth is not None:
            href = attrs.get("href") or ""
            # Ссылки на сам Telegram — это упоминания и хештеги, не источники.
            if href.startswith("http") and "//t.me/" not in href:
                self._post["links"].append(href)
        elif cls & {"tgme_widget_message_photo_wrap", "tgme_widget_message_video_player",
                    "tgme_widget_message_document", "tgme_widget_message_roundvideo"}:
            self._post["has_media"] = True
        elif tag == "br" and self._text_depth is not None:
            self._buf.append("\n")

    def handle_startendtag(self, tag, attrs_list):
        # <br/> не порождает handle_endtag, поэтому глубину не трогаем.
        if tag == "br" and self._post is not None and self._text_depth is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if self._post is not None:
            if self._text_depth is not None and self._depth == self._text_depth:
                self._post["text"] = _clean(("".join(self._buf)))
                self._text_depth = None
            elif self._depth == self._post_depth:
                self._close_post()
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._post is None:
            return
        if self._text_depth is not None:
            self._buf.append(data)
        elif self._post["views"] is None and re.fullmatch(r"\s*[\d.,]+[KMК]?\s*", data or ""):
            self._post["views"] = _views(data)

    def close(self):
        super().close()
        self._close_post()          # последний пост, если тег не закрылся


def _clean(text: str) -> str:
    """Схлопывает пробелы, но сохраняет переводы строк — в постах markettwits
    перенос несёт смысл (заголовок / тело / источник)."""
    text = html.unescape(text)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip() or None


def _views(raw: str) -> int | None:
    """«12.3K» → 12300. Telegram сокращает просмотры, и без разворачивания
    сортировка по популярности сравнивала бы строки, а не числа."""
    s = (raw or "").strip().replace(",", ".").replace("К", "K")
    m = re.fullmatch(r"([\d.]+)([KM]?)", s)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return int(val * {"": 1, "K": 1_000, "M": 1_000_000}[m.group(2)])


def parse_page(html_text: str) -> list[dict]:
    p = ChannelPageParser()
    p.feed(html_text)
    p.close()
    return p.posts


# ── Загрузка ─────────────────────────────────────────────────────────────────

def _user_agent() -> str:
    return str(getattr(config, "NEWS_TG_USER_AGENT", "") or
               "Mozilla/5.0 (compatible; etl-tcs/1.0)")


def fetch_page(channel: str, before: int | None = None) -> tuple[str, int]:
    """Возвращает (html, http_status). Бросает при сетевой ошибке."""
    url = f"{_BASE}/{urllib.parse.quote(channel)}"
    if before:
        url += "?" + urllib.parse.urlencode({"before": before})
    req = urllib.request.Request(url, headers={
        "User-Agent": _user_agent(),
        "Accept-Language": "ru,en;q=0.8",
    })
    timeout = int(getattr(config, "NEWS_TG_TIMEOUT", 20))
    with urllib.request.urlopen(req, timeout=timeout,
                                context=ssl.create_default_context()) as r:
        return r.read().decode("utf-8", "replace"), r.status


# ── Запись ───────────────────────────────────────────────────────────────────

def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        for sql in ALL_NEWS_DDL:
            cur.execute(sql)
    conn.commit()


def _to_row(p: dict) -> dict:
    return {
        "channel": p["channel"],
        "message_id": p["message_id"],
        "posted_at": p["posted_at"],
        "text": p.get("text"),
        "views": p.get("views"),
        "links": p.get("links") or None,
        "has_media": bool(p.get("has_media")),
    }


def save_posts(conn, posts: list[dict]) -> tuple[int, int]:
    """Возвращает (новых, перечитанных). Одна транзакция на страницу."""
    new = 0
    with conn.cursor() as cur:
        for p in posts:
            cur.execute(UPSERT_TG_POST_SQL, _to_row(p))
            row = cur.fetchone()
            if row and row[0]:
                new += 1
    conn.commit()
    return new, len(posts) - new


def last_message_id(conn, channel: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(message_id) FROM news.tg_posts WHERE channel = %s;",
                    (channel,))
        r = cur.fetchone()
    return r[0] if r and r[0] is not None else None


# ── Опрос канала ─────────────────────────────────────────────────────────────

def collect(conn, channel: str, *, max_pages: int | None = None) -> dict:
    """Тянет страницы назад, пока не упрётся в уже известный message_id.

    Смысл обхода назад: при штатном опросе раз в 10 минут хватает одной
    страницы (20 постов ≈ 2 часа канала), но после простоя сервера дыра может
    быть любой длины — тогда добираем страницы, пока не сомкнёмся с тем, что
    уже лежит в базе. Потолок max_pages не даёт этому превратиться в выкачивание
    всего архива при пустой таблице.
    """
    max_pages = int(max_pages if max_pages is not None
                    else getattr(config, "NEWS_TG_MAX_PAGES", 3))
    known = last_message_id(conn, channel)
    stat = {"channel": channel, "pages": 0, "seen": 0, "new": 0, "updated": 0,
            "http_status": None, "ok": False, "error": None}

    before: int | None = None
    try:
        for _ in range(max_pages):
            page, status = fetch_page(channel, before)
            stat["http_status"] = status
            stat["pages"] += 1
            posts = parse_page(page)
            if not posts:
                break
            n, u = save_posts(conn, posts)
            stat["seen"] += len(posts)
            stat["new"] += n
            stat["updated"] += u

            oldest = min(p["message_id"] for p in posts)
            # Сомкнулись с базой либо страница не принесла ничего нового —
            # дальше назад идти незачем.
            if known is not None and oldest <= known:
                break
            if n == 0:
                break
            before = oldest
            time.sleep(1)          # не долбить t.me пачкой запросов подряд
        stat["ok"] = True
    except urllib.error.HTTPError as e:
        stat["http_status"] = e.code
        stat["error"] = f"HTTP {e.code}"
    except Exception as e:                      # noqa: BLE001
        stat["error"] = f"{type(e).__name__}: {e}"
    return stat


def _log_run(conn, stat: dict, started: dt.datetime) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO news.tg_fetch_runs
                    (channel, started_at, finished_at, pages, posts_seen,
                     posts_new, posts_updated, http_status, ok, error)
                VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s);""",
                        (stat["channel"], started, stat["pages"], stat["seen"],
                         stat["new"], stat["updated"], stat["http_status"],
                         stat["ok"], stat["error"]))
        conn.commit()
    except Exception as e:                      # noqa: BLE001
        conn.rollback()
        log.warning("журнал опроса не записан: %s", e)


def channels() -> list[str]:
    raw = str(getattr(config, "NEWS_TG_CHANNELS", "") or "")
    return [c.strip().lstrip("@") for c in raw.replace(",", " ").split() if c.strip()]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _lock():
    """Не даём опросам наслаиваться: медленный ответ t.me не должен
    порождать второй процесс поверх первого, когда крон стреляет каждые 10 минут."""
    f = open(_LOCK, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _stat(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT channel, count(*), min(posted_at), max(posted_at),
                   count(*) FILTER (WHERE posted_at > NOW() - INTERVAL '24 hours')
            FROM news.tg_posts GROUP BY channel ORDER BY channel;""")
        rows = cur.fetchall()
        cur.execute("""
            SELECT channel, started_at, ok, posts_new, coalesce(error, '')
            FROM news.tg_fetch_runs ORDER BY started_at DESC LIMIT 5;""")
        runs = cur.fetchall()
    print(f"\n{'канал':<20}{'постов':>9}{'за 24ч':>9}   период")
    for ch, n, lo, hi, d in rows:
        print(f"{ch:<20}{n:>9}{d:>9}   {lo:%Y-%m-%d %H:%M} → {hi:%Y-%m-%d %H:%M}")
    print(f"\nпоследние опросы:")
    for ch, ts, ok, new, err in runs:
        print(f"  {ts:%Y-%m-%d %H:%M:%S}  {ch:<16}"
              f"{'ok' if ok else 'СБОЙ':<6}+{new:<4}{err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сбор постов Telegram-каналов")
    ap.add_argument("--init", action="store_true", help="создать схему news и выйти")
    ap.add_argument("--channel", action="append", help="канал (можно повторять)")
    ap.add_argument("--pages", type=int, help="потолок страниц за один опрос")
    ap.add_argument("--stat", action="store_true", help="показать собранное")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not a.stat and not a.init and not getattr(config, "NEWS_TG_ENABLED", False):
        log.info("NEWS_TG_ENABLED=0 — сбор выключен")
        return 0

    lock = None if (a.stat or a.init) else _lock()
    if lock is None and not (a.stat or a.init):
        log.info("предыдущий опрос ещё идёт — пропуск")
        return 0

    conn = database.get_connection()
    try:
        ensure_schema(conn)
        if a.init:
            log.info("схема news создана")
            return 0
        if a.stat:
            _stat(conn)
            return 0

        rc = 0
        for ch in (a.channel or channels()):
            started = dt.datetime.now(dt.timezone.utc)
            stat = collect(conn, ch, max_pages=a.pages)
            _log_run(conn, stat, started)
            if stat["ok"]:
                log.info("%s: страниц %d, постов %d, новых %d",
                         ch, stat["pages"], stat["seen"], stat["new"])
            else:
                log.error("%s: сбой — %s", ch, stat["error"])
                rc = 1
        return rc
    finally:
        conn.close()
        database.close_pool()
        if lock:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
