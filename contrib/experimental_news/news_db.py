"""
Доступ к БД для отключённого контура новостей.

Вынесено из database.py: штатный конвейер о news_sentiment ничего не знает.
Пул соединений и контракт транзакций переиспользуются из database — conn
последним необязательным аргументом (None → своя транзакция из пула).

Схему поднимает init_news_schema(): штатный database.init_db() её не создаёт.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2.extras          # noqa: E402
import database                 # noqa: E402
from database import _tx        # noqa: E402

from contrib.experimental_news.news_schema import (  # noqa: E402
    CREATE_NEWS_SQL, UPSERT_NEWS_SQL, LAST_MESSAGE_ID_SQL,
    CREATE_NEWS_CANDLES_VIEW_SQL,
)

log = logging.getLogger("contrib.news_db")


def init_news_schema(conn=None) -> None:
    """Создаёт news_sentiment и view news_with_candles. Вызывать вручную при
    включении контура — в штатный init_db() это больше не входит."""
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(CREATE_NEWS_SQL)
            cur.execute(CREATE_NEWS_CANDLES_VIEW_SQL)
        if conn is not None:
            c.commit()
    log.info("Схема новостей инициализирована (news_sentiment + news_with_candles)")



def get_last_message_id(source: str, conn=None) -> int | None:
    """MAX(message_id) из news_sentiment для источника, или None."""
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(LAST_MESSAGE_ID_SQL, (source,))
            row = cur.fetchone()
    return int(row[0]) if row and row[0] else None


def upsert_news(rows: list[tuple], conn=None) -> int:
    """
    Upsert записей в news_sentiment.
    rows: (message_id, ts, ticker, sentiment, headline, raw_text, source)
    """
    if not rows:
        return 0
    with _tx(conn) as c:
        with c.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_NEWS_SQL, rows, page_size=200)
    return len(rows)


def query_news(ticker: str, limit: int = 50, conn=None) -> list[dict]:
    """Последние новости по тикеру для аналитики."""
    sql = """
        SELECT ts, sentiment, headline, message_id
        FROM news_sentiment
        WHERE ticker = %s
        ORDER BY ts DESC
        LIMIT %s;
    """
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(sql, (ticker, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def query_news_candles(ticker: str, limit: int = 100, conn=None) -> list[dict]:
    """
    Джойн новостей и свечей через VIEW news_with_candles.
    Даёт: новость + overnight следующего дня + интрадей дня новости.
    """
    sql = """
        SELECT news_ts, ticker, sentiment, headline,
               candle_date, open, close,
               ROUND((next_overnight_pct * 100)::numeric, 3) AS next_overnight_pct,
               ROUND((intraday_pct * 100)::numeric, 3)       AS intraday_pct
        FROM news_with_candles
        WHERE ticker = %s
        ORDER BY news_ts DESC
        LIMIT %s;
    """
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(sql, (ticker, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
