from __future__ import annotations
"""
Синхронное подключение к PostgreSQL: инициализация схемы, upsert,
запросы последних дат для инкрементального режима.
"""

import logging
from datetime import date, datetime, timezone

import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError

import config
from models.market_data import (
    CREATE_TABLE_SQL, UPSERT_SQL, LAST_DATE_SQL,
    CREATE_TABLE_5M_SQL, UPSERT_5M_SQL, LAST_TS_5M_SQL,
    CREATE_NEWS_SQL, UPSERT_NEWS_SQL, LAST_MESSAGE_ID_SQL,
    CREATE_NEWS_CANDLES_VIEW_SQL,
)

log = logging.getLogger("database")


def get_connection():
    try:
        kwargs = dict(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
        )
        if config.DB_PASSWORD:
            kwargs["password"] = config.DB_PASSWORD
        return psycopg2.connect(**kwargs)
    except OperationalError as e:
        raise RuntimeError(f"Не удалось подключиться к PostgreSQL: {e}") from e


def init_db(conn) -> None:
    """Создаёт все таблицы, индексы и view если их нет."""
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(CREATE_TABLE_5M_SQL)
        cur.execute(CREATE_NEWS_SQL)
        cur.execute(CREATE_NEWS_CANDLES_VIEW_SQL)
    conn.commit()
    log.info("Схема БД инициализирована (market_data + market_data_5m + news_sentiment)")


# --- Свечи -------------------------------------------------------------------

def get_last_date(conn, ticker: str) -> date | None:
    with conn.cursor() as cur:
        cur.execute(LAST_DATE_SQL, (ticker,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_last_ts_5m(conn, ticker: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(LAST_TS_5M_SQL, (ticker,))
        row = cur.fetchone()
    val = row[0] if row and row[0] else None
    if val and val.tzinfo is None:
        val = val.replace(tzinfo=timezone.utc)
    return val


def upsert_candles(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=500)
    conn.commit()
    return len(rows)


def upsert_candles_5m(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_5M_SQL, rows, page_size=1000)
    conn.commit()
    return len(rows)


# --- Новости -----------------------------------------------------------------

def get_last_message_id(conn, source: str) -> int | None:
    """MAX(message_id) из news_sentiment для источника, или None."""
    with conn.cursor() as cur:
        cur.execute(LAST_MESSAGE_ID_SQL, (source,))
        row = cur.fetchone()
    return int(row[0]) if row and row[0] else None


def upsert_news(conn, rows: list[tuple]) -> int:
    """
    Upsert записей в news_sentiment.
    rows: (message_id, ts, ticker, sentiment, headline, raw_text, source)
    """
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_NEWS_SQL, rows, page_size=200)
    conn.commit()
    return len(rows)


def query_news(conn, ticker: str, limit: int = 50) -> list[dict]:
    """Последние новости по тикеру для аналитики."""
    sql = """
        SELECT ts, sentiment, headline, message_id
        FROM news_sentiment
        WHERE ticker = %s
        ORDER BY ts DESC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, limit))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def query_news_candles(conn, ticker: str, limit: int = 100) -> list[dict]:
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
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, limit))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
