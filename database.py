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
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(CREATE_TABLE_5M_SQL)
    conn.commit()
    log.info("Схема БД инициализирована (market_data + market_data_5m)")


def get_last_date(conn, ticker: str) -> date | None:
    """MAX(date) из market_data для тикера, или None."""
    with conn.cursor() as cur:
        cur.execute(LAST_DATE_SQL, (ticker,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_last_ts_5m(conn, ticker: str) -> datetime | None:
    """MAX(ts) из market_data_5m для тикера, или None."""
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
