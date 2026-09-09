from __future__ import annotations
"""
Работа с PostgreSQL: пул соединений, инициализация схемы, upsert, запросы
последних дат для инкрементального режима.

ТРАНЗАКЦИИ И ПАРАЛЛЕЛЬНОСТЬ
---------------------------
Загрузка тикеров идёт в несколько потоков (config.MAX_CONCURRENT_TICKERS через
asyncio.to_thread). Раньше все потоки писали в ОДИН общий коннект, а каждая
низкоуровневая вставка делала conn.commit() — коммит одного потока фиксировал
незавершённую транзакцию другого. Теперь:

  * соединения выдаёт psycopg2.pool.ThreadedConnectionPool;
  * границу транзакции задаёт контекстный менеджер get_db_connection():
    commit на нормальном выходе, rollback при исключении, возврат в пул всегда;
  * низкоуровневые upsert'ы НЕ коммитят — они пишут в транзакцию вызывающего.

Функции работы с БД принимают conn последним необязательным аргументом:
  * conn=None  → берут коннект из пула и коммитят своей транзакцией;
  * conn задан → пишут в чужую транзакцию и НЕ коммитят (владелец — вызывающий).
"""

import logging
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2 import OperationalError

import config
from models.market_data import (
    CREATE_TABLE_SQL, UPSERT_SQL, LAST_DATE_SQL,
    CREATE_TABLE_5M_SQL, UPSERT_5M_SQL, LAST_TS_5M_SQL,
    CREATE_FORECASTS_SQL, UPSERT_FORECAST_SQL, FORECAST_COLUMNS,
    MIGRATE_MARKET_DATA_SOURCE_SQL,
)
from models.instruments import (
    CREATE_INSTRUMENTS_SQL, UPSERT_INSTRUMENT_SQL, SELECT_LOTS_SQL,
    SELECT_INSTRUMENTS_SQL,
)

log = logging.getLogger("database")


# --- Параметры подключения ---------------------------------------------------

def _connect_kwargs() -> dict:
    kwargs = dict(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
    )
    if config.DB_PASSWORD:
        kwargs["password"] = config.DB_PASSWORD
    return kwargs


# --- Пул соединений ----------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def init_pool(minconn: int | None = None,
              maxconn: int | None = None) -> psycopg2.pool.ThreadedConnectionPool:
    """Создаёт (идемпотентно) потокобезопасный пул соединений."""
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool
    with _pool_lock:
        if _pool is not None and not _pool.closed:
            return _pool
        mn = int(minconn if minconn is not None else getattr(config, "DB_POOL_MIN", 1))
        mx = int(maxconn if maxconn is not None else getattr(config, "DB_POOL_MAX", 5))
        mx = max(mx, mn)
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=mn, maxconn=mx, **_connect_kwargs())
        except OperationalError as e:
            raise RuntimeError(f"Не удалось подключиться к PostgreSQL: {e}") from e
        log.info("Пул соединений PostgreSQL: minconn=%d, maxconn=%d", mn, mx)
        return _pool


def close_pool() -> None:
    """Закрывает все соединения пула (вызывать при завершении процесса)."""
    global _pool
    with _pool_lock:
        if _pool is not None and not _pool.closed:
            _pool.closeall()
            log.info("Пул соединений PostgreSQL закрыт.")
        _pool = None


@contextmanager
def get_db_connection():
    """Коннект из пула на время блока. Commit на нормальном выходе, rollback
    при исключении, возврат в пул в любом случае.

        with get_db_connection() as conn:
            upsert_candles(rows, conn)      # без commit внутри
            upsert_candles_5m(rows5m, conn) # обе пачки — одна транзакция
    """
    pool = init_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 — коннект мог умереть
            log.warning("Rollback не удался — соединение будет пересоздано пулом.")
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def _tx(conn):
    """conn=None → своя транзакция из пула; иначе — транзакция вызывающего."""
    if conn is not None:
        yield conn
    else:
        with get_db_connection() as own:
            yield own


def get_connection():
    """Отдельное (НЕ пуловое) соединение для однопоточных владельцев на весь
    прогон: main.py, run_monitor.py, place_orders. Такой коннект живёт до
    conn.close() и не должен использоваться из нескольких потоков."""
    try:
        return psycopg2.connect(**_connect_kwargs())
    except OperationalError as e:
        raise RuntimeError(f"Не удалось подключиться к PostgreSQL: {e}") from e


# --- Схема -------------------------------------------------------------------

def init_db(conn=None) -> None:
    """Создаёт таблицы и индексы штатного конвейера, если их нет.

    Контур новостей сюда НЕ входит: он отключён и вынесен в
    contrib/experimental_news/ вместе со своей схемой (news_sentiment,
    view news_with_candles) и функциями доступа."""
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TABLE_5M_SQL)
            cur.execute(CREATE_FORECASTS_SQL)
            cur.execute(CREATE_INSTRUMENTS_SQL)
            # Миграции существующих баз (идемпотентны).
            cur.execute(MIGRATE_MARKET_DATA_SOURCE_SQL)
        if conn is not None:
            c.commit()   # DDL у владельца коннекта фиксируем сразу
    log.info("Схема БД инициализирована "
             "(market_data + market_data_5m + forecasts + instruments)")


# --- Свечи -------------------------------------------------------------------

def get_last_date(ticker: str, conn=None) -> date | None:
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(LAST_DATE_SQL, (ticker,))
            row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_last_ts_5m(ticker: str, conn=None) -> datetime | None:
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(LAST_TS_5M_SQL, (ticker,))
            row = cur.fetchone()
    val = row[0] if row and row[0] else None
    if val and val.tzinfo is None:
        val = val.replace(tzinfo=timezone.utc)
    return val


SOURCE_API = "api"
SOURCE_BACKFILL_5M = "backfill_5m"


def upsert_candles(rows: list[tuple], conn=None, source: str = SOURCE_API) -> int:
    """Дневные свечи. НЕ коммитит при переданном conn — транзакцией владеет
    вызывающий (пачка по тикеру фиксируется целиком).

    rows: (ticker, date, open, high, low, close, volume) — колонка source
    добавляется здесь: этим путём идёт ТОЛЬКО официальный бар биржи ('api').
    Реконструкция из 5-минуток пишется своим SQL в services/backfill_daily.py
    и помечается 'backfill_5m'."""
    if not rows:
        return 0
    rows = [(*r, source) for r in rows]
    with _tx(conn) as c:
        with c.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=500)
    return len(rows)


def upsert_candles_5m(rows: list[tuple], conn=None) -> int:
    if not rows:
        return 0
    with _tx(conn) as c:
        with c.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_5M_SQL, rows, page_size=1000)
    return len(rows)


# --- Прогнозы и сигналы ------------------------------------------------------

def save_forecasts(rows: list[dict], conn=None) -> int:
    """
    Upsert рассчитанных прогнозов/сигналов в forecasts по ключу
    (asof_date, ticker, strategy).

    rows — список dict с ключами FORECAST_COLUMNS; отсутствующие поля → NULL.
    raw_payload сериализуется в JSONB. Возвращает число записанных строк.
    """
    if not rows:
        return 0
    payload = []
    for r in rows:
        rec = {k: r.get(k) for k in FORECAST_COLUMNS}
        if not rec.get("asof_date") or not rec.get("ticker") or not rec.get("strategy"):
            continue   # ключевые поля обязательны — молча не пишем мусор
        raw = rec.get("raw_payload")
        rec["raw_payload"] = psycopg2.extras.Json(raw) if raw is not None else None
        payload.append(rec)
    if not payload:
        return 0
    with _tx(conn) as c:
        with c.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_FORECAST_SQL, payload, page_size=200)
    return len(payload)


# --- Справочник инструментов -------------------------------------------------

def save_instruments(rows: list[dict], conn=None) -> int:
    """Upsert справочника инструментов по ключу ticker.

    rows — список dict с ключами колонок instruments; отсутствующие поля → NULL.
    Возвращает число записанных строк.
    """
    if not rows:
        return 0
    cols = ("ticker", "figi", "uid", "name", "class_code", "lot",
            "min_price_increment", "short_enabled", "buy_available",
            "sell_available", "api_trade_available", "for_qual_investor",
            "dlong_client", "dshort_client", "trading_status", "sector")
    payload = []
    for r in rows:
        rec = {k: r.get(k) for k in cols}
        if not rec.get("ticker"):
            continue
        rec["ticker"] = str(rec["ticker"]).upper()
        try:
            rec["lot"] = int(rec.get("lot") or 1)
        except (TypeError, ValueError):
            rec["lot"] = 1
        payload.append(rec)
    if not payload:
        return 0
    with _tx(conn) as c:
        with c.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_INSTRUMENT_SQL, payload,
                                          page_size=100)
    return len(payload)


def get_instrument_lots(conn=None) -> dict[str, int]:
    """{TICKER: lot} из кэша справочника. Пустой словарь, если кэш не наполнен.

    Вызывающий обязан различать «лот неизвестен» и «лот = 1»: молча подставлять
    единицу нельзя, потому что для TGKA это ошибка в 100 000 раз.
    """
    try:
        with _tx(conn) as c:
            with c.cursor() as cur:
                cur.execute(SELECT_LOTS_SQL)
                rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001 — кэша может ещё не быть
        log.warning("Не удалось прочитать кэш инструментов: %s", e)
        return {}
    return {str(tk).upper(): int(lot) for tk, lot, _ in rows if lot}


def get_instruments(conn=None) -> dict[str, dict]:
    """Полный кэш справочника: {TICKER: {lot, tick, short_enabled, ...}}."""
    try:
        with _tx(conn) as c:
            with c.cursor() as cur:
                cur.execute(SELECT_INSTRUMENTS_SQL)
                rows = cur.fetchall()
                names = [d[0] for d in cur.description]
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось прочитать кэш инструментов: %s", e)
        return {}
    out = {}
    for row in rows:
        rec = dict(zip(names, row))
        out[str(rec["ticker"]).upper()] = rec
    return out


def query_forecasts(asof_date, ticker: str | None = None, conn=None) -> list[dict]:
    """Сохранённые прогнозы за дату (опционально по тикеру) — для разбора
    качества сигналов постфактум."""
    sql = """
        SELECT asof_date, ticker, strategy, anchor_price, q10, q50, q90,
               exp_pnl, prob_profit, final_score, verdict, raw_payload, created_at
        FROM forecasts
        WHERE asof_date = %s
          AND (%s IS NULL OR ticker = %s)
        ORDER BY final_score DESC NULLS LAST, ticker;
    """
    tk = ticker.upper() if ticker else None
    with _tx(conn) as c:
        with c.cursor() as cur:
            cur.execute(sql, (asof_date, tk, tk))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
