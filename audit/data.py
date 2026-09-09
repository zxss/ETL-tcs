"""
Слой доступа к данным для квант-аудита.

Отдельный от tft_forecast/features.py намеренно: аудит должен читать сырьё
независимо от того, как его готовит production-конвейер, иначе ошибка в
подготовке признаков останется невидимой (аудит воспроизведёт её же).

Все функции возвращают pandas-объекты и ничего не пишут в БД.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("audit.data")

MSK = dt.timezone(dt.timedelta(hours=3))

# Тикер индекса в market_data (лежит в той же таблице, что и акции).
INDEX_TICKER = "IMOEX"


def _conn():
    import database
    return database.get_connection()


def load_daily(tickers: list[str] | None = None) -> pd.DataFrame:
    """Дневные бары long-форматом: ticker, date, open, high, low, close, volume.

    Возвращает только строки с положительными ценами; строки с нулевым объёмом
    сохраняются (у индекса объёма нет по определению).
    """
    sql = """
        SELECT ticker, date, open, high, low, close, volume
        FROM market_data
        WHERE open > 0 AND high > 0 AND low > 0 AND close > 0
        {flt}
        ORDER BY ticker, date
    """
    params: list = []
    flt = ""
    if tickers:
        flt = "AND ticker = ANY(%s)"
        params.append(list(tickers))

    conn = _conn()
    try:
        df = pd.read_sql(sql.format(flt=flt), conn, params=params or None)
    finally:
        conn.close()

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_5m(tickers: list[str] | None = None,
            since: dt.date | None = None,
            main_session_only: bool = True) -> pd.DataFrame:
    """5-минутные бары с московским временем.

    main_session_only=True оставляет только основную сессию 09:50–18:50 МСК:
    вечёрка имеет принципиально другую ликвидность, и смешивать их при оценке
    спреда нельзя.
    """
    sql = """
        SELECT ticker,
               (ts AT TIME ZONE 'Europe/Moscow') AS ts_msk,
               open, high, low, close, volume
        FROM market_data_5m
        WHERE close > 0
        {tflt}
        {dflt}
        {sflt}
        ORDER BY ticker, ts
    """
    params: list = []
    tflt = dflt = sflt = ""
    if tickers:
        tflt = "AND ticker = ANY(%s)"
        params.append(list(tickers))
    if since:
        dflt = "AND (ts AT TIME ZONE 'Europe/Moscow')::date >= %s"
        params.append(since)
    if main_session_only:
        sflt = ("AND (ts AT TIME ZONE 'Europe/Moscow')::time >= TIME '09:50' "
                "AND (ts AT TIME ZONE 'Europe/Moscow')::time <  TIME '18:50'")

    conn = _conn()
    try:
        df = pd.read_sql(sql.format(tflt=tflt, dflt=dflt, sflt=sflt),
                         conn, params=params or None)
    finally:
        conn.close()

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["ts_msk"] = pd.to_datetime(df["ts_msk"])
    df["date"] = df["ts_msk"].dt.normalize()
    return df


def universe(exclude_index: bool = True) -> list[str]:
    """Тикеры, реально присутствующие в market_data."""
    conn = _conn()
    try:
        df = pd.read_sql(
            "SELECT DISTINCT ticker FROM market_data ORDER BY ticker", conn)
    finally:
        conn.close()
    tk = df["ticker"].tolist()
    if exclude_index:
        tk = [t for t in tk if t != INDEX_TICKER]
    return tk


def to_panel(daily: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Wide-панель date x ticker по одному полю."""
    return daily.pivot(index="date", columns="ticker", values=field).sort_index()


def add_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Добавляет примитивные доходности в процентах, как их считает pnl_engine.

    overnight = open_t / close_{t-1} - 1
    intraday  = close_t / open_t - 1
    total     = close_t / close_{t-1} - 1

    Все три — в процентах, чтобы совпадать с единицами VALIDATION_COST_RT.
    """
    out = []
    for tk, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        prev_close = g["close"].shift(1)
        g["overnight"] = (g["open"] / prev_close - 1.0) * 100.0
        g["intraday"] = (g["close"] / g["open"] - 1.0) * 100.0
        g["total"] = (g["close"] / prev_close - 1.0) * 100.0
        g["log_ret"] = np.log(g["close"] / prev_close)
        g["rub_volume"] = g["close"] * g["volume"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def trading_days(daily: pd.DataFrame, min_tickers: int = 20) -> pd.DatetimeIndex:
    """Даты, в которые торговалось не меньше min_tickers бумаг.

    Защита от «дней-призраков»: одиночная свеча по одной бумаге не должна
    создавать торговый день для всей кросс-секции.
    """
    cnt = daily.groupby("date")["ticker"].nunique()
    return pd.DatetimeIndex(cnt[cnt >= min_tickers].index).sort_values()
