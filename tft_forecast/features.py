"""
Загрузка дневных свечей из PostgreSQL и инженерия признаков для TFT.

Один общий пул данных по всем тикерам — модель учится на кросс-секции
российского рынка, а не на отдельном инструменте. Тикер кодируется
категориальным static-признаком (embedding в модели).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("tft.features")

# Список time-varying числовых признаков (входы энкодера на каждый день).
FEATURE_COLS = [
    "log_ret",       # лог-доходность close-to-close
    "intraday",      # (close/open - 1) * 100
    "overnight",     # (open/prev_close - 1) * 100
    "range_pct",     # (high-low)/prev_close * 100  — реализованный дневной диапазон
    "atr_pct",       # ATR(14) / close * 100
    "vol_z",         # z-оценка объёма за 20 дней
    "ret_mean5",     # средняя доходность за 5 дней
    "ret_std20",     # волатильность доходности за 20 дней
    "dow_sin",       # день недели сегодня (циклический)
    "dow_cos",
    "next_dow_sin",  # день недели ПРОГНОЗИРУЕМОГО дня (known-future: календарь
    "next_dow_cos",  # завтра известен заранее). Ловит Пт→Пн (гэп через выходные).
]

# Целевые переменные:
#   next_low_pct / next_high_pct — коридор (low/high завтра отн. сегодняшнего close);
#   next_overnight / next_intraday / next_total — ПРИМИТИВНЫЕ доходности следующего
#   дня, из которых собирается направленный PnL стратегий (см. directional.py).
TARGET_COLS = [
    "next_low_pct",
    "next_high_pct",
    "next_overnight",   # (next_open / today_close - 1) * 100   → long_overnight
    "next_intraday",    # (next_close / next_open - 1) * 100     → long/short intraday
    "next_total",       # (next_close / today_close - 1) * 100   → short_hold (с минусом)
]

# Минимум наблюдений у тикера, чтобы он попал в обучающий пул.
MIN_ROWS = 120

_EXPORT_SQL = """
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = %s
    ORDER BY date ASC;
"""


@dataclass
class TickerFrame:
    ticker: str
    df: pd.DataFrame          # признаки + цели, уже очищенные
    last_close: float         # close последнего дня (для перевода pct → цена)
    last_date: object         # дата последнего дня


def _load_raw(conn, ticker: str) -> pd.DataFrame | None:
    with conn.cursor() as cur:
        cur.execute(_EXPORT_SQL, (ticker,))
        rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _is_sane(df: pd.DataFrame) -> bool:
    """Грубая проверка масштаба цены — отсекаем битые серии (ср. TGKA)."""
    med = df["close"].median()
    if not np.isfinite(med) or med <= 0.05:
        return False
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False
    return True


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)

    d["log_ret"] = np.log(d["close"] / prev_close)
    d["intraday"] = (d["close"] / d["open"] - 1) * 100
    d["overnight"] = (d["open"] / prev_close - 1) * 100
    d["range_pct"] = (d["high"] - d["low"]) / prev_close * 100

    # ATR(14) на истинном диапазоне
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr_pct"] = tr.rolling(14, min_periods=5).mean() / d["close"] * 100

    vol_mean = d["volume"].rolling(20, min_periods=5).mean()
    vol_std = d["volume"].rolling(20, min_periods=5).std()
    d["vol_z"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    d["ret_mean5"] = d["log_ret"].rolling(5, min_periods=2).mean() * 100
    d["ret_std20"] = d["log_ret"].rolling(20, min_periods=5).std() * 100

    dow = d["date"].dt.dayofweek
    d["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # День недели ПРОГНОЗИРУЕМОГО (следующего) дня — known-future ковариата.
    # Для обучающих строк это dow реального следующего бара (корректно учитывает
    # и выходные, и праздники). Для последней строки следующего бара ещё нет —
    # берём следующий торговый день (пропуская сб/вс): именно его day-of-week мы
    # предсказываем. Это и отделяет прогноз на понедельник от прогноза на среду.
    next_date = d["date"].shift(-1)
    if len(d):
        nb = d["date"].iloc[-1] + pd.Timedelta(days=1)
        while nb.dayofweek >= 5:  # сб(5)/вс(6) → следующий торговый день
            nb += pd.Timedelta(days=1)
        next_date = next_date.copy()
        next_date.iloc[-1] = nb
    ndow = next_date.dt.dayofweek
    d["next_dow"] = ndow.astype("Int64")          # сырой день недели для fallback-бакетов
    d["next_dow_sin"] = np.sin(2 * np.pi * ndow / 5.0)
    d["next_dow_cos"] = np.cos(2 * np.pi * ndow / 5.0)

    # Цели: завтрашние low/high относительно СЕГОДНЯШНЕГО close (без look-ahead:
    # предсказываем будущее, признаки — только до текущего дня включительно).
    nopen = d["open"].shift(-1)
    nclose = d["close"].shift(-1)
    d["next_low_pct"] = (d["low"].shift(-1) / d["close"] - 1) * 100
    d["next_high_pct"] = (d["high"].shift(-1) / d["close"] - 1) * 100
    d["next_overnight"] = (nopen / d["close"] - 1) * 100
    d["next_intraday"] = (nclose / nopen - 1) * 100
    d["next_total"] = (nclose / d["close"] - 1) * 100

    return d


def load_panel(conn, tickers: list[str]) -> list[TickerFrame]:
    """Грузит и подготавливает признаки по всем тикерам. Возвращает список TickerFrame."""
    frames: list[TickerFrame] = []
    for tk in tickers:
        raw = _load_raw(conn, tk)
        if raw is None or len(raw) < MIN_ROWS:
            log.info("  [%s] пропуск: данных нет или < %d строк", tk, MIN_ROWS)
            continue
        if not _is_sane(raw):
            log.warning("  [%s] пропуск: подозрительный масштаб цены (median close=%.4f)",
                        tk, raw["close"].median())
            continue
        feat = _build_features(raw)
        last_close = float(raw["close"].iloc[-1])
        last_date = raw["date"].iloc[-1].date()
        frames.append(TickerFrame(tk, feat, last_close, last_date))
        log.info("  [%s] %d дней, last_close=%.2f (%s)",
                 tk, len(feat), last_close, last_date)
    return frames
