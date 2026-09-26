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

import config
import trading_calendar

log = logging.getLogger("tft.features")

# Горизонт недельного прогноза (торговых дней). Модель обучается предсказывать
# min(low)/max(high)/close за следующие WEEK_H дней НАРЯДУ с дневными целями.
WEEK_H = max(1, int(getattr(config, "WEEK_HORIZON_DAYS", 5)))

# Список time-varying числовых признаков (входы энкодера на каждый день).
FEATURE_COLS = [
    "log_ret",       # лог-доходность close-to-close
    "intraday",      # (close/open - 1) * 100
    "overnight",     # ((open + див) / prev_close - 1) * 100 — полная доходность
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
#   дня, из которых собирается направленный PnL стратегий (см. directional.py);
#   week_low_pct / week_high_pct / week_total — НЕДЕЛЬНЫЙ горизонт (WEEK_H торговых
#   дней вперёд): экстремумы min(low)/max(high) и итоговый close за окно. Модель
#   предсказывает их квантили той же головой — это настоящий multi-horizon
#   прогноз, а не масштабирование дневного по √t.
TARGET_COLS = [
    "next_low_pct",
    "next_high_pct",
    "next_overnight",   # ((next_open + див)/today_close - 1)*100 → long_overnight
    "next_intraday",    # (next_close / next_open - 1) * 100     → long/short intraday
    "next_total",       # (next_close / today_close - 1) * 100   → short_hold (с минусом)
    "week_low_pct",     # (min(low[t+1..t+H]) / today_close - 1) * 100
    "week_high_pct",    # (max(high[t+1..t+H]) / today_close - 1) * 100
    "week_total",       # (close[t+H] / today_close - 1) * 100
]

# Минимум наблюдений у тикера, чтобы он попал в обучающий пул.
MIN_ROWS = 120

# Фильтр сессий подставляется на КАЖДЫЙ вызов (не на импорте): режим
# INCLUDE_WEEKEND_TRADING может меняться между прогонами и в тестах.
_EXPORT_SQL_TMPL = """
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = %s{session_filter}
    ORDER BY date ASC;
"""


@dataclass
class TickerFrame:
    ticker: str
    df: pd.DataFrame          # признаки + цели, уже очищенные
    last_close: float         # close последнего дня (для перевода pct → цена)
    last_date: object         # дата последнего дня


def _load_raw(conn, ticker: str) -> pd.DataFrame | None:
    sql = _EXPORT_SQL_TMPL.format(
        session_filter=trading_calendar.sql_session_filter("date"))
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
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


# Потолок правдоподобной дивдоходности одной выплаты, % от цены. Выше — это не
# дивиденд, а рассогласование масштаба акции: выплата объявлена по СТАРОЙ бумаге,
# а ряд цен идёт по новой после редомициляции или сплита. Живой пример: FIXR
# 13.12.2024 — дивиденд 35,31 ₽ по расписке FIXP против цены акции FIXR 1,30 ₽,
# то есть «дивдоходность» 2708 %. Без потолка поправка внесла бы в обучающую
# выборку ошибку в двадцать семь раз больше той, которую лечит.
# Порог 50 %: настоящий максимум в нашем универсе — VTBR 11.07.2025 с 27,4 %.
MAX_DIV_YIELD_PCT = 50.0


def _div_gap_series(df: pd.DataFrame, payments: list[dict],
                    ticker: str = "?") -> pd.Series:
    """Дивиденд (₽), гэп которого приходится на каждый день ряда.

    Свечи T-Invest не скорректированы на дивиденды: в день отсечки open
    проваливается на величину выплаты. Гэп ложится на открытие ПЕРВОГО торгового
    дня строго после last_buy_date — его и ищем поиском по отсортированным датам
    (та же привязка, что в research/xsec/panel.py).

    Выплата, чей гэп приходится за пределы ряда, отбрасывается. Две отсечки в
    один день (редко, но бывает после объединения выпусков) складываются.
    Выплата, несопоставимая с масштабом цены, отбрасывается с предупреждением —
    см. MAX_DIV_YIELD_PCT.
    """
    div = pd.Series(0.0, index=df.index)
    if not payments:
        return div
    days = df["date"].to_numpy()
    prev_close = df["close"].shift(1)
    for p in payments:
        lb = p.get("last_buy_date")
        amount = float(p.get("dividend_net") or 0.0)
        if lb is None or amount <= 0:
            continue
        i = int(np.searchsorted(days, np.datetime64(pd.Timestamp(lb)), side="right"))
        if not (0 < i < len(div)):
            continue
        px = float(prev_close.iat[i]) if np.isfinite(prev_close.iat[i]) else 0.0
        if px > 0 and amount / px * 100.0 > MAX_DIV_YIELD_PCT:
            log.warning("  [%s] выплата %.4f ₽ на %s отброшена: %.0f %% цены %.4f ₽ — "
                        "масштаб акции не совпадает (сплит/редомициляция)",
                        ticker, amount, lb, amount / px * 100.0, px)
            continue
        div.iat[i] += amount
    return div


def _load_dividends(conn, ticker: str) -> list[dict]:
    """Календарь отсечек по бумаге. Пустой список — поведение как прежде."""
    if not getattr(config, "DIVIDEND_ADJUST_FEATURES", True):
        return []
    try:
        import database
        return database.get_dividends(ticker, conn)
    except Exception as e:  # noqa: BLE001 — таблицы может не быть на старой базе
        log.warning("  [%s] календарь отсечек недоступен: %s", ticker, e)
        return []


def _is_sane(df: pd.DataFrame) -> bool:
    """Грубая проверка масштаба цены — отсекаем битые серии (ср. TGKA)."""
    med = df["close"].median()
    if not np.isfinite(med) or med <= 0.05:
        return False
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False
    return True


def _build_features(df: pd.DataFrame, payments: list[dict] | None = None,
                    ticker: str = "?") -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)

    # Дивидендная поправка. В день отсечки цена падает на выплату, а держатель
    # её получает: полная доходность ночи = (open + D) / prev_close − 1. Без
    # поправки признак overnight в такие дни — не доходность, а дивиденд с
    # обратным знаком (медиана 5,2 %, максимум 20 % против обычных 0,5 %), и в
    # квантильном лоссе он тянет q10 вниз, то есть задирает стопы.
    # Правятся ТОЛЬКО сравнения с ВЧЕРАШНИМ закрытием. Внутридневные
    # (intraday = close/open) дивидендом не задеты: обе цены уже после гэпа.
    div = _div_gap_series(d, payments or [], ticker)
    d["div_gap"] = div

    d["log_ret"] = np.log((d["close"] + div) / prev_close)
    d["intraday"] = (d["close"] / d["open"] - 1) * 100
    d["overnight"] = ((d["open"] + div) / prev_close - 1) * 100
    d["range_pct"] = (d["high"] - d["low"]) / prev_close * 100

    # ATR(14) на истинном диапазоне
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] + div - prev_close).abs(),
            (d["low"] + div - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr_pct"] = tr.rolling(14, min_periods=5).mean() / d["close"] * 100

    vol_mean = d["volume"].rolling(20, min_periods=5).mean()
    vol_std = d["volume"].rolling(20, min_periods=5).std()
    d["vol_z"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    d["ret_mean5"] = d["log_ret"].rolling(5, min_periods=2).mean() * 100
    d["ret_std20"] = d["log_ret"].rolling(20, min_periods=5).std() * 100

    # Период кодирования: 5 в строгом календаре (Пн–Пт), 7 при торговле 7/7.
    dow_period = trading_calendar.dow_period()
    dow = d["date"].dt.dayofweek
    d["dow_sin"] = np.sin(2 * np.pi * dow / dow_period)
    d["dow_cos"] = np.cos(2 * np.pi * dow / dow_period)

    # День недели ПРОГНОЗИРУЕМОГО (следующего) дня — known-future ковариата.
    # Для обучающих строк это dow реального следующего бара (корректно учитывает
    # и выходные, и праздники). Для последней строки следующего бара ещё нет —
    # берём следующий торговый день (пропуская сб/вс): именно его day-of-week мы
    # предсказываем. Это и отделяет прогноз на понедельник от прогноза на среду.
    next_date = d["date"].shift(-1)
    if len(d):
        # Следующий торговый день по ТЕКУЩЕМУ календарю (а не «пропусти сб/вс»):
        # в режиме 7/7 это буквально завтра.
        nb = trading_calendar.next_trading_day(d["date"].iloc[-1].date())
        next_date = next_date.copy()
        next_date.iloc[-1] = pd.Timestamp(nb)
    ndow = next_date.dt.dayofweek
    d["next_dow"] = ndow.astype("Int64")          # сырой день недели для fallback-бакетов
    d["next_dow_sin"] = np.sin(2 * np.pi * ndow / dow_period)
    d["next_dow_cos"] = np.cos(2 * np.pi * ndow / dow_period)

    # Цели: завтрашние low/high относительно СЕГОДНЯШНЕГО close (без look-ahead:
    # предсказываем будущее, признаки — только до текущего дня включительно).
    nopen = d["open"].shift(-1)
    nclose = d["close"].shift(-1)
    # Дивиденд ЗАВТРАШНЕГО гэпа: цели сравнивают завтрашние цены с сегодняшним
    # закрытием, значит поправка нужна та же, что и в признаках.
    ndiv = div.shift(-1).fillna(0.0)
    d["next_low_pct"] = ((d["low"].shift(-1) + ndiv) / d["close"] - 1) * 100
    d["next_high_pct"] = ((d["high"].shift(-1) + ndiv) / d["close"] - 1) * 100
    d["next_overnight"] = ((nopen + ndiv) / d["close"] - 1) * 100
    d["next_intraday"] = (nclose / nopen - 1) * 100
    d["next_total"] = ((nclose + ndiv) / d["close"] - 1) * 100

    # Недельные цели: экстремумы и close за следующие WEEK_H торговых дней.
    # Reverse-rolling: rolling по перевёрнутому ряду в позиции t покрывает
    # оригинальные [t .. t+H-1]; shift(-1) сдвигает окно на [t+1 .. t+H].
    # Последние H строк остаются NaN — dataset их не берёт в обучение.
    fwd_min_low = d["low"][::-1].rolling(WEEK_H, min_periods=WEEK_H).min()[::-1].shift(-1)
    fwd_max_high = d["high"][::-1].rolling(WEEK_H, min_periods=WEEK_H).max()[::-1].shift(-1)
    # Дивиденды, выплаченные ВНУТРИ окна [t+1 .. t+H]. Для week_total поправка
    # точная. Для экстремумов приближённая: истинная зависит от того, до или
    # после отсечки случился минимум. Окно — неделя, отсечка в нём редкость,
    # а без поправки ошибка равна всему дивиденду, с ней — доле от него.
    fwd_div = div[::-1].rolling(WEEK_H, min_periods=WEEK_H).sum()[::-1].shift(-1).fillna(0.0)
    d["week_low_pct"] = ((fwd_min_low + fwd_div) / d["close"] - 1) * 100
    d["week_high_pct"] = ((fwd_max_high + fwd_div) / d["close"] - 1) * 100
    d["week_total"] = ((d["close"].shift(-WEEK_H) + fwd_div) / d["close"] - 1) * 100

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
        feat = _build_features(raw, _load_dividends(conn, tk), tk)
        last_close = float(raw["close"].iloc[-1])
        last_date = raw["date"].iloc[-1].date()
        frames.append(TickerFrame(tk, feat, last_close, last_date))
        log.info("  [%s] %d дней, last_close=%.2f (%s)",
                 tk, len(feat), last_close, last_date)
    return frames
