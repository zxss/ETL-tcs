"""
Слой рыночного контекста (Dashboard 2.0).

Перестаёт оценивать акцию изолированно: добавляет состояние широкого рынка и
риск-фильтры, которые корректируют рейтинг стратегий.

Считает:
  • Market Regime  — режим рынка по IMOEX (EMA50/EMA200): BULL / BEAR / NEUTRAL
  • Rel_Strength   — сила бумаги относительно рынка за 10 дней (RS, %)
  • Vol_Spike      — объём вчера / SMA20(объём) (кратность)
  • ATR_Pctl       — процентиль текущего ATR(14) за 252 дня (0..100)
  • Gap_Down_Prob  — частота гэпов вниз < -0.5% за ~6 месяцев (0..1)
  • Market Breadth — доля бумаг выше своей EMA50
  • Risk Level     — сводный уровень риска рынка

Источник индекса: если в БД есть дневные свечи IMOEX — берём их; иначе строим
синтетический равновзвешенный индекс по вселенной тикеров (proxy).

Полностью graceful: при ошибке возвращает нейтральный контекст.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

log = logging.getLogger("tft.market")

_SQL = """
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = %(tk)s
    ORDER BY date DESC
    LIMIT %(n)s;
"""

_LOOKBACK = 320          # дней истории на тикер (хватает на EMA200 + ATR252)
_ATR_WIN = 14
_ATR_HIST = 252
_GAP_DAYS = 126          # ~6 месяцев торговых дней
_GAP_THRESH = -0.005     # -0.5%


@dataclass
class TickerMarket:
    rs: float | None = None            # относительная сила, %
    vol_spike: float | None = None     # кратность к SMA20
    atr_pctl: float | None = None      # 0..100
    gap_down_prob: float | None = None # 0..1


@dataclass
class MarketContext2:
    regime: str = "NEUTRAL"            # BULL / BEAR / NEUTRAL
    imoex_ret10: float | None = None   # доходность индекса за 10 дней, %
    breadth: float | None = None       # доля бумаг выше EMA50 (0..1)
    atr_pctl_market: float | None = None  # медиана ATR_Pctl по рынку
    risk_level: str = "NORMAL"         # NORMAL / ELEVATED / HIGH
    source: str = "proxy"              # "IMOEX" | "proxy"
    per: dict = field(default_factory=dict)   # ticker -> TickerMarket


# ── Загрузка дневных свечей ────────────────────────────────────────────────────

def _load(conn, ticker: str) -> pd.DataFrame | None:
    try:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"tk": ticker, "n": _LOOKBACK})
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        return None
    if not rows or len(rows) < 30:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)   # был DESC → в хронологию
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty or float(df["close"].median()) <= 0:
        return None
    return df


# ── Метрики на тикер ───────────────────────────────────────────────────────────

def _atr_pctl(df: pd.DataFrame) -> float | None:
    if len(df) < _ATR_WIN + 5:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(_ATR_WIN).mean().dropna()
    if atr.empty:
        return None
    hist = atr.tail(_ATR_HIST)
    cur = float(atr.iloc[-1])
    pctl = float((hist <= cur).mean() * 100.0)
    return pctl


def _vol_spike(df: pd.DataFrame) -> float | None:
    v = df["volume"].dropna()
    if len(v) < 21:
        return None
    sma20 = float(v.iloc[-21:-1].mean())   # 20 дней до «вчера»
    if sma20 <= 0:
        return None
    return float(v.iloc[-1] / sma20)


def _gap_down_prob(df: pd.DataFrame) -> float | None:
    if len(df) < 30:
        return None
    sub = df.tail(_GAP_DAYS + 1)
    gaps = (sub["open"] / sub["close"].shift(1) - 1.0).dropna()
    if gaps.empty:
        return None
    return float((gaps < _GAP_THRESH).mean())


def _ret10(close: pd.Series) -> float | None:
    if len(close) < 11:
        return None
    return float(close.iloc[-1] / close.iloc[-11] - 1.0)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


# ── Сборка рыночного индекса ───────────────────────────────────────────────────

def _build_index(frames: dict[str, pd.DataFrame]) -> pd.Series | None:
    """Равновзвешенный синтетический индекс: среднедневная доходность по
    вселенной → кумулятивный уровень."""
    rets = []
    for df in frames.values():
        s = df.set_index("date")["close"].pct_change()
        rets.append(s)
    if not rets:
        return None
    mat = pd.concat(rets, axis=1).sort_index()
    daily = mat.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + daily).cumprod()
    return level if len(level) >= 11 else None


def _regime(level: pd.Series) -> str:
    if level is None or len(level) < 5:
        return "NEUTRAL"
    ema50 = _ema(level, 50).iloc[-1]
    ema200 = _ema(level, 200).iloc[-1]
    cur = level.iloc[-1]
    if cur > ema50 and ema50 > ema200:
        return "BULL"
    if cur < ema50 and ema50 < ema200:
        return "BEAR"
    return "NEUTRAL"


# ── Публичная точка входа ──────────────────────────────────────────────────────

def compute(conn, tickers: list[str]) -> MarketContext2:
    ctx = MarketContext2()
    try:
        frames: dict[str, pd.DataFrame] = {}
        for tk in tickers:
            df = _load(conn, tk.upper())
            if df is not None:
                frames[tk.upper()] = df
        if not frames:
            return ctx

        # Индекс: IMOEX из БД либо синтетический proxy.
        imoex = _load(conn, "IMOEX")
        if imoex is not None and len(imoex) >= 50:
            level = imoex.set_index("date")["close"].astype(float)
            ctx.source = "IMOEX"
        else:
            level = _build_index(frames)
            ctx.source = "proxy"

        ctx.regime = _regime(level)
        r10 = _ret10(level) if level is not None else None
        ctx.imoex_ret10 = r10 * 100.0 if r10 is not None else None

        # Метрики на тикер + breadth.
        above = 0
        counted = 0
        atr_list = []
        for tk, df in frames.items():
            close = df["close"]
            srs = _ret10(close)
            rs = None
            if srs is not None and r10 is not None:
                rs = (srs - r10) * 100.0
            tm = TickerMarket(
                rs=rs,
                vol_spike=_vol_spike(df),
                atr_pctl=_atr_pctl(df),
                gap_down_prob=_gap_down_prob(df),
            )
            ctx.per[tk] = tm
            if tm.atr_pctl is not None:
                atr_list.append(tm.atr_pctl)
            if len(close) >= 50:
                counted += 1
                if float(close.iloc[-1]) > float(_ema(close, 50).iloc[-1]):
                    above += 1

        ctx.breadth = (above / counted) if counted else None
        ctx.atr_pctl_market = float(np.median(atr_list)) if atr_list else None
        ctx.risk_level = _risk_level(ctx)
        return ctx
    except Exception as e:  # noqa: BLE001
        log.warning("Слой рыночного контекста недоступен (%s) — нейтральный режим.", e)
        return MarketContext2()


def _risk_level(ctx: MarketContext2) -> str:
    atr = ctx.atr_pctl_market
    breadth = ctx.breadth
    if (atr is not None and atr > 85) or (breadth is not None and breadth < 0.35):
        return "HIGH"
    if (atr is not None and atr > 70) or (breadth is not None and breadth < 0.45):
        return "ELEVATED"
    return "NORMAL"
