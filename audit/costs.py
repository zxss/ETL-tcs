"""
Блок 2.1 — реализм рыночного трения.

Ограничение данных, определяющее всю методику: T-Invest отдаёт только OHLCV.
Стакана и тиковых сделок нет, поэтому фактический спред НЕ наблюдаем и может
быть только оценён. Чтобы не выдать одну модель за истину, считаются три
независимые величины и предъявляется диапазон:

  1. tick_floor  — жёсткая нижняя граница: один шаг цены (minPriceIncrement
     из InstrumentsService) к цене. Спред физически не бывает уже.

  2. Corwin-Schultz (2012) на ПЯТИМИНУТНЫХ барах. Оценка построена так, чтобы
     аналитически развести волатильность (растёт со временем) и спред (не
     растёт). На дневных барах она сильно смещена вверх гэпами овернайт, на
     5-минутных работает по назначению. Это основная оценка.

  3. Roll (1984) на 5-минутных барах — S = 2*sqrt(-cov(dp_t, dp_{t+1})).
     Независимая проверка. Смещена вверх любой отрицательной автокорреляцией
     цены, то есть даёт верхнюю границу.

Рабочая оценка half-spread = clip(CS-5m, снизу tick_floor). Стресс-уровень —
95-й перцентиль того же CS-5m по дням (не другой оценки: смешивать базу и
стресс из разных моделей нельзя).

Проскальзывание — линейная модель Amihud: ILLIQ = median(|ret| / rub_volume),
impact(Q) = ILLIQ * Q.

ВАЖНО про объём: market_data.volume выражен в ЛОТАХ, а не в штуках. Рублёвый
оборот равен close * volume * lot_size. Без множителя lot оборот занижен в
lot раз (для TGKA — в 100 000 раз). См. audit/instruments.json.

Итог round-trip:

    CostRT(ticker, Q) = 2*fee + 2*half_spread + 2*impact(Q)
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("audit.costs")

# Комиссия за одну сторону, %. ДОПУЩЕНИЕ: тариф счёта из репозитория не
# наблюдаем. 0,04% — тариф «Трейдер» T-Invest, на нём же построена оценка
# 0,08% RT в config.py, поэтому сравнение идёт при равной комиссии.
DEFAULT_FEE_SIDE_PCT = 0.04

# Ключевая ставка ЦБ для стоимости переноса шорта. ДОПУЩЕНИЕ — уточняется
# параметром; в отчёте фигурирует как сценарий, а не как факт.
DEFAULT_KEY_RATE_PCT = 16.0

BLUE_CHIPS = {"SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN",
              "PLZL", "CHMF", "MOEX", "MTSS", "SNGS", "VTBR", "MAGN", "IRAO"}

_INSTR_PATH = os.path.join(os.path.dirname(__file__), "out", "instruments.json")


# ── Справочник инструментов ──────────────────────────────────────────────────

def _quotation(x) -> float | None:
    if not isinstance(x, dict):
        return None
    return int(x.get("units", 0)) + int(x.get("nano", 0)) / 1e9


def load_instruments(path: str = _INSTR_PATH) -> pd.DataFrame:
    """Метаданные инструментов: lot, шаг цены, доступность шорта, ставки риска.

    Источник — InstrumentsService/ShareBy, снимок в audit/out/instruments.json
    (см. audit/fetch_instruments.py).
    """
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["tick"] = df["min_price_increment"].map(_quotation)
    for c in ("dlong", "dshort", "dlong_client", "dshort_client"):
        if c in df.columns:
            df[c] = df[c].map(_quotation)
    df["lot"] = pd.to_numeric(df["lot"], errors="coerce").fillna(1).astype(int)
    return df[["ticker", "name", "lot", "tick", "short_enabled", "buy_available",
               "sell_available", "api_trade", "dlong", "dshort",
               "dlong_client", "dshort_client"]]


def apply_lot_sizes(daily: pd.DataFrame, instr: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает rub_volume с учётом размера лота.

    market_data.volume — в лотах. Оборот в рублях = close * volume * lot.
    """
    lots = instr.set_index("ticker")["lot"].to_dict()
    out = daily.copy()
    out["lot"] = out["ticker"].map(lots).fillna(1).astype(int)
    out["shares"] = out["volume"] * out["lot"]
    out["rub_volume"] = out["close"] * out["shares"]
    return out


# ── Оценки спреда ────────────────────────────────────────────────────────────

_CS_K = 3.0 - 2.0 * np.sqrt(2.0)


def _cs_from_hl(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Corwin-Schultz по последовательным парам баров. Возвращает спред в долях."""
    hl = np.log(hi / lo) ** 2
    beta = hl[:-1] + hl[1:]
    hi2 = np.maximum(hi[:-1], hi[1:])
    lo2 = np.minimum(lo[:-1], lo[1:])
    gamma = np.log(hi2 / lo2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_K - np.sqrt(gamma / _CS_K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(s < 0, np.nan, s)


def cs_spread_5m(bars5m: pd.DataFrame, min_bars: int = 40) -> pd.DataFrame:
    """Corwin-Schultz на 5-минутных барах, агрегируется в оценку на день.

    Отрицательные значения (частый исход при низкой волатильности) отбрасываются
    как «не оценено», а не заменяются нулём — иначе медиана уезжает вниз.
    """
    rows = []
    for (tk, d), g in bars5m.groupby(["ticker", "date"], sort=False):
        g = g.sort_values("ts_msk")
        hi = g["high"].to_numpy(float)
        lo = g["low"].to_numpy(float)
        if len(g) < min_bars:
            continue
        ok = (hi > 0) & (lo > 0)
        hi, lo = hi[ok], lo[ok]
        if len(hi) < min_bars:
            continue
        s = _cs_from_hl(hi, lo)
        v = np.nanmedian(s) if np.isfinite(s).any() else np.nan
        rows.append((tk, d, v * 100.0 if np.isfinite(v) else np.nan, len(hi)))
    return pd.DataFrame(rows, columns=["ticker", "date", "cs5m_pct", "n_bars"])


def roll_spread_5m(bars5m: pd.DataFrame, min_bars: int = 40) -> pd.DataFrame:
    """Roll (1984) на 5-минутных барах — верхняя граница спреда."""
    rows = []
    for (tk, d), g in bars5m.groupby(["ticker", "date"], sort=False):
        g = g.sort_values("ts_msk")
        p = np.log(g["close"].to_numpy(float))
        if len(p) < min_bars:
            continue
        dp = np.diff(p)
        cov = np.cov(dp[:-1], dp[1:], ddof=1)[0, 1]
        s = 2.0 * np.sqrt(-cov) * 100.0 if cov < 0 else np.nan
        rows.append((tk, d, s, len(dp)))
    return pd.DataFrame(rows, columns=["ticker", "date", "roll5m_pct", "n_obs"])


def zero_return_share(bars5m: pd.DataFrame) -> pd.DataFrame:
    """Доля 5-минутных баров без движения цены (high == low) — признак неликвида."""
    g = bars5m.assign(flat=(bars5m["high"] <= bars5m["low"]).astype(float))
    return g.groupby("ticker")["flat"].mean().rename("flat_bar_share").reset_index()


# ── Проскальзывание ──────────────────────────────────────────────────────────

def amihud_illiq(daily: pd.DataFrame, window_days: int = 126) -> pd.DataFrame:
    """ILLIQ = median(|ret| / rub_volume) и медианный дневной оборот."""
    rows = []
    for tk, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date").tail(window_days)
        rv = g["rub_volume"].replace(0, np.nan)
        r = (g["close"] / g["close"].shift(1) - 1.0).abs()
        illiq = (r / rv).replace([np.inf, -np.inf], np.nan).median()
        rows.append((tk, illiq, rv.median(), float(g["close"].iloc[-1])))
    return pd.DataFrame(rows, columns=["ticker", "illiq", "adv_rub", "last_close"])


def impact_pct(illiq: float, order_rub: float) -> float:
    """Ценовое воздействие ордера размера order_rub, в процентах."""
    if illiq is None or not np.isfinite(illiq):
        return np.nan
    return float(illiq * order_rub * 100.0)


# ── Стоимость переноса шорта ─────────────────────────────────────────────────

def short_carry_pct(days: float = 1.0, key_rate_pct: float = DEFAULT_KEY_RATE_PCT,
                    spread_over_key: float = 5.0) -> float:
    """Стоимость переноса короткой позиции через ночь, % от позиции.

    Ставка = ключевая + маржа брокера, начисляется за календарный день.
    Для intraday_short это ноль, если позиция закрыта внутри сессии, и полная
    ставка, если не закрыта.
    """
    annual = key_rate_pct + spread_over_key
    return annual * days / 365.0


# ── Итоговая матрица издержек ────────────────────────────────────────────────

def build_cost_matrix(daily: pd.DataFrame,
                      bars5m: pd.DataFrame,
                      instr: pd.DataFrame,
                      order_rub: float = 10_000.0,
                      fee_side_pct: float = DEFAULT_FEE_SIDE_PCT) -> pd.DataFrame:
    """Динамическая матрица round-trip издержек по тикерам."""
    cs = cs_spread_5m(bars5m)
    roll = roll_spread_5m(bars5m)
    flat = zero_return_share(bars5m)
    ill = amihud_illiq(daily)

    agg = cs.groupby("ticker")["cs5m_pct"].agg(
        cs_med="median",
        cs_p95=lambda s: s.quantile(0.95),
        cs_days="count").reset_index()
    agg = agg.merge(
        roll.groupby("ticker")["roll5m_pct"].median().rename("roll_med"),
        on="ticker", how="outer")
    agg = agg.merge(flat, on="ticker", how="left")
    agg = agg.merge(ill, on="ticker", how="left")
    agg = agg.merge(
        instr[["ticker", "lot", "tick", "short_enabled", "dshort_client"]],
        on="ticker", how="left")

    # Жёсткий физический пол: один шаг цены.
    agg["tick_floor_pct"] = agg["tick"] / agg["last_close"] * 100.0

    # Рабочая оценка: CS-5m, но не уже одного тика.
    agg["spread_pct"] = np.maximum(
        agg["cs_med"].fillna(agg["tick_floor_pct"]), agg["tick_floor_pct"])
    agg["spread_p95_pct"] = np.maximum(
        agg["cs_p95"].fillna(agg["spread_pct"]), agg["spread_pct"])

    agg["impact_pct"] = [impact_pct(i, order_rub) for i in agg["illiq"]]
    agg["impact_pct"] = agg["impact_pct"].fillna(0.0)

    fee_rt = 2.0 * fee_side_pct
    agg["cost_rt_base"] = fee_rt + agg["spread_pct"] + 2 * agg["impact_pct"]
    agg["cost_rt_stress"] = fee_rt + agg["spread_p95_pct"] + 2 * agg["impact_pct"]
    agg["segment"] = np.where(agg["ticker"].isin(BLUE_CHIPS), "blue_chip", "mid_cap")
    agg["order_rub"] = order_rub
    agg["fee_side_pct"] = fee_side_pct
    return agg.sort_values("cost_rt_base").reset_index(drop=True)


def cost_map_env(matrix: pd.DataFrame, column: str = "cost_rt_base") -> str:
    """Готовая строка для VALIDATION_COST_RT_MAP."""
    return " ".join(
        f"{r.ticker}:{getattr(r, column):.2f}"
        for r in matrix.itertuples()
        if np.isfinite(getattr(r, column)))
