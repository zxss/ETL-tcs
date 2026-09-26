"""
Поштучные издержки круга для боевого контура (аудит 26.09.2026, F12).

До этого рейтинг и контур валидации брали плоские 0,128 % для всех бумаг
(TFT_COST_RT = VALIDATION_COST_RT). Это 0,08 % комиссии «Премиум» плюс МЕДИАННЫЙ по
вселенной спред: ликвидным бумагам издержки завышались, неликвидным — занижались,
и отбор тяготел к MVID, SMLT, AKRN, CBOM, LENT, ETLN.

Источник — матрица audit/out/cost_matrix.csv (audit/costs.py: спред по
Corwin–Schultz с полом в шаг цены, воздействие по Амихуду при заявке 10 000 ₽).
Тот же расчёт, что research/cost_model.py сценарий «base»:
    круг = 2 × 0,04 % комиссии + спред + 2 × воздействие.
Воздействие по Амихуду линейно по объёму, поэтому пересчитывается на реальный
размер позиции (BEST_TRADES_POSITION_RUB), а не берётся при 10 000 ₽.

Бумага вне матрицы или нет файла → плоский TFT_COST_RT (прежнее поведение).
COST_MODEL=flat выключает поштучный расчёт целиком.
"""
from __future__ import annotations

import csv
import logging
import os

import config

log = logging.getLogger("cost_model")

FEE_SIDE_PCT = 0.04                     # тариф «Премиум», % за сделку
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = os.path.join(ROOT, "audit", "out", "cost_matrix.csv")

_cache: dict[str, tuple[float, float, float]] | None = None


def _load() -> dict[str, tuple[float, float, float]]:
    """ticker → (спред %, воздействие % при order_rub, order_rub матрицы)."""
    global _cache
    if _cache is not None:
        return _cache
    out: dict[str, tuple[float, float, float]] = {}
    try:
        with open(MATRIX_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    out[r["ticker"].upper()] = (float(r["spread_pct"]),
                                                float(r["impact_pct"]),
                                                float(r.get("order_rub") or 10000.0))
                except (KeyError, ValueError):
                    continue
    except OSError as e:
        log.warning("Матрица издержек недоступна (%s) — плоские %.3f %%",
                    e, flat_rt())
    _cache = out
    return out


def flat_rt() -> float:
    return float(getattr(config, "TFT_COST_RT", 0.128))


def enabled() -> bool:
    return str(getattr(config, "COST_MODEL", "matrix")).lower() == "matrix"


def round_trip_pct(ticker: str, position_rub: float | None = None) -> float:
    """Круг вход + выход, % от позиции."""
    if not enabled():
        return flat_rt()
    row = _load().get(str(ticker).upper())
    if row is None:
        return flat_rt()
    spread, impact, order_rub = row
    pos = float(position_rub if position_rub is not None
                else getattr(config, "BEST_TRADES_POSITION_RUB", order_rub))
    impact_at_pos = impact * (pos / order_rub) if order_rub > 0 else impact
    return 2.0 * FEE_SIDE_PCT + spread + 2.0 * impact_at_pos


def cost_map(tickers) -> dict[str, float]:
    """{TICKER: круг %} — для контура валидации (VALIDATION_COST_RT_MAP)."""
    return {str(t).upper(): round_trip_pct(t) for t in tickers}
