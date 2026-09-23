"""
Общая обвязка для тестов.

Работает и под pytest (фикстуры ниже), и под stdlib unittest — pytest в
requirements.txt не входит, поэтому фабрики объявлены обычными функциями и
импортируются тестами напрямую, а фикстуры лишь оборачивают их.
"""
from __future__ import annotations

import os
import sys

# Корень проекта в sys.path — тесты запускаются и из tests/, и из корня.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Фабрики тестовых данных (без сети и БД) ──────────────────────────────────

def make_dashboard_row(**over) -> dict:
    """Строка сводного дашборда — вход build_orders / _score_row.

    Значения по умолчанию нейтральные: коридор симметричен вокруг anchor=100,
    рынок NEUTRAL, объём обычный, метрики валидации посередине.
    """
    row = {
        "ticker": "SBER",
        "strategy": "intraday_long",
        "direction": "LONG",
        "selected": True,
        # прогноз
        "anchor_price": 100.0,
        "f_low": 98.0,
        "f_high": 104.0,
        "down": -2.0,          # Downside q0.1, % (нетто) → стоп
        "up": 3.0,
        "exp_pnl": 0.0,
        "prob_profit": 0.5,
        "range_pct": 6.0,
        "coverage": 0.8,
        # контур валидации
        "verdict": None,
        "white_rc": None,
        "spa": None,
        "pbo": None,
        "fdr": None,
        "ruin30": None,
        "lb": None,
        # ликвидность и рыночный контекст
        "liq_score": 50,
        "max_pos": None,
        "regime": "NEUTRAL",
        "rs": 0.0,
        "vol_spike": 1.0,
        "atr_pctl": 50.0,
        "gap_down_prob": None,
        "last_date": None,
        "price_ts": None,
    }
    row.update(over)
    return row


def make_instrument(lot: int = 10, ticker: str = "TEST", step: float = 0.01):
    """Спецификация инструмента T-Invest для _api_quantity."""
    from services.broker.base import Instrument, Quotation

    return Instrument(
        ticker=ticker,
        instrument_uid=f"uid-{ticker}",
        figi=f"figi-{ticker}",
        lot=lot,
        min_price_increment=Quotation.from_float(step),
        currency="rub",
        trading_status="SECURITY_TRADING_STATUS_NORMAL_TRADING",
        api_trade_available=True,
    )


def make_order(quantity_lots=1, lot_size=1, lot_known=True, **over):
    """Минимальный Order для проверки квантования (цены не важны)."""
    from tft_forecast.combined import Order

    kwargs = dict(
        ticker="TEST", strategy="intraday_long", direction="LONG",
        anchor_price=100.0, f_low=98.0, f_high=104.0, down_pct=-2.0,
        entry_price=99.6, better_pct=0.4, stop_price=97.6, stop_pct=2.0,
        tp_price=103.1, tp_pct=3.5,
        lot_size=lot_size, lot_known=lot_known,
        quantity_lots=quantity_lots, total_rub=None, unavailable=False,
    )
    kwargs.update(over)
    return Order(**kwargs)


def make_geom(stop_pct: float, sizable: bool = True, max_pos=None) -> dict:
    """Элемент geoms для _risk_parity_alloc."""
    return {"stop_pct": stop_pct, "sizable": sizable, "max_pos": max_pos}


# ── Фикстуры pytest (если он установлен) ─────────────────────────────────────

try:
    import pytest
except ImportError:            # pytest не обязателен: тесты идут и под unittest
    pytest = None

if pytest is not None:
    @pytest.fixture
    def dashboard_row():
        return make_dashboard_row

    @pytest.fixture
    def instrument():
        return make_instrument

    @pytest.fixture
    def order():
        return make_order

    @pytest.fixture
    def geom():
        return make_geom
