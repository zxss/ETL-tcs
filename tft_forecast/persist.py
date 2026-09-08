"""
Сохранение строк дневного и недельного дашбордов в таблицу forecasts.

Пишется ровно то, что показано пользователю: обе функции вызываются из
combined.py в момент формирования дашборда, до отсечения по top_n — в БД
попадает вся посчитанная выборка, а не только видимый топ.

Ключ upsert'а — (asof_date, ticker, strategy), поэтому повторный прогон за тот
же день перезаписывает прогноз, а не плодит дубли. Дневные строки идут со своей
стратегией (long_overnight / intraday_long / intraday_short), недельный
дашборд — со стратегией STRATEGY_WEEKLY.

Ошибки БД наружу не выпускаются: дашборд не должен падать из-за недоступного
PostgreSQL (config.SAVE_FORECASTS=0 отключает запись совсем).
"""
from __future__ import annotations

import datetime as dt
import logging

import config
import database

log = logging.getLogger("forecast_persist")

STRATEGY_WEEKLY = "weekly"

MSK = dt.timezone(dt.timedelta(hours=3))


def enabled() -> bool:
    return bool(getattr(config, "SAVE_FORECASTS", True))


def asof_date(meta: dict | None):
    """Дата расчёта: as_of из котировочного контекста, иначе сегодня по MSK."""
    as_of = (meta or {}).get("as_of")
    if as_of is not None:
        try:
            return as_of.date() if hasattr(as_of, "date") else as_of
        except Exception:  # noqa: BLE001
            pass
    return dt.datetime.now(MSK).date()


def _num(v):
    """None/NaN → None, иначе float (psycopg2 сам приведёт к NUMERIC)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f   # NaN


def _market(meta: dict | None) -> dict:
    return (meta or {}).get("market", {}) or {}


def _daily_record(r: dict, meta: dict | None, forecasts: dict) -> dict:
    """Строка дневного дашборда → запись forecasts."""
    tk = r["ticker"]
    cor = (forecasts or {}).get(tk) or {}
    return {
        "asof_date": asof_date(meta),
        "ticker": tk,
        "strategy": r["strategy"],
        "anchor_price": _num(r.get("anchor_price")),
        "q10": _num(r.get("f_low")),
        "q50": _num(cor.get("ForecastMed")),
        "q90": _num(r.get("f_high")),
        "exp_pnl": _num(r.get("exp_pnl")),
        "prob_profit": _num(r.get("prob_profit")),
        "final_score": _num(r.get("final_score")),
        "verdict": (r.get("verdict") or None),
        "raw_payload": {
            "horizon": "day",
            "direction": r.get("direction"),
            "selected": r.get("selected"),
            "flags": r.get("flags"),
            "range_pct": _num(r.get("range_pct")),
            "coverage": _num(r.get("coverage")),
            "downside": _num(r.get("down")),
            "upside": _num(r.get("up")),
            # контур валидации
            "white_rc": _num(r.get("white_rc")),
            "spa": _num(r.get("spa")),
            "pbo": _num(r.get("pbo")),
            "fdr_pass": r.get("fdr"),
            "ruin30": _num(r.get("ruin30")),
            "lb_struct": r.get("lb"),
            # ликвидность и рыночный контекст
            "liq_score": _num(r.get("liq_score")),
            "max_pos": _num(r.get("max_pos")),
            "regime": r.get("regime"),
            "rs": _num(r.get("rs")),
            "vol_spike": _num(r.get("vol_spike")),
            "atr_pctl": _num(r.get("atr_pctl")),
            "gap_down_prob": _num(r.get("gap_down_prob")),
            # источник цены
            "price_source": (meta or {}).get("source"),
            "price_stale": (meta or {}).get("stale"),
            "market": _market(meta),
        },
    }


def _weekly_record(r: dict, meta: dict | None, horizon: int) -> dict:
    """Строка недельного дашборда → запись forecasts (strategy='weekly')."""
    anchor = _num(r.get("anchor_price"))
    med5 = _num(r.get("med5"))
    q50 = anchor * (1 + med5 / 100.0) if (anchor and med5 is not None) else None
    return {
        "asof_date": asof_date(meta),
        "ticker": r["ticker"],
        "strategy": STRATEGY_WEEKLY,
        "anchor_price": anchor,
        "q10": _num(r.get("w_low")),
        "q50": _num(q50),
        "q90": _num(r.get("w_high")),
        "exp_pnl": _num(r.get("exp")),
        "prob_profit": _num(r.get("prob")),
        "final_score": None,      # у недельного дашборда своя сортировка — по ExpPnL
        "verdict": None,
        "raw_payload": {
            "horizon": "week",
            "horizon_days": horizon,
            "direction": r.get("dir"),
            "median_pct": med5,
            "week_range_pct": _num(r.get("w_range")),
            "week_coverage": _num(r.get("w_cov")),
            "liq_score": _num(r.get("liq_score")),
            "max_pos": _num(r.get("max_pos")),
            "regime": r.get("regime"),
            "rs": _num(r.get("rs")),
            "vol_spike": _num(r.get("vol_spike")),
            "atr_pctl": _num(r.get("atr_pctl")),
            "price_source": (meta or {}).get("source"),
            "price_stale": (meta or {}).get("stale"),
            "market": _market(meta),
        },
    }


def _save(records: list[dict], what: str) -> int:
    if not records:
        return 0
    try:
        n = database.save_forecasts(records)
        log.info("В forecasts сохранено строк (%s): %d", what, n)
        return n
    except Exception as e:  # noqa: BLE001 — печать дашборда важнее записи
        log.warning("Не удалось сохранить %s-прогноз в БД: %s", what, e)
        return 0


def save_daily(rows: list[dict], meta: dict | None, forecasts: dict | None) -> int:
    """Сохраняет строки дневного дашборда (все посчитанные, не только топ-N)."""
    if not enabled() or not rows:
        return 0
    fc = dict(forecasts or {})
    fc.pop("__meta__", None)
    return _save([_daily_record(r, meta, fc) for r in rows], "дневной")


def save_weekly(rows: list[dict], meta: dict | None, horizon: int) -> int:
    """Сохраняет строки недельного дашборда."""
    if not enabled() or not rows:
        return 0
    return _save([_weekly_record(r, meta, horizon) for r in rows], "недельный")
