"""
Инструментовка Этапа 3 ТЗ — таблица execution_audit.

Что она закрывает: сейчас факт исполнения живёт только в CSV
(data/order_log/<дата>.csv) и не содержит ни прогнозной цены, ни расчётного
проскальзывания, поэтому проверить критерий «фактическое проскальзывание не
превышает расчётное более чем на 20%» нечем. Таблица связывает намерение
(прогноз и цена заявки) с фактом (цена исполнения, комиссия, время удержания,
чистый PnL) в одной строке.

Ключ (order_id) — тот же UUID, который place_orders отправляет брокеру как ключ
идемпотентности, поэтому строка однозначно соответствует заявке.

Подключение (делается на старте Этапа 3, не раньше):
  1. database.init_db — добавить cur.execute(CREATE_EXECUTION_AUDIT_SQL);
  2. services/place_orders.py — вызвать record_intent() сразу после отправки
     лимитки и record_fill() в фазе привязки стопов / при закрытии позиции.

До Этапа 3 модуль ничего не делает и ни на что не влияет.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

log = logging.getLogger("audit.execution")

# DDL живёт в models/market_data.py вместе с остальной схемой проекта —
# здесь только импорт, чтобы источник истины был один.
from models.market_data import CREATE_EXECUTION_AUDIT_SQL  # noqa: E402

_INSERT_INTENT_SQL = """
INSERT INTO execution_audit (
    order_id, account_env, asof_date, ticker, strategy, side,
    final_score, exp_pnl_pct, anchor_price, requested_price,
    expected_slippage_pct, expected_cost_pct, stop_price, target_price,
    qty_lots, lot_size, run_id, phase, raw_payload)
VALUES (%(order_id)s, %(account_env)s, %(asof_date)s, %(ticker)s, %(strategy)s,
        %(side)s, %(final_score)s, %(exp_pnl_pct)s, %(anchor_price)s,
        %(requested_price)s, %(expected_slippage_pct)s, %(expected_cost_pct)s,
        %(stop_price)s, %(target_price)s, %(qty_lots)s, %(lot_size)s,
        %(run_id)s, %(phase)s, %(raw_payload)s)
ON CONFLICT (order_id) DO NOTHING;
"""

_UPDATE_FILL_SQL = """
UPDATE execution_audit SET
    filled = TRUE,
    filled_price = %(filled_price)s,
    filled_at    = %(filled_at)s,
    slippage_rub = %(slippage_rub)s,
    slippage_pct = %(slippage_pct)s,
    fee_rub      = %(fee_rub)s,
    updated_at   = NOW()
WHERE order_id = %(order_id)s;
"""

_UPDATE_EXIT_SQL = """
UPDATE execution_audit SET
    exit_price    = %(exit_price)s,
    exit_at       = %(exit_at)s,
    exit_reason   = %(exit_reason)s,
    hold_time_sec = EXTRACT(EPOCH FROM (%(exit_at)s::timestamptz - filled_at))::int,
    pnl_gross_rub = %(pnl_gross_rub)s,
    pnl_net_rub   = %(pnl_net_rub)s,
    updated_at    = NOW()
WHERE order_id = %(order_id)s;
"""


def init(conn) -> None:
    """Создаёт таблицу, если её нет."""
    with conn.cursor() as cur:
        cur.execute(CREATE_EXECUTION_AUDIT_SQL)
    conn.commit()


def record_intent(conn, *, order_id: str, account_env: str, asof_date: dt.date,
                  ticker: str, strategy: str, side: str,
                  requested_price: float, **opt) -> bool:
    """Пишет намерение сразу после отправки заявки брокеру.

    Возвращает True при успехе. Возврат, а не только лог: журнал, который
    молча не пишется, хуже отсутствующего — вызывающий обязан иметь
    возможность отразить сбой в вердикте фазы.
    """
    payload = {
        "order_id": order_id, "account_env": account_env, "asof_date": asof_date,
        "ticker": ticker, "strategy": strategy, "side": side,
        "requested_price": requested_price,
        "final_score": opt.get("final_score"),
        "exp_pnl_pct": opt.get("exp_pnl_pct"),
        "anchor_price": opt.get("anchor_price"),
        "expected_slippage_pct": opt.get("expected_slippage_pct"),
        "expected_cost_pct": opt.get("expected_cost_pct"),
        "stop_price": opt.get("stop_price"),
        "target_price": opt.get("target_price"),
        "qty_lots": opt.get("qty_lots"),
        "lot_size": opt.get("lot_size"),
        "run_id": opt.get("run_id"),
        "phase": opt.get("phase"),
        "raw_payload": json.dumps(opt.get("raw") or {}, ensure_ascii=False),
    }
    try:
        with conn.cursor() as cur:
            cur.execute(_INSERT_INTENT_SQL, payload)
        conn.commit()
        return True
    except Exception as e:  # noqa: BLE001 — журнал не должен ронять торговлю
        conn.rollback()
        log.warning("execution_audit: не удалось записать намерение %s: %s", order_id, e)
        return False


def record_fill(conn, *, order_id: str, filled_price: float,
                filled_at: dt.datetime, requested_price: float,
                side: str, fee_rub: float | None = None,
                qty_shares: float | None = None) -> bool:
    """Фиксирует факт исполнения и фактическое проскальзывание.

    Знак: для покупки проскальзывание положительно, когда залили ДОРОЖЕ
    заявки, для продажи — когда ДЕШЕВЛЕ. Поэтому множитель по стороне сделки,
    иначе шорты давали бы зеркальный знак и среднее по портфелю схлопывалось
    бы к нулю на ровном месте.
    """
    if not requested_price:
        log.warning("execution_audit: нулевая цена заявки %s — проскальзывание "
                    "не считается", order_id)
        return False
    sign = 1.0 if side.upper() == "BUY" else -1.0
    slip_pct = (filled_price / requested_price - 1.0) * 100.0 * sign
    slip_rub = ((filled_price - requested_price) * sign *
                (qty_shares or 0.0))
    try:
        with conn.cursor() as cur:
            cur.execute(_UPDATE_FILL_SQL, {
                "order_id": order_id, "filled_price": filled_price,
                "filled_at": filled_at, "slippage_rub": slip_rub,
                "slippage_pct": slip_pct, "fee_rub": fee_rub})
        conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        log.warning("execution_audit: не удалось записать заливку %s: %s", order_id, e)
        return False


def record_exit(conn, *, order_id: str, exit_price: float,
                exit_at: dt.datetime, exit_reason: str,
                pnl_gross_rub: float, pnl_net_rub: float) -> None:
    """Закрывает строку: цена выхода, причина, время удержания, PnL."""
    try:
        with conn.cursor() as cur:
            cur.execute(_UPDATE_EXIT_SQL, {
                "order_id": order_id, "exit_price": exit_price,
                "exit_at": exit_at, "exit_reason": exit_reason,
                "pnl_gross_rub": pnl_gross_rub, "pnl_net_rub": pnl_net_rub})
        conn.commit()
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        log.warning("execution_audit: не удалось записать выход %s: %s", order_id, e)


# ── Отчёты Этапа 3 ───────────────────────────────────────────────────────────

STAGE3_REPORT_SQL = """
SELECT
    COUNT(*)                                             AS orders,
    COUNT(*) FILTER (WHERE filled)                       AS filled,
    ROUND(AVG(CASE WHEN filled THEN 1 ELSE 0 END), 3)    AS fill_rate,
    ROUND(AVG(slippage_pct)          FILTER (WHERE filled), 4) AS avg_slippage_pct,
    ROUND(AVG(expected_slippage_pct) FILTER (WHERE filled), 4) AS avg_expected_slippage_pct,
    ROUND(AVG(slippage_pct) FILTER (WHERE filled) /
          NULLIF(AVG(expected_slippage_pct) FILTER (WHERE filled), 0), 3)
                                                         AS slippage_ratio,
    ROUND(SUM(pnl_net_rub), 2)                           AS pnl_net_rub,
    ROUND(AVG(hold_time_sec) / 60.0, 1)                  AS avg_hold_min,
    COUNT(*) FILTER (WHERE exit_reason = 'target')       AS exits_target,
    COUNT(*) FILTER (WHERE exit_reason = 'stop')         AS exits_stop
FROM execution_audit
WHERE account_env = %s AND asof_date >= %s;
"""
