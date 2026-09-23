"""
Замыкание строк execution_audit по факту брокера: заливка входа и выход.

Зачем отдельный модуль. Намерение пишется в момент выставления лимитки, а
заливка и выход случаются позже и в другой фазе: ночная лимитка заливается в
вечернюю сессию, стоп срабатывает утром до CLOSE. Раньше строку замыкал только
CLOSE и только при живой позиции, поэтому 5 из 7 ночных заливок r3/r4
(15–16.09) остались в журнале с filled=false, а выходы по стопу — без причины.

Источники факта:
  * вход — GetOrderState по id заявки брокера (цена, лоты, комиссия);
  * время заливки и выход по стопу — операции по счёту (сделки и комиссии).

Причины выхода совпадают с отчётом Этапа 3 (audit/execution_audit.py):
  stop, target — сработал SL / TP; stop_breach — аварийный выход по рынку,
  цена уже за стопом; square_off — CLEANUP; overnight_close — CLOSE;
  closed_externally — позиция закрыта без известной нам ноги.
"""
from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("exec_journal")

MSK = dt.timezone(dt.timedelta(hours=3))


def utc_iso(ts: str | dt.datetime | None) -> str:
    """ISO UTC c «Z» для запросов к API. Наивное время реестра — МСК (сервер)."""
    if ts is None:
        t = dt.datetime.now(dt.timezone.utc)
    elif isinstance(ts, dt.datetime):
        t = ts
    else:
        t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=MSK)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(raw) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00")) if raw else None
    except ValueError:
        return None


def _row(conn, order_id: str):
    with conn.cursor() as cur:
        cur.execute("SELECT requested_price, filled, filled_price, filled_at, fee_rub "
                    "FROM execution_audit WHERE order_id = %s;", (order_id,))
        return cur.fetchone()


def _is_long(rec: dict) -> bool:
    return (rec.get("exit_direction") or "SELL") == "SELL"


def journal_fill(broker, account_id: str, conn, rec: dict) -> bool:
    """Записать заливку входа. True — строка в журнале заполнена (сейчас или раньше)."""
    from audit import execution_audit as ea
    oid = rec.get("order_id")
    if conn is None or not oid:
        return False
    try:
        row = _row(conn, oid)
        if not row:
            return False
        if row[1]:
            return True
        st = broker.get_order_state(account_id=account_id, order_id=oid)
        lots = int(st.lots_executed or 0)
        if lots <= 0 or not st.executed_price:
            return False
        shares = lots * int(rec.get("lot") or 1)
        entry_side = "BUY" if _is_long(rec) else "SELL"
        at = None
        try:
            trades, _ = broker.get_trades(account_id, utc_iso(rec.get("created")),
                                          rec.get("instrument_uid"))
            first = next((t for t in trades if t.side == entry_side), None)
            at = parse_ts(first.at) if first else None
        except Exception as e:                   # noqa: BLE001 — время не критично
            log.info("%s: операции недоступны, время заливки — текущее (%s)",
                     rec.get("ticker"), e)
        return ea.record_fill(conn, order_id=oid, filled_price=float(st.executed_price),
                              filled_at=at or dt.datetime.now(dt.timezone.utc),
                              requested_price=float(row[0] or 0.0), side=entry_side,
                              fee_rub=st.executed_commission, qty_shares=shares)
    except Exception as e:                       # noqa: BLE001 — журнал не роняет торговлю
        _rollback(conn)
        log.warning("%s: заливка не записана в журнал: %s", rec.get("ticker"), e)
        return False


def journal_exit(broker, account_id: str, conn, rec: dict, reason: str, *,
                 exit_price: float | None = None, exit_at: dt.datetime | None = None,
                 exit_fee: float | None = None) -> bool:
    """Записать выход. Без exit_price цена и комиссия берутся из операций
    (выход по SL/TP: у условной заявки своей цены исполнения нет)."""
    from audit import execution_audit as ea
    oid = rec.get("order_id")
    if conn is None or not oid:
        return False
    if not journal_fill(broker, account_id, conn, rec):
        log.warning("%s: вход не записан — выход в журнал не пишется", rec.get("ticker"))
        return False
    try:
        row = _row(conn, oid)
        entry_px, filled_at, entry_fee = float(row[2]), row[3], float(row[4] or 0.0)
        st = broker.get_order_state(account_id=account_id, order_id=oid)
        shares = int(st.lots_executed or 0) * int(rec.get("lot") or 1)
        exit_side = "SELL" if _is_long(rec) else "BUY"
        if exit_price is None:
            since = utc_iso(filled_at or rec.get("created"))
            trades, _ = broker.get_trades(account_id, since, rec.get("instrument_uid"))
            out = [t for t in trades if t.side == exit_side]
            if not out:
                log.warning("%s: сделок выхода в операциях нет — выход не записан",
                            rec.get("ticker"))
                return False
            qty = sum(t.quantity for t in out)
            exit_price = sum(t.price * t.quantity for t in out) / qty
            exit_at = exit_at or parse_ts(out[-1].at)
            _, exit_fee = broker.get_trades(account_id, out[0].at, rec.get("instrument_uid"))
        sign = 1.0 if _is_long(rec) else -1.0
        gross = (float(exit_price) - entry_px) * shares * sign
        net = gross - entry_fee - float(exit_fee or 0.0)
        return ea.record_exit(conn, order_id=oid, exit_price=float(exit_price),
                              exit_at=exit_at or dt.datetime.now(dt.timezone.utc),
                              exit_reason=reason, pnl_gross_rub=round(gross, 4),
                              pnl_net_rub=round(net, 4))
    except Exception as e:                       # noqa: BLE001
        _rollback(conn)
        log.warning("%s: выход не записан в журнал: %s", rec.get("ticker"), e)
        return False


def _rollback(conn) -> None:
    try:
        conn.rollback()
    except Exception:                            # noqa: BLE001
        pass
