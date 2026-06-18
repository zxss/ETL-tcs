"""
services/place_orders.py — автозаявки в T-Invest Sandbox по топ-N сигналам
сводного дашборда. ДВУХФАЗНАЯ модель.

Почему две фазы: цена входа — лимитка ВНУТРИ прогнозного коридора (далеко от
спота: SHORT у F.High, LONG у F.Low). Такая заявка почти никогда не заливается
мгновенно — она встаёт в очередь. Ждать её исполнения синхронно в одном запуске
бессмысленно (блокировка на минуты, стоп так и не ставится). Поэтому:

  ФАЗА 1 (по умолчанию):  python3 -m services.place_orders
    Ставит ТОЛЬКО лимитные заявки и сразу выходит. Для каждой записывает в
    «реестр ожидающих стопов» (data/order_log/pending_stops.json): order_id,
    instrument_uid, направление и цену будущего STOP_LOSS.

  ФАЗА 2 (позже, можно по cron):  python3 -m services.place_orders --attach-stops
    Смотрит открытые позиции (GetPositions) и активные стопы. Для каждой
    ЗАЛИВШЕЙСЯ позиции из реестра, у которой ещё нет стопа, выставляет STOP_LOSS.
    Незалитые лимитки остаются в реестре до следующего прохода.

Риск раннего стопа (ТЗ §7.5.3) решён архитектурно: стоп физически не может
появиться раньше факта исполнения входа, т.к. ставится отдельной фазой по
факту наличия позиции.

  --immediate-stop  — старое поведение: лимитка + стоп сразу в одном запуске
                      (для prod с гарантированной заливкой; риск принимается).

CLI:
  python3 -m services.place_orders --dry-run --top-n 3
  python3 -m services.place_orders --top-n 3            # фаза 1
  python3 -m services.place_orders --attach-stops       # фаза 2
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import config
import database
import tft_forecast
from services import run_validation
from services.broker import (
    BrokerClient,
    BrokerError,
    Instrument,
    NotSupportedError,
    Quotation,
    TinkoffProdClient,
    TinkoffSandboxClient,
    new_order_id,
)
from tft_forecast.combined import Order, build_orders, select_top_rows

log = logging.getLogger("place_orders")

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "order_log"


# ── Pipeline (от БД до Order) ─────────────────────────────────────────────────


def compute_orders(top_n: int, position_rub: float, entry_frac: float,
                   *, tp_frac: float = 1.0, quiet: bool = True) -> tuple[list[Order], dict]:
    """От подключения к БД до готового списка Order (+meta для журнала)."""
    conn = database.get_connection()
    try:
        val_rows = run_validation.run(conn, quiet=quiet)
        forecasts = tft_forecast.run(conn, quiet=quiet) or {}
    finally:
        conn.close()

    meta = dict(forecasts.get("__meta__") or {})
    universe = [k for k in forecasts if k != "__meta__"] or config.VALIDATION_TICKERS
    top_rows = select_top_rows(
        val_rows, forecasts, universe, config.VALIDATION_STRATS,
        show_all=getattr(config, "SHOW_ALL_INTRADAY", False), top_n=top_n,
    )
    orders = build_orders(top_rows, position_rub, entry_frac, tp_frac=tp_frac)
    meta.update(forecast_universe=len(universe), top_n_requested=top_n,
                orders_built=len(orders))
    return orders, meta


# ── Сводка / подтверждение ────────────────────────────────────────────────────


def print_summary(account_id: str, env: str, orders: list[Order],
                  position_rub: float) -> None:
    if env == "PROD":
        print("\n" + "!" * 88)
        print(f"!!  ВНИМАНИЕ: БОЕВОЙ КОНТУР — РЕАЛЬНЫЕ ДЕНЬГИ. Счёт №{account_id}")
        print("!" * 88)
    else:
        print(f"\nКонтур: {env} | Счёт №{account_id}")
    print("-" * 88)
    for i, o in enumerate(orders, 1):
        side  = "Покупка (LONG)" if o.direction == "LONG" else "Продажа (SHORT)"
        entry = f"{o.entry_price:.4f}" if o.entry_price else "—"
        stop  = f"{o.stop_price:.4f}"  if o.stop_price  else "—"
        tp    = f"{o.tp_price:.4f}"    if o.tp_price   else "—"
        qty   = o.quantity_lots if o.quantity_lots is not None else "—"
        marks = []
        if o.unavailable:   marks.append("N/A")
        if not o.lot_known: marks.append("LOT?")
        if o.tp_price is None: marks.append("без TP")
        flag = f"  [{'/'.join(marks)}]" if marks else ""
        print(f"{i:>2}. {o.ticker:<6} | {side:<16} | Кол-во: {qty} лот. | "
              f"Вход: {entry} | Стоп: {stop} | Профит: {tp}{flag}")
    print("-" * 88)
    placeable = sum(1 for o in orders if o.is_placeable)
    print(f"К выставлению: {placeable}. Будет пропущено: {len(orders) - placeable}.")
    print(f"Целевой размер позиции: {position_rub:,.0f} ₽ на бумагу.")
    print("Стоп (STOP_LOSS) и профит (TAKE_PROFIT) ставятся фазой --attach-stops "
          "по факту исполнения входа.")


def confirm(prompt: str = "Выставить заявки? (y/n): ") -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


# ── Журнал (аудит) ─────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    "ts", "env", "account_id", "ticker", "direction", "action",
    "order_id", "qty_lots_api", "qty_shares", "price", "status", "info",
]


def _open_log():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOG_DIR / f"{dt.date.today().isoformat()}.csv"
    new = not path.exists()
    fp = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fp, fieldnames=_LOG_FIELDS)
    if new:
        w.writeheader()
    return w, fp


def _logrow(writer: csv.DictWriter, **kw) -> None:
    row = {k: kw.get(k, "") for k in _LOG_FIELDS}
    row["ts"] = dt.datetime.now().isoformat(timespec="seconds")
    writer.writerow(row)


# ── Реестр ожидающих стопов (между фазами) ───────────────────────────────────

_PENDING_PATH = _LOG_DIR / "pending_stops.json"


def _load_pending() -> dict:
    if not _PENDING_PATH.exists():
        return {}
    try:
        return json.loads(_PENDING_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Не прочитать %s (%s) — начинаю с пустого реестра.",
                    _PENDING_PATH, e)
        return {}


def _save_pending(data: dict) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PENDING_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# ── Конвертация дашборд-лотов → API-лотов ────────────────────────────────────


def _api_quantity(o: Order, inst: Instrument) -> tuple[int, int, list[str]]:
    """Инвариант — число АКЦИЙ = lots_dashboard × lot_size_dashboard.
    Делим на инструментный лот API (inst.lot) → quantity для PostOrder.
    Возвращает (api_quantity_lots, shares, warnings)."""
    warnings: list[str] = []
    shares = (o.quantity_lots or 0) * o.lot_size
    if shares <= 0:
        return 0, 0, ["нулевое число акций"]
    if not o.lot_known:
        warnings.append(f"лот по дашборду = {o.lot_size}(?) — не подтверждён")
    if shares % inst.lot != 0:
        warnings.append(f"shares={shares} не делится на API-лот={inst.lot} → округляем вниз")
    return max(1, shares // inst.lot), shares, warnings


# ── ФАЗА 1: только лимитки ────────────────────────────────────────────────────


def _occupied_uids(broker: BrokerClient, account_id: str) -> tuple[set[str], list[str]]:
    """instrument_uid, которые уже «заняты»: активная лимитка / открытая
    позиция / активный стоп. Защита от задвоения при повторном запуске фазы 1.
    Возвращает (uids, notes) — notes для печати, что именно учтено."""
    notes: list[str] = []
    occupied: set[str] = set()
    try:
        ord_uids = broker.get_active_order_instrument_uids(account_id)
        occupied |= ord_uids
        notes.append(f"активных заявок: {len(ord_uids)}")
    except BrokerError as e:
        notes.append(f"заявки не проверены ({e})")
    pos_uids = {p.instrument_uid for p in broker.get_positions(account_id) if p.is_open}
    occupied |= pos_uids
    notes.append(f"открытых позиций: {len(pos_uids)}")
    try:
        stop_uids = broker.get_active_stop_instrument_uids(account_id)
        occupied |= stop_uids
        notes.append(f"активных стопов: {len(stop_uids)}")
    except NotSupportedError:
        notes.append("стопы не проверены (GetStopOrders недоступен)")
    return occupied, notes


def place_limits(broker: BrokerClient, account_id: str, orders: list[Order], *,
                 dry_run: bool, immediate_stop: bool, force: bool,
                 writer: csv.DictWriter, env: str) -> None:
    """Ставит лимитные заявки. Если immediate_stop=False (по умолчанию) —
    записывает будущий стоп в реестр ожидающих. Если True — ставит стоп сразу.

    Защита от задвоения: если по инструменту уже есть активная заявка / позиция /
    стоп — заявка ПРОПУСКАЕТСЯ (если не передан force=True)."""
    pending = _load_pending()
    acc_list = pending.setdefault(account_id, [])

    # снимок занятых инструментов (один раз перед циклом)
    if force:
        occupied: set[str] = set()
        print("[WARN]  --force: защита от задвоения отключена.")
    else:
        occupied, notes = _occupied_uids(broker, account_id)
        print(f"Защита от задвоения: {', '.join(notes)}.")

    for o in orders:
        tk = o.ticker
        if not o.is_placeable:
            print(f"[SKIP]  {tk}: пропуск (unavailable={o.unavailable}, "
                  f"qty={o.quantity_lots}, entry={o.entry_price}).")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="skip", info=f"unavailable={o.unavailable}")
            continue

        # инструмент
        try:
            inst = broker.find_instrument(tk)
        except BrokerError as e:
            print(f"[ERROR] {tk}: инструмент недоступен: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="find_instrument", status="error", info=str(e))
            continue

        # защита от задвоения
        if inst.instrument_uid in occupied:
            print(f"[SKIP]  {tk}: уже есть активная заявка/позиция/стоп — "
                  f"не дублирую (--force для обхода).")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="skip_duplicate", status="skipped",
                    info="active order/position/stop exists")
            continue

        if inst.trading_status != "SECURITY_TRADING_STATUS_NORMAL_TRADING":
            print(f"[WARN]  {tk}: торговый статус {inst.trading_status}.")

        api_q, shares, qwarns = _api_quantity(o, inst)
        if api_q <= 0:
            print(f"[ERROR] {tk}: количество 0.")
            continue
        for w in qwarns:
            print(f"[WARN]  {tk}: {w}")

        entry_q = Quotation.from_float(o.entry_price, inst.min_price_increment)
        stop_q  = Quotation.from_float(o.stop_price,  inst.min_price_increment)
        tp_q    = (Quotation.from_float(o.tp_price, inst.min_price_increment)
                   if o.tp_price else None)
        if abs(entry_q.as_float() - o.entry_price) > 1e-9:
            print(f"[INFO]  {tk}: вход {o.entry_price:.6f} → {entry_q.as_float():.6f} "
                  f"(шаг {inst.min_price_increment.as_float()}).")

        # лимитка
        if dry_run:
            tp_txt = f", TAKE_PROFIT @ {tp_q.as_float()}" if tp_q else " (без TP)"
            print(f"[DRY]   {tk}: LIMIT {o.order_direction} qty={api_q} @ {entry_q.as_float()}"
                  f"  → STOP_LOSS {o.exit_direction} @ {stop_q.as_float()}{tp_txt} "
                  f"({'сразу' if immediate_stop else 'в реестр ожидающих'})")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    direction=o.order_direction, action="limit",
                    qty_lots_api=api_q, qty_shares=shares,
                    price=entry_q.as_float(), status="dry-run")
            continue

        order_id = new_order_id()
        try:
            st = broker.post_limit_order(
                account_id=account_id, instrument=inst,
                direction=o.order_direction, quantity_lots=api_q,
                price=entry_q, order_id=order_id)
        except BrokerError as e:
            print(f"[ERROR] {tk}: лимитка не выставлена: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    direction=o.order_direction, action="limit",
                    qty_lots_api=api_q, price=entry_q.as_float(),
                    status="error", info=str(e))
            continue
        order_id = st.order_id or order_id
        occupied.add(inst.instrument_uid)  # не задвоить тем же тикером в этом же прогоне
        print(f"[OK]    {tk}: лимитка {order_id} → {st.execution_report_status} "
              f"({st.lots_executed}/{st.lots_requested}).")
        _logrow(writer, env=env, account_id=account_id, ticker=tk,
                direction=o.order_direction, action="limit",
                order_id=order_id, qty_lots_api=api_q, qty_shares=shares,
                price=entry_q.as_float(), status=st.execution_report_status)

        rec = {
            "order_id":       order_id,
            "ticker":         tk,
            "instrument_uid": inst.instrument_uid,
            "lot":            inst.lot,
            "api_qty":        api_q,
            "exit_direction": o.exit_direction,
            "stop_units":     stop_q.units,
            "stop_nano":      stop_q.nano,
            "tp_units":       tp_q.units if tp_q else None,
            "tp_nano":        tp_q.nano  if tp_q else None,
            "created":        dt.datetime.now().isoformat(timespec="seconds"),
            "stop_placed":    False,
            "stop_order_id":  None,
            "tp_placed":      tp_q is None,   # нет TP-цены → считаем «нечего ставить»
            "tp_order_id":    None,
            "closed":         False,
        }
        if immediate_stop:
            sid = _place_conditional(broker, account_id, tk, inst, o.exit_direction,
                                     api_q, stop_q, "STOP_LOSS", writer, env)
            rec["stop_placed"], rec["stop_order_id"] = bool(sid), sid
            if tp_q:
                tid = _place_conditional(broker, account_id, tk, inst, o.exit_direction,
                                         api_q, tp_q, "TAKE_PROFIT", writer, env)
                rec["tp_placed"], rec["tp_order_id"] = bool(tid), tid
        acc_list.append(rec)

    if not dry_run:
        _save_pending(pending)
        if not immediate_stop:
            n = sum(1 for r in acc_list if not (r["stop_placed"] and r["tp_placed"]))
            if n:
                print(f"\n→ {n} позиц. ждут SL/TP. Когда лимитки зальются, выполните:")
                print("    python3 -m services.place_orders --attach-stops")


_KIND_RU = {"STOP_LOSS": "стоп", "TAKE_PROFIT": "профит"}


def _place_conditional(broker: BrokerClient, account_id: str, ticker: str,
                       inst: Instrument, direction: str, qty: int, price_q: Quotation,
                       order_type: str, writer: csv.DictWriter, env: str) -> Optional[str]:
    """Ставит одну условную заявку (STOP_LOSS или TAKE_PROFIT).
    Возвращает stop_order_id при успехе, иначе None."""
    label = _KIND_RU.get(order_type, order_type)
    action = order_type.lower()
    try:
        sid = broker.post_stop_order(
            account_id=account_id, instrument=inst, direction=direction,
            quantity_lots=qty, stop_price=price_q, order_id=new_order_id(),
            order_type=order_type)
    except NotSupportedError as e:
        print(f"[ERROR] {ticker}: {label} ({order_type}) не поддержан брокером. ({e})")
        _logrow(writer, env=env, account_id=account_id, ticker=ticker,
                direction=direction, action=action, qty_lots_api=qty,
                price=price_q.as_float(), status="unsupported", info=str(e))
        return None
    except BrokerError as e:
        print(f"[ERROR] {ticker}: {label} не выставлен: {e}")
        _logrow(writer, env=env, account_id=account_id, ticker=ticker,
                direction=direction, action=action, qty_lots_api=qty,
                price=price_q.as_float(), status="error", info=str(e))
        return None
    print(f"[OK]    {ticker}: {label} {sid} @ {price_q.as_float()} ({order_type}).")
    _logrow(writer, env=env, account_id=account_id, ticker=ticker,
            direction=direction, action=action, order_id=sid,
            qty_lots_api=qty, price=price_q.as_float(), status="placed")
    return sid


# ── ФАЗА 2: привязка стопов к залившимся позициям ────────────────────────────


def _exit_dir(r: dict) -> str:
    """Направление выходных заявок из записи (совместимо со старым ключом)."""
    return r.get("exit_direction") or r.get("stop_direction") or "BUY"


def attach_stops(broker: BrokerClient, account_id: str, *,
                 dry_run: bool, writer: csv.DictWriter, env: str) -> None:
    """ФАЗА 2: для залившихся позиций ставит STOP_LOSS и TAKE_PROFIT.

    ВНИМАНИЕ: это НЕ нативный OCO. SL и TP — две независимые условные заявки.
    Когда одна срабатывает и закрывает позицию, вторая остаётся «висеть» и при
    касании своей цены откроет ОБРАТНУЮ позицию. Поэтому здесь же выполняется
    reconcile: если позиция, на которую были выставлены SL/TP, теперь закрыта —
    оставшийся sibling-стоп снимается."""
    pending = _load_pending()
    acc_list = pending.get(account_id, [])
    active = [r for r in acc_list if not r.get("closed")]
    if not active:
        print("Реестр пуст — нечего привязывать.")
        return

    positions = {p.instrument_uid: p.balance_shares
                 for p in broker.get_positions(account_id) if p.is_open}
    try:
        stop_orders = broker.get_active_stop_orders(account_id)
    except NotSupportedError:
        stop_orders = []
        print("[WARN]  GetStopOrders не поддержан — дедуп стопов по реестру.")
    existing_sl = {s.instrument_uid for s in stop_orders if s.kind == "STOP_LOSS"}
    existing_tp = {s.instrument_uid for s in stop_orders if s.kind == "TAKE_PROFIT"}

    print(f"Записей в работе: {len(active)}. Открытых позиций: {len(positions)}.")
    for r in active:
        tk, uid = r["ticker"], r["instrument_uid"]
        bal = positions.get(uid, 0.0)
        bracketed = r.get("stop_placed") or r.get("tp_placed")

        # ── reconcile: позиция закрыта, но брекет был выставлен → снять остаток
        if abs(bal) < 1e-9:
            if bracketed:
                _reconcile_closed(broker, account_id, r, stop_orders, dry_run, writer, env)
            else:
                print(f"[WAIT]  {tk}: лимитка {r['order_id']} ещё не залилась — в очереди.")
            continue

        lot = int(r.get("lot", 1)) or 1
        qty = max(1, int(abs(bal) // lot))
        if qty != r.get("api_qty"):
            print(f"[INFO]  {tk}: позиция {abs(bal):.0f} шт → {qty} лот "
                  f"(заявлено {r.get('api_qty')}).")
        edir = _exit_dir(r)

        inst = None
        if not dry_run:
            try:
                inst = broker.find_instrument(tk)
            except BrokerError as e:
                print(f"[ERROR] {tk}: инструмент недоступен: {e}")
                continue

        # ── STOP_LOSS
        if not r.get("stop_placed"):
            stop_q = Quotation(units=int(r["stop_units"]), nano=int(r["stop_nano"]))
            if uid in existing_sl:
                print(f"[SKIP]  {tk}: STOP_LOSS уже есть — помечаю выполненным.")
                r["stop_placed"] = True
            elif dry_run:
                print(f"[DRY]   {tk}: STOP_LOSS {edir} qty={qty} @ {stop_q.as_float()}")
            else:
                sid = _place_conditional(broker, account_id, tk, inst, edir,
                                         qty, stop_q, "STOP_LOSS", writer, env)
                if sid:
                    r["stop_placed"], r["stop_order_id"] = True, sid

        # ── TAKE_PROFIT
        if not r.get("tp_placed") and r.get("tp_units") is not None:
            tp_q = Quotation(units=int(r["tp_units"]), nano=int(r["tp_nano"]))
            if uid in existing_tp:
                print(f"[SKIP]  {tk}: TAKE_PROFIT уже есть — помечаю выполненным.")
                r["tp_placed"] = True
            elif dry_run:
                print(f"[DRY]   {tk}: TAKE_PROFIT {edir} qty={qty} @ {tp_q.as_float()}")
            else:
                tid = _place_conditional(broker, account_id, tk, inst, edir,
                                         qty, tp_q, "TAKE_PROFIT", writer, env)
                if tid:
                    r["tp_placed"], r["tp_order_id"] = True, tid

    if not dry_run:
        _save_pending(pending)


def _reconcile_closed(broker: BrokerClient, account_id: str, r: dict,
                      stop_orders: list, dry_run: bool,
                      writer: csv.DictWriter, env: str) -> None:
    """Позиция закрыта (сработал SL или TP). Снимаем оставшийся sibling-стоп,
    чтобы он не открыл обратную позицию, и помечаем запись закрытой."""
    tk, uid = r["ticker"], r["instrument_uid"]
    leftovers = [s for s in stop_orders if s.instrument_uid == uid]
    if dry_run:
        print(f"[DRY]   {tk}: позиция закрыта — снять {len(leftovers)} оставшихся стоп(ов).")
        return
    for s in leftovers:
        try:
            broker.cancel_stop_order(account_id=account_id, stop_order_id=s.stop_order_id)
            print(f"[OCO]   {tk}: снят оставшийся {s.kind} {s.stop_order_id[:8]}… "
                  f"(позиция уже закрыта).")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="oco_cancel", order_id=s.stop_order_id,
                    status="cancelled", info=s.kind)
        except BrokerError as e:
            print(f"[WARN]  {tk}: не снять {s.kind} {s.stop_order_id[:8]}…: {e}")
    r["closed"] = True
    print(f"[DONE]  {tk}: сделка закрыта, запись архивирована.")


# ── Подготовка брокера/счёта ──────────────────────────────────────────────────


def _make_broker_and_account(sandbox: bool) -> tuple[BrokerClient, str, str]:
    """Контур по умолчанию — PROD (боевой счёт PROD_ACCOUNT_ID). sandbox=True —
    тестовый контур (создаёт/пополняет виртуальный счёт)."""
    if sandbox:
        broker: BrokerClient = TinkoffSandboxClient()
        pref = config.SANDBOX_ACCOUNT_ID
        account_id = broker.open_or_get_account(preferred_id=pref)
        if not pref:  # свежесозданный sandbox-счёт — пополнить
            try:
                broker.pay_in(account_id, config.SANDBOX_PAYIN_RUB,
                              currency=config.SANDBOX_PAYIN_CURRENCY)
            except BrokerError as e:
                print(f"[WARN] Не удалось пополнить sandbox-счёт: {e}")
        return broker, account_id, "SANDBOX"

    # PROD — верифицируем боевой счёт (FULL access), без создания/пополнения
    broker = TinkoffProdClient()
    account_id = broker.open_or_get_account(preferred_id=config.PROD_ACCOUNT_ID)
    return broker, account_id, "PROD"


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    p = argparse.ArgumentParser(
        description="Автозаявки в T-Invest (двухфазно). КОНТУР ПО УМОЛЧАНИЮ — PROD "
                    "(реальные деньги); --sandbox для тестового контура.")
    p.add_argument("--sandbox", action="store_true",
                   help="Тестовый контур (виртуальные деньги). По умолчанию — PROD.")
    p.add_argument("--attach-stops", action="store_true",
                   help="ФАЗА 2: привязать STOP_LOSS к залившимся позициям из реестра.")
    p.add_argument("--top-n", type=int,
                   default=int(getattr(config, "BEST_TRADES_TOP_N", 10)))
    p.add_argument("--position", type=float,
                   default=float(getattr(config, "BEST_TRADES_POSITION_RUB", 10_000.0)))
    p.add_argument("--entry-frac", type=float,
                   default=float(getattr(config, "LIMIT_ENTRY_FRACTION", 0.8)))
    p.add_argument("--tp-frac", type=float,
                   default=float(getattr(config, "LIMIT_TP_FRACTION", 1.0)),
                   help="Цель take-profit как доля пути до дальней границы коридора (0..1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Пройти pipeline без реальной отправки в брокер.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Не спрашивать y/n (для cron).")
    p.add_argument("--immediate-stop", action="store_true",
                   help="ФАЗА 1: ставить стоп сразу за лимиткой (риск раннего стопа).")
    p.add_argument("--force", action="store_true",
                   help="ФАЗА 1: отключить защиту от задвоения (ставить лимитку, "
                        "даже если по инструменту уже есть заявка/позиция/стоп).")
    args = p.parse_args(argv)

    try:
        broker, account_id, env = _make_broker_and_account(args.sandbox)
    except BrokerError as e:
        print(f"[ERROR] Счёт/контур: {e}", file=sys.stderr)
        return 1

    writer, fp = _open_log()
    try:
        # ── ФАЗА 2 ──
        if args.attach_stops:
            print(f"Контур: {env} | Счёт №{account_id} | ФАЗА 2: привязка стопов")
            attach_stops(broker, account_id, dry_run=args.dry_run,
                         writer=writer, env=env)
            return 0

        # ── ФАЗА 1 ──
        log.info("Запуск pipeline (валидация + forecast)...")
        orders, _meta = compute_orders(args.top_n, args.position, args.entry_frac,
                                       tp_frac=args.tp_frac)
        if not orders:
            print("Нет торговых сигналов.")
            return 0

        print_summary(account_id, env, orders, args.position)
        prompt = ("ВЫСТАВИТЬ РЕАЛЬНЫЕ ЗАЯВКИ на боевом счёте? (y/n): "
                  if env == "PROD" else "Выставить заявки? (y/n): ")
        if args.dry_run:
            print("[DRY-RUN] реальные ордера НЕ отправляются.")
        elif not args.no_confirm and not confirm(prompt):
            print("[INFO] Выставление заявок отменено пользователем.")
            return 0

        place_limits(broker, account_id, orders,
                     dry_run=args.dry_run, immediate_stop=args.immediate_stop,
                     force=args.force, writer=writer, env=env)
    finally:
        fp.close()
    print("\nГотово. Журнал: data/order_log/<дата>.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
