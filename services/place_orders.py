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

  ФАЗА 3 (перед концом сессии):  python3 -m services.place_orders --square-off
    Закрывает по рынку ВНУТРИДНЕВНЫЕ позиции (intraday_long / intraday_short),
    предварительно сняв их SL и TP. Позиции long_overnight не трогает.
    Без этой фазы 79,1% внутридневных позиций доживают до закрытия сессии и
    переносятся через ночь: торгуется не та стратегия, которую валидировали,
    плюс 0,0575% за ночь на перенос шорта. Цена дефекта на реплее — 14,6 п.п.
    итоговой доходности. Cron: 35 18 * * 1-5 (см. INTRADAY_SQUARE_OFF_TIME).

Риск раннего стопа (ТЗ §7.5.3) решён архитектурно: стоп физически не может
появиться раньше факта исполнения входа, т.к. ставится отдельной фазой по
факту наличия позиции.

  --immediate-stop  — старое поведение: лимитка + стоп сразу в одном запуске
                      (для prod с гарантированной заливкой; риск принимается).

СИНХРОНИЗАЦИЯ (по умолчанию, ТЗ): обычный прогон приводит портфель к актуальным
сигналам ОДНОЙ командой:
  1) закрывает по рынку позиции, по которым сигнал исчез/сменил направление;
  2) снимает неактуальные лимитки и осиротевшие стопы;
  3) выставляет заявки по новым сигналам;
  4) ЖДЁТ исполнения лимиток (--wait-fill, по умолч. 30s);
  5) привязывает SL/TP к залившимся позициям (+OCO-reconcile);
  6) проверяет, что каждая открытая позиция защищена SL и TP.
Перед любыми изменениями печатается ПЛАН и спрашивается y/n.
--place-only возвращает старое поведение (только выставить заявки, без закрытий).

СВЕЖЕСТЬ ДАННЫХ: перед расчётом свечи догружаются (load_history + backfill из 5M),
затем проверяется, что дневные догнали последнюю завершённую сессию. Если нет —
StaleDataError, заявки НЕ ставятся (защита от торговли по устаревшему прогнозу).
--skip-refresh пропускает догрузку (проверка свежести остаётся).

КОНТУР: по умолчанию SANDBOX (виртуальные деньги). Боевой счёт — только с явным
--prod. Связка --prod --no-confirm (cron) требует ALLOW_UNATTENDED_PROD=1.

CLI:
  python3 -m services.place_orders --top-n 10 --dry-run  # план синхронизации (sandbox)
  python3 -m services.place_orders --top-n 10            # синхронизация портфеля
  python3 -m services.place_orders --top-n 10 --wait-fill 60   # ждать заливки до 60s
  python3 -m services.place_orders --top-n 10 --place-only     # только выставить заявки
  python3 -m services.place_orders --attach-stops        # только привязать SL/TP
  python3 -m services.place_orders --square-off          # закрыть внутридневные
  python3 -m services.place_orders --prod --top-n 10     # БОЕВОЙ контур
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import config
import database
import tft_forecast
from services import run_validation
from services.data_freshness import StaleDataError, ensure_fresh_data
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
from tft_forecast.combined import (
    Order, build_orders, select_top_rows, warn_unvalidated,
)

log = logging.getLogger("place_orders")

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "order_log"


# ── Pipeline (от БД до Order) ─────────────────────────────────────────────────


def compute_orders(top_n: int, position_rub: float, entry_frac: float,
                   *, tp_frac: float | None = None, quiet: bool = True,
                   refresh: bool = True,
                   budget_rub: float | None = None) -> tuple[list[Order], dict]:
    """От подключения к БД до готового списка Order (+meta для журнала).
    Перед расчётом догружает свечи и проверяет их свежесть (StaleDataError).

    tp_frac=None → берётся config.LIMIT_TP_FRACTION. Раньше здесь стояла
    захардкоженная 1.0, из-за чего программный вызов (в обход CLI) целился в
    дальнюю границу коридора независимо от конфигурации: цель достигалась
    в 1,9% сделок. Источник значения по умолчанию должен быть один.
    """
    if tp_frac is None:
        tp_frac = float(getattr(config, "LIMIT_TP_FRACTION", 0.5))
    conn = database.get_connection()
    try:
        last_date = ensure_fresh_data(conn, refresh=refresh)
        val_rows = run_validation.run(conn, quiet=quiet)
        forecasts = tft_forecast.run(conn, quiet=quiet) or {}
    finally:
        conn.close()

    meta = dict(forecasts.get("__meta__") or {})
    universe = [k for k in forecasts if k != "__meta__"] or config.VALIDATION_TICKERS
    # Валидация продолжает считать ВСЕ стратегии (config.VALIDATION_STRATS),
    # а торгуются только разрешённые: select_top_rows применяет специализацию
    # Пути А — см. combined.apply_strategy_specialisation.
    top_rows = select_top_rows(
        val_rows, forecasts, universe, config.VALIDATION_STRATS,
        show_all=getattr(config, "SHOW_ALL_INTRADAY", False), top_n=top_n,
    )
    orders = build_orders(top_rows, position_rub, entry_frac, tp_frac=tp_frac,
                          budget_rub=budget_rub)
    rejected = sum(1 for r in top_rows if r.get("verdict") == "REJECTED")
    meta.update(forecast_universe=len(universe), top_n_requested=top_n,
                orders_built=len(orders), data_last_date=str(last_date),
                budget_rub=budget_rub, top_rows=top_rows,
                rejected_candidates=rejected)
    return orders, meta


# ── Сводка / подтверждение ────────────────────────────────────────────────────


def print_summary(account_id: str, env: str, orders: list[Order],
                  position_rub: float, budget_rub: float | None = None) -> None:
    if env == "PROD":
        print("\n" + "!" * 88)
        print(f"!!  ВНИМАНИЕ: БОЕВОЙ КОНТУР — РЕАЛЬНЫЕ ДЕНЬГИ. Счёт №{account_id}")
        print("!" * 88)
    else:
        print(f"\nКонтур: {env} | Счёт №{account_id}")
    print("-" * 100)
    use_budget = bool(budget_rub and budget_rub > 0)
    for i, o in enumerate(orders, 1):
        side  = "Покупка (LONG)" if o.direction == "LONG" else "Продажа (SHORT)"
        entry = f"{o.entry_price:.4f}" if o.entry_price else "—"
        stop  = f"{o.stop_price:.4f}"  if o.stop_price  else "—"
        tp    = f"{o.tp_price:.4f}"    if o.tp_price   else "—"
        qty   = o.quantity_lots if o.quantity_lots is not None else "—"
        # рублёвый риск позиции = сумма × стоп% (для контроля риск-паритета)
        risk = (o.total_rub * o.stop_pct / 100.0
                if (o.total_rub and o.stop_pct) else None)
        sum_s  = f"{o.total_rub:,.0f}₽" if o.total_rub else "—"
        risk_s = f"риск {risk:,.0f}₽" if risk is not None else "риск —"
        marks = []
        if o.unavailable:   marks.append("N/A")
        if not o.lot_known: marks.append("LOT?")
        if o.tp_price is None: marks.append("без TP")
        flag = f"  [{'/'.join(marks)}]" if marks else ""
        print(f"{i:>2}. {o.ticker:<6} | {side:<16} | {qty} лот | сумма {sum_s:>11} | "
              f"{risk_s:>13} | Вход {entry} | Стоп {stop} | Профит {tp}{flag}")
    print("-" * 100)
    placeable = sum(1 for o in orders if o.is_placeable)
    print(f"К выставлению: {placeable}. Будет пропущено: {len(orders) - placeable}.")
    if use_budget:
        spent = sum(o.total_rub for o in orders if o.total_rub)
        risks = [o.total_rub * o.stop_pct / 100.0
                 for o in orders if o.total_rub and o.stop_pct]
        print(f"Бюджет: {budget_rub:,.0f} ₽ | задействовано: {spent:,.0f} ₽ "
              f"({spent / budget_rub * 100:.0f}%) | остаток: {budget_rub - spent:,.0f} ₽.")
        if risks:
            print(f"Режим РИСК-ПАРИТЕТ: риск на позицию {min(risks):,.0f}–{max(risks):,.0f} ₽ "
                  f"(равный вклад в просадку; вес ∝ 1/стоп%).")
    else:
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
_LOCK_PATH = _LOG_DIR / "pending_stops.lock"


class RegistryError(RuntimeError):
    """Реестр стопов нечитаем. Фаза обязана упасть, а не работать с пустым:
    пустой реестр значит «стопов не ставить, интрадей не закрывать»."""


class RegistryBusy(RuntimeError):
    """Реестр занят другим процессом дольше таймаута."""


@contextlib.contextmanager
def registry_lock(*, timeout_s: float | None = None, wait: bool = True):
    """Эксклюзивная блокировка реестра на всё время фазы (fcntl.flock).

    Фазы, монитор protect и ручной CLI читают реестр, ходят к брокеру и пишут
    реестр обратно. Без блокировки параллельный protect мог бы, например,
    заново поставить стоп на позицию, которую CLEANUP в этот момент закрывает.
    wait=False — не ждать (protect просто пропускает прогон).

    Внутри процесса брать один раз: flock на новом дескрипторе того же файла
    блокирует и собственный процесс.
    """
    import fcntl
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if timeout_s is None:
        timeout_s = float(getattr(config, "REGISTRY_LOCK_TIMEOUT_SEC", 600))
    # Только чтение: flock записи не требует, а lock-файл, созданный другим
    # пользователем (например, тестами хука выкладки), не должен ронять фазу.
    fd = os.open(_LOCK_PATH, os.O_RDONLY | os.O_CREAT, 0o666)
    fp = os.fdopen(fd, "r")
    try:
        deadline = time.monotonic() + (timeout_s if wait else 0.0)
        while True:
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RegistryBusy(f"реестр {_PENDING_PATH.name} занят другим процессом")
                time.sleep(0.5)
        yield
    finally:
        fp.close()                               # закрытие дескриптора снимает flock


def _load_pending() -> dict:
    if not _PENDING_PATH.exists():
        return {}
    try:
        return json.loads(_PENDING_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise RegistryError(f"реестр {_PENDING_PATH} нечитаем ({e}) — фаза остановлена; "
                            f"восстановить из резервной копии {_PENDING_PATH.name}.bak") from e


def _save_pending(data: dict) -> None:
    """Атомарная запись: временный файл в том же каталоге → fsync → os.replace.
    Обрыв процесса посреди записи оставляет прежний реестр целым. Прежняя
    версия сохраняется рядом как .bak."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _PENDING_PATH.with_name(f".{_PENDING_PATH.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        if _PENDING_PATH.exists():
            # копия, а не перенос: между двумя переносами реестра не было бы
            # вовсе, и обрыв в этот момент выглядел бы как пустой реестр
            try:
                shutil.copyfile(_PENDING_PATH, _PENDING_PATH.with_name(_PENDING_PATH.name + ".bak"))
            except OSError:
                pass
        os.replace(tmp, _PENDING_PATH)
    finally:
        if tmp.exists():
            tmp.unlink()


# ── Гард маржинального шорта ─────────────────────────────────────────────────


def _short_blocked(o: Order, inst: Instrument) -> bool:
    """True → вход SHORT по бумаге, у которой брокер запретил маржинальный шорт.

    Источник истины — живой shortEnabledFlag из ShareBy, а не список в
    combined.py: список отсекает сигнал раньше, а этот гард ловит случаи, когда
    брокер снял бумагу с маржиналки, а список ещё не обновили. Проверяется
    только ВХОД: выход из лонга — тоже SELL, но он маржи не требует.
    """
    return o.direction == "SHORT" and not inst.short_enabled


# ── Конвертация дашборд-лотов → API-лотов ────────────────────────────────────


def _api_quantity(o: Order, inst: Instrument) -> tuple[int, int, list[str]]:
    """Инвариант — число АКЦИЙ = lots_dashboard × lot_size_dashboard.
    Делим на инструментный лот API (inst.lot) → quantity для PostOrder.
    Возвращает (api_quantity_lots, shares, warnings).

    Округление ТОЛЬКО ВНИЗ: если посчитанных акций не хватает даже на один
    API-лот, возвращаем 0 и заявка не выставляется (вызывающий код проверяет
    api_q <= 0). Прежний max(1, ...) в этом случае поднимал размер до целого
    лота — то есть покупал БОЛЬШЕ, чем посчитала модель (при shares=1 и
    inst.lot=10 — в 10 раз), обходя защиту вызывающего и ломая риск-сайзинг."""
    warnings: list[str] = []
    shares = (o.quantity_lots or 0) * o.lot_size
    if shares <= 0:
        return 0, 0, ["нулевое число акций"]
    if not o.lot_known:
        warnings.append(f"лот по дашборду = {o.lot_size}(?) — не подтверждён")
    if shares < inst.lot:
        warnings.append(f"shares={shares} меньше API-лота={inst.lot} → "
                        f"размер 0, заявка не выставляется")
        return 0, shares, warnings
    if shares % inst.lot != 0:
        warnings.append(f"shares={shares} не делится на API-лот={inst.lot} → округляем вниз")
    return shares // inst.lot, shares, warnings


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
                 writer: csv.DictWriter, env: str,
                 report: list | None = None) -> list[dict]:
    """Ставит лимитные заявки. Если immediate_stop=False (по умолчанию) —
    записывает будущий стоп в реестр ожидающих. Если True — ставит стоп сразу.

    Защита от задвоения: если по инструменту уже есть активная заявка / позиция /
    стоп — заявка ПРОПУСКАЕТСЯ (если не передан force=True).

    Возвращает список записей реестра, выставленных В ЭТОМ прогоне (для ожидания
    исполнения и последующей привязки стопов).

    report — необязательный список, куда дописывается (тикер, исход) по каждой
    заявке: placed / skip_* / error_*. Нужен вызывающему, чтобы отличить отказ
    брокера (частичный провал фазы) от намеренного пропуска дубля."""
    pending = _load_pending()
    acc_list = pending.setdefault(account_id, [])
    placed: list[dict] = []  # записи, выставленные именно в этом прогоне

    def _rep(tk: str, status: str) -> None:
        if report is not None:
            report.append((tk, status))

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
            _rep(tk, "skip_unplaceable")
            continue

        # инструмент
        try:
            inst = broker.find_instrument(tk)
        except BrokerError as e:
            print(f"[ERROR] {tk}: инструмент недоступен: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="find_instrument", status="error", info=str(e))
            _rep(tk, "error_instrument")
            continue

        # защита от задвоения
        if inst.instrument_uid in occupied:
            print(f"[SKIP]  {tk}: уже есть активная заявка/позиция/стоп — "
                  f"не дублирую (--force для обхода).")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="skip_duplicate", status="skipped",
                    info="active order/position/stop exists")
            _rep(tk, "skip_duplicate")
            continue

        if _short_blocked(o, inst):
            print(f"[SKIP SHORT] {tk}: шорт недоступен у брокера "
                  f"(shortEnabledFlag=false) — заявка не выставляется.")
            log.info("[SKIP SHORT] %s: шорт недоступен у брокера", tk)
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="skip_short", status="skipped",
                    info="shortEnabledFlag=false")
            _rep(tk, "skip_short")
            continue

        if inst.trading_status != "SECURITY_TRADING_STATUS_NORMAL_TRADING":
            print(f"[WARN]  {tk}: торговый статус {inst.trading_status}.")

        api_q, shares, qwarns = _api_quantity(o, inst)
        if api_q <= 0:
            print(f"[ERROR] {tk}: количество 0.")
            _rep(tk, "error_quantity")
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
            _rep(tk, "dry_run")
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
            _rep(tk, "error_broker")
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
            "strategy":       o.strategy,
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
        placed.append(rec)
        _rep(tk, "placed")

    if not dry_run:
        _save_pending(pending)
        if not immediate_stop:
            n = sum(1 for r in acc_list if not (r["stop_placed"] and r["tp_placed"]))
            if n:
                print(f"\n→ {n} позиц. ждут SL/TP. Когда лимитки зальются, выполните:")
                print("    python3 -m services.place_orders --attach-stops")
    return placed


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
                 dry_run: bool, writer: csv.DictWriter, env: str,
                 conn=None, report: dict | None = None) -> None:
    """ФАЗА 2: для залившихся позиций ставит STOP_LOSS и TAKE_PROFIT.

    ВНИМАНИЕ: это НЕ нативный OCO. SL и TP — две независимые условные заявки.
    Когда одна срабатывает, вторая остаётся и при касании своей цены откроет
    ОБРАТНУЮ позицию (17.09: стоп SNGS сработал в 07:06, тейк висел до 09:10).
    Поэтому каждый прогон, в порядке:

      1. OCO по истории условных заявок: нога EXECUTED → вторая снимается,
         запись закрывается с причиной stop/target. Не ждём, пока брокер покажет
         нулевую позицию: по факту исполнения ноги позиция уже закрыта.
      2. Позиция закрыта без известной ноги → снять остатки (closed_externally).
      3. Позиция есть → записать заливку в execution_audit (conn).
      4. Цена уже за стопом → стоп не ставить (брокер отклонит, 15.09 SMLT:
         30099), а сразу закрыть позицию по рынку (stop_breach).
      5. Иначе поставить недостающие SL и TP.

    report (необязательный dict) получает списки oco_closed, breach_exits,
    fills_journaled, closed_externally — для уведомлений и вердикта."""
    from services import exec_journal
    rep = report if report is not None else {}
    for k in ("oco_closed", "breach_exits", "closed_externally", "breach_failed",
              "flat_unconfirmed"):
        rep.setdefault(k, [])
    rep.setdefault("fills_journaled", 0)

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
    history = _stop_history(broker, account_id, active)
    existing_sl = {s.instrument_uid for s in stop_orders if s.kind == "STOP_LOSS"}
    existing_tp = {s.instrument_uid for s in stop_orders if s.kind == "TAKE_PROFIT"}

    print(f"Записей в работе: {len(active)}. Открытых позиций: {len(positions)}.")
    for r in active:
        tk, uid = r["ticker"], r["instrument_uid"]
        bal = positions.get(uid, 0.0)

        # ── 1. OCO: одна нога исполнилась → снять вторую
        fired = _fired_leg(r, history)
        if fired:
            kind, rec_ = fired
            if abs(bal) > 1e-9:
                print(f"[WARN]  {tk}: {kind} исполнен, а брокер ещё показывает "
                      f"позицию {bal:.0f} шт — снимаю парную ногу всё равно.")
            if dry_run:
                print(f"[DRY]   {tk}: {kind} исполнен — снять парную ногу.")
                continue
            if _cancel_uid_stops(broker, account_id, r, stop_orders, writer, env, "oco_cancel"):
                r["closed"] = True
                r["closed_reason"] = "stop" if kind == "STOP_LOSS" else "target"
                exec_journal.journal_exit(broker, account_id, conn, r, r["closed_reason"],
                                          exit_at=exec_journal.parse_ts(rec_.activated_at))
                rep["oco_closed"].append(tk)
                print(f"[OCO]   {tk}: {kind} исполнен, парная нога снята, запись закрыта.")
            continue

        bracketed = r.get("stop_placed") or r.get("tp_placed")

        # ── 2. позиция закрыта без известной ноги
        if abs(bal) < 1e-9:
            if bracketed:
                # Ноль у брокера — ещё не закрытие. Снимать защиту можно только
                # по факту сделки выхода (см. exit_confirmed).
                if exit_confirmed(broker, account_id, r) is False:
                    print(f"[HOLD]  {tk}: брокер показывает ноль, но сделки выхода "
                          f"нет — запись и условные заявки сохранены.")
                    rep["flat_unconfirmed"].append(tk)
                    continue
                _reconcile_closed(broker, account_id, r, stop_orders, dry_run, writer, env)
                if r.get("closed") and not dry_run:
                    r["closed_reason"] = r.get("closed_reason") or "closed_externally"
                    exec_journal.journal_exit(broker, account_id, conn, r, "closed_externally")
                    rep["closed_externally"].append(tk)
            else:
                print(f"[WAIT]  {tk}: лимитка {r['order_id']} ещё не залилась — в очереди.")
            continue

        # ── 3. заливка в журнал исполнения
        if conn is not None and not dry_run and not r.get("fill_journaled"):
            if exec_journal.journal_fill(broker, account_id, conn, r):
                r["fill_journaled"] = True
                rep["fills_journaled"] += 1

        lot = int(r.get("lot", 1)) or 1
        qty = int(abs(bal) // lot)
        if qty <= 0:
            print(f"[ERROR] {tk}: позиция {abs(bal):.0f} шт меньше лота {lot} — "
                  f"защиту поставить нельзя.")
            continue
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

        # ── 4–5. STOP_LOSS (или аварийный выход, если цена уже за стопом)
        if not r.get("stop_placed"):
            stop_q = Quotation(units=int(r["stop_units"]), nano=int(r["stop_nano"]))
            if uid in existing_sl:
                print(f"[SKIP]  {tk}: STOP_LOSS уже есть — помечаю выполненным.")
                r["stop_placed"] = True
            elif dry_run:
                print(f"[DRY]   {tk}: STOP_LOSS {edir} qty={qty} @ {stop_q.as_float()}")
            else:
                if _stop_breached(broker, uid, edir, stop_q.as_float()):
                    _breach_exit(broker, account_id, r, inst, bal, stop_orders,
                                 writer, env, conn, rep)
                    continue
                sid = _place_conditional(broker, account_id, tk, inst, edir,
                                         qty, stop_q, "STOP_LOSS", writer, env)
                if sid:
                    r["stop_placed"], r["stop_order_id"] = True, sid
                elif _stop_breached(broker, uid, edir, stop_q.as_float()):
                    # отказ брокера пришёл, потому что цена успела пройти стоп
                    _breach_exit(broker, account_id, r, inst, bal, stop_orders,
                                 writer, env, conn, rep)
                    continue

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


def _stop_history(broker: BrokerClient, account_id: str, active: list[dict]) -> dict:
    """{stop_order_id: StopOrderRecord} по записям с выставленными ногами.
    Контур без истории → пустой словарь: OCO сработает по нулевой позиции."""
    from services import exec_journal
    with_legs = [r for r in active if r.get("stop_order_id") or r.get("tp_order_id")]
    if not with_legs:
        return {}
    since = min(exec_journal.utc_iso(r.get("created")) for r in with_legs)
    try:
        return {h.stop_order_id: h for h in broker.get_stop_order_history(account_id, since)}
    except (NotSupportedError, BrokerError) as e:
        print(f"[WARN]  история условных заявок недоступна ({e}) — OCO по позиции.")
        return {}


def _fired_leg(r: dict, history: dict):
    """(вид, запись истории) исполнившейся ноги или None."""
    for kind, key in (("STOP_LOSS", "stop_order_id"), ("TAKE_PROFIT", "tp_order_id")):
        h = history.get(r.get(key) or "")
        if h is not None and h.status == "EXECUTED":
            return kind, h
    return None


def _cancel_uid_stops(broker: BrokerClient, account_id: str, r: dict, stop_orders: list,
                      writer: csv.DictWriter, env: str, action: str) -> bool:
    """Снять все активные условные заявки по инструменту записи. True — сняты все."""
    ok = True
    for s in [s for s in stop_orders if s.instrument_uid == r["instrument_uid"]]:
        try:
            broker.cancel_stop_order(account_id=account_id, stop_order_id=s.stop_order_id)
            print(f"[CANCEL] {r['ticker']}: снят {s.kind} {s.stop_order_id[:8]}…")
            _logrow(writer, env=env, account_id=account_id, ticker=r["ticker"],
                    action=action, order_id=s.stop_order_id, status="cancelled", info=s.kind)
        except BrokerError as e:
            ok = False
            print(f"[ERROR] {r['ticker']}: не снять {s.kind} {s.stop_order_id[:8]}…: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=r["ticker"],
                    action=action, order_id=s.stop_order_id, status="error", info=str(e))
    return ok


def _stop_breached(broker: BrokerClient, uid: str, exit_direction: str, stop: float) -> bool:
    """Цена уже за стопом: для лонга (выход SELL) последняя ≤ стопа, для шорта ≥.
    Нет цены → False: без цены аварийно закрывать нельзя, ставим стоп как обычно."""
    try:
        last = broker.get_last_price(uid)
    except (NotSupportedError, BrokerError):
        return False
    if not last or not stop:
        return False
    return last <= stop if exit_direction == "SELL" else last >= stop


def _breach_exit(broker: BrokerClient, account_id: str, r: dict, inst: Instrument,
                 balance: float, stop_orders: list, writer: csv.DictWriter, env: str,
                 conn, rep: dict) -> bool:
    """Аварийный выход по рынку: цена прошла стоп раньше, чем он был поставлен.

    Сначала снимаются условные заявки по бумаге (тейк на пустой позиции открыл
    бы обратную), без этого рыночная заявка не отправляется. Исполнение
    дожидается по GetOrderState."""
    from services import exec_journal
    tk = r["ticker"]
    print(f"[BREACH] {tk}: цена уже за стопом — закрываю позицию по рынку.")
    if not _cancel_uid_stops(broker, account_id, r, stop_orders, writer, env, "breach_cancel_stop"):
        rep["breach_failed"].append(tk)
        return False
    lot = int(r.get("lot") or 1) or 1
    lots = int(abs(balance) // lot)
    edir = _close_direction(balance)
    try:
        posted = broker.post_market_order(account_id=account_id, instrument=inst,
                                          direction=edir, quantity_lots=lots,
                                          order_id=new_order_id())
        st = _await_order(broker, account_id, posted.order_id)
    except BrokerError as e:
        print(f"[ERROR] {tk}: аварийный выход не прошёл: {e}")
        _logrow(writer, env=env, account_id=account_id, ticker=tk, direction=edir,
                action="stop_breach_exit", qty_lots_api=lots, status="error", info=str(e))
        rep["breach_failed"].append(tk)
        return False
    _logrow(writer, env=env, account_id=account_id, ticker=tk, direction=edir,
            action="stop_breach_exit", order_id=st.order_id, qty_lots_api=lots,
            price=st.executed_price, status=st.execution_report_status)
    if not st.is_filled:
        rep["breach_failed"].append(tk)
        return False
    r["closed"], r["closed_reason"] = True, "stop_breach"
    exec_journal.journal_exit(broker, account_id, conn, r, "stop_breach",
                              exit_price=st.executed_price,
                              exit_fee=st.executed_commission)
    rep["breach_exits"].append(tk)
    return True


def _await_order(broker: BrokerClient, account_id: str, order_id: str, *,
                 timeout_s: float = 20.0):
    """GetOrderState до конечного статуса или таймаута."""
    deadline = time.monotonic() + timeout_s
    st = broker.get_order_state(account_id=account_id, order_id=order_id)
    while not st.is_terminal and time.monotonic() < deadline:
        time.sleep(1.0)
        st = broker.get_order_state(account_id=account_id, order_id=order_id)
    return st


def exit_confirmed(broker: BrokerClient, account_id: str, r: dict) -> bool | None:
    """Есть ли по записи ФАКТИЧЕСКАЯ сделка выхода.

    Нулевой баланс у брокера сам по себе не означает, что позиция закрыта.
    23.09.2026, боевой счёт: тейк по UPRO сработал в 22:57, с 22:58 GetPositions
    показывал ноль, а в 23:53 позиция вернулась — продажа прошла только утром
    в 09:15. За эти 55 минут PROTECT успел снять стоп и закрыть запись, и ночь
    позиция простояла без защиты; утром CLOSE увидел её и уронил тест в halt.

    True  — выход подтверждён сделками;
    False — сделок выхода нет, данные брокера ещё не устоялись: запись и
            условные заявки трогать нельзя;
    None  — операции недоступны, подтвердить нечем (решение — на вызывающем).
    """
    from services import exec_journal
    uid = r.get("instrument_uid")
    if not uid:
        return None
    try:
        trades, _ = broker.get_trades(account_id,
                                      exec_journal.utc_iso(r.get("created")), uid)
    except (NotSupportedError, BrokerError, AttributeError) as e:
        print(f"[WARN]  {r.get('ticker')}: операции недоступны ({e}) — "
              f"выход подтвердить нечем.")
        return None
    side = _exit_dir(r)
    return any(t.side == side and t.quantity for t in trades)


def _reconcile_closed(broker: BrokerClient, account_id: str, r: dict,
                      stop_orders: list, dry_run: bool,
                      writer: csv.DictWriter, env: str) -> None:
    """Позиция закрыта, а какая нога сработала — неизвестно. Снимаем оставшиеся
    условные заявки и закрываем запись, только если сняты все."""
    tk = r["ticker"]
    leftovers = [s for s in stop_orders if s.instrument_uid == r["instrument_uid"]]
    if dry_run:
        print(f"[DRY]   {tk}: позиция закрыта — снять {len(leftovers)} оставшихся стоп(ов).")
        return
    if not _cancel_uid_stops(broker, account_id, r, stop_orders, writer, env, "oco_cancel"):
        print(f"[WARN]  {tk}: не все стопы сняты — запись остаётся открытой.")
        return
    r["closed"] = True
    print(f"[DONE]  {tk}: сделка закрыта, запись архивирована.")


# ── ФАЗА 3: закрытие внутридневных позиций перед концом сессии ────────────────

# Стратегии, которые ПО ОПРЕДЕЛЕНИЮ не переносятся через ночь.
INTRADAY_STRATEGIES = frozenset({"intraday_long", "intraday_short"})

MSK = dt.timezone(dt.timedelta(hours=3))


def _square_off_time() -> dt.time:
    raw = str(getattr(config, "INTRADAY_SQUARE_OFF_TIME", "18:35")).strip()
    try:
        hh, mm = raw.split(":")
        return dt.time(int(hh), int(mm))
    except (ValueError, AttributeError):
        log.warning("INTRADAY_SQUARE_OFF_TIME=%r не разобрано — беру 18:35.", raw)
        return dt.time(18, 35)


def _is_intraday(rec: dict) -> bool | None:
    """True/False по записи реестра; None — стратегия неизвестна (старая запись).

    Различать важно: молча считать запись без стратегии внутридневной нельзя —
    так можно закрыть позицию long_overnight, которая обязана жить через ночь.
    """
    st = rec.get("strategy")
    if not st:
        return None
    return st in INTRADAY_STRATEGIES


def square_off_intraday(broker: BrokerClient, account_id: str, *,
                        dry_run: bool = False, no_confirm: bool = False,
                        writer: csv.DictWriter, env: str,
                        force: bool = False, now: dt.datetime | None = None,
                        close_unregistered: bool = False, exclude_uids=(),
                        conn=None, report: dict | None = None) -> int:
    """Закрывает по рынку внутридневные позиции перед закрытием основной сессии.

    Зачем: в двухфазной модели позиция выходит ТОЛЬКО по STOP_LOSS или
    TAKE_PROFIT. На симуляции 5-минутного пути цены 79,1% внутридневных заливок
    не достигают ни того, ни другого и переносятся через ночь.

    Строгий порядок (аудит r4, 17.09):
      1. снять активные лимитки на вход и условные заявки по закрываемым бумагам —
         иначе лимитка зальётся уже после закрытия и уйдёт в ночь, а оставшийся
         стоп откроет обратную позицию;
      2. взять ФАКТИЧЕСКИЕ позиции у брокера;
      3. закрыть по рынку;
      4. дождаться, пока позиция у брокера станет нулевой (опрос до
         SQUARE_OFF_FILL_TIMEOUT_SEC).

    Позиции long_overnight НЕ ТРОГАЮТСЯ. close_unregistered=True (фаза CLEANUP
    Этапа 2) — закрываются и позиции без записи в реестре или без стратегии:
    на 18:20 у контура не бывает законных позиций, кроме интрадея. В ручном CLI
    (по умолчанию False) такие позиции не трогаются: они могут быть ручными.
    exclude_uids — инструменты, которые не закрываются никогда (паи казначейства).

    report получает cancelled_orders, cancelled_stops, closed, orphans,
    not_flat, kept_overnight. Возвращает число закрытых позиций.
    """
    from services import exec_journal
    rep = report if report is not None else {}
    for k in ("closed", "orphans", "not_flat", "kept_overnight"):
        rep.setdefault(k, [])
    rep.setdefault("cancelled_orders", 0)
    rep.setdefault("cancelled_stops", 0)

    if not getattr(config, "INTRADAY_SQUARE_OFF_ENABLED", True) and not force:
        print("INTRADAY_SQUARE_OFF_ENABLED=0 — закрытие по концу сессии отключено.")
        return 0

    now = now or dt.datetime.now(MSK)
    target = _square_off_time()
    if now.time() < target and not force:
        print(f"Сейчас {now:%H:%M} МСК, закрытие назначено на {target:%H:%M} — рано. "
              f"--force чтобы закрыть сейчас.")
        return 0

    pending = _load_pending()
    acc_list = pending.get(account_id, [])
    active = [r for r in acc_list if not r.get("closed")]
    reg = {r.get("instrument_uid"): r for r in active if r.get("instrument_uid")}
    exclude = set(exclude_uids or ())
    night = {u for u, r in reg.items() if r.get("strategy") == "long_overnight"}

    def _wanted(uid: str) -> bool:
        """Бумага подлежит закрытию: интрадей по реестру или (в CLEANUP) чужая."""
        if uid in exclude or uid in night:
            return False
        flag = _is_intraday(reg.get(uid) or {})
        return flag is True or (close_unregistered and flag is None)

    # ── 1. лимитки на вход и условные заявки
    try:
        active_orders = broker.get_active_orders(account_id)
    except (NotSupportedError, BrokerError) as e:
        active_orders = []
        print(f"[WARN]  активные заявки не получены: {e}")
    try:
        stop_orders = broker.get_active_stop_orders(account_id)
    except NotSupportedError:
        stop_orders = []
        print("[WARN]  GetStopOrders не поддержан — стопы снять не смогу.")
    orders_to_cancel = [o for o in active_orders if _wanted(o.instrument_uid)]
    stops_to_cancel = [s for s in stop_orders if _wanted(s.instrument_uid)]

    if dry_run:
        positions = [p for p in broker.get_positions(account_id)
                     if p.is_open and _wanted(p.instrument_uid)]
        print(f"[DRY-RUN] снять лимиток {len(orders_to_cancel)}, стопов "
              f"{len(stops_to_cancel)}, закрыть позиций {len(positions)}; "
              f"реальные заявки не отправляются.")
        return 0
    if not no_confirm:
        prompt = ("ЗАКРЫТЬ интрадей по рынку на БОЕВОМ счёте? (y/n): " if env == "PROD"
                  else "Закрыть интрадей по рынку? (y/n): ")
        if not confirm(prompt):
            print("[INFO] Закрытие отменено пользователем.")
            return 0

    failed_stop_uids: set[str] = set()
    for o in orders_to_cancel:
        try:
            broker.cancel_order(account_id=account_id, order_id=o.order_id)
            rep["cancelled_orders"] += 1
            _logrow(writer, env=env, account_id=account_id,
                    ticker=(reg.get(o.instrument_uid) or {}).get("ticker", o.instrument_uid[:8]),
                    action="squareoff_cancel_order", order_id=o.order_id, status="cancelled")
        except BrokerError as e:
            print(f"[WARN]  не снята лимитка {o.order_id[:8]}…: {e}")
    for s in stops_to_cancel:
        tk = (reg.get(s.instrument_uid) or {}).get("ticker", s.instrument_uid[:8])
        try:
            broker.cancel_stop_order(account_id=account_id, stop_order_id=s.stop_order_id)
            rep["cancelled_stops"] += 1
            print(f"[CANCEL] {tk}: снят {s.kind} {s.stop_order_id[:8]}…")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="squareoff_cancel_stop", order_id=s.stop_order_id,
                    status="cancelled", info=s.kind)
            rec = reg.get(s.instrument_uid)
            if rec is not None:
                # стоп снят: если позицию закрыть не выйдет, protect поставит заново
                key = "stop" if s.kind == "STOP_LOSS" else "tp"
                rec[f"{key}_placed"], rec[f"{key}_order_id"] = False, None
        except BrokerError as e:
            failed_stop_uids.add(s.instrument_uid)
            print(f"[ERROR] {tk}: не снять {s.kind}: {e}")

    # ── 2. фактические позиции у брокера
    positions = {p.instrument_uid: p for p in broker.get_positions(account_id) if p.is_open}
    # Внутридневная запись без позиции: лимитка не залилась и снята выше. Запись
    # закрывается, иначе монитор ждал бы её заливки вечно.
    for uid, rec in reg.items():
        if uid not in positions and _is_intraday(rec) is True:
            reason = "intraday_not_filled"
            if exec_journal.journal_fill(broker, account_id, conn, rec):
                # залилась и уже закрыта, а монитор не успел это записать
                exec_journal.journal_exit(broker, account_id, conn, rec, "closed_externally")
                reason = "closed_externally"
            rec["closed"], rec["closed_reason"] = True, reason
    if not positions:
        print("Открытых позиций нет — закрывать нечего.")
        _save_pending(pending)
        return 0

    targets = []
    for uid, pos in positions.items():
        rec = reg.get(uid)
        if uid in exclude:
            continue
        if uid in night:
            rep["kept_overnight"].append(rec.get("ticker"))
            continue
        flag = _is_intraday(rec or {})
        if flag is True or (close_unregistered and flag is None):
            if flag is None:
                rep["orphans"].append((rec or {}).get("ticker") or uid[:8])
            targets.append((uid, pos, rec))
        elif flag is None:
            print(f"[SKIP]  позиция {uid[:8]}… не принадлежит системе (нет в реестре "
                  f"или без стратегии) — не трогаю.")
    if rep["kept_overnight"]:
        print(f"[KEEP]  ночные позиции остаются открытыми: {', '.join(rep['kept_overnight'])}")
    if not targets:
        print("Внутридневных позиций к закрытию нет.")
        _save_pending(pending)
        return 0

    # ── 3. закрытие по рынку (без снятых стопов не закрываем: обратная позиция хуже)
    print(f"\nЗАКРЫТИЕ ВНУТРИДНЕВНЫХ ПОЗИЦИЙ ({now:%H:%M} МСК, контур {env})")
    sent = []
    for uid, pos, rec in targets:
        tk = (rec or {}).get("ticker") or uid[:8]
        if uid in failed_stop_uids:
            print(f"[ERROR] {tk}: стопы не сняты — позиция не закрывается.")
            rep["not_flat"].append(tk)
            continue
        if _market_close(broker, account_id, pos, tk, writer, env):
            sent.append((uid, tk, rec))
        else:
            rep["not_flat"].append(tk)

    # ── 4. ждать нулевой позиции у брокера
    timeout = float(getattr(config, "SQUARE_OFF_FILL_TIMEOUT_SEC", 30))
    remaining = _await_flat(broker, account_id, {u for u, _, _ in sent}, timeout_s=timeout)
    closed = 0
    for uid, tk, rec in sent:
        if uid in remaining:
            print(f"[ERROR] {tk}: за {timeout:.0f} с позиция у брокера не обнулилась.")
            rep["not_flat"].append(tk)
            continue
        closed += 1
        rep["closed"].append(tk)
        if rec is not None:
            rec["closed"], rec["closed_reason"] = True, "square_off"
            exec_journal.journal_exit(broker, account_id, conn, rec, "square_off")

    _save_pending(pending)
    print(f"\nЗакрыто внутридневных позиций: {closed} из {len(targets)}.")
    return closed


def _await_flat(broker: BrokerClient, account_id: str, uids: set[str], *,
                timeout_s: float, poll_s: float = 1.0) -> set[str]:
    """Опрашивает позиции, пока все uids не обнулятся. Возвращает необнулённые."""
    remaining = set(uids)
    deadline = time.monotonic() + timeout_s
    while remaining:
        open_uids = {p.instrument_uid for p in broker.get_positions(account_id) if p.is_open}
        remaining &= open_uids
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    return remaining


# ── СИНХРОНИЗАЦИЯ ПОРТФЕЛЯ К СИГНАЛАМ (ТЗ) ────────────────────────────────────


def _pos_direction(balance_shares: float) -> str:
    """LONG / SHORT по знаку остатка позиции."""
    return "LONG" if balance_shares > 0 else "SHORT"


def _close_direction(balance_shares: float) -> str:
    """Направление заявки для ЗАКРЫТИЯ позиции: лонг→SELL, шорт→BUY."""
    return "SELL" if balance_shares > 0 else "BUY"


def _mark_registry_closed(account_id: str, closed_uids: set[str]) -> None:
    """Пометить записи реестра закрытыми для инструментов, которые мы закрыли
    по рынку (чтобы attach_stops не считал их ожидающими)."""
    if not closed_uids:
        return
    pending = _load_pending()
    changed = False
    for r in pending.get(account_id, []):
        if r.get("instrument_uid") in closed_uids and not r.get("closed"):
            r["closed"] = True
            changed = True
    if changed:
        _save_pending(pending)


def _wait_for_fills(broker: BrokerClient, account_id: str, placed: list[dict], *,
                    wait_s: float, poll_s: float) -> None:
    """Подождать исполнения только что выставленных лимиток (до wait_s сек),
    опрашивая позиции каждые poll_s. Выходит раньше, когда все залились.
    Нужно, чтобы стопы привязались в ТОМ ЖЕ прогоне, а не следующим."""
    pending_uids = {r["instrument_uid"] for r in placed}
    if not pending_uids or wait_s <= 0:
        return
    print(f"\n— Ожидание исполнения лимиток (до {wait_s:.0f}s) —")
    deadline = time.monotonic() + wait_s
    while True:
        open_uids = {p.instrument_uid for p in broker.get_positions(account_id) if p.is_open}
        remaining = pending_uids - open_uids
        print(f"  залилось {len(pending_uids) - len(remaining)}/{len(pending_uids)}; "
              f"ждём {len(remaining)}…")
        if not remaining:
            print("  все лимитки исполнены.")
            return
        if time.monotonic() >= deadline:
            print(f"  таймаут: {len(remaining)} не залилось — стопы по ним поставит "
                  f"следующий прогон (--attach-stops).")
            return
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


def _verify_protection(broker: BrokerClient, account_id: str, ticker_of) -> None:
    """Финальная проверка: у каждой открытой позиции есть и SL, и TP."""
    print("\n— Проверка защиты позиций —")
    positions = [p for p in broker.get_positions(account_id) if p.is_open]
    if not positions:
        print("  открытых позиций нет.")
        return
    try:
        stops = broker.get_active_stop_orders(account_id)
    except NotSupportedError:
        print("  GetStopOrders недоступен — проверка пропущена.")
        return
    sl = {s.instrument_uid for s in stops if s.kind == "STOP_LOSS"}
    tp = {s.instrument_uid for s in stops if s.kind == "TAKE_PROFIT"}
    for p in positions:
        miss = [n for n, have in (("SL", p.instrument_uid in sl),
                                  ("TP", p.instrument_uid in tp)) if not have]
        mark = "OK" if not miss else "⚠ НЕТ " + "/".join(miss)
        print(f"  {ticker_of(p.instrument_uid):<6} {_pos_direction(p.balance_shares):<5} "
              f"{abs(p.balance_shares):.0f} шт  {mark}")


def _market_close(broker: BrokerClient, account_id: str, pos, ticker: str,
                  writer: csv.DictWriter, env: str) -> bool:
    """Отправить рыночную заявку на закрытие позиции. True — заявка принята
    (исполнение проверяет вызывающий по позициям брокера)."""
    try:
        inst = broker.find_instrument_by_uid(pos.instrument_uid)
    except BrokerError as e:
        print(f"[ERROR] {ticker}: не закрыть — инструмент по uid не найден: {e}")
        return False
    lot = int(inst.lot or 1) or 1
    qty = int(abs(pos.balance_shares) // lot)
    if qty <= 0:
        # Прежний max(1, …) продал бы целый лот при остатке меньше лота — то есть
        # открыл бы обратную позицию на разницу.
        print(f"[ERROR] {ticker}: остаток {abs(pos.balance_shares):.0f} шт меньше лота {lot}.")
        return False
    if abs(pos.balance_shares) % lot:
        print(f"[WARN]  {ticker}: остаток не кратен лоту {lot} — закрываю {qty} лот.")
    edir = _close_direction(pos.balance_shares)
    try:
        st = broker.post_market_order(account_id=account_id, instrument=inst,
                                      direction=edir, quantity_lots=qty,
                                      order_id=new_order_id())
    except BrokerError as e:
        print(f"[ERROR] {ticker}: market-close не прошёл: {e}")
        _logrow(writer, env=env, account_id=account_id, ticker=ticker,
                direction=edir, action="close_market", qty_lots_api=qty,
                status="error", info=str(e))
        return False
    print(f"[CLOSE] {ticker}: market {edir} qty={qty} → {st.execution_report_status}")
    _logrow(writer, env=env, account_id=account_id, ticker=ticker,
            direction=edir, action="close_market", order_id=st.order_id,
            qty_lots_api=qty, status=st.execution_report_status)
    return True


def _print_sync_plan(closes: list, cancel_ord: list, cancel_stp: list,
                     to_place: list, kept: list, ticker_of, env: str,
                     account_id: str) -> None:
    if env == "PROD":
        print("\n" + "!" * 88)
        print(f"!!  СИНХРОНИЗАЦИЯ БОЕВОГО ПОРТФЕЛЯ — РЕАЛЬНЫЕ ДЕНЬГИ. Счёт №{account_id}")
        print("!" * 88)
    else:
        print(f"\nКонтур: {env} | Счёт №{account_id} | СИНХРОНИЗАЦИЯ ПОРТФЕЛЯ")

    print(f"\n[=] Оставить без изменений (сигнал совпал): {len(kept)}")
    for uid, tk, d in kept:
        print(f"    {tk:<6} {d}")

    print(f"\n[X] ЗАКРЫТЬ по рынку (сигнал исчез/сменил направление): {len(closes)}")
    for p in closes:
        print(f"    {ticker_of(p.instrument_uid):<6} {_pos_direction(p.balance_shares):<5} "
              f"{abs(p.balance_shares):.0f} шт → {_close_direction(p.balance_shares)}")

    print(f"\n[-] Снять лимитки (сигнал исчез/перевёрнут): {len(cancel_ord)}")
    for ao in cancel_ord:
        print(f"    {ticker_of(ao.instrument_uid):<6} {ao.direction:<4} {ao.lots} лот  "
              f"id={ao.order_id[:8]}…")

    print(f"\n[-] Снять осиротевшие стопы: {len(cancel_stp)}")
    for s in cancel_stp:
        print(f"    {ticker_of(s.instrument_uid):<6} {s.kind:<12} id={s.stop_order_id[:8]}…")

    print(f"\n[+] Выставить новые заявки: {len(to_place)}")
    for o in to_place:
        entry = f"{o.entry_price:.4f}" if o.entry_price else "—"
        stop  = f"{o.stop_price:.4f}"  if o.stop_price  else "—"
        tp    = f"{o.tp_price:.4f}"    if o.tp_price   else "—"
        side  = "BUY (LONG)" if o.direction == "LONG" else "SELL (SHORT)"
        print(f"    {o.ticker:<6} {side:<13} вход {entry} | стоп {stop} | профит {tp}")


def sync_portfolio(broker: BrokerClient, account_id: str, orders: list[Order], *,
                   dry_run: bool, no_confirm: bool, immediate_stop: bool,
                   writer: csv.DictWriter, env: str,
                   wait_s: float = 0.0, poll_s: float = 5.0) -> None:
    """Привести портфель к целевым сигналам (ТЗ):
      3. закрыть позиции, по которым сигнал исчез/сменил направление (по рынку);
      4. выставить заявки по новым сигналам (place_limits с дедупом);
      5. установить/сопроводить SL и TP (attach_stops);
      6. снять осиротевшие защитные заявки и неактуальные лимитки.
    """
    # ── 1. целевые инструменты (uid → Order/Instrument), направления
    targets: dict[str, tuple[Order, Instrument]] = {}
    uid2ticker: dict[str, str] = {}
    for o in orders:
        if not o.is_placeable:
            continue
        try:
            inst = broker.find_instrument(o.ticker)
        except BrokerError as e:
            print(f"[WARN]  {o.ticker}: пропуск в синхронизации — {e}")
            continue
        if _short_blocked(o, inst):
            print(f"[SKIP SHORT] {o.ticker}: шорт недоступен у брокера "
                  f"(shortEnabledFlag=false) — цель исключена из синхронизации.")
            log.info("[SKIP SHORT] %s: шорт недоступен у брокера", o.ticker)
            continue
        targets[inst.instrument_uid] = (o, inst)
        uid2ticker[inst.instrument_uid] = o.ticker
    target_dir = {uid: od.direction for uid, (od, _) in targets.items()}

    # ── 2. текущее состояние
    positions = [p for p in broker.get_positions(account_id) if p.is_open]
    active_orders = broker.get_active_orders(account_id)
    try:
        stop_orders = broker.get_active_stop_orders(account_id)
    except NotSupportedError:
        stop_orders = []

    def ticker_of(uid: str) -> str:
        if uid not in uid2ticker:
            try:
                uid2ticker[uid] = broker.find_instrument_by_uid(uid).ticker
            except BrokerError:
                uid2ticker[uid] = uid[:8]
        return uid2ticker[uid]

    # ── 3. позиции: оставить совпавшие по направлению, остальные закрыть
    closes, kept = [], []
    kept_pos_uids: set[str] = set()
    for p in positions:
        cur = _pos_direction(p.balance_shares)
        if target_dir.get(p.instrument_uid) == cur:
            kept_pos_uids.add(p.instrument_uid)
            kept.append((p.instrument_uid, ticker_of(p.instrument_uid), cur))
        else:
            closes.append(p)

    # ── 4. лимитки: снять не-целевые и перевёрнутые по направлению
    cancel_ord = []
    for ao in active_orders:
        tgt = targets.get(ao.instrument_uid)
        if tgt is None or tgt[0].order_direction != ao.direction:
            cancel_ord.append(ao)
    keep_ord_uids = {ao.instrument_uid for ao in active_orders if ao not in cancel_ord}

    # ── 5. стопы: осиротевшие — нет удерживаемой позиции под ними
    cancel_stp = [s for s in stop_orders if s.instrument_uid not in kept_pos_uids]

    # ── 6. новые заявки: цель, которая не удерживается и не в активной лимитке
    to_place = [od for uid, (od, _) in targets.items()
                if uid not in kept_pos_uids and uid not in keep_ord_uids]

    _print_sync_plan(closes, cancel_ord, cancel_stp, to_place, kept,
                     ticker_of, env, account_id)

    if dry_run:
        print("\n[DRY-RUN] синхронизация ничего не меняет.")
        return
    if not (closes or cancel_ord or cancel_stp or to_place):
        print("\nПортфель уже соответствует сигналам — изменений нет.")
        return

    prompt = ("\nПРИМЕНИТЬ синхронизацию на БОЕВОМ счёте (закрытия по рынку + заявки)? (y/n): "
              if env == "PROD" else "\nПрименить синхронизацию? (y/n): ")
    if not no_confirm and not confirm(prompt):
        print("[INFO] Синхронизация отменена пользователем.")
        return

    # ── исполнение: снять стопы → снять лимитки → закрыть → выставить → привязать
    for s in cancel_stp:
        tk = ticker_of(s.instrument_uid)
        try:
            broker.cancel_stop_order(account_id=account_id, stop_order_id=s.stop_order_id)
            print(f"[-]     {tk}: снят стоп {s.kind} {s.stop_order_id[:8]}…")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    action="cancel_stop", order_id=s.stop_order_id,
                    status="cancelled", info=s.kind)
        except BrokerError as e:
            print(f"[WARN]  {tk}: не снять стоп {s.stop_order_id[:8]}…: {e}")

    for ao in cancel_ord:
        tk = ticker_of(ao.instrument_uid)
        try:
            broker.cancel_order(account_id=account_id, order_id=ao.order_id)
            print(f"[-]     {tk}: снята лимитка {ao.order_id[:8]}…")
            _logrow(writer, env=env, account_id=account_id, ticker=tk,
                    direction=ao.direction, action="cancel_order",
                    order_id=ao.order_id, status="cancelled")
        except BrokerError as e:
            print(f"[WARN]  {tk}: не снять лимитку {ao.order_id[:8]}…: {e}")

    closed_uids = set()
    for p in closes:
        if _market_close(broker, account_id, p, ticker_of(p.instrument_uid), writer, env):
            closed_uids.add(p.instrument_uid)
    _mark_registry_closed(account_id, closed_uids)

    # новые заявки — place_limits сам сделает дедуп по СВЕЖЕМУ состоянию
    print("\n— Выставление заявок по новым сигналам —")
    placed = place_limits(broker, account_id, orders, dry_run=False,
                          immediate_stop=immediate_stop, force=False,
                          writer=writer, env=env)

    # подождать, пока лимитки у рынка зальются, чтобы привязать стопы СЕЙЧАС
    if not immediate_stop:
        _wait_for_fills(broker, account_id, placed, wait_s=wait_s, poll_s=poll_s)

    # привязка/подчистка SL+TP к залившимся позициям
    print("\n— Привязка стоп-лосс / тейк-профит —")
    attach_stops(broker, account_id, dry_run=False, writer=writer, env=env)

    # финальная проверка: каждая позиция защищена SL и TP
    _verify_protection(broker, account_id, ticker_of)


# ── Подготовка брокера/счёта ──────────────────────────────────────────────────


def _make_broker_and_account(prod: bool) -> tuple[BrokerClient, str, str]:
    """Контур по умолчанию — SANDBOX (виртуальный счёт: создаётся/пополняется).
    prod=True — боевой счёт (реальные деньги), только по явному флагу --prod."""
    if not prod:
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
    account_id = broker.open_or_get_account(preferred_id=config.require_prod_account_id())
    return broker, account_id, "PROD"


def _guard_unattended_prod(prod: bool, no_confirm: bool) -> None:
    """Связка «боевой контур + без подтверждения» — только по явному флагу
    окружения ALLOW_UNATTENDED_PROD=1 (осознанный запуск по cron)."""
    if prod and no_confirm and os.getenv("ALLOW_UNATTENDED_PROD", "") != "1":
        raise RuntimeError(
            "Выполнение на PROD без подтверждения запрещено без флага "
            "ALLOW_UNATTENDED_PROD=1")


def _resolve_env(prod_flag: bool) -> bool:
    """Итоговый контур: config.TRADING_MODE плюс флаг --prod.

    TRADING_MODE может только ОГРАНИЧИТЬ права запуска, но не расширить:
    при TRADING_MODE=sandbox флаг --prod отклоняется, а при TRADING_MODE=prod
    боевой контур всё равно требует явного --prod. Обратного флага (--sandbox,
    включающего боевой режим) не существует по построению.
    """
    mode = str(getattr(config, "TRADING_MODE", "sandbox")).strip().lower()
    if mode not in ("sandbox", "prod"):
        raise RuntimeError(
            f"TRADING_MODE={mode!r} не распознан: допустимо sandbox или prod.")
    if prod_flag and mode != "prod":
        raise RuntimeError(
            "Флаг --prod отклонён: TRADING_MODE=sandbox. Боевой контур требует "
            "И переменной TRADING_MODE=prod, И флага --prod — два независимых "
            "подтверждения намерения торговать реальными деньгами.")
    return prod_flag


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    p = argparse.ArgumentParser(
        description="Автозаявки в T-Invest (двухфазно). КОНТУР ПО УМОЛЧАНИЮ — "
                    "SANDBOX (виртуальные деньги); боевой контур только с --prod.")
    g_env = p.add_mutually_exclusive_group()
    g_env.add_argument("--sandbox", action="store_true",
                       help="Тестовый контур (виртуальные деньги) — режим по умолчанию.")
    g_env.add_argument("--prod", action="store_true",
                       help="БОЕВОЙ контур: реальный счёт, реальные деньги. "
                            "В связке с --no-confirm требует ALLOW_UNATTENDED_PROD=1.")
    p.add_argument("--attach-stops", action="store_true",
                   help="ФАЗА 2: привязать STOP_LOSS к залившимся позициям из реестра.")
    p.add_argument("--square-off", action="store_true",
                   help="ФАЗА 3: закрыть по рынку ВНУТРИДНЕВНЫЕ позиции перед "
                        "концом основной сессии (снимает SL/TP, затем закрывает). "
                        "Позиции long_overnight не трогаются. По умолчанию "
                        "срабатывает только после INTRADAY_SQUARE_OFF_TIME; "
                        "--force закрывает немедленно.")
    p.add_argument("--top-n", type=int,
                   default=int(getattr(config, "BEST_TRADES_TOP_N", 10)))
    p.add_argument("--position", type=float,
                   default=float(getattr(config, "BEST_TRADES_POSITION_RUB", 10_000.0)))
    p.add_argument("--budget", type=float, default=None,
                   help="Общий бюджет в ТЫСЯЧАХ ₽ (--budget 100 = 100 000 ₽). "
                        "Распределяется по бумагам РИСК-ПАРИТЕТОМ (вес ∝ 1/стоп%%), "
                        "перекрывает --position.")
    p.add_argument("--entry-frac", type=float,
                   default=float(getattr(config, "LIMIT_ENTRY_FRACTION", 0.8)))
    p.add_argument("--tp-frac", type=float,
                   default=float(getattr(config, "LIMIT_TP_FRACTION", 0.5)),
                   help="Цель take-profit как доля пути до дальней границы коридора (0..1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Пройти pipeline без реальной отправки в брокер.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Не спрашивать y/n (для cron).")
    p.add_argument("--place-only", action="store_true",
                   help="Только выставить заявки по сигналам, НЕ закрывать позиции "
                        "и не снимать заявки (старое поведение без синхронизации).")
    p.add_argument("--skip-refresh", action="store_true",
                   help="Не догружать свечи перед расчётом (если main.py только что "
                        "отработал). Проверка свежести всё равно выполняется.")
    p.add_argument("--wait-fill", type=float,
                   default=float(getattr(config, "ORDER_FILL_WAIT_SEC", 30.0)),
                   help="Секунд ждать исполнения лимиток перед привязкой стопов "
                        "(0 — не ждать). По умолч. из config.ORDER_FILL_WAIT_SEC.")
    p.add_argument("--immediate-stop", action="store_true",
                   help="ФАЗА 1: ставить стоп сразу за лимиткой (риск раннего стопа).")
    p.add_argument("--force", action="store_true",
                   help="ФАЗА 1: отключить защиту от задвоения (ставить лимитку, "
                        "даже если по инструменту уже есть заявка/позиция/стоп).")
    p.add_argument("--force-trade-unvalidated", action="store_true",
                   help="Подтвердить торговлю стратегиями с вердиктом REJECTED. "
                        "Без флага при STRICT_VALIDATION_GATE=0 печатается "
                        "предупреждение и (в интерактивном режиме) спрашивается "
                        "подтверждение; при STRICT_VALIDATION_GATE=1 такие "
                        "сигналы отсекаются ещё в select_top_rows.")
    args = p.parse_args(argv)

    try:
        use_prod = _resolve_env(args.prod)
    except RuntimeError as e:
        print(f"[ОТКАЗ] {e}", file=sys.stderr)
        return 1
    _guard_unattended_prod(use_prod, args.no_confirm)

    try:
        broker, account_id, env = _make_broker_and_account(use_prod)
    except (BrokerError, ValueError) as e:
        print(f"[ERROR] Счёт/контур: {e}", file=sys.stderr)
        return 1

    writer, fp = _open_log()
    try:
        # ── ФАЗА 3: закрытие внутридневных позиций ──
        if args.square_off:
            print(f"Контур: {env} | Счёт №{account_id} | ФАЗА 3: закрытие внутридневных")
            with registry_lock():
                square_off_intraday(broker, account_id, dry_run=args.dry_run,
                                    no_confirm=args.no_confirm, writer=writer, env=env,
                                    force=args.force)
            return 0

        # ── ФАЗА 2 ──
        if args.attach_stops:
            print(f"Контур: {env} | Счёт №{account_id} | ФАЗА 2: привязка стопов")
            with registry_lock():
                attach_stops(broker, account_id, dry_run=args.dry_run,
                             writer=writer, env=env)
            return 0

        # ── ФАЗА 1 ──
        log.info("Запуск pipeline (валидация + forecast)...")
        budget_rub = args.budget * 1000.0 if args.budget and args.budget > 0 else None
        try:
            orders, _meta = compute_orders(args.top_n, args.position, args.entry_frac,
                                           tp_frac=args.tp_frac,
                                           refresh=not args.skip_refresh,
                                           budget_rub=budget_rub)
        except StaleDataError as e:
            print(f"[ОТКАЗ] Устаревшие данные: {e}", file=sys.stderr)
            print("        Прогоните `python3 main.py` и повторите.", file=sys.stderr)
            return 2
        if not orders:
            print("Нет торговых сигналов.")
            return 0

        # Гейт валидации: если среди кандидатов есть REJECTED, об этом обязаны
        # сказать вслух. Раньше такие заявки уходили в стакан молча.
        n_rejected = warn_unvalidated(_meta.get("top_rows") or [], env=env,
                                      force=args.force_trade_unvalidated)
        if n_rejected and not args.force_trade_unvalidated and not args.dry_run:
            if args.no_confirm:
                print("[ОТКАЗ] Есть сигналы с вердиктом REJECTED, а --no-confirm "
                      "не оставляет возможности подтвердить их вручную.",
                      file=sys.stderr)
                print("        Добавьте --force-trade-unvalidated (осознанно) "
                      "или включите STRICT_VALIDATION_GATE=1.", file=sys.stderr)
                return 3
            if not confirm("Торговать непроверенными стратегиями? (y/n): "):
                print("[INFO] Отменено: сигналы не прошли валидацию.")
                return 0

        print_summary(account_id, env, orders, args.position, budget_rub=budget_rub)

        # ── СИНХРОНИЗАЦИЯ (по умолчанию): закрыть выпавшие сигналы + выставить новые
        if not args.place_only:
            with registry_lock():
                sync_portfolio(broker, account_id, orders,
                               dry_run=args.dry_run, no_confirm=args.no_confirm,
                               immediate_stop=args.immediate_stop, writer=writer, env=env,
                               wait_s=args.wait_fill,
                               poll_s=float(getattr(config, "ORDER_FILL_POLL_SEC", 5.0)))
            return 0

        # ── --place-only: старое поведение (только заявки, без закрытий)
        prompt = ("ВЫСТАВИТЬ РЕАЛЬНЫЕ ЗАЯВКИ на боевом счёте? (y/n): "
                  if env == "PROD" else "Выставить заявки? (y/n): ")
        if args.dry_run:
            print("[DRY-RUN] реальные ордера НЕ отправляются.")
        elif not args.no_confirm and not confirm(prompt):
            print("[INFO] Выставление заявок отменено пользователем.")
            return 0

        with registry_lock():
            place_limits(broker, account_id, orders,
                         dry_run=args.dry_run, immediate_stop=args.immediate_stop,
                         force=args.force, writer=writer, env=env)
    finally:
        fp.close()
    print("\nГотово. Журнал: data/order_log/<дата>.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
