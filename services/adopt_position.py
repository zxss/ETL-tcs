"""services/adopt_position.py — взять ручную позицию под защиту контура.

Зачем. PROTECT и CLEANUP видят только позиции, записанные в реестр
data/order_log/pending_stops.json: `naked` считается по записям реестра,
пересечённым с открытыми позициями. Позиция, открытая руками — в приложении
T-Invest, по сигналу стороннего аналитика, — для контура невидима. Стоп ей
никто не поставит, в строке «без стопа» она не появится, и тишина в логе не
будет означать, что всё под контролем.

Команда создаёт запись реестра для УЖЕ ОТКРЫТОЙ позиции. Дальше её ведёт
обычный PROTECT (каждые 5 минут): поставит STOP_LOSS, при наличии цели —
TAKE_PROFIT, снимет парную ногу при срабатывании одной, запишет выход.

Заявку на вход команда НЕ выставляет: вход — ваше решение и ваше действие.

Что делает поле strategy (см. place_orders.square_off_intraday):
  intraday_long / intraday_short — CLEANUP закроет позицию по рынку в конце
      основной сессии (в турнирном контуре это 18:23);
  manual (по умолчанию) и любое другое непустое имя — CLEANUP не трогает,
      позиция остаётся через ночь (появится плата за перенос).
Пустая стратегия недопустима: запись без неё CLEANUP считает чужой и в
Этапе 2 закрывает по рынку (close_unregistered=True).

Примеры:
    python3 -m services.adopt_position --prod -t VTBR --stop 52.50
    python3 -m services.adopt_position --prod -t VTBR --stop 52.50 --take 49.80 \
        --strategy intraday_short
    python3 -m services.adopt_position --prod --list
    python3 -m services.adopt_position --prod --forget VTBR
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import uuid

import config
from services.broker.base import BrokerError, Quotation

STRATEGIES = ("manual", "intraday_long", "intraday_short", "long_overnight")
_INTRADAY = ("intraday_long", "intraday_short")


# ── Чистая логика (тестируется без брокера) ──────────────────────────────────

def exit_direction(balance_shares: float) -> str:
    """Направление ЗАКРЫВАЮЩЕЙ заявки: шорт закрывается покупкой."""
    return "BUY" if balance_shares < 0 else "SELL"


def check_levels(*, balance_shares: float, last: float,
                 stop: float, take: float | None) -> list[str]:
    """Стоп и цель на правильной стороне от цены. Список ошибок (пустой — ок).

    Стоп с неправильной стороны — не опечатка в журнале, а мгновенный убыток:
    PROTECT увидит «цена уже за стопом» и закроет позицию по рынку тем же
    прогоном (ветка _stop_breached → _breach_exit в place_orders).
    """
    errs: list[str] = []
    if balance_shares == 0:
        return ["позиция нулевая — защищать нечего"]
    short = balance_shares < 0
    side = "шорт" if short else "лонг"
    if stop <= 0:
        errs.append("стоп должен быть положительным")
    elif short and stop <= last:
        errs.append(f"{side}: стоп {stop} должен быть ВЫШЕ цены {last}")
    elif not short and stop >= last:
        errs.append(f"{side}: стоп {stop} должен быть НИЖЕ цены {last}")
    if take is not None:
        if take <= 0:
            errs.append("цель должна быть положительной")
        elif short and take >= last:
            errs.append(f"{side}: цель {take} должна быть НИЖЕ цены {last}")
        elif not short and take <= last:
            errs.append(f"{side}: цель {take} должна быть ВЫШЕ цены {last}")
    return errs


def build_record(*, order_id: str, ticker: str, instrument_uid: str, lot: int,
                 balance_shares: float, stop_q: Quotation, tp_q: Quotation | None,
                 strategy: str, journal: bool = False,
                 created: dt.datetime | None = None) -> dict:
    """Запись реестра в том же формате, что пишет place_orders.place_limits."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy должна быть одной из {STRATEGIES}, дано {strategy!r}")
    lot = int(lot) or 1
    return {
        "order_id":       order_id,
        "ticker":         ticker.upper(),
        "strategy":       strategy,
        "instrument_uid": instrument_uid,
        "lot":            lot,
        "api_qty":        int(abs(balance_shares) // lot),
        "exit_direction": exit_direction(balance_shares),
        "stop_units":     stop_q.units,
        "stop_nano":      stop_q.nano,
        "tp_units":       tp_q.units if tp_q else None,
        "tp_nano":        tp_q.nano  if tp_q else None,
        "created":        (created or dt.datetime.now()).isoformat(timespec="seconds"),
        "stop_placed":    False,
        "stop_order_id":  None,
        "tp_placed":      tp_q is None,          # нет цели → ставить нечего
        "tp_order_id":    None,
        "closed":         False,
        # у ручной позиции нет строки в execution_audit: журнал заливки
        # пропускаем, иначе PROTECT будет ходить в БД каждый прогон впустую
        "fill_journaled": not journal,
        "adopted":        True,                  # след: запись заведена вручную
    }


def describe(rec: dict, *, last: float | None = None) -> str:
    """Человекочитаемая сводка записи — то, что подтверждает пользователь."""
    stop = Quotation(units=int(rec["stop_units"]), nano=int(rec["stop_nano"])).as_float()
    qty = rec["api_qty"] * rec["lot"]
    side = "ШОРТ" if rec["exit_direction"] == "BUY" else "ЛОНГ"
    out = [f"{rec['ticker']}: {side} {qty} шт ({rec['api_qty']} лот × {rec['lot']})",
           f"  стратегия   {rec['strategy']}"
           + ("  → CLEANUP закроет в конце сессии" if rec["strategy"] in _INTRADAY
              else "  → CLEANUP не трогает, позиция уходит в ночь"),
           f"  стоп        {stop}"]
    if rec["tp_units"] is not None:
        tp = Quotation(units=int(rec["tp_units"]), nano=int(rec["tp_nano"])).as_float()
        out.append(f"  цель        {tp}")
    else:
        out.append("  цель        не задана")
    if last:
        risk = abs(stop - last) * qty
        out.append(f"  цена сейчас {last} → риск до стопа ≈ {risk:,.2f} ₽")
    return "\n".join(out)


# ── Работа с реестром ────────────────────────────────────────────────────────

def _registry_api():
    from services.place_orders import (_load_pending, _save_pending,  # noqa: PLC0415
                                       registry_lock)
    return _load_pending, _save_pending, registry_lock


def active_records(account_id: str) -> list[dict]:
    load, _, lock = _registry_api()
    with lock():
        return [r for r in load().get(account_id, []) if not r.get("closed")]


def add_record(account_id: str, rec: dict) -> None:
    """Добавить запись под блокировкой реестра. Дубль по инструменту — отказ."""
    load, save, lock = _registry_api()
    with lock():
        pending = load()
        acc = pending.setdefault(account_id, [])
        dup = next((r for r in acc if not r.get("closed")
                    and r.get("instrument_uid") == rec["instrument_uid"]), None)
        if dup:
            raise ValueError(
                f"{rec['ticker']}: в реестре уже есть активная запись "
                f"({dup.get('order_id')}, стратегия {dup.get('strategy')}) — "
                f"сначала закройте её (--forget) или дождитесь выхода")
        acc.append(rec)
        save(pending)


def forget(account_id: str, ticker: str) -> int:
    """Закрыть записи по тикеру. Условные заявки у брокера НЕ снимаются."""
    load, save, lock = _registry_api()
    with lock():
        pending = load()
        n = 0
        for r in pending.get(account_id, []):
            if r.get("ticker", "").upper() == ticker.upper() and not r.get("closed"):
                r["closed"] = True
                r["closed_reason"] = "forgotten"
                n += 1
        if n:
            save(pending)
        return n


# ── CLI ──────────────────────────────────────────────────────────────────────

def _broker(prod: bool):
    from services.place_orders import _make_broker_and_account
    broker, account_id, env = _make_broker_and_account(prod)
    if prod:
        from services.stage2_demo import contour_ok
        bad = contour_ok(env)
        if bad:
            raise SystemExit(f"контур не согласован: {bad}")
        want = str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip()
        if account_id != want:
            raise SystemExit(f"счёт брокера {account_id} ≠ PROD_ACCOUNT_ID {want}")
    return broker, account_id, env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m services.adopt_position",
        description="Взять уже открытую позицию под защиту PROTECT. "
                    "Заявку на вход НЕ выставляет.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sandbox", action="store_true", help="контур песочницы (по умолчанию)")
    g.add_argument("--prod", action="store_true", help="боевой счёт PROD_ACCOUNT_ID")
    p.add_argument("-t", "--ticker", help="тикер уже открытой позиции")
    p.add_argument("--stop", type=float, help="цена STOP_LOSS")
    p.add_argument("--take", type=float, default=None, help="цена TAKE_PROFIT (необязательно)")
    p.add_argument("--strategy", choices=STRATEGIES, default="manual",
                   help="manual — CLEANUP не трогает (по умолчанию); "
                        "intraday_* — CLEANUP закроет в конце сессии")
    p.add_argument("--order-id", default=None,
                   help="id входной заявки у брокера, если известен — тогда "
                        "заливка попадёт в журнал исполнения")
    p.add_argument("--list", action="store_true", help="показать активные записи реестра")
    p.add_argument("--forget", metavar="TICKER", help="закрыть записи по тикеру")
    p.add_argument("--dry-run", action="store_true", help="только показать запись")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждения")
    a = p.parse_args(argv)

    broker, account_id, env = _broker(a.prod)
    print(f"Контур {env}, счёт {account_id}\n")

    if a.list:
        recs = active_records(account_id)
        if not recs:
            print("Активных записей нет.")
            return 0
        for r in recs:
            print(describe(r) + f"\n  запись      {r.get('order_id')}"
                                f"{'  (ручная)' if r.get('adopted') else ''}\n")
        return 0

    if a.forget:
        n = forget(account_id, a.forget)
        print(f"Закрыто записей: {n}." if n else "Активных записей по тикеру нет.")
        if n:
            print("ВНИМАНИЕ: условные заявки у брокера не сняты — снимите их "
                  "сами, иначе оставшаяся нога при касании цены откроет "
                  "обратную позицию.")
        return 0

    if not a.ticker or a.stop is None:
        p.error("нужны --ticker и --stop (либо --list / --forget)")

    try:
        inst = broker.find_instrument(a.ticker)
    except BrokerError as e:
        print(f"[ERROR] {a.ticker}: {e}")
        return 1

    pos = next((x for x in broker.get_positions(account_id)
                if x.instrument_uid == inst.instrument_uid and x.is_open), None)
    if pos is None:
        print(f"[ERROR] {a.ticker}: открытой позиции нет. Команда берёт под защиту "
              f"уже открытую позицию — сначала откройте её.")
        return 1

    last = float(broker.get_last_price(inst.instrument_uid) or 0)
    errs = check_levels(balance_shares=pos.balance_shares, last=last,
                        stop=a.stop, take=a.take)
    if errs:
        for e in errs:
            print(f"[ERROR] {e}")
        return 1

    step = inst.min_price_increment
    rec = build_record(
        order_id=a.order_id or f"adopted-{uuid.uuid4()}",
        ticker=inst.ticker, instrument_uid=inst.instrument_uid, lot=inst.lot,
        balance_shares=pos.balance_shares,
        stop_q=Quotation.from_float(a.stop, step),
        tp_q=Quotation.from_float(a.take, step) if a.take is not None else None,
        strategy=a.strategy, journal=bool(a.order_id))

    print(describe(rec, last=last))
    if a.dry_run:
        print("\nDRY-RUN — в реестр ничего не записано.")
        return 0
    if not a.yes:
        if input("\nЗаписать в реестр? [y/N] ").strip().lower() not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return 1
    try:
        add_record(account_id, rec)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    print(f"\nЗапись добавлена. Ближайший прогон PROTECT поставит стоп"
          f"{' и цель' if rec['tp_units'] is not None else ''}. Проверить:\n"
          f"    python3 -m services.stage2_demo protect"
          f"{' --prod' if a.prod else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
