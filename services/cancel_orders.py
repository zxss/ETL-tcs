"""
services/cancel_orders.py — снятие активных заявок и стоп-заявок T-Invest.

Зачем: лимитки, выставленные глубоко в прогнозном коридоре (большой
--entry-frac), могут висеть днями, так и не залившись. Чтобы переставить их
ближе к рынку (меньший --entry-frac), сначала надо снять старые — иначе защита
от задвоения в place_orders их пропустит ([SKIP]). Этот скрипт снимает заявки
и/или стопы, по желанию с фильтром по тикеру.

Сначала ВСЕГДА печатает превью того, что будет снято, и спрашивает y/n
(на боевом счёте — с баннером). --dry-run только показывает, ничего не трогает.

КОНТУР ПО УМОЛЧАНИЮ — PROD (боевой счёт). --sandbox для теста.

Запуск:
    python3 -m services.cancel_orders --dry-run          # показать, что снялось бы
    python3 -m services.cancel_orders                    # снять ВСЁ (заявки + стопы)
    python3 -m services.cancel_orders --orders-only      # только лимитные заявки
    python3 -m services.cancel_orders --stops-only       # только стоп-заявки
    python3 -m services.cancel_orders --ticker ROSN      # только по одному тикеру
    python3 -m services.cancel_orders --sandbox          # песочница
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import config
from services.account_status import _resolve_tickers, snapshot
from services.broker import BrokerError, TinkoffProdClient, TinkoffSandboxClient
from services.broker.base import Quotation

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "order_log"
_PENDING_PATH = _LOG_DIR / "pending_stops.json"


# ── Журнал (тот же CSV-формат, что у place_orders) ────────────────────────────

_LOG_FIELDS = ["ts", "env", "account_id", "ticker", "direction", "action",
               "order_id", "qty_lots_api", "qty_shares", "price", "status", "info"]


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


# ── Сбор того, что снимать ────────────────────────────────────────────────────


def _collect(snap: dict, tk: dict, *, want_orders: bool, want_stops: bool,
             ticker: str) -> tuple[list[dict], list[dict]]:
    """Отфильтровать заявки/стопы по флагам и (опц.) тикеру. Возвращает
    (orders, stops) — списки сырых dict из снимка, готовые к снятию."""
    flt = ticker.upper().strip() if ticker else ""

    def _match(uid: str) -> bool:
        return not flt or tk.get(uid, "").upper() == flt

    orders = ([o for o in snap["orders"] if _match(o.get("instrumentUid", ""))]
              if want_orders else [])
    stops = ([s for s in snap["stops"] if _match(s.get("instrumentUid", ""))]
             if want_stops else [])
    return orders, stops


def _preview(orders: list[dict], stops: list[dict], tk: dict, env: str,
             account_id: str) -> None:
    if env == "PROD":
        print("\n" + "!" * 88)
        print(f"!!  БОЕВОЙ КОНТУР — РЕАЛЬНЫЕ ДЕНЬГИ. Счёт №{account_id}")
        print("!" * 88)
    else:
        print(f"\nКонтур: {env} | Счёт №{account_id}")

    print(f"\nК снятию — лимитные заявки: {len(orders)}")
    for o in orders:
        uid = o.get("instrumentUid", "")
        d = (o.get("direction", "") or "").replace("ORDER_DIRECTION_", "")
        px = Quotation.from_payload(o.get("initialSecurityPrice")).as_float()
        print(f"  {tk.get(uid, uid[:8]):<6} {d:<4} {o.get('lotsRequested')} лот "
              f"@ {px:.4f}  id={o.get('orderId', '')[:8]}…")

    print(f"\nК снятию — стоп-заявки: {len(stops)}")
    for s in stops:
        uid = s.get("instrumentUid", "")
        d = (s.get("direction", "") or "").replace("STOP_ORDER_DIRECTION_", "")
        sp = Quotation.from_payload(s.get("stopPrice")).as_float()
        print(f"  {tk.get(uid, uid[:8]):<6} {d:<4} {s.get('lotsRequested')} лот "
              f"стоп@ {sp:.4f}  id={s.get('stopOrderId', '')[:8]}…")


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


def _clear_registry(account_id: str) -> int:
    """Удалить записи аккаунта из реестра ожидающих стопов (после полного
    снятия — иначе attach-stops будет вечно ждать снятые лимитки). Возвращает
    число удалённых записей."""
    if not _PENDING_PATH.exists():
        return 0
    try:
        data = json.loads(_PENDING_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    n = len(data.get(account_id, []))
    if account_id in data:
        data[account_id] = []
        _PENDING_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return n


# ── Снятие ────────────────────────────────────────────────────────────────────


def cancel(broker, account_id: str, orders: list[dict], stops: list[dict], *,
           tk: dict, dry_run: bool, writer: csv.DictWriter, env: str) -> tuple[int, int]:
    """Снять отобранные заявки и стопы. Возвращает (снято_заявок, снято_стопов)."""
    n_ord = n_stp = 0

    for o in orders:
        uid = o.get("instrumentUid", "")
        oid = o.get("orderId", "")
        name = tk.get(uid, uid[:8])
        if dry_run:
            print(f"[DRY]   {name}: снять лимитку {oid[:8]}…")
            continue
        try:
            broker.cancel_order(account_id=account_id, order_id=oid)
            n_ord += 1
            print(f"[OK]    {name}: лимитка {oid[:8]}… снята.")
            _logrow(writer, env=env, account_id=account_id, ticker=name,
                    action="cancel_order", order_id=oid, status="cancelled")
        except BrokerError as e:
            print(f"[ERROR] {name}: не снять лимитку {oid[:8]}…: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=name,
                    action="cancel_order", order_id=oid, status="error", info=str(e))

    for s in stops:
        uid = s.get("instrumentUid", "")
        sid = s.get("stopOrderId", "")
        name = tk.get(uid, uid[:8])
        if dry_run:
            print(f"[DRY]   {name}: снять стоп {sid[:8]}…")
            continue
        try:
            broker.cancel_stop_order(account_id=account_id, stop_order_id=sid)
            n_stp += 1
            print(f"[OK]    {name}: стоп {sid[:8]}… снят.")
            _logrow(writer, env=env, account_id=account_id, ticker=name,
                    action="cancel_stop", order_id=sid, status="cancelled")
        except BrokerError as e:
            print(f"[ERROR] {name}: не снять стоп {sid[:8]}…: {e}")
            _logrow(writer, env=env, account_id=account_id, ticker=name,
                    action="cancel_stop", order_id=sid, status="error", info=str(e))

    return n_ord, n_stp


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Снятие активных заявок/стопов T-Invest. По умолчанию — "
                    "боевой счёт, снимаются И заявки, И стопы.")
    p.add_argument("--sandbox", action="store_true", help="тестовый контур")
    p.add_argument("--account", default="", help="ID счёта (по умолч. из config).")
    p.add_argument("--ticker", default="", help="снять только по этому тикеру.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--orders-only", action="store_true", help="только лимитные заявки")
    g.add_argument("--stops-only", action="store_true", help="только стоп-заявки")
    p.add_argument("--dry-run", action="store_true",
                   help="показать, что снялось бы, ничего не трогая")
    p.add_argument("--no-confirm", action="store_true", help="не спрашивать y/n (для cron)")
    args = p.parse_args(argv)

    want_orders = not args.stops_only
    want_stops = not args.orders_only

    if args.sandbox:
        c = TinkoffSandboxClient()
        acc = args.account or config.SANDBOX_ACCOUNT_ID
        env = "SANDBOX"
        if not acc:
            accs = c._post("SandboxService/GetSandboxAccounts").get("accounts", [])
            if not accs:
                print("Нет sandbox-счетов. Укажите --account.", file=sys.stderr)
                return 1
            acc = accs[0]["id"]
            print(f"(счёт не задан — взят первый sandbox: {acc})")
    else:
        c = TinkoffProdClient()
        acc = args.account or config.PROD_ACCOUNT_ID
        env = "PROD"
        if not acc:
            print("Не задан боевой счёт (PROD_ACCOUNT_ID).", file=sys.stderr)
            return 1

    try:
        snap = snapshot(c, acc, sandbox=args.sandbox)
    except BrokerError as e:
        print(f"[ERROR] Снимок счёта: {e}", file=sys.stderr)
        return 1

    uids = {o.get("instrumentUid") for o in snap["orders"] if o.get("instrumentUid")}
    uids |= {s.get("instrumentUid") for s in snap["stops"] if s.get("instrumentUid")}
    tk = _resolve_tickers(c, [u for u in uids if u])

    orders, stops = _collect(snap, tk, want_orders=want_orders,
                             want_stops=want_stops, ticker=args.ticker)
    _preview(orders, stops, tk, env, acc)

    if not orders and not stops:
        print("\nНечего снимать.")
        return 0

    if args.dry_run:
        print("\n[DRY-RUN] ничего не снято.")
        return 0

    prompt = ("\nСНЯТЬ перечисленное на БОЕВОМ счёте? (y/n): " if env == "PROD"
              else "\nСнять перечисленное? (y/n): ")
    if not args.no_confirm and not _confirm(prompt):
        print("[INFO] Отменено пользователем.")
        return 0

    writer, fp = _open_log()
    try:
        n_ord, n_stp = cancel(c, acc, orders, stops, tk=tk,
                              dry_run=False, writer=writer, env=env)
    finally:
        fp.close()

    print(f"\nСнято: лимиток {n_ord}, стопов {n_stp}.")

    # полное снятие (без фильтров) → почистить реестр ожидающих стопов,
    # иначе attach-stops будет вечно ждать уже снятые лимитки
    if want_orders and want_stops and not args.ticker:
        cleared = _clear_registry(acc)
        if cleared:
            print(f"Реестр ожидающих стопов очищен ({cleared} записей) — "
                  f"можно переставлять заявки заново.")

    print("Журнал: data/order_log/<дата>.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
