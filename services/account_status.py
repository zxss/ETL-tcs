"""
services/account_status.py — состояние счёта T-Invest (read-only).

Показывает: стоимость портфеля, свободные деньги, открытые позиции (с тикерами),
активные лимитные заявки и активные стоп-заявки. Ничего не меняет.

КОНТУР ПО УМОЛЧАНИЮ — PROD (боевой счёт PROD_ACCOUNT_ID). --sandbox для теста.

Запуск:
    python3 -m services.account_status                 # боевой счёт
    python3 -m services.account_status --sandbox       # песочница
    python3 -m services.account_status --account <id>  # конкретный счёт
    python3 -m services.account_status --json          # сырой JSON-снимок
"""
from __future__ import annotations

import argparse
import json
import sys

import config
from services.broker import TinkoffProdClient, TinkoffSandboxClient
from services.broker.base import Quotation


def _money(m: dict | None) -> float:
    return Quotation.from_payload(m).as_float() if m else 0.0


def _resolve_tickers(c, uids: list[str]) -> dict[str, str]:
    """instrument_uid → ticker (вызовов мало, кэш не нужен)."""
    out: dict[str, str] = {}
    for uid in uids:
        try:
            d = c._post("InstrumentsService/GetInstrumentBy",
                        {"idType": "INSTRUMENT_ID_TYPE_UID", "id": uid})
            out[uid] = d.get("instrument", {}).get("ticker", uid[:8])
        except Exception:  # noqa: BLE001 — не критично для просмотра
            out[uid] = uid[:8]
    return out


def snapshot(c, account_id: str, *, sandbox: bool) -> dict:
    """Сырой снимок счёта. Отличается только метод списка заявок:
    sandbox → SandboxService/GetSandboxOrders, prod → OrdersService/GetOrders.
    Портфель/позиции/стопы — общие сервисы (OperationsService/StopOrdersService)."""
    orders_method = "SandboxService/GetSandboxOrders" if sandbox else "OrdersService/GetOrders"
    return {
        "account_id": account_id,
        "sandbox":    sandbox,
        "portfolio": c._post("OperationsService/GetPortfolio", {"accountId": account_id}),
        "positions": c._post("OperationsService/GetPositions", {"accountId": account_id}),
        "orders":    c._post(orders_method, {"accountId": account_id}).get("orders", []),
        "stops":     c._post("StopOrdersService/GetStopOrders", {"accountId": account_id}).get("stopOrders", []),
    }


def print_status(c, snap: dict) -> None:
    acc, pf, pos = snap["account_id"], snap["portfolio"], snap["positions"]
    orders, stops = snap["orders"], snap["stops"]
    env = "SANDBOX" if snap["sandbox"] else "PROD (РЕАЛЬНЫЕ ДЕНЬГИ)"

    uids = {p.get("instrumentUid") for p in pos.get("securities", []) if p.get("instrumentUid")}
    uids |= {o.get("instrumentUid") for o in orders if o.get("instrumentUid")}
    uids |= {s.get("instrumentUid") for s in stops if s.get("instrumentUid")}
    tk = _resolve_tickers(c, [u for u in uids if u])

    print(f"\nСЧЁТ {env}: {acc}")
    print("=" * 64)
    print("Стоимость портфеля:")
    print(f"  Акции:   {_money(pf.get('totalAmountShares')):>14,.2f}")
    print(f"  Деньги:  {_money(pf.get('totalAmountCurrencies')):>14,.2f}")
    print(f"  ИТОГО:   {_money(pf.get('totalAmountPortfolio')):>14,.2f}")

    print("\nСвободные деньги:")
    for m in pos.get("money", []):
        print(f"  {m.get('currency', '?').upper()}: {Quotation.from_payload(m).as_float():,.2f}")

    secs = [p for p in pos.get("securities", []) if float(p.get("balance", 0) or 0)]
    print(f"\nПозиции по бумагам: {len(secs)}")
    for p in secs:
        uid = p.get("instrumentUid", "")
        bal = float(p.get("balance", 0) or 0)
        side = "LONG" if bal > 0 else "SHORT"
        print(f"  {tk.get(uid, uid[:8]):<6} {bal:+.0f} шт  ({side})")

    print(f"\nАктивные лимитные заявки: {len(orders)}")
    for o in orders:
        uid = o.get("instrumentUid", "")
        d = (o.get("direction", "") or "").replace("ORDER_DIRECTION_", "")
        st = (o.get("executionReportStatus", "") or "").replace("EXECUTION_REPORT_STATUS_", "")
        px = Quotation.from_payload(o.get("initialSecurityPrice")).as_float()
        print(f"  {tk.get(uid, uid[:8]):<6} {d:<4} {o.get('lotsRequested')} лот "
              f"@ {px:.4f}  [{st}]  id={o.get('orderId', '')[:8]}…")

    print(f"\nАктивные стоп-заявки: {len(stops)}")
    for s in stops:
        uid = s.get("instrumentUid", "")
        d = (s.get("direction", "") or "").replace("STOP_ORDER_DIRECTION_", "")
        sp = Quotation.from_payload(s.get("stopPrice")).as_float()
        print(f"  {tk.get(uid, uid[:8]):<6} {d:<4} {s.get('lotsRequested')} лот "
              f"стоп@ {sp:.4f}  id={s.get('stopOrderId', '')[:8]}…")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Состояние счёта T-Invest (read-only). "
                                            "По умолчанию — боевой счёт.")
    p.add_argument("--sandbox", action="store_true", help="тестовый контур")
    p.add_argument("--account", default="", help="ID счёта (по умолч. из config).")
    p.add_argument("--json", action="store_true", help="сырой JSON-снимок")
    args = p.parse_args(argv)

    if args.sandbox:
        c = TinkoffSandboxClient()
        acc = args.account or config.SANDBOX_ACCOUNT_ID
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
        if not acc:
            print("Не задан боевой счёт (PROD_ACCOUNT_ID).", file=sys.stderr)
            return 1

    snap = snapshot(c, acc, sandbox=args.sandbox)
    if args.json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        print_status(c, snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
