"""
services/sandbox_status.py — состояние счёта в T-Invest Sandbox (read-only).

Показывает: стоимость портфеля, свободные деньги, открытые позиции (с тикерами),
активные лимитные заявки и активные стоп-заявки. Ничего не меняет.

Запуск:
    python3 -m services.sandbox_status                 # счёт из SANDBOX_ACCOUNT_ID
    python3 -m services.sandbox_status --account <id>  # конкретный счёт
    python3 -m services.sandbox_status --json          # сырой JSON-снимок
"""
from __future__ import annotations

import argparse
import json
import sys

import config
from services.broker import TinkoffSandboxClient
from services.broker.base import Quotation


def _money(m: dict | None) -> float:
    return Quotation.from_payload(m).as_float() if m else 0.0


def _resolve_tickers(c: TinkoffSandboxClient, uids: list[str]) -> dict[str, str]:
    """instrument_uid → ticker (одним вызовом на каждый, кэш не нужен — их мало)."""
    out: dict[str, str] = {}
    for uid in uids:
        try:
            d = c._post("InstrumentsService/GetInstrumentBy",
                        {"idType": "INSTRUMENT_ID_TYPE_UID", "id": uid})
            out[uid] = d.get("instrument", {}).get("ticker", uid[:8])
        except Exception:  # noqa: BLE001 — не критично для просмотра
            out[uid] = uid[:8]
    return out


def snapshot(c: TinkoffSandboxClient, account_id: str) -> dict:
    """Собрать сырой снимок счёта (для --json и для печати)."""
    return {
        "account_id": account_id,
        "portfolio": c._post("OperationsService/GetPortfolio", {"accountId": account_id}),
        "positions": c._post("OperationsService/GetPositions", {"accountId": account_id}),
        "orders":    c._post("SandboxService/GetSandboxOrders", {"accountId": account_id}).get("orders", []),
        "stops":     c._post("StopOrdersService/GetStopOrders", {"accountId": account_id}).get("stopOrders", []),
    }


def print_status(c: TinkoffSandboxClient, snap: dict) -> None:
    acc = snap["account_id"]
    pf  = snap["portfolio"]
    pos = snap["positions"]
    orders = snap["orders"]
    stops  = snap["stops"]

    # собрать все UID для расшифровки тикеров
    uids = {p.get("instrumentUid") for p in pos.get("securities", []) if p.get("instrumentUid")}
    uids |= {o.get("instrumentUid") for o in orders if o.get("instrumentUid")}
    uids |= {s.get("instrumentUid") for s in stops if s.get("instrumentUid")}
    tk = _resolve_tickers(c, [u for u in uids if u])

    print(f"\nСЧЁТ SANDBOX: {acc}")
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
    p = argparse.ArgumentParser(description="Состояние счёта T-Invest Sandbox (read-only)")
    p.add_argument("--account", default=config.SANDBOX_ACCOUNT_ID or "",
                   help="ID sandbox-счёта (по умолч. config.SANDBOX_ACCOUNT_ID).")
    p.add_argument("--json", action="store_true", help="сырой JSON-снимок")
    args = p.parse_args(argv)

    c = TinkoffSandboxClient()
    acc = args.account
    if not acc:
        # счёта не задано — берём первый существующий
        accs = c._post("SandboxService/GetSandboxAccounts").get("accounts", [])
        if not accs:
            print("Нет sandbox-счетов. Создайте через place_orders или укажите --account.",
                  file=sys.stderr)
            return 1
        acc = accs[0]["id"]
        print(f"(счёт не задан — взят первый: {acc})")

    snap = snapshot(c, acc)
    if args.json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        print_status(c, snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
