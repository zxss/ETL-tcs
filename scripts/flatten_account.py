"""
Закрыть позиции счёта по рынку — безопасно (замена /tmp/flatten.py, 26.09.2026).

Что было не так у прежнего скрипта. Он сначала снимал ВСЕ условные заявки, потом
пытался продать и при первой ошибке падал. 26.09 (суббота, биржа закрыта, бумаги в
режиме DEALER_NORMAL_TRADING) продажа не прошла, а стопы уже были сняты: позиции
боевого счёта остались на выходные без защиты, реестр об этом не знал.

Теперь:
  1. ПРЕДПРОВЕРКА. По каждой позиции — статус торгов у брокера. Бумага продаётся,
     только если идут БИРЖЕВЫЕ торги (SECURITY_TRADING_STATUS_NORMAL_TRADING) и
     рыночная заявка разрешена. Во внебиржевом режиме брокера (DEALER_*) рыночная
     заявка либо отклоняется, либо исполняется по котировке дилера — не трогаем.
     Нечего продавать → ничего не снимаем и выходим.
  2. ПО ОДНОЙ БУМАГЕ: снять её условные и лимитные заявки → продать по рынку →
     дождаться статуса. Ошибка по одной бумаге не останавливает остальные.
  3. РЕЕСТР. Закрываются только записи тех бумаг, которые реально ушли в ноль.
     У непроданных сбрасываются флаги защиты — PROTECT перевыставит стопы.
  4. Весь прогон под блокировкой реестра: PROTECT не вмешается посередине.

Запуск (боевой счёт — только сам пользователь):
    cd /opt/etl-tcs && set -a && . .env.tournament && set +a
    .venv/bin/python -m scripts.flatten_account --prod          # холостой
    .venv/bin/python -m scripts.flatten_account --prod --go     # отправить
Песочница: без --prod. Паи фонда (TMON) — только с --fund. Биржа закрыта —
только с --dealer (сделка с брокером по его котировке).
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

import config
from services.broker.base import BrokerError
from services.place_orders import (_load_pending, _make_broker_and_account,
                                   _save_pending, registry_lock)
from services.stage2_demo import contour_ok

EXCHANGE_OK = "SECURITY_TRADING_STATUS_NORMAL_TRADING"
DEALER_OK = "SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING"


def trading_status(br, uid: str, dealer: bool = False) -> dict:
    """Статус торгов по бумаге. Любая ошибка → «торговать нельзя».

    dealer=True разрешает и внебиржевой режим брокера (биржа закрыта, сделка с
    брокером по ЕГО котировке — обычно шире биржевого спреда). 26.09.2026 в таком
    режиме прошла рыночная продажа 1 шт AKRN по 17 364 ₽.
    """
    try:
        s = br._post("MarketDataService/GetTradingStatus", {"instrumentId": uid})
    except BrokerError as e:
        return {"ok": False, "why": f"статус недоступен: {e}"}
    status = s.get("tradingStatus") or "?"
    allowed = (EXCHANGE_OK, DEALER_OK) if dealer else (EXCHANGE_OK,)
    if status not in allowed:
        return {"ok": False, "why": status}
    if not s.get("marketOrderAvailableFlag") or not s.get("apiTradeAvailableFlag"):
        return {"ok": False, "why": "рыночная заявка через API недоступна"}
    return {"ok": True, "why": status}


def positions(br, acc: str, fund: bool) -> list[tuple]:
    fund_tk = str(getattr(config, "TREASURY_TICKER", "TMON")).upper().rstrip("@")
    out = []
    for p in br.get_positions(acc):
        if not p.is_open:
            continue
        inst = br.find_instrument_by_uid(p.instrument_uid)
        tk = (inst.ticker or "").upper()
        if tk.rstrip("@") == fund_tk and not fund:
            continue
        out.append((tk, inst, p.instrument_uid, p.balance_shares))
    return out


def wait_status(br, acc: str, order_id: str, timeout_s: float = 20.0) -> str:
    """Дождаться терминального статуса рыночной заявки (в тот же день он есть)."""
    t0, status = time.time(), "?"
    while time.time() - t0 < timeout_s:
        try:
            st = br.get_order_state(account_id=acc, order_id=order_id)
            status = st.execution_report_status
            if status.endswith(("_FILL", "_REJECTED", "_CANCELLED")):
                return status
        except BrokerError as e:
            status = f"статус недоступен: {e}"
        time.sleep(1.0)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Закрыть позиции счёта по рынку")
    ap.add_argument("--prod", action="store_true", help="боевой счёт (реальные деньги)")
    ap.add_argument("--go", action="store_true", help="отправить заявки (иначе холостой)")
    ap.add_argument("--fund", action="store_true", help="продать и паи фонда")
    ap.add_argument("--dealer", action="store_true",
                    help="разрешить внебиржевой режим брокера (биржа закрыта, цена — "
                         "котировка брокера, спред шире)")
    a = ap.parse_args(argv)

    br, acc, env = _make_broker_and_account(a.prod)
    bad = contour_ok(env)
    if bad:
        print(f"контур не согласован: {bad}")
        return 2
    if a.prod and acc != str(getattr(config, "PROD_ACCOUNT_ID", "")).strip():
        print("счёт не совпадает с PROD_ACCOUNT_ID — отказ")
        return 2

    with registry_lock():
        pos = positions(br, acc, a.fund)
        stops = br.get_active_stop_orders(acc)
        orders = br.get_active_orders(acc)
        print(f"контур {env} · позиций {len(pos)} · условных {len(stops)} · лимиток {len(orders)}")
        plan = []
        for tk, inst, uid, bal in pos:
            st = trading_status(br, uid, a.dealer)
            lots = int(abs(bal) // int(inst.lot or 1))
            mark = "✓" if st["ok"] and lots > 0 else "✗"
            print(f"  {mark} {tk:<6} {bal:+.0f} шт = {lots} лот · {st['why']}")
            if st["ok"] and lots > 0:
                plan.append((tk, inst, uid, bal, lots))
        if a.dealer:
            print("  ВНИМАНИЕ: --dealer — продажа по котировке брокера вне биржи, "
                  "цена может быть заметно хуже биржевой")
        if not plan:
            print("\nпродавать сейчас нечего или биржа закрыта — ничего не тронуто"
                  + ("" if a.dealer else " (вне биржи — только с --dealer)"))
            return 1 if pos else 0
        if not a.go:
            print(f"\nхолостой прогон: к продаже {len(plan)} из {len(pos)} (добавьте --go)")
            return 0

        sold_uids = set()
        for tk, inst, uid, bal, lots in plan:
            for s in [s for s in stops if s.instrument_uid == uid]:
                try:
                    br.cancel_stop_order(account_id=acc, stop_order_id=s.stop_order_id)
                    print(f"  {tk}: снята условная {s.kind}")
                except BrokerError as e:
                    print(f"  {tk}: условную {s.kind} снять не удалось: {e}")
            for o in [o for o in orders if getattr(o, "instrument_uid", None) == uid]:
                try:
                    br.cancel_order(account_id=acc, order_id=o.order_id)
                    print(f"  {tk}: снята лимитка {o.order_id[:8]}…")
                except BrokerError as e:
                    print(f"  {tk}: лимитку снять не удалось: {e}")
            side = "SELL" if bal > 0 else "BUY"
            try:
                st = br.post_market_order(account_id=acc, instrument=inst, direction=side,
                                          quantity_lots=lots, order_id=str(uuid.uuid4()))
                final = wait_status(br, acc, st.order_id)
                print(f"  {tk}: {side} {lots} лот → {final}")
            except BrokerError as e:
                print(f"  {tk}: ЗАЯВКА ОТКЛОНЕНА: {e}")

        # реестр: закрыть проданное, у остального сбросить защиту для PROTECT
        left = {uid: bal for _tk, _i, uid, bal in positions(br, acc, a.fund)}
        pending = _load_pending()
        closed = rearm = 0
        for r in pending.get(acc, []):
            if r.get("closed"):
                continue
            uid = r.get("instrument_uid")
            if uid in {p[2] for p in plan} and abs(left.get(uid, 0.0)) < 1e-9:
                r["closed"], r["closed_reason"] = True, "manual_close"
                sold_uids.add(uid)
                closed += 1
            elif uid in left:
                r["stop_placed"] = r["tp_placed"] = False
                r["stop_order_id"] = r["tp_order_id"] = None
                rearm += 1
        _save_pending(pending)
        print(f"\nреестр: закрыто {closed}, к перезащите PROTECT {rearm}")
        print(f"осталось позиций: {len(left)}")
        for uid, bal in left.items():
            print(f"  {uid[:8]}…: {bal:+.0f} шт")
    return 0


if __name__ == "__main__":
    sys.exit(main())
