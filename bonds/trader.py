"""
Поток «портфель ВДО» — живая система в ПЕСОЧНИЦЕ T-Invest.

Что подаём:  ISS (срез рынка, ликвидность за 10 торговых дней), API брокера (торгуемость,
             флаги, рейтинги), smart-lab (рейтинг, эмитент), ключевую ставку ЦБ,
             правила — последняя bonds/rules_wNN.json.
Что запускаем:
  python -m bonds.trader rebalance [--dry-run] [--if-due]   раз в месяц: целевой портфель
                                                            → заявки (продажи, затем покупки)
  python -m bonds.trader monitor [--dry-run]                каждый день: выходы (стоп, рейтинг,
                                                            остановка, оферта) и дозаявка цели
  python -m bonds.trader status                             стоимость, структура, лимиты
  python -m bonds.trader flags --out flags.json              флаги брокера для бэктеста
Что получаем:  <BONDS_DIR>/target_YYYY-MM_wNN.json (цель месяца, версии не перезаписываются),
               ledger.jsonl (каждый запуск: сделки, проверки), account.txt (счёт песочницы).

ТОЛЬКО ПЕСОЧНИЦА: клиент TinkoffSandboxClient, боевого пути нет; чужие счета отклоняются.
Сделки — лимитные, только при биржевых торгах (SECURITY_TRADING_STATUS_NORMAL_TRADING),
цена облигации в API — в процентах номинала (сверено с ISS 27.09.2026).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config                                                          # noqa: E402
from bonds.core import (build_portfolio, check_portfolio, exit_signals,  # noqa: E402
                        load_rules, rating_name, worst_rating)
from bonds.sources import iss, smartlab, tinvest                       # noqa: E402
from services.broker.base import BrokerError, Instrument, Quotation    # noqa: E402
from services.broker.tinkoff_sandbox import TinkoffSandboxClient, new_order_id  # noqa: E402
from tools.flowkit import Checks, versioned                            # noqa: E402

log = logging.getLogger("bonds")
MSK = dt.timezone(dt.timedelta(hours=3))
DATA = os.getenv("BONDS_DIR", os.path.join(ROOT, "data", "bonds"))
CACHE = os.path.join(DATA, "cache")
LEDGER = os.path.join(DATA, "ledger.jsonl")
ACCOUNT_FILE = os.path.join(DATA, "account.txt")
CAPITAL_RUB = 1_000_000.0
# счета других контуров песочницы — модуль облигаций их не трогает
FOREIGN_ACCOUNT_FILES = ("/opt/etl-r4/audit/momentum/account.txt",
                         "/opt/research-data/levmom/account.txt")
LIQ_DAYS = 10
EXCHANGE_OK = "SECURITY_TRADING_STATUS_NORMAL_TRADING"


def _jsonl(rec: dict) -> None:
    os.makedirs(DATA, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ── счёт ─────────────────────────────────────────────────────────────────────

def open_account(br) -> str:
    pref = open(ACCOUNT_FILE, encoding="utf-8").read().strip() if os.path.exists(ACCOUNT_FILE) else ""
    forbidden = {str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip(),
                 str(getattr(config, "SANDBOX_ACCOUNT_ID", "") or "").strip()}
    for path in FOREIGN_ACCOUNT_FILES:
        if os.path.exists(path):
            forbidden.add(open(path, encoding="utf-8").read().strip())
    forbidden.discard("")
    if pref in forbidden:
        raise RuntimeError("счёт модуля облигаций совпадает с чужим счётом — отказ")
    acc = br.open_or_get_account(preferred_id=pref)
    if acc in forbidden:
        raise RuntimeError("получен чужой счёт — отказ")
    if acc != pref:
        br.pay_in(acc, CAPITAL_RUB, currency="rub")
        os.makedirs(DATA, exist_ok=True)
        with open(ACCOUNT_FILE, "w", encoding="utf-8") as f:  # flow-state: номер счёта
            f.write(acc + "\n")
        log.warning("новый счёт песочницы %s, пополнен на %.0f ₽", acc, CAPITAL_RUB)
    return acc


# ── данные ───────────────────────────────────────────────────────────────────

def key_rate(day: dt.date) -> float:
    try:
        import database
        _, r = database.get_key_rate(day)
        if r:
            return float(r)
    except Exception:  # noqa: BLE001
        pass
    from services.load_macro import fetch
    rows = fetch(day - dt.timedelta(days=120), day)
    return float(sorted(rows, key=lambda x: x["date"])[-1]["key_rate_pct"])


def liquidity(today: dt.date) -> dict[str, float]:
    """Медиана дневного оборота за LIQ_DAYS торговых дней (история ISS, кэш по датам)."""
    days = [d for d in iss.trading_days(today - dt.timedelta(days=30), today - dt.timedelta(days=1), CACHE)]
    vals: dict[str, list] = {}
    for d in days[-LIQ_DAYS:]:
        for r in iss.history_on(d, CACHE):
            vals.setdefault(r["SECID"], []).append(r.get("VALUE") or 0)
    return {k: statistics.median(v + [0] * (LIQ_DAYS - len(v))) for k, v in vals.items()}


def universe(br, today: dt.date) -> tuple[list[dict], dict]:
    snap = iss.snapshot()
    tf = tinvest.flags(br)
    try:
        sl = smartlab.ratings(CACHE)
    except OSError as e:
        log.warning("smart-lab недоступен (%s) — только рейтинги брокера", e)
        sl = {}
    liq = liquidity(today)
    bonds = []
    for s in snap:
        f = tf.get(s["isin"] or s["secid"])
        if not f or s.get("face_unit") not in ("SUR", "RUB"):
            continue
        slr = sl.get(s["isin"], {})
        texts = [r.split(":", 1)[-1] for r in f["ratings"]] + ([slr["rating"]] if slr.get("rating") else [])
        bonds.append({**s, **{k: f[k] for k in ("qual_only", "required_tests", "floating", "subordinated",
                                                  "perpetual", "sector", "negative_watch", "uid", "figi",
                                                  "lot", "nominal", "min_step", "sell_ok")},
                      "issue_size_rub": f.get("issue_size_rub") or s.get("issue_size_rub"),
                      "tradable": f["tradable"], "rating_num": worst_rating(texts),
                      "ratings": texts, "issuer": slr.get("issuer") or _issuer(s.get("name")),
                      "value_rub": liq.get(s["secid"], 0.0)})
    return bonds, {"snapshot": len(snap), "with_flags": len(bonds), "smartlab": len(sl)}


def _issuer(name) -> str:
    import re
    m = re.match(r"[^\d]+", str(name or ""))
    return (m.group(0) if m else str(name)).strip(" -").lower()


# ── позиции и стоимость ──────────────────────────────────────────────────────

def money_total(br, acc: str) -> float:
    data = br._post("SandboxService/GetSandboxPositions", {"accountId": acc})
    return sum(Quotation.from_payload(m).as_float() for key in ("money", "blocked")
               for m in (data.get(key) or []) if m.get("currency") == "rub")


def holdings(br, acc: str) -> dict[str, float]:
    data = br._post("SandboxService/GetSandboxPositions", {"accountId": acc})
    return {it["instrumentUid"]: float(it.get("balance", 0) or 0)
            for it in (data.get("securities") or []) if float(it.get("balance", 0) or 0) > 0}


def unit_cost(b: dict) -> float:
    """₽ за одну облигацию: цена (% номинала) × номинал + НКД."""
    return (b.get("price_pct") or 0) / 100 * (b.get("nominal") or 1000) + (b.get("accrued") or 0)


def nav(br, acc: str, by_uid: dict) -> dict:
    money = money_total(br, acc)
    pos = holdings(br, acc)
    vals = {uid: qty * unit_cost(by_uid[uid]) for uid, qty in pos.items() if uid in by_uid}
    return {"money": money, "positions": pos, "values": vals, "nav": money + sum(vals.values()),
            "unknown": [u for u in pos if u not in by_uid]}


# ── заявки ───────────────────────────────────────────────────────────────────

def _instrument(b: dict) -> Instrument:
    return Instrument(ticker=b["secid"], instrument_uid=b["uid"], figi=b.get("figi") or "",
                      lot=int(b.get("lot") or 1),
                      min_price_increment=Quotation.from_float(b.get("min_step") or 0.01),
                      currency="rub", trading_status="", api_trade_available=True,
                      short_enabled=False, class_code="TQCB")


def _limit_price(br, b: dict, side: str, slip_pct: float) -> float | None:
    """Покупка — по лучшему ask, но не дороже last×(1+slip); продажа — зеркально."""
    q = lambda x: int(x.get("units", 0)) + int(x.get("nano", 0)) / 1e9 if x else None
    try:
        ob = br._post("MarketDataService/GetOrderBook", {"instrumentId": b["uid"], "depth": 1})
        st = br._post("MarketDataService/GetTradingStatus", {"instrumentId": b["uid"]})
    except BrokerError:
        return None
    if st.get("tradingStatus") != EXCHANGE_OK or not st.get("limitOrderAvailableFlag"):
        return None
    last = b.get("price_pct")
    if side == "BUY":
        cands = [x for x in (q((ob.get("asks") or [{}])[0].get("price")),
                             last * (1 + slip_pct / 100) if last else None) if x]
        p = min(cands) if cands else None
    else:
        cands = [x for x in (q((ob.get("bids") or [{}])[0].get("price")),
                             last * (1 - slip_pct / 100) if last else None) if x]
        p = max(cands) if cands else None
    if not p:
        return None
    step = b.get("min_step") or 0.01
    return round(math.floor(p / step + 1e-9) * step if side == "BUY" else math.ceil(p / step - 1e-9) * step, 6)


def place(br, acc: str, b: dict, side: str, lots: int, rules: dict, dry: bool) -> dict:
    px = _limit_price(br, b, side, rules["execution"]["limit_slippage_pct"])
    rec = {"secid": b["secid"], "side": side, "lots": lots, "price_pct": px}
    if px is None:
        return {**rec, "status": "нет биржевых торгов — отложено"}
    if dry:
        return {**rec, "status": "DRY"}
    try:
        st = br.post_limit_order(account_id=acc, instrument=_instrument(b), direction=side,
                                 quantity_lots=lots, price=Quotation.from_float(px),
                                 order_id=new_order_id())
        return {**rec, "status": st.execution_report_status}
    except BrokerError as e:
        return {**rec, "status": f"ОШИБКА {e}"}


def trade_to_target(br, acc: str, target: dict, by_uid: dict, by_secid: dict, rules: dict,
                    dry: bool, exits: dict | None = None) -> list:
    """Разница с целью → заявки: сперва продажи (деньги под покупки), потом покупки.
    Активные заявки снимаются перед расчётом — пересчёт идёт от факта позиций."""
    if not dry:
        for o in br.get_active_orders(acc):
            try:
                br.cancel_order(account_id=acc, order_id=o.order_id)
            except BrokerError:
                pass
    state = nav(br, acc, by_uid)
    want_lots = {}
    for secid, w in target["weights"].items():
        b = by_secid.get(secid)
        if not b or (exits and secid in exits):
            continue
        lot_cost = unit_cost(b) * int(b.get("lot") or 1)
        want_lots[b["uid"]] = int(w * state["nav"] // lot_cost) if lot_cost > 0 else 0
    orders = []
    for uid, qty in state["positions"].items():
        b = by_uid.get(uid)
        if not b:
            continue
        cur = int(qty // int(b.get("lot") or 1))
        d = want_lots.get(uid, 0) - cur
        if d < 0:
            orders.append(place(br, acc, b, "SELL", -d, rules, dry))
    for uid, lots in want_lots.items():
        cur = int(state["positions"].get(uid, 0) // int(by_uid[uid].get("lot") or 1))
        if lots - cur > 0:
            orders.append(place(br, acc, by_uid[uid], "BUY", lots - cur, rules, dry))
    return orders


# ── команды ──────────────────────────────────────────────────────────────────

def latest_target(month: str | None = None) -> dict | None:
    pat = f"target_{month}_w*.json" if month else "target_*_w*.json"
    files = sorted(glob.glob(os.path.join(DATA, pat)))
    return json.load(open(files[-1], encoding="utf-8")) if files else None


def cmd_rebalance(dry: bool, if_due: bool) -> int:
    now = dt.datetime.now(MSK)
    month = now.strftime("%Y-%m")
    if if_due and latest_target(month):
        log.info("цель на %s уже есть — ребаланс месяца сделан", month)
        return 0
    rules = load_rules()
    br = TinkoffSandboxClient()
    acc = open_account(br)
    bonds, meta = universe(br, now.date())
    by_uid = {b["uid"]: b for b in bonds}
    by_secid = {b["secid"]: b for b in bonds}
    key = key_rate(now.date())
    held = {by_uid[u]["secid"] for u in holdings(br, acc) if u in by_uid}
    res = build_portfolio(bonds, rules, key, use_credit=True, holdings=held)
    chk = Checks("bonds-rebalance")
    chk.eq("ограничения цели соблюдены", res["violations"], [])
    chk.ok("вес цели ≤ 1 − резерв", sum(p["weight"] for p in res["portfolio"])
           <= 1 - rules["portfolio"]["cash_reserve"] + 1e-9)
    target = {"month": month, "rules": rules["_path"], "key_rate": key, "created": now.isoformat(),
              "universe": meta, "eligible": res["eligible"], "excluded": res["excluded"],
              "weights": {p["secid"]: p["weight"] for p in res["portfolio"]},
              "picks": [{k: p.get(k) for k in ("secid", "name", "issuer", "sector", "yield_pct",
                                                "duration_y", "price_pct", "days_to_event", "score")}
                        | {"rating": rating_name(p.get("rating_num"))} for p in res["portfolio"]],
              "checks": chk.as_dict()}
    if chk.exit_code():
        print(chk.report())
        return 1
    orders = trade_to_target(br, acc, target, by_uid, by_secid, rules, dry)
    if not dry:
        path = versioned(DATA, f"target_{month}", "json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(target, f, ensure_ascii=False, indent=1, default=str)
        _jsonl({"ts": now.isoformat(), "kind": "rebalance", "target": os.path.basename(path),
                "orders": orders})
    report(target, orders, dry)
    return 0


def month_exits(month: str) -> dict:
    """Выходы, уже сделанные в этом месяце: выпуск не докупается до следующего ребаланса,
    даже если цена вернулась выше стопа (иначе стоп превращается в пилу)."""
    out = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") == "monitor" and str(r.get("ts", "")).startswith(month):
                out.update(r.get("exits") or {})
    return out


def report(target: dict, orders: list, dry: bool) -> None:
    from collections import Counter
    picks = target["picks"]
    print(f"{target['month']}: ставка {target['key_rate']} %, вселенная {target['universe']}, "
          f"прошли фильтры {target['eligible']}, в портфеле {len(picks)}"
          + (" (ПРОБНЫЙ ПРОГОН, заявки не отправлялись)" if dry else ""))
    print("исключены:", ", ".join(f"{k} {v}" for k, v in sorted(target["excluded"].items(), key=lambda x: -x[1])))
    for p in picks:
        print(f"  {p['secid']}  {str(p['name'])[:14]:<14} {p['rating']:<5} доходн. {p['yield_pct']:5.1f} %  "
              f"дюр. {p['duration_y']:.2f}  цена {p['price_pct']:6.2f}  {p['issuer']} / {p['sector']}")
    if picks:
        n = len(picks)
        print(f"средняя доходность {sum(p['yield_pct'] for p in picks) / n:.1f} %, "
              f"дюрация {sum(p['duration_y'] for p in picks) / n:.2f}; сектора "
              + ", ".join(f"{k} {v}" for k, v in Counter(p["sector"] for p in picks).most_common()))
    print(f"заявок {len(orders)}:", dict(Counter(o["status"] for o in orders)))


def cmd_monitor(dry: bool) -> int:
    now = dt.datetime.now(MSK)
    rules = load_rules()
    br = TinkoffSandboxClient()
    acc = open_account(br)
    bonds, _ = universe(br, now.date())
    by_uid = {b["uid"]: b for b in bonds}
    by_secid = {b["secid"]: b for b in bonds}
    key = key_rate(now.date())
    exits = month_exits(now.strftime("%Y-%m"))
    for uid in holdings(br, acc):
        b = by_uid.get(uid)
        if b:
            why = exit_signals(b, rules, key, use_credit=True)
            if why:
                exits[b["secid"]] = why
    target = latest_target()
    orders = trade_to_target(br, acc, target, by_uid, by_secid, rules, dry, exits) if target else []
    rec = {"ts": now.isoformat(), "kind": "monitor", "exits": exits, "orders": orders}
    if not dry:
        _jsonl(rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str)[:4000])
    return 0


def cmd_status() -> int:
    rules = load_rules()
    br = TinkoffSandboxClient()
    acc = open_account(br)
    bonds, _ = universe(br, dt.date.today())
    by_uid = {b["uid"]: b for b in bonds}
    st = nav(br, acc, by_uid)
    port = []
    for uid, v in st["values"].items():
        b = by_uid[uid]
        port.append({"secid": b["secid"], "issuer": b["issuer"], "sector": b.get("sector"),
                     "weight": v / st["nav"] if st["nav"] else 0, "duration_y": b.get("duration_y") or 0,
                     "rating": rating_name(b.get("rating_num")), "yield_pct": b.get("yield_pct")})
    print(f"счёт {acc}: стоимость {st['nav']:,.0f} ₽ ({(st['nav']/CAPITAL_RUB-1)*100:+.2f} %), "
          f"деньги {st['money']:,.0f} ₽, выпусков {len(port)}".replace(",", " "))
    for p in sorted(port, key=lambda x: -x["weight"]):
        print(f"  {p['secid']}  {p['weight']*100:5.1f} %  {p['rating']:<5} доходн. {p['yield_pct'] or 0:5.1f} %  "
              f"дюр. {p['duration_y']:.2f}  {p['issuer']} / {p['sector']}")
    bad = check_portfolio(port, rules)
    print("ограничения:", "соблюдены" if not bad else "; ".join(bad))
    if st["unknown"]:
        print("позиции вне вселенной (проверить вручную):", st["unknown"])
    return 0


def cmd_flags(out: str) -> int:
    br = TinkoffSandboxClient()
    tf = tinvest.flags(br)
    try:
        sl = smartlab.ratings(CACHE)
    except OSError:
        sl = {}
    for isin, f in tf.items():
        if sl.get(isin, {}).get("issuer"):
            f["issuer"] = sl[isin]["issuer"]
    with open(out, "w", encoding="utf-8") as f:  # flow-state: выгрузка флагов для бэктеста
        json.dump(tf, f, ensure_ascii=False)
    print(f"флаги брокера: {len(tf)} выпусков → {out}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rebalance", "monitor", "status", "flags"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--if-due", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "flags.json"))
    a = ap.parse_args(argv)
    if a.cmd == "rebalance":
        return cmd_rebalance(a.dry_run, a.if_due)
    if a.cmd == "monitor":
        return cmd_monitor(a.dry_run)
    if a.cmd == "flags":
        return cmd_flags(a.out)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())
