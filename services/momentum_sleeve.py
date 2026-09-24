"""
Моментум-рукав контура r4: форвард-проверка на ОТДЕЛЬНОМ счёте песочницы.

Зачем (24.09.2026). Кросс-секционный моментум — единственный сигнал программы,
который прошёл предрегистрированную отложенную выборку
(research/longhist/factors: 12-1 +11,5 %/год t 1,88; аутсайдеры месяца
−18,6 %/год t −2,42) и остался после издержек. Но в деньгах он НЕ доказан:
лонг-онли на той же выборке проиграл TMON 25 п.п., а хеджированный вариант
обошёл TMON лишь на 5,8 п.п. при t 0,85 (research/longhist/portfolio). Чистой
истории не осталось, поэтому единственный способ добрать доказательства —
проверка вперёд. Судить не раньше 12 ребалансов.

ПОЧЕМУ ОТДЕЛЬНЫЙ СЧЁТ. Фаза CLEANUP в 18:20 закрывает по рынку всё, кроме паёв
казначейства и открытых ночных записей реестра (square_off_intraday с
close_unregistered=True): «на 18:20 у контура не бывает законных позиций, кроме
интрадея». Месячные позиции на общем счёте были бы сметены в первый же вечер и
подняли бы ложную тревогу про позиции вне реестра. Уборка привязана к номеру
счёта, поэтому рукав торгует на своём sandbox-счёте — и НИ ОДНА строка торговой
логики r4 (stage2_demo, place_orders, предполётные проверки) не меняется.

ТОЛЬКО ПЕСОЧНИЦА. Боевого пути в этом модуле нет по построению: клиент только
TinkoffSandboxClient, флага --prod не существует, а счёт сверяется с
PROD_ACCOUNT_ID и со счётом самого r4 и отклоняется при совпадении.

Правило (заморожено 24.09.2026, до первого прогона):
  универс      топ-50 по среднему обороту за 60 дней (торговалась ≥ 50 из 60);
  сигнал       доходность d−252…d−21 (12-1), берём верхний квинтиль;
  фильтр       выкидываем нижний квинтиль доходности d−21…d (аутсайдеры месяца);
  портфель     MOMENTUM_NAMES бумаг равным весом, ребаланс раз в месяц;
  хедж         шорт ближнего фьючерса на индекс (IMOEX), нотионал ≈ лонг-ноге;
  эталон       TMON на том же капитале (пишется в журнал каждый ребаланс).

Запуск (сервер, из /opt/etl-tcs):
    python -m services.momentum_sleeve rebalance [--dry-run]
    python -m services.momentum_sleeve status
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

import config
import database
from services.broker.base import BrokerError, Instrument, Quotation
from services.broker.tinkoff_sandbox import TinkoffSandboxClient, new_order_id

log = logging.getLogger("services.momentum_sleeve")
MSK = dt.timezone(dt.timedelta(hours=3))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "audit", "momentum")
LEDGER = os.path.join(OUT_DIR, "ledger.jsonl")
ACCOUNT_FILE = os.path.join(OUT_DIR, "account.txt")   # номер счёта песочницы, не секрет

TOP_N, LIQ_WIN, LIQ_MIN = 50, 60, 50
LOOKBACK, SKIP = 252, 21
QUINT = 0.2
MAX_ABS_R = 0.40
BENCH_TICKER = "TMON"
FUT_MIN_DAYS = 14                       # ближе к экспирации не входим — роллим


def _cfg(name: str, default):
    v = os.getenv(name, "")
    if v == "":
        v = getattr(config, name, default)
    return type(default)(v) if not isinstance(v, type(default)) else v


def capital_rub() -> float:
    return float(_cfg("MOMENTUM_CAPITAL_RUB", 1_500_000.0))


def names_count() -> int:
    return int(_cfg("MOMENTUM_NAMES", 10))


def long_share() -> float:
    """Доля капитала в лонг-ногу; остальное — запас под ГО фьючерса."""
    return float(_cfg("MOMENTUM_LONG_SHARE", 0.8))


# ── Счёт: только песочница и только свой ─────────────────────────────────────

def open_account(broker: TinkoffSandboxClient) -> str:
    """Отдельный sandbox-счёт рукава. Совпадение со счётом r4 или с боевым — отказ."""
    pref = str(_cfg("MOMENTUM_ACCOUNT_ID", "")).strip()
    if not pref and os.path.exists(ACCOUNT_FILE):        # .env не трогаем: свой файл
        pref = open(ACCOUNT_FILE, encoding="utf-8").read().strip()
    prod = str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip()
    r4 = str(getattr(config, "SANDBOX_ACCOUNT_ID", "") or "").strip()
    if prod and pref == prod:
        raise RuntimeError("MOMENTUM_ACCOUNT_ID совпадает с боевым счётом — отказ")
    if r4 and pref == r4:
        raise RuntimeError("MOMENTUM_ACCOUNT_ID совпадает со счётом r4: уборка 18:20 "
                           "сметёт месячные позиции — нужен отдельный счёт")
    account_id = broker.open_or_get_account(preferred_id=pref)
    if account_id == prod:
        raise RuntimeError("получен боевой счёт — отказ")
    if r4 and account_id == r4:
        raise RuntimeError("получен счёт r4 — отказ")
    if not pref or account_id != pref:
        broker.pay_in(account_id, capital_rub(), currency="rub")
        log.warning("новый счёт песочницы %s пополнен на %.0f ₽ (счета песочницы живут "
                    "3 месяца от последнего обращения)", account_id, capital_rub())
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ACCOUNT_FILE, "w", encoding="utf-8") as f:
        f.write(account_id + "\n")
    return account_id


# ── Сигнал ───────────────────────────────────────────────────────────────────

SQL = """
SELECT ticker, date, close, volume FROM market_data
WHERE close > 0 AND date >= %(f)s AND date <= %(t)s ORDER BY ticker, date
"""


def targets(conn, day: dt.date) -> dict:
    """Целевой состав лонг-ноги на дату (правило заморожено в шапке модуля)."""
    df = pd.read_sql(SQL, conn, params={"f": day - dt.timedelta(days=430), "t": day})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("close", "volume"):
        df[c] = df[c].astype(float)
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    value = (close * df.pivot(index="date", columns="ticker", values="volume")).sort_index()
    if len(close) < LOOKBACK + 5:
        raise RuntimeError(f"истории мало: {len(close)} дней, нужно {LOOKBACK + 5}")
    ret = close.pct_change().where(lambda x: x.abs() <= MAX_ABS_R)
    cum = (1.0 + ret.fillna(0.0)).cumprod()
    liq = value.rolling(LIQ_WIN, min_periods=LIQ_MIN).mean().iloc[-1].dropna()
    uni = [t for t in liq.sort_values(ascending=False).index[:TOP_N]
           if pd.notna(close[t].iloc[-1])]
    k = max(1, int(round(len(uni) * QUINT)))
    mom = (cum[uni].iloc[-1 - SKIP] / cum[uni].iloc[-1 - LOOKBACK] - 1).dropna()
    m1 = (cum[uni].iloc[-1] / cum[uni].iloc[-1 - SKIP] - 1).dropna()
    losers = set(m1.sort_values().index[:k])
    ranked = [t for t in mom.sort_values(ascending=False).index if t not in losers]
    picks = ranked[:names_count()]
    return {"asof": str(close.index[-1]), "universe": len(uni), "excluded_losers": len(losers),
            "picks": picks, "momentum": {t: round(float(mom[t]) * 100, 2) for t in picks},
            "last_price": {t: float(close[t].iloc[-1]) for t in picks}}


# ── Фьючерс на индекс (в торговом контуре его резолвинга нет — только здесь) ──

def near_index_future(broker: TinkoffSandboxClient) -> Instrument:
    data = broker._post("InstrumentsService/Futures",
                        {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})
    today = dt.date.today()
    cand = []
    for f in data.get("instruments", []) or []:
        if f.get("basicAsset") != "IMOEX" or not f.get("apiTradeAvailableFlag"):
            continue
        exp = str(f.get("expirationDate") or "")[:10]
        if not exp:
            continue
        days = (dt.date.fromisoformat(exp) - today).days
        if days >= FUT_MIN_DAYS:
            cand.append((days, f))
    if not cand:
        raise RuntimeError("нет доступного фьючерса на индекс")
    _, f = min(cand)
    return Instrument(ticker=f["ticker"], instrument_uid=f["uid"], figi=f.get("figi", ""),
                      lot=int(f.get("lot", 1)),
                      min_price_increment=Quotation.from_payload(f.get("minPriceIncrement")),
                      currency=f.get("currency", "rub"),
                      trading_status=f.get("tradingStatus", ""),
                      api_trade_available=bool(f.get("apiTradeAvailableFlag")),
                      short_enabled=bool(f.get("shortEnabledFlag", True)),
                      class_code=f.get("classCode", "SPBFUT"))


# ── Ребаланс ─────────────────────────────────────────────────────────────────

def _px(broker, uid: str) -> float | None:
    try:
        return broker.get_last_price(uid)
    except BrokerError:
        return None


def plan_rebalance(broker, account_id: str, tgt: dict) -> dict:
    """Текущие позиции → список сделок до целевого портфеля. Заявок не шлёт."""
    fut = near_index_future(broker)
    fut_px = _px(broker, fut.instrument_uid) or 0.0
    per_name = capital_rub() * long_share() / max(1, len(tgt["picks"]))
    want: dict[str, dict] = {}
    for tk in tgt["picks"]:
        ins = broker.find_instrument(tk)
        px = _px(broker, ins.instrument_uid) or tgt["last_price"][tk]
        lots = int(per_name // (px * ins.lot)) if px > 0 else 0
        want[ins.instrument_uid] = {"ticker": tk, "ins": ins, "lots": lots, "price": px}
    long_notional = sum(w["lots"] * w["ins"].lot * w["price"] for w in want.values())
    fut_lots = int(round(long_notional / fut_px)) if fut_px > 0 else 0
    want[fut.instrument_uid] = {"ticker": fut.ticker, "ins": fut, "lots": -fut_lots,
                                "price": fut_px}

    have = {p.instrument_uid: p.balance_shares for p in broker.get_positions(account_id)
            if p.is_open}
    trades = []
    for uid, w in want.items():
        cur_lots = int(round(have.pop(uid, 0.0) / w["ins"].lot))
        delta = w["lots"] - cur_lots
        if delta:
            trades.append({"ticker": w["ticker"], "uid": uid, "ins": w["ins"],
                           "direction": "BUY" if delta > 0 else "SELL",
                           "lots": abs(delta), "price": w["price"],
                           "target_lots": w["lots"], "current_lots": cur_lots})
    for uid, bal in have.items():                       # вышли из состава — закрыть
        if abs(bal) < 1e-9:
            continue
        try:
            ins = broker.find_instrument_by_uid(uid)
        except BrokerError as e:
            log.warning("не опознан инструмент %s: %s — пропуск", uid[:8], e)
            continue
        lots = int(round(abs(bal) / ins.lot))
        if lots:
            trades.append({"ticker": ins.ticker, "uid": uid, "ins": ins,
                           "direction": "SELL" if bal > 0 else "BUY", "lots": lots,
                           "price": _px(broker, uid) or 0.0,
                           "target_lots": 0, "current_lots": int(round(bal / ins.lot))})
    return {"future": fut.ticker, "future_price": fut_px, "future_lots": -fut_lots,
            "long_notional_rub": round(long_notional, 2), "per_name_rub": round(per_name, 2),
            "trades": trades}


def bench_price(broker) -> float | None:
    """Цена эталона. TMON через ShareBy не ищется: тот же фонд торгуется как
    TMON@ на СПБ — берём листинг так же, как это делает казначейство."""
    try:
        from services.treasury import choose_listing
        item = choose_listing(broker.find_instrument_listings(BENCH_TICKER), BENCH_TICKER)
        return _px(broker, item["uid"]) if item and item.get("uid") else None
    except Exception as e:                       # noqa: BLE001
        log.warning("эталон %s недоступен: %s", BENCH_TICKER, e)
        return None


def equity(broker, account_id: str) -> dict:
    """Оценка капитала рукава: деньги + рыночная стоимость позиций."""
    money = broker.get_money_rub(account_id)
    total, rows = money, []
    for p in broker.get_positions(account_id):
        if not p.is_open:
            continue
        px = _px(broker, p.instrument_uid) or 0.0
        val = p.balance_shares * px
        total += val
        rows.append({"uid": p.instrument_uid, "shares": p.balance_shares,
                     "price": px, "value_rub": round(val, 2)})
    return {"money_rub": round(money, 2), "positions": rows, "equity_rub": round(total, 2)}


def run_rebalance(dry_run: bool) -> int:
    broker = TinkoffSandboxClient()
    account_id = open_account(broker)
    conn = database.get_connection()
    try:
        tgt = targets(conn, dt.date.today())
    finally:
        conn.close()
    plan = plan_rebalance(broker, account_id, tgt)
    before = equity(broker, account_id)
    run_id = dt.datetime.now(MSK).strftime("MOMENTUM-%Y%m%d-%H%M%S")
    run_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log.info("[MOMENTUM] счёт %s, состав: %s", account_id, ", ".join(tgt["picks"]))
    log.info("[MOMENTUM] лонг %.0f ₽, хедж %s %d конт. по %.0f",
             plan["long_notional_rub"], plan["future"], plan["future_lots"], plan["future_price"])
    report = []
    for t in plan["trades"]:
        line = {k: t[k] for k in ("ticker", "direction", "lots", "current_lots", "target_lots")}
        if dry_run:
            line["status"] = "dry-run"
        else:
            try:
                st = broker.post_market_order(account_id=account_id, instrument=t["ins"],
                                              direction=t["direction"],
                                              quantity_lots=t["lots"],
                                              order_id=new_order_id())
                line["status"] = st.execution_report_status or "posted"
                line["order_id"] = st.order_id
                line["lots_executed"] = st.lots_executed
                line["executed_price"] = st.executed_price
            except BrokerError as e:
                line["status"] = f"error: {e}"
                log.error("[MOMENTUM] %s %s x%d: %s", t["direction"], t["ticker"], t["lots"], e)
        report.append(line)
        log.info("[MOMENTUM] %-6s %-5s %3d лот → %s", t["direction"], t["ticker"],
                 t["lots"], line["status"])

    after = equity(broker, account_id) if not dry_run else before
    bench_px = bench_price(broker)
    rec = {"run_id": run_id, "ts": dt.datetime.now(MSK).isoformat(timespec="seconds"),
           "account_id": account_id, "dry_run": dry_run, "picks": tgt["picks"],
           "asof": tgt["asof"], "future": plan["future"], "future_lots": plan["future_lots"],
           "long_notional_rub": plan["long_notional_rub"],
           "equity_before_rub": before["equity_rub"], "equity_after_rub": after["equity_rub"],
           "bench_ticker": BENCH_TICKER, "bench_price": bench_px,
           "trades": len(report), "errors": sum(1 for r in report if str(r["status"]).startswith("error"))}
    _write(os.path.join(run_dir, "plan.json"),
           {"targets": tgt, "future": plan["future"], "future_lots": plan["future_lots"],
            "per_name_rub": plan["per_name_rub"],
            "trades": [{k: v for k, v in t.items() if k != "ins"} for t in plan["trades"]]})
    _write(os.path.join(run_dir, "result.json"), {**rec, "report": report, "after": after})
    if not dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _notify(rec, report)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 1 if rec["errors"] else 0


def _write(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def _notify(rec: dict, report: list) -> None:
    try:
        from services import notify
    except Exception:                            # noqa: BLE001
        return
    n = len(rec["picks"])
    lines = [f"<b>Моментум-рукав</b> (песочница, отдельный счёт)",
             f"ребаланс {rec['run_id']}, бумаг {n}, сделок {rec['trades']}"
             + (f", ОШИБОК {rec['errors']}" if rec["errors"] else ""),
             f"состав: {', '.join(rec['picks'])}",
             f"хедж: {rec['future']} {rec['future_lots']} конт.",
             f"капитал {rec['equity_after_rub']:,.0f} ₽".replace(",", " ")]
    try:
        notify.send("\n".join(lines))
    except Exception as e:                       # noqa: BLE001
        log.warning("уведомление не ушло: %s", e)


def run_status() -> int:
    broker = TinkoffSandboxClient()
    account_id = open_account(broker)
    eq = equity(broker, account_id)
    hist = []
    if os.path.exists(LEDGER):
        hist = [json.loads(x) for x in open(LEDGER, encoding="utf-8") if x.strip()]
    out = {"account_id": account_id, **eq, "rebalances": len(hist),
           "first": hist[0]["ts"] if hist else None, "last": hist[-1]["ts"] if hist else None}
    if hist:
        e0 = hist[0]["equity_after_rub"]
        out["equity_change_pct"] = round((eq["equity_rub"] / e0 - 1) * 100, 2) if e0 else None
        b0, b1 = hist[0].get("bench_price"), hist[-1].get("bench_price")
        if b0 and b1:
            out["bench_change_pct"] = round((b1 / b0 - 1) * 100, 2)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Моментум-рукав r4 (только песочница)")
    ap.add_argument("command", choices=["rebalance", "status"])
    ap.add_argument("--dry-run", action="store_true", help="посчитать и показать сделки, не отправляя")
    a = ap.parse_args()
    if str(getattr(config, "TRADING_MODE", "sandbox")).strip().lower() == "prod":
        log.warning("TRADING_MODE=prod — рукав всё равно идёт в песочницу (боевого пути нет)")
    return run_rebalance(a.dry_run) if a.command == "rebalance" else run_status()


if __name__ == "__main__":
    raise SystemExit(main())
