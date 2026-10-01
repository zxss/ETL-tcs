"""
Моментум с плечом на фьючерсах — форвард-тест в ПЕСОЧНИЦЕ (запрос 27.09.2026:
«найди с плечом максимальную доходность, запусти тест в песочнице»).

Стратегия (research/levmom/backtest.py, 2015-05…2026-09):
  * вселенная — 35 бумаг робота, у которых есть фьючерс на FORTS;
  * сигнал — доходность 12-1 (d−252…d−21, с дивидендами), топ-10 равными весами;
  * лонг — ФЬЮЧЕРСЫ на эти 10 бумаг, шорт — мини-фьючерс на индекс (тот же нотионал);
  * плечо L = 2,25 — максимум, при котором ГО занимает ≤ 75 % капитала и худший
    исторический месяц (−7,4 % на 1×) не приводит к маржин-коллу;
  * свободные деньги сверх ГО и запаса — в TMON (фонд денежного рынка);
  * ребаланс раз в месяц, ролл за FUT_MIN_DAYS до экспирации, ежедневная проверка
    загрузки ГО с разгрузкой.
Бэктест при L = 2,25 — около 26 % годовых на всей истории, но с 06.2024 при ставке
~19 % стратегия ПРОИГРЫВАЛА фонду (спред сдулся, ГО блокирует деньги). Это тест
максимальной доходности, а не подтверждённая стратегия.

ТОЛЬКО ПЕСОЧНИЦА: клиент TinkoffSandboxClient, боевого пути нет; свой счёт,
который сверяется с боевым, r4 и счётом старого рукава.

Запуск (сервер, из /opt/research-vrp, код брокера — из /opt/etl-tcs):
    python -m research.levmom.sleeve rebalance [--dry-run] [--if-due]
    python -m research.levmom.sleeve check [--dry-run]
    python -m research.levmom.sleeve status
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys

sys.path.insert(1, "/opt/etl-tcs")

import config                                              # noqa: E402
import database                                            # noqa: E402
from services.broker.base import BrokerError, Instrument, Quotation  # noqa: E402
from services.broker.tinkoff_sandbox import TinkoffSandboxClient, new_order_id  # noqa: E402

log = logging.getLogger("levmom")
MSK = dt.timezone(dt.timedelta(hours=3))
OUT_DIR = os.getenv("LEVMOM_DIR", "/opt/research-data/levmom")
LEDGER = os.path.join(OUT_DIR, "ledger.jsonl")
DAILY = os.path.join(OUT_DIR, "daily.jsonl")
ACCOUNT_FILE = os.path.join(OUT_DIR, "account.txt")

CAPITAL_RUB = 1_000_000.0
LEVERAGE = 2.25
TOP_N = 10
LOOKBACK, SKIP = 252, 21
FUT_MIN_DAYS = 10                  # ближе к экспирации — роллим в следующий контракт
CASH_BUFFER = 0.10                 # доля капитала деньгами сверх ГО (вариационная маржа)
GO_ALARM = 0.90                    # ГО + запас > 90 % капитала → разгрузка
UNIVERSE = ("AFLT ALRS ASTR BSPB CBOM ENPG FEES FIXR FLOT GMKN HYDR IRAO LENT LKOH "
            "MAGN MGNT MOEX MVID PHOR POSI ROSN RTKM RUAL SMLT TATN UPRO VKCO VTBR "
            "X5 YDEX SBER GAZP SNGS SNGSP MTSS").split()
# базовый актив фьючерса в API → тикер акции
BASIC_TO_TICKER = {"SBRF": "SBER", "GAZR": "GAZP", "SNGR": "SNGS", "SNGP": "SNGSP",
                   "MTSI": "MTSS"}


def _write_jsonl(path: str, rec: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ── счёт ─────────────────────────────────────────────────────────────────────

def open_account(br: TinkoffSandboxClient) -> str:
    pref = open(ACCOUNT_FILE, encoding="utf-8").read().strip() \
        if os.path.exists(ACCOUNT_FILE) else ""
    forbidden = {str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip(),
                 str(getattr(config, "SANDBOX_ACCOUNT_ID", "") or "").strip()}
    old = "/opt/etl-r4/audit/momentum/account.txt"
    if os.path.exists(old):
        forbidden.add(open(old, encoding="utf-8").read().strip())
    forbidden.discard("")
    if pref in forbidden:
        raise RuntimeError("счёт рукава совпадает с чужим счётом — отказ")
    acc = br.open_or_get_account(preferred_id=pref)
    if acc in forbidden:
        raise RuntimeError("получен чужой счёт — отказ")
    if acc != pref:
        br.pay_in(acc, CAPITAL_RUB, currency="rub")
        log.warning("новый счёт песочницы %s пополнен на %.0f ₽", acc, CAPITAL_RUB)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(ACCOUNT_FILE, "w", encoding="utf-8") as f:  # flow-state
            f.write(acc + "\n")
    return acc


# ── сигнал ───────────────────────────────────────────────────────────────────

def momentum_picks(conn, day: dt.date) -> dict:
    """Топ-10 по доходности 12-1 с дивидендами из market_data + dividends."""
    import numpy as np
    import pandas as pd
    df = pd.read_sql("SELECT ticker, date, close FROM market_data "
                     "WHERE close > 0 AND date <= %(d)s AND ticker = ANY(%(t)s) "
                     "AND extract(isodow FROM date) < 6 ORDER BY date",
                     conn, params={"d": day, "t": list(UNIVERSE)})
    px = df.pivot(index="date", columns="ticker", values="close").astype(float).sort_index()
    div = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    dv = pd.read_sql("SELECT ticker, last_buy_date, dividend_net FROM dividends "
                     "WHERE ticker = ANY(%(t)s)", conn, params={"t": list(UNIVERSE)})
    days = px.index.values
    for r in dv.itertuples():
        if r.ticker in div.columns:
            i = np.searchsorted(days, r.last_buy_date, side="right")
            if 0 < i < len(days) and float(r.dividend_net) / float(px[r.ticker].iloc[i - 1]) < 0.5:
                div.iat[i, div.columns.get_loc(r.ticker)] += float(r.dividend_net)
    ret = ((px + div) / px.shift(1) - 1).where(lambda x: x.abs() <= 0.4)
    win = ret.iloc[-LOOKBACK:-SKIP]
    ok = win.notna().sum() >= 0.8 * len(win)
    mom = ((1 + win.fillna(0)).prod() - 1)[ok].sort_values(ascending=False)
    return {"asof": str(px.index[-1]), "picks": list(mom.index[:TOP_N]),
            "momentum": {k: round(float(v), 4) for k, v in mom.items()}}


# ── фьючерсы ─────────────────────────────────────────────────────────────────

def _q(x) -> float:
    return (int(x.get("units", 0) or 0) + int(x.get("nano", 0) or 0) / 1e9) if x else 0.0


def futures_map(br) -> dict:
    """{тикер акции | 'IMOEX': (ближайший контракт ≥ FUT_MIN_DAYS до экспирации)}."""
    data = br._post("InstrumentsService/Futures",
                    {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})
    today = dt.date.today()
    best: dict[str, tuple] = {}
    for f in data.get("instruments", []) or []:
        if not f.get("apiTradeAvailableFlag"):
            continue
        exp = str(f.get("expirationDate") or "")[:10]
        if not exp:
            continue
        days = (dt.date.fromisoformat(exp) - today).days
        if days < FUT_MIN_DAYS or days > 400:
            continue
        basic = str(f.get("basicAsset") or "").upper()
        key = "IMOEX" if basic in ("IMOEX", "MIX") else BASIC_TO_TICKER.get(basic, basic)
        if key != "IMOEX" and key not in UNIVERSE:
            continue
        cand = (days, f)
        # индекс: из контрактов одной экспирации — самый мелкий (мини MX)
        if key in best:
            d0, f0 = best[key]
            if (days, _q(f.get("basicAssetSize"))) >= (d0, _q(f0.get("basicAssetSize"))):
                continue
        best[key] = cand
    return {k: v[1] for k, v in best.items()}


def to_instrument(f: dict) -> Instrument:
    return Instrument(ticker=f["ticker"], instrument_uid=f["uid"], figi=f.get("figi", ""),
                      lot=int(f.get("lot", 1)),
                      min_price_increment=Quotation.from_payload(f.get("minPriceIncrement")),
                      currency=f.get("currency", "rub"),
                      trading_status=f.get("tradingStatus", ""),
                      api_trade_available=bool(f.get("apiTradeAvailableFlag")),
                      short_enabled=bool(f.get("shortEnabledFlag", True)),
                      class_code=f.get("classCode", "SPBFUT"))


def contract_info(br, uid: str) -> dict:
    """Стоимость контракта в ₽ и ГО на покупку/продажу (GetFuturesMargin)."""
    m = br._post("InstrumentsService/GetFuturesMargin", {"instrumentId": uid})
    px = br.get_last_price(uid) or 0.0
    step, step_rub = _q(m.get("minPriceIncrement")), _q(m.get("minPriceIncrementAmount"))
    value = px / step * step_rub if step > 0 else 0.0
    return {"price": px, "value_rub": value,
            "go_buy": _q(m.get("initialMarginOnBuy")), "go_sell": _q(m.get("initialMarginOnSell"))}


def positions(br, acc: str) -> dict[str, float]:
    """uid → баланс, включая фьючерсы (клиент песочницы видит только securities)."""
    data = br._post("SandboxService/GetSandboxPositions", {"accountId": acc})
    out: dict[str, float] = {}
    for section in ("securities", "futures"):
        for it in (data.get(section) or []):
            uid, bal = it.get("instrumentUid") or "", float(it.get("balance", 0) or 0)
            if uid and abs(bal) > 1e-9:
                out[uid] = out.get(uid, 0.0) + bal
    return out


def fund_instrument(br):
    from services.treasury import choose_listing
    item = choose_listing(br.find_instrument_listings("TMON"), "TMON")
    return br.find_instrument_by_uid(item["uid"]) if item and item.get("uid") else None


def money_parts(br, acc: str) -> dict:
    """Рубли счёта по полям брокера (исправлено 01.10.2026).

    В песочнице ГО под фьючерсы УЖЕ входит в positions.money (money 851 298 ₽ =
    portfolio.totalAmountCurrencies), а positions.blocked показывает его отдельно.
    Прежняя формула money + blocked считала ГО дважды: капитал 1,46 млн вместо 0,97,
    и ребаланс 01.10 построил бы плечо ≈ 3,4 вместо 2,25. Деньги под активные
    заявки на покупку (лимитки на паи фонда) из money вычитаются — их возвращаем
    по initialOrderPrice."""
    data = br._post("SandboxService/GetSandboxPositions", {"accountId": acc})
    rub = lambda key: sum(Quotation.from_payload(m).as_float()
                          for m in (data.get(key) or []) if m.get("currency") == "rub")
    orders = br._post("SandboxService/GetSandboxOrders", {"accountId": acc}).get("orders") or []
    reserved = sum(Quotation.from_payload(o.get("initialOrderPrice")).as_float()
                   for o in orders if o.get("direction") == "ORDER_DIRECTION_BUY")
    return {"money": rub("money"), "blocked": rub("blocked"), "reserved_orders": reserved}


def open_varmargin(br, acc: str) -> float:
    """Нерасчётная вариационная маржа открытых фьючерсов (до клиринга в деньгах её нет)."""
    p = br._post("SandboxService/GetSandboxPortfolio", {"accountId": acc})
    return sum(Quotation.from_payload(x.get("varMargin")).as_float()
               for x in (p.get("positions") or []) if x.get("varMargin"))


def equity(br, acc: str) -> dict:
    """Капитал = деньги (ГО в них уже есть) + резерв под заявки на покупку + паи фонда
    + открытая вариационная маржа. go_broker — фактически заблокированное ГО."""
    mp = money_parts(br, acc)
    vm = open_varmargin(br, acc)
    fund = fund_instrument(br)
    pos = positions(br, acc)
    fund_val = 0.0
    if fund and fund.instrument_uid in pos:
        fund_val = pos[fund.instrument_uid] * (br.get_last_price(fund.instrument_uid) or 0.0)
    money = mp["money"] + mp["reserved_orders"]
    return {"money": money, "fund": fund_val, "varmargin": vm, "equity": money + fund_val + vm,
            "go_broker": max(mp["blocked"] - mp["reserved_orders"], 0.0),
            "positions": pos, "fund_ins": fund}


# ── план и исполнение ────────────────────────────────────────────────────────

def plan(br, acc: str, picks: list[str], eq: float) -> dict:
    fm = futures_map(br)
    idx = fm.get("IMOEX")
    if idx is None:
        raise RuntimeError("нет фьючерса на индекс")
    want: dict[str, dict] = {}
    per_name = LEVERAGE * eq / len(picks)
    long_notional, go = 0.0, 0.0
    for tk in picks:
        f = fm.get(tk)
        if f is None:
            log.warning("%s: фьючерса нет — пропуск", tk)
            continue
        ci = contract_info(br, f["uid"])
        n = int(round(per_name / ci["value_rub"])) if ci["value_rub"] > 0 else 0
        want[f["uid"]] = {"ticker": tk, "future": f["ticker"], "ins": to_instrument(f),
                          "contracts": n, "info": ci}
        long_notional += n * ci["value_rub"]
        go += n * ci["go_buy"]
    ci = contract_info(br, idx["uid"])
    k = int(round(long_notional / ci["value_rub"])) if ci["value_rub"] > 0 else 0
    want[idx["uid"]] = {"ticker": "IMOEX", "future": idx["ticker"], "ins": to_instrument(idx),
                        "contracts": -k, "info": ci}
    go += k * ci["go_sell"]
    have = dict(positions(br, acc))
    fund = fund_instrument(br)
    if fund:
        have.pop(fund.instrument_uid, None)
    trades = []
    for uid, w in want.items():
        cur = int(round(have.pop(uid, 0.0) / w["ins"].lot))
        d = w["contracts"] - cur
        if d:
            trades.append({"uid": uid, "name": w["future"], "ins": w["ins"],
                           "side": "BUY" if d > 0 else "SELL", "lots": abs(d)})
    for uid, bal in have.items():                    # старые контракты / вышедшие из топа
        ins = br.find_instrument_by_uid(uid)
        lots = int(round(abs(bal) / ins.lot))
        if lots:
            trades.append({"uid": uid, "name": ins.ticker, "ins": ins,
                           "side": "SELL" if bal > 0 else "BUY", "lots": lots})
    fund_target = max(0.0, eq - go - CASH_BUFFER * eq)
    return {"want": {w["future"]: w["contracts"] for w in want.values()},
            "long_notional": long_notional, "short_notional": k * ci["value_rub"],
            "go": go, "fund_target": fund_target, "trades": trades,
            "leverage_real": long_notional / eq if eq else 0.0}


def execute(br, acc: str, trades: list, dry: bool) -> list:
    out = []
    # сперва закрытия и сокращения, потом открытия — меньше пиковое ГО
    order = sorted(trades, key=lambda t: 0 if t["side"] == "SELL" and "IMOEX" not in t["name"] else 1)
    for t in order:
        if dry:
            out.append({**{k: t[k] for k in ("name", "side", "lots")}, "status": "DRY"})
            continue
        try:
            st = br.post_market_order(account_id=acc, instrument=t["ins"], direction=t["side"],
                                      quantity_lots=t["lots"], order_id=new_order_id())
            out.append({"name": t["name"], "side": t["side"], "lots": t["lots"],
                        "status": st.execution_report_status})
        except BrokerError as e:
            out.append({"name": t["name"], "side": t["side"], "lots": t["lots"],
                        "status": f"ОШИБКА {e}"})
    return out


def settle_fund(br, acc: str, target_rub: float, dry: bool) -> dict:
    """Довести паи TMON до целевой суммы (продать при нехватке денег под ГО)."""
    fund = fund_instrument(br)
    if not fund:
        return {"fund": "недоступен"}
    px = br.get_last_price(fund.instrument_uid) or 0.0
    have = positions(br, acc).get(fund.instrument_uid, 0.0)
    want = int(target_rub // (px * fund.lot)) if px > 0 else 0
    d = want - int(round(have / fund.lot))
    if d == 0 or dry:
        return {"fund_lots_delta": d, "dry": dry}
    try:
        st = br.post_market_order(account_id=acc, instrument=fund, direction="BUY" if d > 0 else "SELL",
                                  quantity_lots=abs(d), order_id=new_order_id())
        return {"fund_lots_delta": d, "status": st.execution_report_status}
    except BrokerError as e:
        return {"fund_lots_delta": d, "status": f"ОШИБКА {e}"}


def due(now: dt.datetime) -> bool:
    if not os.path.exists(LEDGER):
        return True
    for line in open(LEDGER, encoding="utf-8"):
        if line.strip() and json.loads(line).get("kind") == "rebalance" \
                and json.loads(line).get("ts", "")[:7] == now.strftime("%Y-%m"):
            return False
    return True


def cmd_rebalance(dry: bool, if_due: bool) -> int:
    now = dt.datetime.now(MSK)
    if if_due and not due(now):
        log.info("ребаланс этого месяца уже сделан")
        return 0
    br = TinkoffSandboxClient()
    acc = open_account(br)
    conn = database.get_connection()
    sig = momentum_picks(conn, now.date())
    eq = equity(br, acc)
    p = plan(br, acc, sig["picks"], eq["equity"])
    # деньги под ГО: если паёв слишком много — продать до сделок по фьючерсам
    pre = settle_fund(br, acc, min(eq["fund"], p["fund_target"]), dry)
    res = execute(br, acc, p["trades"], dry)
    post = settle_fund(br, acc, p["fund_target"], dry)
    rec = {"ts": now.isoformat(timespec="seconds"), "kind": "rebalance", "dry": dry,
           "account": acc, "equity": eq["equity"], "signal": sig, "leverage": LEVERAGE,
           "leverage_real": round(p["leverage_real"], 3), "go": round(p["go"], 2),
           "want": p["want"], "fund_target": round(p["fund_target"], 2),
           "fund_pre": pre, "fund_post": post, "trades": res}
    if not dry:
        _write_jsonl(LEDGER, rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str)[:4000])
    return 0 if all(not str(t["status"]).startswith("ОШИБКА") for t in res) else 1


def cmd_check(dry: bool) -> int:
    """Ежедневно: ролл истекающих контрактов и разгрузка при перегрузке ГО."""
    now = dt.datetime.now(MSK)
    br = TinkoffSandboxClient()
    acc = open_account(br)
    eq = equity(br, acc)
    pos = eq["positions"]
    fm = futures_map(br)
    active = {f["uid"] for f in fm.values()}
    go = 0.0
    roll = False
    for uid, bal in pos.items():
        if eq["fund_ins"] and uid == eq["fund_ins"].instrument_uid:
            continue
        if uid not in active:
            roll = True                       # контракт ближе FUT_MIN_DAYS к экспирации
        try:
            ci = contract_info(br, uid)
            go += abs(bal) * (ci["go_buy"] if bal > 0 else ci["go_sell"])
        except BrokerError:
            pass
    go_used = eq["go_broker"] or go            # фактическое ГО брокера; оценка — если его нет
    load = (go_used + CASH_BUFFER * eq["equity"]) / eq["equity"] if eq["equity"] else 0.0
    rec = {"ts": now.isoformat(timespec="seconds"), "kind": "check", "equity": eq["equity"],
           "money": eq["money"], "fund": eq["fund"], "varmargin": round(eq["varmargin"], 2),
           "go": round(go_used, 2), "go_est": round(go, 2),
           "go_load": round(load, 3), "roll": roll}
    if (roll or load > GO_ALARM) and os.path.exists(LEDGER):
        # тот же состав, перерасчёт под текущий капитал: ролл и/или разгрузка
        last = [json.loads(x) for x in open(LEDGER, encoding="utf-8") if x.strip()]
        picks = next((r["signal"]["picks"] for r in reversed(last)
                      if r.get("kind") == "rebalance"), None)
        if picks:
            p = plan(br, acc, picks, eq["equity"])
            rec["fund_pre"] = settle_fund(br, acc, min(eq["fund"], p["fund_target"]), dry)
            rec["trades"] = execute(br, acc, p["trades"], dry)
            rec["fund_post"] = settle_fund(br, acc, p["fund_target"], dry)
            rec["reason"] = ("ролл " if roll else "") + ("разгрузка ГО" if load > GO_ALARM else "")
    if not dry:
        _write_jsonl(DAILY, rec)
    print(json.dumps(rec, ensure_ascii=False, default=str))
    return 0


def cmd_status() -> int:
    br = TinkoffSandboxClient()
    acc = open_account(br)
    eq = equity(br, acc)
    start = CAPITAL_RUB
    print(f"счёт {acc}: капитал {eq['equity']:,.0f} ₽ ({(eq['equity']/start-1)*100:+.2f} %), "
          f"деньги {eq['money']:,.0f}, паи {eq['fund']:,.0f}".replace(",", " "))
    for uid, bal in eq["positions"].items():
        try:
            print(f"  {br.find_instrument_by_uid(uid).ticker:<8} {bal:+.0f}")
        except BrokerError:
            print(f"  {uid[:8]}… {bal:+.0f}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rebalance", "check", "status"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--if-due", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "rebalance":
        return cmd_rebalance(a.dry_run, a.if_due)
    if a.cmd == "check":
        return cmd_check(a.dry_run)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())
