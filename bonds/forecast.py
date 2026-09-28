"""
Поток «прогноз облигаций на 12 месяцев»: лонг и шорт (описательная модель, не испытание).

Модель: у каждой бумаги ожидаемая доходность на год = её доходность к оферте/погашению
минус «недобор» её корзины по спреду к ключевой ставке, измеренный на истории: сколько
обещанной доходности в среднем съедали дефолты, остановки торгов и падение цен.

Калибровка: концы месяцев 2020-12 … 2025-08, каждый рублёвый выпуск TQCB с оборотом
≥ 1 млн ₽ в этот день, не флоатер/квал/тест/субординация/бессрочный, до события ≥ 60 дней;
фактическая доходность за 12 месяцев — цепочкой месячных срезов ISS (цена + НКД + купоны +
амортизация); выпуск, пропавший со среза, досчитывается по дневной истории
(bond_return: погашение, остановка, дефолт −50 %); после погашения — деньги по ставке,
после остановки/дефолта — 0.
Проверка устойчивости: недобор, откалиброванный на окнах 2020-12…2023-02, предсказывает
окна 2023-03…2025-08 (ранговая корреляция и порядок корзин) — вне выборки калибровки.

Что подаём:  кэш ISS и флаги брокера (как bonds.backtest), живой срез (как bonds.trader).
Что запускаем: python -m bonds.forecast --cache <кэш> --flags flags.json
Что получаем:  <cache>/results/bonds_forecast_wNN.json — корзины, проверка вне выборки,
               прогноз по каждой бумаге (лонг — выше ставки; шорт — ниже нуля, с акцией
               эмитента, если её можно шортить), самопроверки.

Шорт облигаций у брокера недоступен (shortEnabledFlag = 0 у всех 1 437 рублёвых выпусков
TQCB, 28.09.2026) — шорт-сторона показывает, что не держать, и прокси через акцию эмитента.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bonds.backtest import FLOAT_WORDS, bond_return, key_rates, month_ends, rate_on  # noqa: E402
from bonds.core import load_rules, rating_name                                     # noqa: E402
from bonds.sources import iss                                                       # noqa: E402
from tools.flowkit import Checks, save_state, versioned                             # noqa: E402

CAL_START, CAL_LAST_START = dt.date(2020, 12, 1), dt.date(2025, 8, 31)
SPLIT = dt.date(2023, 2, 28)             # окна со стартом ≤ SPLIT — «ранние», позже — «поздние»
H = 12
EDGES = [-math.inf, 0, 3, 6, 10, 15, 20, 30, math.inf]
# Корзины 30…50 и 50…∞ объединены 28.09.2026 ПОСЛЕ первого прогона (w02): в них 47 и 60
# наблюдений, а «доходность − недобор» давала одному эмитенту противоположные ответы
# (БалтЛизП19 при спреде 49 п.п. +36,6 %, БалтЛизП15 при 53 п.п. −171 %). В верхней
# корзине прогноз — средний факт корзины, а не доходность бумаги минус недобор.
TOP_FROM = 30
CRASH = -0.20
SHORT_FEE_RUB_DAY, SHORT_POS_RUB = 45.0, 50_000.0   # «от 45 ₽/день» тарифа; для крупных — не известно


def bucket(spread: float) -> str:
    for lo, hi in zip(EDGES, EDGES[1:]):
        if lo <= spread < hi:
            return f"{'−∞' if lo == -math.inf else int(lo)}…{'∞' if hi == math.inf else int(hi)}"
    return "?"


def px(r):
    return r.get("CLOSE") or r.get("LEGALCLOSEPRICE")


def value(r):
    return px(r) / 100 * r["FACEVALUE"] + (r.get("ACCINT") or 0)


def eligible_hist(r, day, flags, min_value):
    if (r.get("FACEUNIT") or "RUB") not in ("RUB", "SUR") or not px(r) or not r.get("FACEVALUE"):
        return None
    f = flags.get(r["SECID"], {})
    det = str(r.get("COUPON_DETAILS") or "").lower()
    if any(w in det for w in FLOAT_WORDS) or f.get("floating") or f.get("qual_only") or f.get("required_tests") \
            or f.get("subordinated") or f.get("perpetual"):
        return None
    ev = iss.next_event(r, day)
    if ev is None or (ev - day).days < 60 or (r.get("VALUE") or 0) < min_value:
        return None
    has_offer = ev != iss._date(r.get("MATDATE"))
    y = (r.get("YIELDTOOFFER") if has_offer else None) or r.get("YIELDCLOSE")
    return None if y is None else {"y": float(y), "event": ev}


def chain(secid, i, ends, snap, rates, cache, cal):
    """Доходность за H месяцев от конца месяца ends[i] цепочкой срезов; (r, rf, событие)."""
    cur = snap[i][secid]
    g, rf_g, state, note = 1.0, 1.0, "alive", ""
    for j in range(i, i + H):
        rf = rate_on(rates, ends[j]) / 100 * (ends[j + 1] - ends[j]).days / 365
        rf_g *= 1 + rf
        if state == "redeemed":
            g *= 1 + rf
            continue
        if state == "dead":
            continue
        nxt = snap[j + 1].get(secid)
        if nxt is not None:
            cash = 0.0
            if (nxt.get("ACCINT") or 0) + 1e-9 < (cur.get("ACCINT") or 0):
                cash += cur.get("COUPONVALUE") or cur.get("ACCINT") or 0
            if nxt["FACEVALUE"] + 1e-9 < cur["FACEVALUE"]:
                cash += cur["FACEVALUE"] - nxt["FACEVALUE"]
            g *= (value(nxt) + cash) / value(cur)
            cur = nxt
            continue
        r, n = bond_return(secid, ends[j], ends[j + 1], cache, cal=cal)
        g *= 1 + r
        ev = iss.next_event(cur, ends[j])
        if n == "погашен" or (n.startswith("остановка") and ev and ev <= ends[j + 1] + dt.timedelta(days=10)):
            state, note = "redeemed", note or "погашение/оферта"
        elif n:
            state, note = "dead", n
        else:
            rows = [x for x in iss.daily(secid, ends[j], ends[j + 1], cache) if px(x) and x.get("FACEVALUE")]
            if rows:
                last = dict(rows[-1])
                last.setdefault("COUPONVALUE", cur.get("COUPONVALUE"))
                last["OFFERDATE"], last["PUTOPTIONDATE"] = cur.get("OFFERDATE"), cur.get("PUTOPTIONDATE")
                last["CALLOPTIONDATE"], last["BUYBACKDATE"] = cur.get("CALLOPTIONDATE"), cur.get("BUYBACKDATE")
                cur = last
    return g - 1, rf_g - 1, note


def indexed_secids(snap) -> set:
    """Выпуски, у которых номинал хоть раз ВЫРОС между срезами: привязка к золоту/валюте/
    индексации или сбой данных (RU000A102QL3: 125 000 → 11 382 488). Их доходность
    несопоставима с обычной ВДО — исключаются из калибровки и прогноза (28.09.2026)."""
    out, last = set(), {}
    for sn in snap:
        for sid, r in sn.items():
            f = r.get("FACEVALUE") or 0
            if sid in last and f > last[sid] * 1.001:
                out.add(sid)
            last[sid] = f
    return out


def calibrate(cache, flags, rules):
    rates = key_rates()
    cal = iss.trading_days(CAL_START - dt.timedelta(days=40), dt.date(2026, 9, 1), cache)
    ends = [d for d in month_ends(cal) if d >= CAL_START]
    snap = [{r["SECID"]: r for r in iss.history_on(d, cache) if px(r) and r.get("FACEVALUE")} for d in ends]
    indexed = indexed_secids(snap)
    obs = []
    min_value = rules["liquidity"]["min_day_value_rub"]
    for i, d in enumerate(ends):
        if d > CAL_LAST_START or i + H >= len(ends):
            break
        key = rate_on(rates, d)
        for sid, r in snap[i].items():
            if sid in indexed:
                continue
            e = eligible_hist(r, d, flags, min_value)
            if not e:
                continue
            ret, rf, note = chain(sid, i, ends, snap, rates, cache, cal)
            obs.append({"start": str(d), "secid": sid, "y": e["y"], "spread": e["y"] - key, "price": px(r),
                        "b": bucket(e["y"] - key), "r": ret, "rf": rf, "note": note})
    return obs, ends, indexed


def bucket_table(obs):
    out = {}
    for b in sorted({o["b"] for o in obs}, key=lambda s: float(s.split("…")[0].replace("−∞", "-inf"))):
        x = [o for o in obs if o["b"] == b]
        r = np.array([o["r"] for o in x])
        out[b] = {"n": len(x), "months": len({o["start"] for o in x}),
                  "promised": float(np.mean([o["y"] for o in x])), "real": float(r.mean() * 100),
                  "median": float(np.median(r) * 100), "p10": float(np.percentile(r, 10) * 100),
                  "excess": float(np.mean([o["r"] - o["rf"] for o in x]) * 100),
                  "crash": float(np.mean(r < CRASH)), "gap": float(np.mean([o["y"] for o in x]) - r.mean() * 100)}
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def out_of_sample(obs):
    early = [o for o in obs if dt.date.fromisoformat(o["start"]) <= SPLIT]
    late = [o for o in obs if dt.date.fromisoformat(o["start"]) > SPLIT]
    gap = {b: v["gap"] for b, v in bucket_table(early).items()}
    fallback = st.mean(gap.values())
    early_real = {b: v["real"] for b, v in bucket_table(early).items()}
    pred = [early_real.get(o["b"], 0.0) if o["spread"] >= TOP_FROM else o["y"] - gap.get(o["b"], fallback) for o in late]
    real = [o["r"] * 100 for o in late]
    by_month = {}
    for o, p in zip(late, pred):
        by_month.setdefault(o["start"], []).append((p, o["r"] * 100))
    ics = [spearman([a for a, _ in v], [b for _, b in v]) for v in by_month.values() if len(v) >= 10]
    long_ = [o["r"] - o["rf"] for o, p in zip(late, pred) if p > (o["rf"] * 100)]
    rest = [o["r"] - o["rf"] for o, p in zip(late, pred) if p <= (o["rf"] * 100)]
    neg = [o["r"] for o, p in zip(late, pred) if p < 0]
    return {"early_n": len(early), "late_n": len(late), "early_gap": gap,
            "late_table": bucket_table(late), "ic_mean": float(np.mean(ics)), "ic_months": len(ics),
            "ic_t": float(np.mean(ics) / (np.std(ics, ddof=1) / math.sqrt(len(ics)))),
            "mae_pp": float(np.mean(np.abs(np.array(pred) - np.array(real)))),
            "long_excess": float(np.mean(long_) * 100) if long_ else None, "long_n": len(long_),
            "rest_excess": float(np.mean(rest) * 100) if rest else None,
            "neg_n": len(neg), "neg_real": float(np.mean(neg) * 100) if neg else None,
            "neg_crash": float(np.mean(np.array(neg) < CRASH)) if neg else None}


def live(table, rules, indexed):
    from bonds import trader
    from services.broker.tinkoff_sandbox import TinkoffSandboxClient
    br = TinkoffSandboxClient()
    today = dt.date.today()
    bonds, meta = trader.universe(br, today)
    key = trader.key_rate(today)
    raw = {x["uid"]: x for x in br._post("InstrumentsService/Bonds", {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})["instruments"]}
    shares = br._post("InstrumentsService/Shares", {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})["instruments"]
    by_logo = {}
    for s in shares:
        lg = (s.get("brand") or {}).get("logoName")
        if lg and s.get("classCode") == "TQBR":
            by_logo[lg] = {"ticker": s["ticker"], "short_ok": bool(s.get("shortEnabledFlag"))}
    held = {}
    try:
        held = {p["secid"] for p in (trader.latest_target() or {}).get("picks", [])}
    except Exception:  # noqa: BLE001
        pass
    gaps = {b: v["gap"] for b, v in table.items()}
    out, lq = [], rules["liquidity"]["min_day_value_rub"]
    for b in bonds:
        if b.get("yield_pct") is None or not b.get("price_pct") or b.get("floating") or b.get("qual_only") \
                or b.get("required_tests") or b.get("subordinated") or b.get("perpetual") or not b.get("tradable"):
            continue
        if (b.get("days_to_event") or 0) < 60 or (b.get("value_rub") or 0) < lq:
            continue
        sp = b["yield_pct"] - key
        bk = bucket(sp)
        if bk not in gaps:
            continue
        rb = raw.get(b["uid"], {})
        q = lambda x: (int((x or {}).get("units", 0) or 0) + int((x or {}).get("nano", 0) or 0) / 1e9)
        if b["secid"] in indexed or q(rb.get("nominal")) > q(rb.get("initialNominal")) * 1.001:
            continue
        stock = by_logo.get((rb.get("brand") or {}).get("logoName"))
        e = table[bk]["real"] if sp >= TOP_FROM else b["yield_pct"] - gaps[bk]
        out.append({"secid": b["secid"], "name": b.get("name"), "issuer": b.get("issuer"), "rating": rating_name(b.get("rating_num")),
                    "yield": b["yield_pct"], "price": b["price_pct"], "duration": b.get("duration_y"), "event": b.get("event"),
                    "spread": sp, "bucket": bk, "expected": e, "model": "факт корзины" if sp >= TOP_FROM else "доходность − недобор", "expected_excess": e - key,
                    "crash_prob": table[bk]["crash"], "bond_short_ok": bool(rb.get("shortEnabledFlag")),
                    "stock": stock, "in_sandbox": b["secid"] in held, "value_rub": b.get("value_rub")})
    short_fee = SHORT_FEE_RUB_DAY * 365 / SHORT_POS_RUB * 100
    return {"asof": str(today), "key": key, "universe": meta, "bonds": out, "short_fee_pct_year": short_fee}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    flags = json.load(open(a.flags, encoding="utf-8"))
    rules = load_rules()
    obs, ends, indexed = calibrate(a.cache, flags, rules)
    table = bucket_table(obs)
    oos = out_of_sample(obs)
    now = live(table, rules, indexed)
    chk = Checks("bonds-forecast")
    chk.eq("наблюдений в корзинах = всего", sum(v["n"] for v in table.values()), len(obs))
    chk.ok("старт окон не позже 2025-08", all(o["start"] <= str(CAL_LAST_START) for o in obs))
    chk.ok("ставка за 12 мес. правдоподобна (3…30 %)", all(0.03 <= o["rf"] <= 0.30 for o in obs))
    chk.ok("доходность ≥ −100 %", all(o["r"] >= -1.0 - 1e-9 for o in obs))
    chk.ok("нет наблюдений по выпускам с растущим номиналом", not any(o["secid"] in indexed for o in obs))
    chk.ok("доходность за год ≤ +300 % (иначе сбой данных)", all(o["r"] <= 3.0 for o in obs),
           str(sorted(((o["r"], o["secid"], o["start"]) for o in obs), reverse=True)[:3]))
    for b, v in table.items():
        x = [o for o in obs if o["b"] == b]
        chk.close(f"недобор {b} пересчитан", v["gap"], float(np.mean([o["y"] for o in x]) - np.mean([o["r"] for o in x]) * 100), tol=1e-9)
    chk.eq("ранние + поздние = всего", oos["early_n"] + oos["late_n"], len(obs))
    chk.ok("прогноз = доходность − недобор (верхняя корзина — её факт)", all(abs(b["expected"] - (
        table[b["bucket"]]["real"] if b["spread"] >= TOP_FROM else b["yield"] - table[b["bucket"]]["gap"])) < 1e-9 for b in now["bonds"]))
    chk.ok("прогноз ≥ −100 %", all(b["expected"] >= -100 for b in now["bonds"]))
    path = versioned(a.out or os.path.join(a.cache, "results"), "bonds_forecast", "json")
    save_state(path, params={"H": H, "EDGES": [str(e) for e in EDGES], "SPLIT": SPLIT, "CRASH": CRASH,
                             "CAL": [CAL_START, CAL_LAST_START], "short_fee_rub_day": SHORT_FEE_RUB_DAY},
               results={"table": table, "oos": oos, "now": now, "n_obs": len(obs), "indexed": sorted(indexed),
                        "obs": obs}, checks=chk,
               inputs={"cache": a.cache, "flags": a.flags})
    print(chk.report())
    print(f"наблюдений {len(obs)}, окон {len({o['start'] for o in obs})}")
    for b, v in table.items():
        print(f"  спред {b:>7}: n {v['n']:5d}  обещано {v['promised']:5.1f}  факт {v['real']:+6.1f} (медиана {v['median']:+6.1f}, "
              f"p10 {v['p10']:+6.1f})  к ставке {v['excess']:+6.1f}  обвал {v['crash']:.0%}  недобор {v['gap']:5.1f}")
    print(f"вне выборки: IC {oos['ic_mean']:+.3f} (t {oos['ic_t']:+.2f}, {oos['ic_months']} мес.), ошибка {oos['mae_pp']:.1f} п.п.; "
          f"«лонг» к ставке {oos['long_excess']:+.1f} п.п. (n {oos['long_n']}) против остальных {oos['rest_excess']:+.1f}; "
          f"прогноз < 0: n {oos['neg_n']}, факт {oos['neg_real'] if oos['neg_real'] is None else round(oos['neg_real'],1)} %, обвал {oos['neg_crash']}")
    L = sorted([b for b in now["bonds"] if b["expected_excess"] > 0], key=lambda x: -x["expected_excess"])
    S = sorted([b for b in now["bonds"] if b["expected"] < 0], key=lambda x: x["expected"])
    print(f"сейчас: ставка {now['key']}, бумаг {len(now['bonds'])}, лонг (прогноз > ставки) {len(L)}, прогноз < 0: {len(S)}")
    print("результат:", path)
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
