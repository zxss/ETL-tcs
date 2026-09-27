"""
Бэктест портфеля ВДО строго по bonds/PREREG.md (коммит bf613b1) и rules_w01.json.

Что подаём:  ISS (история срезов и дневная история выпусков), ключевую ставку ЦБ,
             необязательно — флаги брокера по ISIN (--flags, JSON из bonds.trader flags).
Что запускаем: python -m bonds.backtest --cache <каталог кэша> [--flags flags.json]
Что получаем:  <out>/bonds_backtest_wNN.json — параметры, самопроверки, помесячный ряд,
               итоги; в stdout — сводка. Версии не перезаписываются.

Приближения (раскрыты в отчёте): кредитный фильтр выключен (истории рейтингов нет);
эмитент — по короткому названию выпуска (в истории ISS эмитента нет); объём выпуска
и сектор — только из флагов брокера для живых выпусков; флоатеры — по полю
COUPON_DETAILS и флагам брокера.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bonds.core import build_portfolio, load_rules            # noqa: E402
from bonds.sources import iss                                  # noqa: E402
from tools.flowkit import Checks, save_state, versioned        # noqa: E402

START, END = dt.date(2020, 12, 1), dt.date(2026, 8, 31)
FIRST_HALF_END = dt.date(2023, 12, 31)
N_TRIALS = 287
FLOAT_WORDS = ("ключев", "ruonia", "ипц", "кбд", "плава")


def key_rates() -> list[tuple[dt.date, float]]:
    from services.load_macro import fetch
    rows = fetch(dt.date(2020, 1, 1), dt.date.today())
    return sorted((dt.date.fromisoformat(r["date"]), float(r["key_rate_pct"])) for r in rows)


def rate_on(rates, day):
    past = [r for d, r in rates if d <= day]
    return past[-1] if past else rates[0][1]


def issuer_of(name: str | None) -> str:
    """Эмитент по короткому названию выпуска: буквы до первой цифры («ДОМ.РФ15об» → «дом.рф»)."""
    m = re.match(r"[^\d]+", str(name or ""))
    return (m.group(0) if m else str(name)).strip(" -").lower()


def to_bond(r: dict, asof: dt.date, flags: dict) -> dict | None:
    if (r.get("FACEUNIT") or "RUB") not in ("RUB", "SUR"):
        return None
    ev = iss.next_event(r, asof)
    has_offer = ev is not None and ev != iss._date(r.get("MATDATE"))
    y = (r.get("YIELDTOOFFER") if has_offer else None) or r.get("YIELDCLOSE")
    price = r.get("CLOSE") or r.get("LEGALCLOSEPRICE")
    det = str(r.get("COUPON_DETAILS") or "").lower()
    f = flags.get(r.get("SECID"), {})
    return {"secid": r["SECID"], "issuer": issuer_of(r.get("SHORTNAME")),   # одно правило для живых и погашенных
            "sector": f.get("sector"), "yield_pct": y, "price_pct": price,
            "duration_y": (r.get("DURATION") or 0) / 365 if r.get("DURATION") else None,
            "days_to_event": (ev - asof).days if ev else None,
            "value_rub": r.get("VALUE") or 0, "bid_ask_pct": None,
            "issue_size_rub": f.get("issue_size_rub"),
            "floating": any(w in det for w in FLOAT_WORDS) or f.get("floating", False),
            "subordinated": f.get("subordinated", False), "perpetual": f.get("perpetual", False),
            "qual_only": f.get("qual_only", False), "required_tests": f.get("required_tests", False),
            "tradable": True, "rating_num": None}


def bond_return(secid: str, a: dt.date, b: dt.date, cache: str,
                cal: list[dt.date] | None = None) -> tuple[float, str]:
    """Доходность выпуска с закрытия a по закрытие b: (цена+НКД+купоны+амортизация).
    cal — календарь торговых дней: тогда «остановка торгов» = пропущены ≥ 3 последних
    торговых дня окна (PREREG_distress); без него — прежнее правило «> 7 дней» (№ 287)."""
    rows = [r for r in iss.daily(secid, a, b, cache)
            if (r.get("CLOSE") or r.get("LEGALCLOSEPRICE")) and r.get("FACEVALUE")]
    if not rows:
        return 0.0, "нет цены на входе"
    px = lambda r: (r.get("CLOSE") or r.get("LEGALCLOSEPRICE"))
    v0 = px(rows[0]) / 100 * rows[0]["FACEVALUE"] + (rows[0].get("ACCINT") or 0)
    cash, prev = 0.0, rows[0]
    for r in rows[1:]:
        if (r.get("ACCINT") or 0) + 1e-9 < (prev.get("ACCINT") or 0):
            cash += prev.get("COUPONVALUE") or prev.get("ACCINT") or 0     # выплата купона
        if r["FACEVALUE"] + 1e-9 < prev["FACEVALUE"]:
            cash += prev["FACEVALUE"] - r["FACEVALUE"]                      # амортизация
        prev = r
    last = rows[-1]
    last_day = dt.date.fromisoformat(last["TRADEDATE"])
    mat = iss._date(last.get("MATDATE"))
    v1 = px(last) / 100 * last["FACEVALUE"] + (last.get("ACCINT") or 0)
    note = ""
    if cal is not None:
        wd = [x for x in cal if a <= x <= b]
        stale = len(wd) >= 3 and last_day < wd[-3]
    else:
        stale = (b - last_day).days > 7
    if stale:                                     # перестал торговаться до конца периода
        if mat and mat <= b:
            cash += last["FACEVALUE"] + (last.get("ACCINT") or 0)          # погашение
            v1, note = 0.0, "погашен"
        else:
            note = "остановка торгов"
            if px(last) < 50:
                v1 *= 0.5
                note = "дефолт: −50 % к последней цене"
    return (v1 + cash) / v0 - 1.0, note


def month_ends(days: list[dt.date]) -> list[dt.date]:
    out = []
    for d, n in zip(days, days[1:]):
        if n.month != d.month:
            out.append(d)
    return out


def run(cache: str, flags: dict, rules: dict, start=START, end=END) -> dict:
    rates = key_rates()
    days = iss.trading_days(start, end + dt.timedelta(days=40), cache)
    ends = [d for d in month_ends(days) if d <= end]
    nxt = {d: days[days.index(d) + 1] for d in ends}
    rows, held, weights_prev = [], set(), {}
    cost_rt = rules["execution"]["assumed_round_trip_pct_backtest"] / 100
    for k, d in enumerate(ends[:-1]):
        key = rate_on(rates, d)
        bonds = [b for b in (to_bond(r, d, flags) for r in iss.history_on(d, cache)) if b]
        res = build_portfolio(bonds, rules, key, use_credit=False, holdings=held)
        port = res["portfolio"]
        a, b = nxt[d], nxt[ends[k + 1]]
        w_new = {p["secid"]: p["weight"] for p in port}
        turnover = sum(abs(w_new.get(s, 0) - weights_prev.get(s, 0)) for s in set(w_new) | set(weights_prev))
        rets, notes = {}, {}
        for p in port:
            rets[p["secid"]], notes[p["secid"]] = bond_return(p["secid"], a, b, cache)
        rf = key / 100 * (b - a).days / 365
        invested = sum(w_new.values())
        gross = sum(w_new[s] * rets[s] for s in w_new) + (1 - invested) * rf
        rows.append({"rebalance": str(a), "until": str(b), "key_rate": key,
                     "universe": res["universe"], "eligible": res["eligible"],
                     "n": len(port), "turnover": turnover, "gross": gross, "rf": rf,
                     "violations": res["violations"], "excluded": res["excluded"],
                     "worst_bond": min(rets.values()) if rets else None,
                     "events": {s: n for s, n in notes.items() if n}, "picks": list(w_new)})
        held, weights_prev = set(w_new), w_new
    return {"months": rows, "cost_rt": cost_rt}


def stats(rows, cost_rt, mult) -> dict:
    ex = np.array([r["gross"] - r["turnover"] * cost_rt * mult / 2 - r["rf"] for r in rows])
    net = np.array([r["gross"] - r["turnover"] * cost_rt * mult / 2 for r in rows])
    eq = np.cumprod(1 + net)
    t = float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))) if len(ex) > 2 else float("nan")
    first = np.array([e for e, r in zip(ex, rows) if dt.date.fromisoformat(r["rebalance"]) <= FIRST_HALF_END])
    second = np.array([e for e, r in zip(ex, rows) if dt.date.fromisoformat(r["rebalance"]) > FIRST_HALF_END])
    years = len(net) / 12
    rf_ann = float(np.prod([1 + r["rf"] for r in rows]) ** (1 / years) - 1)
    return {"cost_mult": mult, "months": len(ex), "mean_excess_m": float(ex.mean()), "t": t,
            "win_share": float((ex > 0).mean()), "worst_month": float(ex.min()),
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "annual_net": float(eq[-1] ** (1 / years) - 1), "annual_rf": rf_ann,
            "first_half_mean": float(first.mean()) if len(first) else None,
            "second_half_mean": float(second.mean()) if len(second) else None,
            "dsr": dsr(ex, N_TRIALS)}


def dsr(x, n):
    from scipy.stats import norm
    x = np.asarray(x, float)
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    v = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1)
    em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / n) + em * norm.ppf(1 - 1 / (n * math.e)))
    return float(norm.cdf((sr - sr0) / math.sqrt(v)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--flags", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", default=None, help="только для проверочных прогонов")
    ap.add_argument("--end", default=None, help="только для проверочных прогонов")
    a = ap.parse_args(argv)
    start = dt.date.fromisoformat(a.start) if a.start else START
    end = dt.date.fromisoformat(a.end) if a.end else END
    rules = load_rules()
    flags = json.load(open(a.flags, encoding="utf-8")) if a.flags else {}
    r = run(a.cache, flags, rules, start, end)
    rows = r["months"]
    s1, s2 = stats(rows, r["cost_rt"], 1.0), stats(rows, r["cost_rt"], 2.0)
    passed = bool(s1["t"] >= 2 and s1["mean_excess_m"] > 0 and (s1["first_half_mean"] or -1) > 0
                  and (s1["second_half_mean"] or -1) > 0 and s2["mean_excess_m"] > 0
                  and s1["dsr"] >= 0.95)
    chk = Checks("bonds-backtest")
    chk.eq("месяцев в статистике = в ряду", s1["months"], len(rows))
    chk.eq("ряд покрывает весь период (нет обрыва календаря)", len(rows),
           (end.year - start.year) * 12 + end.month - start.month)
    chk.ok("ограничения портфеля соблюдены каждый месяц", all(not m["violations"] for m in rows))
    chk.ok("число выпусков ≤ N", all(m["n"] <= rules["portfolio"]["n_bonds"] for m in rows))
    chk.ok("оборот в [0, 2]", all(0 <= m["turnover"] <= 2 + 1e-9 for m in rows))
    chk.close("среднее избытка пересчитано", s1["mean_excess_m"], float(np.mean(
        [m["gross"] - m["turnover"] * r["cost_rt"] / 2 - m["rf"] for m in rows])), tol=1e-12)
    chk.ok("ПРОЙДЕН ⇒ все условия", (not passed) or (s1["t"] >= 2 and s1["dsr"] >= 0.95))
    out_dir = a.out or os.path.join(a.cache, "results")
    path = versioned(out_dir, "bonds_backtest", "json")
    save_state(path, params={"rules": rules["_path"], "START": start, "END": end,
                             "preregistered_period": start == START and end == END,
                             "N_TRIALS": N_TRIALS, "prereg": "bonds/PREREG.md"},
               results={"stats_x1": s1, "stats_x2": s2, "passed": passed, "months": rows},
               checks=chk, inputs={"cache": a.cache, "flags": a.flags})
    print(chk.report())
    pct = lambda v, f="+.3f": "—" if v is None else format(v * 100, f)
    for s in (s1, s2):
        print(f"издержки ×{s['cost_mult']:g}: месяцев {s['months']}, избыток {s['mean_excess_m']*100:+.3f} %/мес, "
              f"t {s['t']:+.2f}, в плюс {s['win_share']:.0%}, худший {s['worst_month']*100:+.2f} %, "
              f"просадка {s['max_dd']*100:.1f} %, годовых {s['annual_net']*100:.1f} % при ставке "
              f"{s['annual_rf']*100:.1f} %; 2021–23 {pct(s['first_half_mean'])}, "
              f"2024–26 {pct(s['second_half_mean'])}; DSR {s['dsr']:.3f}")
    print(f"средний отбор: {np.mean([m['eligible'] for m in rows]):.0f} прошедших фильтры из "
          f"{np.mean([m['universe'] for m in rows]):.0f}; ПОРОГ: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}")
    print("результат:", path)
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
