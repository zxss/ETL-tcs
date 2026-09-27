"""
Испытания № 288–289 по bonds/PREREG_distress.md (коммит 4a7e109): ВДО без потолка
доходности и покупка на просадках цены («новостной фон»), окно 2022-01 … 2026-08.

Что подаём:  тот же кэш ISS и флаги брокера, что у bonds.backtest.
Что запускаем: python -m bonds.distress --cache <кэш> --flags flags.json
Что получаем:  <cache>/results/bonds_distress_wNN.json — параметры, самопроверки,
               помесячные ряды трёх вариантов (эталон w01, № 288, № 289), таблицы по
               выпускам и событийный анализ просадок; сводка в stdout.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bonds.backtest import (FLOAT_WORDS, bond_return, dsr, issuer_of, key_rates,  # noqa: E402
                            month_ends, rate_on, to_bond)
from bonds.core import build_portfolio, load_rules            # noqa: E402
from bonds.sources import iss                                  # noqa: E402
from tools.flowkit import Checks, save_state, versioned        # noqa: E402

START, END = dt.date(2021, 12, 1), dt.date(2026, 8, 31)       # первый ребаланс — январь 2022
FIRST_HALF_END = dt.date(2023, 12, 31)
N_TRIALS = 289
DIP = -0.10                  # падение чистой цены за месяц
DIP_N, DIP_HOLD, CASH = 20, 3, 0.05
DIP_W = (1 - CASH) / DIP_HOLD / DIP_N
MONTHS_EXPECTED = 56


def px(r):
    return r.get("CLOSE") or r.get("LEGALCLOSEPRICE")


class Market:
    """Календарь, ставки, срезы на концы месяцев и кэш доходностей выпусков по окнам."""

    def __init__(self, cache, flags, start=None, end=None):
        global START, END
        START, END = start or START, end or END
        self.cache, self.flags = cache, flags
        self.rates = key_rates()
        self.cal = iss.trading_days(START - dt.timedelta(days=62), END + dt.timedelta(days=200), cache)
        allends = month_ends(self.cal)
        self.ends = [d for d in allends if START <= d <= END]
        self.prev = {d: allends[allends.index(d) - 1] for d in self.ends}
        self.nxt = {d: self.cal[self.cal.index(d) + 1] for d in allends if d != self.cal[-1]}
        self.allends = allends
        self._ret = {}

    def rows(self, d):
        return iss.history_on(d, self.cache)

    def ret(self, secid, a, b):
        k = (secid, a, b)
        if k not in self._ret:
            self._ret[k] = bond_return(secid, a, b, self.cache, cal=self.cal)
        return self._ret[k]

    def window(self, k):
        """Окно месяца k: с первого торгового дня после конца месяца k до такого же дня следующего."""
        d = self.ends[k]
        nd = self.allends[self.allends.index(d) + 1]
        return self.nxt[d], self.nxt[nd]


def simulate(m: Market, select, hold: int, cost_rt: float) -> dict:
    """Когорты: select(k, held) → {secid: вес}; когорта держится hold месяцев."""
    cohorts, rows, prev_w, held, spells = [], [], {}, set(), []
    for k in range(len(m.ends) - 1):            # последний конец месяца — только граница окна
        a, b = m.window(k)
        if b > m.cal[-1]:
            break
        new, meta = select(k, held)
        cohorts.append({"start": k, "w": new, "meta": meta})
        live = [c for c in cohorts if k - c["start"] < hold]
        w = {}
        for c in live:
            for s, x in c["w"].items():
                w[s] = w.get(s, 0) + x
        key = rate_on(m.rates, m.ends[k])
        rf = key / 100 * (b - a).days / 365
        rets = {s: m.ret(s, a, b) for s in w}
        gross = sum(w[s] * rets[s][0] for s in w) + (1 - sum(w.values())) * rf
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        for c in live:
            for s in c["w"]:
                spells.append({"cohort": c["start"], "k": k, "secid": s, "ret": rets[s][0], "note": rets[s][1]})
        rows.append({"rebalance": str(a), "until": str(b), "key_rate": key, "rf": rf, "gross": gross,
                     "turnover": turnover, "n": len(w), "invested": sum(w.values()),
                     "max_w": max(w.values()) if w else 0, "new": len(new), "meta": meta})
        prev_w, held = w, set(new)
    return {"months": rows, "spells": spells, "cohorts": [{"start": c["start"], "w": c["w"]} for c in cohorts],
            "cost_rt": cost_rt}


def stats(rows, cost_rt, mult):
    ex = np.array([r["gross"] - r["turnover"] * cost_rt * mult / 2 - r["rf"] for r in rows])
    net = np.array([r["gross"] - r["turnover"] * cost_rt * mult / 2 for r in rows])
    eq = np.cumprod(1 + net)
    first = [e for e, r in zip(ex, rows) if dt.date.fromisoformat(r["rebalance"]) <= FIRST_HALF_END]
    second = [e for e, r in zip(ex, rows) if dt.date.fromisoformat(r["rebalance"]) > FIRST_HALF_END]
    years = len(net) / 12
    return {"cost_mult": mult, "months": len(ex), "mean_excess_m": float(ex.mean()),
            "t": float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))),
            "win_share": float((ex > 0).mean()), "worst_month": float(ex.min()), "best_month": float(ex.max()),
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "annual_net": float(eq[-1] ** (1 / years) - 1),
            "annual_rf": float(np.prod([1 + r["rf"] for r in rows]) ** (1 / years) - 1),
            "total_net": float(eq[-1] - 1), "total_rf": float(np.prod([1 + r["rf"] for r in rows]) - 1),
            "first_half_mean": float(np.mean(first)), "second_half_mean": float(np.mean(second)),
            "dsr": dsr(ex, N_TRIALS)}


def gate(s1, s2):
    return bool(s1["mean_excess_m"] > 0 and s1["t"] >= 2 and s1["first_half_mean"] > 0
                and s1["second_half_mean"] > 0 and s2["mean_excess_m"] > 0 and s1["dsr"] >= 0.95)


def rules_select(m, rules):
    def sel(k, held):
        d = m.ends[k]
        key = rate_on(m.rates, d)
        bonds = [b for b in (to_bond(r, d, m.flags) for r in m.rows(d)) if b]
        res = build_portfolio(bonds, rules, key, use_credit=False, holdings=held)
        spread = {p["secid"]: (p["yield_pct"] - key) for p in res["portfolio"]}
        info = {p["secid"]: {"name": None, "spread": spread[p["secid"]], "price": p["price_pct"],
                             "yield": p["yield_pct"]} for p in res["portfolio"]}
        return ({p["secid"]: p["weight"] for p in res["portfolio"]},
                {"eligible": res["eligible"], "violations": res["violations"], "info": info})
    return sel


def dip_candidates(m, k, rules):
    d = m.ends[k]
    key = rate_on(m.rates, d)
    prev = {r["SECID"]: px(r) for r in m.rows(m.prev[d]) if px(r)}
    u = rules["universe"]
    out = []
    for r in m.rows(d):
        b = to_bond(r, d, m.flags)
        if not b or not b.get("price_pct") or not prev.get(b["secid"]):
            continue
        drop = b["price_pct"] / prev[b["secid"]] - 1
        if drop > DIP:
            continue
        if b["qual_only"] or b["required_tests"] or b["floating"] or b["subordinated"] or b["perpetual"]:
            continue
        if b.get("issue_size_rub") is not None and b["issue_size_rub"] < u["min_issue_size_rub"]:
            continue
        if b.get("days_to_event") is None or b["days_to_event"] < u["min_days_to_event"]:
            continue
        if (b.get("value_rub") or 0) < rules["liquidity"]["min_day_value_rub"] or b.get("yield_pct") is None:
            continue
        out.append({**b, "drop": drop, "prev_price": prev[b["secid"]], "spread": b["yield_pct"] - key,
                    "name": r.get("SHORTNAME"), "month_end": str(d)})
    return sorted(out, key=lambda x: (x["drop"], x["secid"]))


def dip_select(m, rules):
    def sel(k, held):
        picked, issuers = [], set()
        for c in dip_candidates(m, k, rules):
            if len(picked) >= DIP_N or c["issuer"] in issuers:
                continue
            picked.append(c)
            issuers.add(c["issuer"])
        info = {c["secid"]: {k2: c[k2] for k2 in ("name", "drop", "prev_price", "price_pct", "spread",
                                                    "yield_pct", "issuer")} for c in picked}
        return {c["secid"]: DIP_W for c in picked}, {"info": info}
    return sel


def event_study(m, rules):
    ev = []
    for k in range(len(m.ends)):
        for c in dip_candidates(m, k, rules):
            d = m.ends[k]
            a = m.nxt[d]
            i = m.allends.index(d)
            rec = {"month_end": str(d), "secid": c["secid"], "name": c["name"], "drop": c["drop"],
                   "price": c["price_pct"], "spread": c["spread"]}
            for h in (1, 3, 6):
                if i + h >= len(m.allends) or m.allends[i + h] > END + dt.timedelta(days=31):
                    continue
                b = m.nxt.get(m.allends[i + h])
                if not b or b > m.cal[-1]:
                    continue
                r, note = m.ret(c["secid"], a, b)
                rf = rate_on(m.rates, d) / 100 * (b - a).days / 365
                rec[f"r{h}"], rec[f"ex{h}"], rec[f"note{h}"] = r, r - rf, note
            ev.append(rec)
    return ev


def by_bond(res, m):
    """Держания по выпускам: месяцев, накопленная доходность, события."""
    out = {}
    for s in res["spells"]:
        o = out.setdefault((s["cohort"], s["secid"]), {"secid": s["secid"], "entry": res["months"][s["cohort"]]["rebalance"],
                                                         "months": 0, "growth": 1.0, "notes": []})
        o["months"] += 1
        o["growth"] *= 1 + s["ret"]
        if s["note"]:
            o["notes"].append(s["note"])
    for (c, sid), o in out.items():
        o["ret"] = o.pop("growth") - 1
        o.update(res["months"][c]["meta"]["info"].get(sid, {}))
    return list(out.values())


def merge_spells_288(res):
    """№ 288: подряд идущие месяцы одного выпуска — одно держание."""
    by, cur = [], {}
    months = res["months"]
    for k, mo in enumerate(months):
        info = mo["meta"]["info"]
        for sid in list(cur):
            if sid not in info:
                by.append(cur.pop(sid))
        for sid, inf in info.items():
            r = next(s for s in res["spells"] if s["k"] == k and s["secid"] == sid)
            if sid not in cur:
                cur[sid] = {"secid": sid, "entry": mo["rebalance"], "months": 0, "growth": 1.0,
                            "entry_spread": inf["spread"], "entry_price": inf["price"], "entry_yield": inf["yield"],
                            "max_spread": inf["spread"], "notes": []}
            c = cur[sid]
            c["months"] += 1
            c["growth"] *= 1 + r["ret"]
            c["max_spread"] = max(c["max_spread"], inf["spread"])
            if r["note"]:
                c["notes"].append(r["note"])
    by += list(cur.values())
    for c in by:
        c["ret"] = c.pop("growth") - 1
    return by


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="проверка кода на 2021-02…2021-07 (вне окна испытания), результат не засчитывается")
    a = ap.parse_args(argv)
    flags = json.load(open(a.flags, encoding="utf-8"))
    global MONTHS_EXPECTED
    if a.smoke:
        MONTHS_EXPECTED = 5
        m = Market(a.cache, flags, dt.date(2021, 2, 1), dt.date(2021, 7, 31))
    else:
        m = Market(a.cache, flags)
    rules = load_rules()
    cost = rules["execution"]["assumed_round_trip_pct_backtest"] / 100
    r288 = copy.deepcopy(rules)
    r288["yield"]["max_spread_to_key_pp"] = float("inf")

    base = simulate(m, rules_select(m, rules), 1, cost)
    t288 = simulate(m, rules_select(m, r288), 1, cost)
    t289 = simulate(m, dip_select(m, rules), DIP_HOLD, cost)
    ev = event_study(m, rules)

    names = {}
    for d in m.ends:
        for r in m.rows(d):
            names.setdefault(r["SECID"], r.get("SHORTNAME"))
    spells288 = merge_spells_288(t288)
    for s in spells288:
        s["name"] = names.get(s["secid"])
    dist288 = [s for s in spells288 if s["max_spread"] > 15]
    picks289 = by_bond(t289, m)

    S = {}
    for nm, res in (("w01", base), ("288", t288), ("289", t289)):
        s1, s2 = stats(res["months"], cost, 1), stats(res["months"], cost, 2)
        S[nm] = {"x1": s1, "x2": s2, "passed": gate(s1, s2)}

    def evs(h):
        x = np.array([e[f"ex{h}"] for e in ev if f"ex{h}" in e])
        r = np.array([e[f"r{h}"] for e in ev if f"r{h}" in e])
        bad = sum(1 for e in ev if "дефолт" in str(e.get(f"note{h}", "")))
        halt = sum(1 for e in ev if "остановка" in str(e.get(f"note{h}", "")) or "дефолт" in str(e.get(f"note{h}", "")))
        return {"n": int(len(x)), "mean_ret": float(r.mean()) if len(r) else None,
                "median_ret": float(np.median(r)) if len(r) else None,
                "mean_excess": float(x.mean()) if len(x) else None,
                "t_naive": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else None,
                "share_pos": float((x > 0).mean()) if len(x) else None, "defaults": bad, "halts": halt}
    ES = {f"{h}m": evs(h) for h in (1, 3, 6)}

    chk = Checks("bonds-distress")
    for nm, res in (("w01", base), ("288", t288), ("289", t289)):
        chk.eq(f"{nm}: месяцев {MONTHS_EXPECTED}", len(res["months"]), MONTHS_EXPECTED)
        chk.ok(f"{nm}: вложено ≤ 95 %", all(r["invested"] <= 1 - CASH + 1e-9 for r in res["months"]))
        chk.ok(f"{nm}: вес выпуска ≤ 5 %", all(r["max_w"] <= 0.05 + 1e-9 for r in res["months"]))
    chk.ok("288: ограничения портфеля соблюдены", all(not r["meta"]["violations"] for r in t288["months"]))
    chk.ok("289: ≤ 20 бумаг в когорте", all(len(c["w"]) <= DIP_N for c in t289["cohorts"]))
    chk.ok("289: один эмитент в когорте", all(len({(r["meta"]["info"][s]["issuer"]) for s in r["meta"]["info"]})
                                              == len(r["meta"]["info"]) for r in t289["months"]))
    chk.ok("289: все покупки — падение ≥ 10 %", all(i["drop"] <= DIP + 1e-12 for r in t289["months"]
                                                  for i in r["meta"]["info"].values()))
    for nm, res in (("288", t288), ("289", t289)):
        chk.close(f"{nm}: среднее избытка пересчитано", S[nm]["x1"]["mean_excess_m"], float(np.mean(
            [r["gross"] - r["turnover"] * cost / 2 - r["rf"] for r in res["months"]])), tol=1e-12)
        chk.ok(f"{nm}: ПРОЙДЕН ⇒ все условия", (not S[nm]["passed"]) or (S[nm]["x1"]["t"] >= 2 and S[nm]["x1"]["dsr"] >= 0.95))

    out_dir = a.out or os.path.join(a.cache, "results")
    path = versioned(out_dir, "bonds_distress_smoke" if a.smoke else "bonds_distress", "json")
    for res in (base, t288, t289):
        for r in res["months"]:
            r["meta"] = {k: v for k, v in r["meta"].items() if k != "info"} | {"picks": list(r["meta"]["info"])}
    save_state(path, params={"prereg": "bonds/PREREG_distress.md", "smoke": a.smoke, "START": START, "END": END, "DIP": DIP,
                             "DIP_N": DIP_N, "DIP_HOLD": DIP_HOLD, "N_TRIALS": N_TRIALS, "rules": rules["_path"]},
               results={"stats": S, "event_study": ES, "distress_288": dist288, "picks_289": picks289,
                        "events": ev, "months": {"w01": base["months"], "288": t288["months"], "289": t289["months"]}},
               checks=chk, inputs={"cache": a.cache, "flags": a.flags})
    print(chk.report())
    for nm, lab in (("w01", "эталон rules_w01"), ("288", "№ 288 без потолка"), ("289", "№ 289 просадки")):
        for mult in ("x1", "x2"):
            s = S[nm][mult]
            print(f"{lab:<18} {mult}: {s['annual_net']*100:5.1f} %/год против ставки {s['annual_rf']*100:4.1f} %; "
                  f"избыток {s['mean_excess_m']*100:+.3f} %/мес t {s['t']:+.2f}; 2022–23 {s['first_half_mean']*100:+.3f}, "
                  f"2024–26 {s['second_half_mean']*100:+.3f}; худший {s['worst_month']*100:+.2f} %, просадка {s['max_dd']*100:.1f} %; DSR {s['dsr']:.3f}")
        print(f"{'':<18} ПОРОГ: {'ПРОЙДЕН' if S[nm]['passed'] else 'не пройден'}")
    for h, e in ES.items():
        if not e["n"]:
            print(f"события просадок {h}: нет данных")
            continue
        print(f"события просадок {h}: n {e['n']}, средняя {e['mean_ret']*100:+.2f} %, медиана {e['median_ret']*100:+.2f} %, "
              f"избыток {e['mean_excess']*100:+.2f} % (t {e['t_naive']:+.2f}), в плюс к ставке {e['share_pos']:.0%}, "
              f"остановок {e['halts']}, дефолтов −50 % {e['defaults']}")
    print(f"№ 288: держаний со спредом > 15 п.п.: {len(dist288)}; № 289: покупок {len(picks289)}")
    print("результат:", path)
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
