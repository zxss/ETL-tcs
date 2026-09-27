"""
Бэктест премии за волатильность строго по research/vrp/PREREG.md (коммит 4bb2e22).

Ничего из параметров здесь не подбирается: структура, вход, издержки, тайминг,
выборки и порог записаны в предрегистрации до просмотра цен.

Запуск на сервере после выгрузки (research.vrp.download):
    python -m research.vrp.backtest --data /opt/research-data/vrp
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from research.vrp.secid import parse

WING = 0.10                  # крылья ±10 % от фьючерса в день входа
LOOKBACK_DAYS = 30           # вход — первый торговый день ≥ E − 30 календарных дней
COST_PCT = 0.05              # 5 % премии ноги
COST_TICKS = 2               # но не меньше 2 шагов цены
FEE_RUB = 2.0                # ₽ за контракт на вход, на ногу
TICK = {"RTS": 5.0, "Si": 1.0}
PREFIX = {"RTS": "RI", "Si": "Si"}
DEV_END = dt.date(2021, 12, 31)
N_TRIALS = 286
HAR_MIN = 250
H = 21


# ── данные ───────────────────────────────────────────────────────────────────

def _f(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_options(data: str, asset: str):
    """{(date, secid): settle}, метаданные серий, экспирации месячных серий."""
    px, last_seen = {}, {}
    for path in sorted(glob.glob(os.path.join(data, f"options_{asset}_*.csv"))):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = dt.date.fromisoformat(r["TRADEDATE"])
                sid = r["SECID"]
                last_seen[sid] = max(last_seen.get(sid, d), d)
                s = _f(r["SETTLEPRICE"])
                if s:
                    px[(d, sid)] = s
    meta = {}
    for sid, last in last_seen.items():
        p = parse(sid, last.year)
        if p and not p["weekly"]:
            p["expiry"] = last
            meta[sid] = p
    # экспирация месяца — самая частая последняя дата среди его месячных серий
    by_month = defaultdict(lambda: defaultdict(int))
    for sid, p in meta.items():
        by_month[(p["year"], p["month"])][p["expiry"]] += 1
    expiries = {ym: max(c.items(), key=lambda kv: kv[1])[0] for ym, c in by_month.items()}
    # серии, ещё живые на конец выгрузки, «последним днём» получают последний день
    # данных — это не экспирация, такие месяцы исключаются
    last_data = max(d for d, _ in px) if px else None
    expiries = {ym: e for ym, e in expiries.items() if last_data and e < last_data}
    return px, meta, expiries


def load_futures(data: str, asset: str) -> dict:
    """{secid: {date: settle}} только одиночные контракты (без спредов)."""
    # старые контракты ISS отдаёт с суффиксом года (RIH4_2014): суффикс снимается,
    # десятилетия внутри одного кода разделяет близость даты к экспирации
    rx = re.compile(rf"^({PREFIX[asset]}[HMUZ]\d)(?:_\d{{4}})?$")
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(data, f"futures_{asset}_*.csv"))):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = rx.match(r["SECID"])
                if m:
                    s = _f(r["SETTLEPRICE"])
                    if s:
                        out[m.group(1)][dt.date.fromisoformat(r["TRADEDATE"])] = s
    return out


def underlying_code(asset: str, year: int, month: int) -> tuple[str, int]:
    q = next(qm for qm in (3, 6, 9, 12) if qm >= month)
    return f"{PREFIX[asset]}{'HMUZ'[q // 3 - 1]}{year % 10}", year


def fut_price(fut: dict, code: str, day: dt.date, near: dt.date, after=False):
    """Цена контракта code в день day; десятилетие кода — по близости к near."""
    series = fut.get(code, {})
    if day in series and abs((day - near).days) < 400:
        return series[day], day
    if after:        # закрытие биржи: первая доступная цена после day
        later = sorted(d for d in series if d > day and (d - near).days < 400)
        if later:
            return series[later[0]], later[0]
    return None, None


def iss_index(sec: str) -> dict:
    out, s = {}, 0
    while True:
        u = (f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/"
             f"{sec}.json?iss.meta=off&from=2013-01-01&start={s}"
             f"&history.columns=TRADEDATE,CLOSE")
        h = json.loads(urllib.request.urlopen(u, timeout=60).read())["history"]["data"]
        for d, c in h:
            if c:
                out[dt.date.fromisoformat(d)] = float(c)
        if len(h) < 100:
            return out
        s += len(h)


def key_rate_history() -> list[tuple[dt.date, float]]:
    """Ключевая ставка ЦБ с 2013 (SOAP DailyInfo.KeyRate), по возрастанию дат."""
    env = ('<?xml version="1.0" encoding="utf-8"?><soap:Envelope '
           'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
           '<KeyRate xmlns="http://web.cbr.ru/"><fromDate>2013-09-01</fromDate>'
           f'<ToDate>{dt.date.today()}</ToDate></KeyRate></soap:Body></soap:Envelope>')
    req = urllib.request.Request(
        "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx", data=env.encode(),
        headers={"Content-Type": "text/xml; charset=utf-8",
                 "SOAPAction": '"http://web.cbr.ru/KeyRate"'})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=60).read())
    out = []
    for node in root.iter():
        if node.tag.endswith("KR"):
            d = r = None
            for ch in node:
                t = ch.tag.rsplit("}", 1)[-1]
                if t == "DT":
                    d = dt.date.fromisoformat(ch.text[:10])
                elif t == "Rate":
                    r = float(ch.text)
            if d and r:
                out.append((d, r))
    return sorted(out)


def rate_on(rates, day) -> float:
    lo = [r for d, r in rates if d <= day]
    return lo[-1] if lo else rates[0][1]


# ── HAR и подразумеваемая волатильность ──────────────────────────────────────

class Har:
    """HAR по дневным доходностям: прогноз 21-дневной реализованной волы, % годовых.

    Признаки на день j: RV_d, среднее за 5 и за 22 дня квадратов доходностей
    (годовых); цель — среднее за j+1..j+21. МНК на расширяющемся окне: для прогноза
    в день i берутся только наблюдения, чья цель целиком закончилась до i.
    """

    def __init__(self, returns: dict):
        self.days = sorted(returns)
        sq = np.array([returns[d] for d in self.days]) ** 2 * 252
        n = len(sq)
        c = np.concatenate([[0.0], np.cumsum(sq)])
        idx = np.arange(n)
        self.xd = sq
        self.xw = np.where(idx >= 4, (c[idx + 1] - c[np.maximum(idx - 4, 0)]) / 5, np.nan)
        self.xm = np.where(idx >= 21, (c[idx + 1] - c[np.maximum(idx - 21, 0)]) / 22, np.nan)
        self.y = np.where(idx + H < n, (c[np.minimum(idx + 1 + H, n)] - c[idx + 1]) / H, np.nan)
        self.pos = {d: i for i, d in enumerate(self.days)}

    def at(self, day: dt.date) -> float | None:
        """Прогноз на день входа (по данным ДО этого дня включительно − 1)."""
        prior = [d for d in self.days if d < day]
        if not prior:
            return None
        i = self.pos[prior[-1]]
        js = np.arange(21, i - H + 1)
        if len(js) < HAR_MIN:
            return None
        X = np.column_stack([np.ones(len(js)), self.xd[js], self.xw[js], self.xm[js]])
        beta, *_ = np.linalg.lstsq(X, self.y[js], rcond=None)
        x = np.array([1.0, self.xd[i], self.xw[i], self.xm[i]])
        return math.sqrt(max(float(x @ beta), 1e-6)) * 100


def log_returns(closes: dict) -> dict:
    days = sorted(closes)
    return {days[k]: math.log(closes[days[k]] / closes[days[k - 1]]) for k in range(1, len(days))}


def same_contract_returns(fut: dict) -> dict:
    """Доходность дня — медиана по контрактам, торговавшимся и вчера, и сегодня:
    без ложных скачков на смене ближайшего контракта."""
    per_day = defaultdict(list)
    for ser in fut.values():
        ds = sorted(ser)
        for k in range(1, len(ds)):
            if (ds[k] - ds[k - 1]).days <= 5:
                per_day[ds[k]].append(math.log(ser[ds[k]] / ser[ds[k - 1]]))
    return {d: float(np.median(v)) for d, v in per_day.items()}


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black76(F, K, T, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) if kind == "C" else max(K - F, 0)
    d1 = (math.log(F / K) + sigma * sigma * T / 2) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return F * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def implied_vol(price, F, K, T, kind):
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if black76(F, K, T, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2 * 100


# ── циклы ────────────────────────────────────────────────────────────────────

def cycles(asset, px, meta, expiries, fut, usdrub, rates, cost_mult=1.0):
    trade_days = sorted({d for d, _ in px})
    by_exp = defaultdict(list)
    for sid, p in meta.items():
        by_exp[(p["year"], p["month"])].append(sid)
    out, skipped = [], defaultdict(int)
    for (y, m), E in sorted(expiries.items()):
        entry = next((d for d in trade_days if d >= E - dt.timedelta(days=LOOKBACK_DAYS)), None)
        if entry is None or entry >= E:
            skipped["нет дня входа"] += 1
            continue
        code, _ = underlying_code(asset, y, m)
        F, _ = fut_price(fut, code, entry, E)
        if not F:
            skipped["нет цены фьючерса на входе"] += 1
            continue
        calls = {meta[s]["strike"]: px.get((entry, s)) for s in by_exp[(y, m)]
                 if meta[s]["type"] == "C" and px.get((entry, s))}
        puts = {meta[s]["strike"]: px.get((entry, s)) for s in by_exp[(y, m)]
                if meta[s]["type"] == "P" and px.get((entry, s))}
        both = sorted(set(calls) & set(puts))
        if not both:
            skipped["нет пары колл/пут"] += 1
            continue
        K0 = min(both, key=lambda k: abs(k - F))
        up = [k for k in calls if k > K0]
        dn = [k for k in puts if k < K0]
        if not up or not dn:
            skipped["нет крыльев"] += 1
            continue
        Kc = min(up, key=lambda k: abs(k - F * (1 + WING)))
        Kp = min(dn, key=lambda k: abs(k - F * (1 - WING)))
        legs = [calls[K0], puts[K0], calls[Kc], puts[Kp]]
        credit = legs[0] + legs[1] - legs[2] - legs[3]
        # издержки в пунктах цены: 5 % премии ноги, не меньше 2 шагов, + сбор
        rub_per_pt = 1.0 if asset == "Si" else 0.02 * usdrub.get(entry, 75.0)
        cost = sum(max(COST_PCT * p, COST_TICKS * TICK[asset]) for p in legs)
        cost += 4 * FEE_RUB / rub_per_pt
        cost *= cost_mult
        max_loss = max(Kc - K0, K0 - Kp) - credit + cost
        if credit <= 0 or max_loss <= 0:
            skipped["вырожденная структура"] += 1
            continue
        S, s_day = fut_price(fut, code, E, E, after=True)
        if not S:
            skipped["нет цены фьючерса на экспирации"] += 1
            continue
        pnl = credit - abs(S - K0) + max(S - Kc, 0) + max(Kp - S, 0) - cost
        days = (E - entry).days
        rf = rate_on(rates, entry) / 100 * days / 365
        T = max(days, 1) / 365
        iv = (implied_vol(calls[K0], F, K0, T, "C") + implied_vol(puts[K0], F, K0, T, "P")) / 2
        out.append({"expiry": E.isoformat(), "entry": entry.isoformat(), "F": F, "K0": K0,
                    "Kc": Kc, "Kp": Kp, "credit": credit, "cost": cost,
                    "max_loss": max_loss, "S": S, "settle_day": s_day.isoformat(),
                    "pnl": pnl, "ret": pnl / max_loss, "rf": rf,
                    "excess": pnl / max_loss - rf, "iv_atm": iv})
    return out, dict(skipped)


# ── статистика ───────────────────────────────────────────────────────────────

def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def p_one_sided(t, n):
    from scipy.stats import t as tdist
    return float(1 - tdist.cdf(t, n - 1)) if np.isfinite(t) else 1.0


def dsr(x, n_trials):
    from scipy.stats import norm
    x = np.asarray(x, float)
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    v = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1)
    em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / n_trials)
                          + em * norm.ppf(1 - 1 / (n_trials * math.e)))
    return float(norm.cdf((sr - sr0) / math.sqrt(v)))


def summarize(name, cyc):
    x = np.array([c["excess"] for c in cyc])
    dev = np.array([c["excess"] for c in cyc if dt.date.fromisoformat(c["expiry"]) <= DEV_END])
    hold = np.array([c["excess"] for c in cyc if dt.date.fromisoformat(c["expiry"]) > DEV_END])
    t = tstat(x)
    return {"trial": name, "n": len(x), "mean_excess": float(x.mean()) if len(x) else None,
            "t": t, "p": p_one_sided(t, len(x)) if len(x) > 2 else 1.0,
            "worst": float(x.min()) if len(x) else None,
            "win_share": float((x > 0).mean()) if len(x) else None,
            "dev_n": len(dev), "dev_mean": float(dev.mean()) if len(dev) else None,
            "hold_n": len(hold), "hold_mean": float(hold.mean()) if len(hold) else None,
            "dsr": dsr(x, N_TRIALS),
            "annual_excess": float(x.mean() * 12) if len(x) else None}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args(argv)
    rates = key_rate_history()
    rvi = iss_index("RVI")
    rtsi = iss_index("RTSI")
    har_rts = Har(log_returns(rtsi))
    results, all_cycles = [], {}
    si_fut = load_futures(a.data, "Si")
    usdrub = {}
    for code, ser in si_fut.items():
        for d, p in ser.items():
            usdrub.setdefault(d, p / 1000.0)
    for asset in ("RTS", "Si"):
        px, meta, expiries = load_options(a.data, asset)
        fut = load_futures(a.data, asset)
        har = Har(same_contract_returns(fut)) if asset == "Si" else har_rts
        for mult in (1.0, 2.0):
            cyc, skipped = cycles(asset, px, meta, expiries, fut, usdrub, rates, mult)
            fam = "A-RI" if asset == "RTS" else "B-Si"
            timed = []
            for c in cyc:
                e = dt.date.fromisoformat(c["entry"])
                fc = har.at(e)
                implied = rvi.get(e) if asset == "RTS" else c["iv_atm"]
                if fc is not None and implied is not None and implied >= fc:
                    timed.append(c)
            for name, sub in ((f"{fam} безусловное", cyc), (f"{fam} тайминг HAR", timed)):
                s = summarize(name, sub)
                s["cost_mult"] = mult
                s["skipped"] = skipped
                results.append(s)
                all_cycles[f"{name} ×{mult:g}"] = sub
    base = [r for r in results if r["cost_mult"] == 1.0]
    adj = holm(np.array([r["p"] for r in base]))
    for r, p in zip(base, adj):
        r["p_holm"] = float(p)
    x2 = {r["trial"]: r for r in results if r["cost_mult"] == 2.0}
    for r in base:
        r2 = x2[r["trial"]]
        r["passed"] = bool(
            r["n"] > 2 and r["mean_excess"] > 0 and r["t"] >= 2
            and (r["dev_mean"] or -1) > 0 and (r["hold_mean"] or -1) > 0
            and r["p_holm"] < 0.05 and np.isfinite(r["dsr"]) and r["dsr"] >= 0.95
            and (r2["mean_excess"] or -1) > 0)
    out = os.path.join(a.data, "backtest_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "cycles": all_cycles}, f, ensure_ascii=False,
                  indent=1, default=str)
    print("=" * 96)
    print(f"{'испытание':<22}{'изд.':>5}{'циклов':>7}{'изб./цикл':>10}{'t':>7}"
          f"{'Холм p':>8}{'худший':>9}{'в плюс':>8}{'разраб.':>9}{'отлож.':>9}{'DSR':>7}{'ПРОЙДЕН':>9}")
    for r in results:
        print(f"{r['trial']:<22}{r['cost_mult']:>4g}×{r['n']:>7}"
              f"{(r['mean_excess'] or 0):>+10.3f}{r['t']:>+7.2f}"
              f"{r.get('p_holm', float('nan')):>8.3f}{(r['worst'] or 0):>+9.2f}"
              f"{(r['win_share'] or 0):>8.0%}{(r['dev_mean'] or 0):>+9.3f}"
              f"{(r['hold_mean'] or 0):>+9.3f}{r['dsr']:>7.3f}"
              f"{('ДА' if r.get('passed') else 'нет' if 'passed' in r else ''):>9}")
    print("пропуски циклов:", base[0]["skipped"], "|", base[2]["skipped"])
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
