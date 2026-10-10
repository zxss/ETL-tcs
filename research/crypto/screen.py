"""
Срез рынка криптовалют на сегодня — ОПИСАТЕЛЬНО, не испытание и не прогноз.

Что подаём:  живые данные Binance (API data-api.binance.vision), CoinGecko (капитализация,
             FDV, падение от максимума), GeckoTerminal (пулы сети TON); необязательно —
             --trials <crypto_trials_wNN.json>: какие сигналы прошли проверку.
Что запускаем: python -m research.crypto.screen --out <каталог> [--trials …]
Что получаем:  <out>/crypto_screen_wNN.json и .md: ETH, BTC, телеграм-токены, ликвидные
               альты с факторами (моментум 28 дней без последнего, 7 дней, волатильность
               30 дней, оборот) и пометкой «в лонге по прошедшему проверку сигналу».
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.parse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from research.crypto.download import exclusion, get  # noqa: E402
from tools.flowkit import Checks, save_state, versioned  # noqa: E402

API = "https://data-api.binance.vision/api/v3"
CG = "https://api.coingecko.com/api/v3"
GT = "https://api.geckoterminal.com/api/v2"
TG_IDS = {"the-open-network": "TON", "notcoin": "NOT", "dogs-2": "DOGS", "hamster-kombat": "HMSTR",
          "catizen": "CATI", "major": "MAJOR", "blum": "BLUM", "x-empire": "X"}
MIN_LIQ = 5e6


def klines(sym: str, n: int = 400):
    data = json.loads(get(f"{API}/klines?" + urllib.parse.urlencode({"symbol": sym, "interval": "1d", "limit": n})))
    return [(dt.datetime.fromtimestamp(int(r[0]) / (1e6 if int(r[0]) > 1e14 else 1e3), dt.timezone.utc).date(),
             float(r[4]), float(r[7])) for r in data]


def factors(rows):
    if len(rows) < 61:
        return None
    rows = rows[:-1] if rows[-1][0] >= dt.datetime.now(dt.timezone.utc).date() else rows   # без незакрытого дня
    c = np.array([r[1] for r in rows]); v = np.array([r[2] for r in rows])
    r = c[1:] / c[:-1] - 1
    f = lambda k: float(c[-1] / c[-1 - k] - 1) if len(c) > k else None
    return {"date": str(rows[-1][0]), "close": float(c[-1]), "r7": f(7), "r30": f(30), "r90": f(90), "r365": f(365),
            "mom28": float(c[-2] / c[-29] - 1), "vol30": float(np.std(r[-30:], ddof=1) * math.sqrt(365)),
            "liq30": float(np.median(v[-30:])), "dd365": float(c[-1] / c[-366:].max() - 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", default=None)
    a = ap.parse_args(argv)
    passed = set()
    if a.trials:
        fac = json.load(open(a.trials, encoding="utf-8"))["results"]["factors"]
        passed = {k for k, v in fac.items() if v.get("passed")}
    info = json.loads(get(f"{API}/exchangeInfo?permissions=SPOT"))
    syms = [s["symbol"] for s in info["symbols"] if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
            and not exclusion(s["symbol"])]
    res = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for s, rows in zip(syms, ex.map(lambda x: klines(x), syms)):
            f = factors(rows)
            if f:
                res[s] = f
    univ = {s: f for s, f in res.items() if f["liq30"] >= MIN_LIQ}
    k = max(1, len(univ) // 5)
    rank = lambda key, rev: [s for s, _ in sorted(univ.items(), key=lambda x: x[1][key], reverse=rev)[:k]]
    q = {"mom": rank("mom28", True), "rev": rank("r7", False), "lowvol": rank("vol30", False)}
    for s, f in univ.items():
        f["quintile_flags"] = [n for n, lst in q.items() if s in lst]
        f["long_validated"] = [n for n in f["quintile_flags"] if n in passed]
    cg = []
    for page in (1, 2):
        cg += json.loads(get(f"{CG}/coins/markets?vs_currency=usd&per_page=250&page={page}&price_change_percentage=30d,1y"))
        time.sleep(3)
    cgm = {}
    for x in cg:
        cgm.setdefault(x["symbol"].upper(), x)
    tg = json.loads(get(f"{CG}/coins/markets?vs_currency=usd&ids={','.join(TG_IDS)}&price_change_percentage=7d,30d,1y"))
    pools = json.loads(get(f"{GT}/networks/ton/pools?page=1&sort=h24_volume_usd_desc"))["data"]

    def cgi(x):
        return {"mc": x.get("market_cap"), "fdv": x.get("fully_diluted_valuation"),
                "fdv_mc": (x["fully_diluted_valuation"] / x["market_cap"]) if x.get("market_cap") and x.get("fully_diluted_valuation") else None,
                "ath_change": x.get("ath_change_percentage"), "r30": x.get("price_change_percentage_30d_in_currency"),
                "r1y": x.get("price_change_percentage_1y_in_currency"), "price": x.get("current_price"),
                "volume": x.get("total_volume")}
    for s, f in univ.items():
        m = cgm.get(s[:-4])
        f["cg"] = cgi(m) if m else None
    out = {"asof": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "validated": sorted(passed),
           "universe": len(univ), "pairs": len(res),
           "majors": {s: res.get(s) for s in ("BTCUSDT", "ETHUSDT")},
           "telegram": [{"id": x["id"], "symbol": TG_IDS[x["id"]], **cgi(x),
                         "binance": res.get(TG_IDS[x["id"]] + "USDT")} for x in tg],
           "ton_pools": [{"name": p["attributes"]["name"], "vol24": float(p["attributes"]["volume_usd"]["h24"] or 0),
                          "liquidity": float(p["attributes"]["reserve_in_usd"] or 0),
                          "ch24": p["attributes"]["price_change_percentage"].get("h24"),
                          "created": p["attributes"]["pool_created_at"][:10]} for p in pools],
           "alts": univ}
    chk = Checks("crypto-screen")
    chk.ok("ETH и BTC получены", all(out["majors"].values()))
    chk.ok("вселенная ≥ 30 ликвидных пар", len(univ) >= 30, str(len(univ)))
    chk.ok("квинтиль = вселенная/5", all(len(v) == k for v in q.values()))
    path = versioned(a.out, "crypto_screen", "json")
    save_state(path, params={"MIN_LIQ": MIN_LIQ, "validated": sorted(passed)}, results=out, checks=chk)
    print(chk.report())
    print(f"пар {len(res)}, ликвидных {len(univ)}, прошедших проверку сигналов: {sorted(passed) or 'нет'}")
    print("результат:", path)
    return chk.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
