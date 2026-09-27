"""
MOEX ISS (публичный, без токена) — рыночные данные облигаций.

  snapshot()          текущий срез доски TQCB: цена, доходность к событию, дюрация,
                      bid/offer, оборот, купон, даты оферт и погашения;
  history_on(date)    срез истории за дату (все выпуски), для бэктеста;
  daily(secid, a, b)  дневная история выпуска: цена, НКД, номинал, купон.

История кэшируется в <cache>/iss/…json: повторный бэктест сеть не трогает.
Страница ISS — не больше 100 строк; сеть рвётся — повторы с паузой.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

ISS = "https://iss.moex.com/iss"
BOARD = "TQCB"


def _get(url: str, tries: int = 6) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except (OSError, ValueError) as e:
            if k == tries - 1:
                raise RuntimeError(f"ISS не ответил: {url} ({e})") from e
            time.sleep(min(60, 2 ** k))
    return {}


def _paged(url: str, block: str) -> list[dict]:
    rows, s = [], 0
    while True:
        j = _get(f"{url}&start={s}")
        cols, data = j[block]["columns"], j[block]["data"]
        rows += [dict(zip(cols, r)) for r in data]
        cur = j.get(f"{block}.cursor")
        total = cur["data"][0][1] if cur else None
        s += len(data)
        if not data or (total is not None and s >= total) or (total is None and len(data) < 100):
            return rows
        time.sleep(0.15)


def _cached(cache: str | None, name: str, fetch):
    if cache:
        path = os.path.join(cache, "iss", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    data = fetch()
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:  # flow-state: кэш сырых данных
            json.dump(data, f, ensure_ascii=False)
    return data


def _date(x):
    try:
        return dt.date.fromisoformat(str(x)[:10]) if x and str(x)[:4] != "0000" else None
    except ValueError:
        return None


def next_event(row: dict, asof: dt.date) -> dt.date | None:
    """Ближайшее будущее событие: оферта, put/call, выкуп или погашение."""
    ds = [_date(row.get(k)) for k in ("OFFERDATE", "PUTOPTIONDATE", "CALLOPTIONDATE",
                                       "BUYBACKDATE", "MATDATE")]
    ds = [d for d in ds if d and d > asof]
    return min(ds) if ds else None


def snapshot() -> list[dict]:
    """Текущий срез TQCB: securities + marketdata по SECID."""
    j = _get(f"{ISS}/engines/stock/markets/bonds/boards/{BOARD}/securities.json?iss.meta=off")
    sec = {r[0]: dict(zip(j["securities"]["columns"], r)) for r in j["securities"]["data"]}
    md = {r[0]: dict(zip(j["marketdata"]["columns"], r)) for r in j["marketdata"]["data"]}
    today = dt.date.today()
    out = []
    for sid, s in sec.items():
        m = md.get(sid, {})
        ev = next_event(s, today)
        has_offer = ev is not None and ev != _date(s.get("MATDATE"))
        y = (m.get("YIELDTOOFFER") if has_offer else None) or m.get("YIELD") \
            or s.get("YIELDATPREVWAPRICE")
        price = m.get("LAST") or m.get("WAPRICE") or s.get("PREVWAPRICE") or s.get("PREVPRICE")
        bid, ask = m.get("BID"), m.get("OFFER")
        out.append({
            "secid": sid, "isin": s.get("ISIN"), "name": s.get("SHORTNAME"),
            "face_unit": s.get("FACEUNIT"), "face": s.get("FACEVALUE"),
            "yield_pct": y, "price_pct": price,
            "duration_y": (m.get("DURATION") or 0) / 365 if m.get("DURATION") else None,
            "days_to_event": (ev - today).days if ev else None, "event": str(ev) if ev else None,
            "value_rub": m.get("VALTODAY") or 0, "bid": bid, "ask": ask,
            "bid_ask_pct": (ask - bid) / ask * 100 if bid and ask else None,
            "issue_size_rub": (s.get("ISSUESIZEPLACED") or 0) * (s.get("FACEVALUE") or 0) or None,
            "coupon_pct": s.get("COUPONPERCENT"), "accrued": s.get("ACCRUEDINT"),
            "lot": s.get("LOTSIZE"), "halted": m.get("TRADINGSTATUS") not in (None, "T", "N"),
            "listlevel": s.get("LISTLEVEL"),
        })
    return out


def history_on(day: dt.date, cache: str | None) -> list[dict]:
    """Срез истории ISS за дату по доске TQCB (все выпуски)."""
    cols = ("BOARDID,SECID,SHORTNAME,CLOSE,LEGALCLOSEPRICE,ACCINT,FACEVALUE,FACEUNIT,"
            "YIELDCLOSE,YIELDTOOFFER,DURATION,VALUE,NUMTRADES,MATDATE,OFFERDATE,"
            "PUTOPTIONDATE,CALLOPTIONDATE,BUYBACKDATE,COUPONPERCENT,COUPONVALUE,COUPON_DETAILS")
    def fetch():
        rows = _paged(f"{ISS}/history/engines/stock/markets/bonds/boards/{BOARD}/securities.json"
                      f"?iss.meta=off&date={day}&history.columns={cols}", "history")
        return rows
    return _cached(cache, f"hist_{day}.json", fetch)


def daily(secid: str, a: dt.date, b: dt.date, cache: str | None) -> list[dict]:
    """Дневная история выпуска на [a, b] — цена, НКД, номинал, купон."""
    cols = "TRADEDATE,CLOSE,LEGALCLOSEPRICE,ACCINT,FACEVALUE,COUPONVALUE,VALUE,MATDATE"
    def fetch():
        return _paged(f"{ISS}/history/engines/stock/markets/bonds/boards/{BOARD}/securities/"
                      f"{secid}.json?iss.meta=off&from={a}&till={b}&history.columns={cols}",
                      "history")
    return _cached(cache, f"daily_{secid}_{a}_{b}.json", fetch)


def trading_days(a: dt.date, b: dt.date, cache: str | None) -> list[dt.date]:
    """Торговые дни — по истории индекса корпоративных облигаций RUCBITR."""
    def fetch():
        return _paged(f"{ISS}/history/engines/stock/markets/index/securities/RUCBITR.json"
                      f"?iss.meta=off&from={a}&till={b}&history.columns=TRADEDATE", "history")
    rows = _cached(cache, f"days_{a}_{b}.json", fetch)
    return sorted({dt.date.fromisoformat(r["TRADEDATE"]) for r in rows})
