"""
Кэрри фьючерсов против TMON (ТЗ пользователя 24.09.2026, гипотеза B3 —
казначейство, не прогноз).

Связка «купить спот + продать квартальный фьючерс до экспирации» фиксирует
доходность F/S − 1 без рыночного риска (расчётный фьючерс исполняется по
фиксингу, спот продаётся по тому же фиксингу/рынку). Если она стабильно выше
TMON после издержек и маржи — у простаивающих денег есть доходность лучше.
Инструменты: юань (CR ↔ CNYRUB_TOM) и золото (GL ↔ GLDRUB_TOM). Акционный
cash-and-carry уже проверен (H5, +1,1 %/год, t 2,5 — не прошёл) — не повторяем.

Расчёт на каждый вход (ближний контракт, до экспирации 20…100 дней):
  locked   = F/S − 1 − издержки (спот туда-обратно 2×(0,04 комиссия + 0,02
             спред) + фьючерс 2×0,01) = F/S − 1 − 0,14 %;
  капитал  = S × (1 + ГО), ГО 15 % не приносит дохода (консервативно);
  tmon     = TMON[exp]/TMON[t] − 1 за тот же срок (фактический, ex-post);
             до листинга TMON — RUSFAR, начисленный по дням, минус 0,4 %/год.
  excess   = locked/(1+ГО) − tmon, в %/год.
Для t — только непересекающиеся входы (раз в квартал на новом контракте, за
~70 дней до экспирации). 2 испытания (юань, золото).
Запуск (сервер): python -m research.treasury.carry
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

log = logging.getLogger("research.treasury.carry")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "treasury")
ISS = "https://iss.moex.com/iss"
PAIRS = {"CR": "CNYRUB_TOM", "GL": "GLDRUB_TOM"}
COST_PCT = 0.14
GO = 0.15
TMON_FEE = 0.4
ENTRY_DAYS = 70


def iss_history(path: str, d_from: str, cols: str) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        url = f"{ISS}/history/{path}.json?from={d_from}&start={start}&iss.meta=off&history.columns={cols}"
        with urllib.request.urlopen(url, timeout=30) as r:
            j = json.load(r)
        data = j["history"]["data"]
        rows += data
        if len(data) < 100:
            break
        start += len(data)
        time.sleep(0.1)
    return pd.DataFrame(rows, columns=cols.split(","))


def expiry(secid: str) -> dt.date | None:
    url = f"{ISS}/securities/{secid}.json?iss.meta=off&iss.only=description"
    with urllib.request.urlopen(url, timeout=30) as r:
        j = json.load(r)
    for row in j["description"]["data"]:
        if row[0] == "LSTDELDATE":
            return dt.date.fromisoformat(row[2])
    return None


def fut_closes(conn, root: str) -> pd.DataFrame:
    q = """SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
                  (array_agg(close ORDER BY ts DESC))[1] AS close
           FROM research_fut_5m WHERE ticker LIKE %(r)s AND close > 0
             AND (ts AT TIME ZONE 'Europe/Moscow')::time <= '18:45'
           GROUP BY 1, 2"""
    df = pd.read_sql(q, conn, params={"r": root + "%"})
    df["close"] = df["close"].astype(float)
    return df.pivot(index="d", columns="ticker", values="close").sort_index()


def money_index(d_from: str) -> tuple[pd.Series, pd.Series]:
    """Индекс денежного рынка по дням: TMON где есть, до него RUSFAR − комиссия."""
    rf = iss_history("engines/stock/markets/index/securities/RUSFAR", d_from, "TRADEDATE,CLOSE")
    rf["TRADEDATE"] = pd.to_datetime(rf["TRADEDATE"]).dt.date
    rf = rf.dropna().set_index("TRADEDATE")["CLOSE"].astype(float)
    days = pd.date_range(rf.index.min(), dt.date.today(), freq="D").date
    rate = rf.reindex(days).ffill()
    idx = ((rate - TMON_FEE) / 100.0 / 365.0 + 1.0).cumprod()
    tm = iss_history("engines/stock/markets/shares/boards/TQBR/securities/TMON", d_from, "TRADEDATE,CLOSE")
    tm["TRADEDATE"] = pd.to_datetime(tm["TRADEDATE"]).dt.date
    tm = tm.dropna().set_index("TRADEDATE")["CLOSE"].astype(float)
    return pd.Series(idx.to_numpy(), index=days), tm.reindex(days).ffill()


def mm_return(rusfar_idx: pd.Series, tmon: pd.Series, a: dt.date, b: dt.date) -> tuple[float, str]:
    if pd.notna(tmon.get(a)) and pd.notna(tmon.get(b)):
        return tmon[b] / tmon[a] - 1.0, "TMON"
    return rusfar_idx[b] / rusfar_idx[a] - 1.0, "RUSFAR"


def run_pair(conn, root: str, spot_id: str, rusfar_idx, tmon) -> dict:
    fut = fut_closes(conn, root)
    sp = iss_history(f"engines/currency/markets/selt/boards/CETS/securities/{spot_id}",
                     str(fut.index.min()), "TRADEDATE,CLOSE")
    sp["TRADEDATE"] = pd.to_datetime(sp["TRADEDATE"]).dt.date
    spot = sp.dropna().set_index("TRADEDATE")["CLOSE"].astype(float)
    exps = {tk: expiry(tk) for tk in fut.columns}
    log.info("%s: %d контрактов, спот %d дней", root, len(exps), len(spot))
    daily, entries = [], []
    last_entry_tk = None
    for d in fut.index:
        if d not in spot.index:
            continue
        live = [(exps[tk], tk) for tk in fut.columns
                if exps.get(tk) and pd.notna(fut.at[d, tk]) and 20 <= (exps[tk] - d).days <= 100]
        if not live:
            continue
        ex, tk = min(live)
        days_to = (ex - d).days
        if ex > max(rusfar_idx.index):
            continue
        f, s = fut.at[d, tk], spot[d]
        locked = (f / s - 1.0) * 100.0 - COST_PCT
        mm, src = mm_return(rusfar_idx, tmon, d, ex)
        ann = 365.0 / days_to
        rec = {"d": d, "tk": tk, "days": days_to, "implied_ann_pct": (f / s - 1.0) * 100.0 * ann,
               "locked_ann_pct": locked / (1 + GO) * ann, "mm_ann_pct": mm * 100.0 * ann,
               "mm_src": src}
        rec["excess_ann_pct"] = rec["locked_ann_pct"] - rec["mm_ann_pct"]
        daily.append(rec)
        if tk != last_entry_tk and days_to <= ENTRY_DAYS:
            entries.append(rec)
            last_entry_tk = tk
    dd = pd.DataFrame(daily)
    en = pd.DataFrame(entries)
    out = {"days": len(dd), "share_days_excess_pos": float((dd["excess_ann_pct"] > 0).mean()) if len(dd) else None,
           "daily_mean": dd[["implied_ann_pct", "locked_ann_pct", "mm_ann_pct", "excess_ann_pct"]].mean().to_dict()
           if len(dd) else {}}
    if len(en) >= 4:
        s = en["excess_ann_pct"]
        t = float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))) if s.std(ddof=1) else None
        out["entries"] = {"n": len(s), "mean_excess_ann_pct": float(s.mean()), "t": t,
                          "positive": int((s > 0).sum())}
        out["entry_rows"] = en.assign(d=en["d"].astype(str)).round(2).to_dict("records")
    # по годам — режим ставок менялся
    if len(dd):
        dd["y"] = [x.year for x in dd["d"]]
        out["by_year"] = dd.groupby("y")[["implied_ann_pct", "mm_ann_pct", "excess_ann_pct"]].mean().round(2).to_dict("index")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        rusfar_idx, tmon = money_index("2022-01-01")
        res = {"params": {"cost_pct": COST_PCT, "go": GO, "tmon_fee_pct": TMON_FEE},
               **{r: run_pair(conn, r, s, rusfar_idx, tmon) for r, s in PAIRS.items()}}
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "carry_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
