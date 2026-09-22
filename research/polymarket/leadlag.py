"""
Опережает ли Polymarket IMOEX внутри торгового дня (часовые данные).

Объявлено до замера (22.09.2026): основной замер — лаг +1 час
(изменение индекса «мир» за час h ↔ доходность IMOEX за час h+1),
только часы основной сессии 10:00–18:00 МСК, внутри одного дня.
Лаги −1 и 0 — справочно (кто за кем идёт). Одно испытание в реестр.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.polymarket import analyze as an              # noqa: E402

SQL = """SELECT ts, close FROM market_data_5m WHERE ticker = 'IMOEX' AND close > 0 ORDER BY ts"""


def main() -> int:
    import database
    markets, h = an.load()
    peace = an.class_index(markets, h, "PEACE")
    conn = database.get_connection()
    try:
        b = pd.read_sql(SQL, conn)
    finally:
        conn.close()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    first = b.groupby(b["ts"].dt.date)["ts"].min().dt.tz_convert("Europe/Moscow").dt.time
    print("первый бар IMOEX (МСК), последние дни:", sorted(set(first.tail(20).astype(str))))
    px = b.set_index("ts")["close"].astype(float).resample("h").last()
    r = (px / px.shift(1) - 1) * 100
    msk = r.index.tz_convert("Europe/Moscow")
    sess = pd.Series((msk.hour >= 11) & (msk.hour <= 18) & (msk.dayofweek < 5), index=r.index)
    df = pd.DataFrame({"imoex": r, "peace": peace.reindex(r.index)})
    df["day"] = msk.date
    out = {}
    for lag in (-1, 0, 1):
        x = df.groupby("day")["peace"].shift(lag)         # lag 1: пис часа h−1 против IMOEX часа h
        d = pd.DataFrame({"x": x, "y": df["imoex"]})[sess].dropna()
        d = d[d["x"] != 0]
        n = len(d)
        c = float(d["x"].corr(d["y"]))
        t = c * math.sqrt(n - 2) / math.sqrt(1 - c * c)
        out[f"lag{lag:+d}"] = {"n": n, "r": c, "t": t}
        print(f"лаг {lag:+d} ч: часов {n}, r {c:+.3f}, t {t:+.2f}")
    with open(os.path.join(an.OUT_DIR, "leadlag.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(an.TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 9,
                            "stage": "polymarket-leadlag", "trials": 1, "revision": rev}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
