"""
Вероятности Polymarket («мир» / «эскалация») против IMOEX.

Схема заморожена до анализа (коммитом вместе с этим файлом):
  • индекс класса — взвешенное (√объёма) среднее ЧАСОВЫХ изменений вероятности
    по активным рынкам класса; изменение берётся внутри одного рынка, поэтому
    смена сроков лестницы (перемирие к маю → к июню) не даёт скачков;
    рынок активен, если до срока ≥ 21 дня и 0,02 < p < 0,98;
  • окна по часам MOEX (UTC): закрытие 15:50 (18:50 МСК), «до открытия»
    03:50 (06:50 МСК — раньше любой сессии, окно не перекрывает торги);
  • 8 замеров, поправка Холма:
      A same   — Δиндекса за день торгов ↔ IMOEX close→close того же дня;
      B gap    — Δ за ночь (закрытие → 06:50) ↔ гэп IMOEX (open/close вчера);
      C intra  — Δ за ночь ↔ IMOEX open→close того же дня (продолжение после гэпа);
      D next   — Δ за день ↔ IMOEX close→close следующего дня.
  Предсказательные — C и D; A и B — одновременные.

Запуск (на сервере): python -m research.polymarket.analyze
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.polymarket.fetch import OUT_DIR, _ts          # noqa: E402

log = logging.getLogger("research.polymarket.analyze")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
MIN_DAYS_TO_END = 21
P_LO, P_HI = 0.02, 0.98
CLOSE_UTC, PREOPEN_UTC = dt.time(15, 50), dt.time(3, 50)
CLASSES = ("PEACE", "WAR")
# Уточнение отбора ПО ТЕКСТУ вопросов, до замера (просмотр списка 204 рынков):
# место встречи, «не встретятся», Нобелевка, гонки с GTA VI, комбо, встречи
# Трампа с Зеленским/Си, миротворцы — не вероятность мира.
EXCLUDE = re.compile(r"meet next in|not meet|nobel|gta|combo|trump.{0,30}zelensk(?!.{0,40}putin)|xi meeting|peacekeeping", re.I)
# «Эскалация» — только выход за пределы Украины; ход боёв (Россия возьмёт город,
# контрнаступление, Крым) — отдельная тема, в индекс не смешивается.
WAR_SCOPE = re.compile(r"nato|us x russia|invade|strike on a nato|article 5|troops fighting|drone", re.I)


def eligible(m: dict) -> bool:
    q = m["question"]
    if EXCLUDE.search(q):
        return False
    if m["cls"] == "WAR" and not WAR_SCOPE.search(q):
        return False
    return True


def load() -> tuple[list[dict], pd.DataFrame]:
    with open(os.path.join(OUT_DIR, "markets.json"), encoding="utf-8") as f:
        markets = [m for m in json.load(f) if eligible(m)]
    h = pd.read_csv(os.path.join(OUT_DIR, "history_hourly.csv"), dtype={"token": str})
    h["ts"] = pd.to_datetime(h["t"], unit="s", utc=True).dt.floor("h")
    return markets, h


def class_index(markets: list[dict], h: pd.DataFrame, cls: str) -> pd.Series:
    """Часовое изменение индекса класса (п.п. вероятности)."""
    num, den = None, None
    for m in markets:
        if m["cls"] != cls:
            continue
        g = h[h["token"] == m["yes_token"]].drop_duplicates("ts", keep="last").set_index("ts")["p"]
        if len(g) < 48:
            continue
        idx = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
        p = g.reindex(idx).ffill()
        dp = p.diff()
        end = _ts(m.get("end"))
        alive = (p.shift(1) > P_LO) & (p.shift(1) < P_HI)
        if end:
            left = (end - idx.asi8 // 10**9) / 86400.0
            alive &= pd.Series(left >= MIN_DAYS_TO_END, index=idx)
        w = math.sqrt(max(m["volume"], 1.0))
        contrib = (dp.where(alive) * w).fillna(0.0)
        weight = pd.Series(np.where(alive & dp.notna(), w, 0.0), index=idx)
        num = contrib if num is None else num.add(contrib, fill_value=0.0)
        den = weight if den is None else den.add(weight, fill_value=0.0)
    out = (num / den.replace(0.0, np.nan)) * 100.0
    return out.dropna()


def window_sum(ix: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float | None:
    s = ix[(ix.index > a) & (ix.index <= b)]
    return float(s.sum()) if len(s) else None


def imoex(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT date, open, close FROM market_data WHERE ticker = 'IMOEX' "
                     "AND close > 0 ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[[d.weekday() < 5 for d in df["date"]]].drop_duplicates("date", keep="last")
    df["open"], df["close"] = df["open"].astype(float), df["close"].astype(float)
    df["prev_close"] = df["close"].shift(1)
    df["prev_date"] = df["date"].shift(1)
    df["cc"] = (df["close"] / df["prev_close"] - 1) * 100
    df["gap"] = (df["open"] / df["prev_close"] - 1) * 100
    df["oc"] = (df["close"] / df["open"] - 1) * 100
    df["cc_next"] = df["cc"].shift(-1)
    return df.dropna(subset=["prev_date"])


def windows(ix: pd.Series, days: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in days.itertuples(index=False):
        c0 = pd.Timestamp.combine(r.prev_date, CLOSE_UTC).tz_localize("UTC")
        o1 = pd.Timestamp.combine(r.date, PREOPEN_UTC).tz_localize("UTC")
        c1 = pd.Timestamp.combine(r.date, CLOSE_UTC).tz_localize("UTC")
        rows.append({"date": r.date, "d_day": window_sum(ix, c0, c1), "d_night": window_sum(ix, c0, o1)})
    return pd.DataFrame(rows).set_index("date")


def corr_t(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x, y], axis=1).dropna()
    df = df[df.iloc[:, 0] != 0]
    n = len(df)
    if n < 20:
        return {"n": n}
    r = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
    from scipy import stats
    rho = float(stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1]).statistic)
    return {"n": n, "r": r, "t": t, "p": float(2 * stats.t.sf(abs(t), n - 2)), "spearman": rho}


def holm(p: dict) -> dict:
    items = sorted((v, k) for k, v in p.items() if v is not None)
    out, run, m = {}, 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    markets, h = load()
    conn = database.get_connection()
    try:
        days = imoex(conn)
    finally:
        conn.close()
    res, pvals, levels = {"markets": {c: sum(m["cls"] == c for m in markets) for c in CLASSES}}, {}, {}
    for cls in CLASSES:
        ix = class_index(markets, h, cls)
        levels[cls] = ix.cumsum()
        w = windows(ix, days).join(days.set_index("date")[["cc", "gap", "oc", "cc_next"]])
        w = w[w.index >= ix.index.min().date()]
        tests = {"A_same": corr_t(w["d_day"], w["cc"]), "B_gap": corr_t(w["d_night"], w["gap"]),
                 "C_intra": corr_t(w["d_night"], w["oc"]), "D_next": corr_t(w["d_day"], w["cc_next"])}
        for k, v in tests.items():
            pvals[f"{cls}:{k}"] = v.get("p")
        big = w.dropna(subset=["d_night"])
        thr = big["d_night"].abs().quantile(0.95) if len(big) else None
        ev = big[big["d_night"].abs() >= thr] if thr else big.iloc[0:0]
        res[cls] = {"hours": int(len(ix)), "from": str(ix.index.min()), "to": str(ix.index.max()),
                    "days": int(w["d_day"].notna().sum()), "tests": tests,
                    "big_night_moves": {"threshold_pp": thr, "n": int(len(ev)),
                                        "up_mean_gap": float(ev[ev["d_night"] > 0]["gap"].mean()) if len(ev) else None,
                                        "down_mean_gap": float(ev[ev["d_night"] < 0]["gap"].mean()) if len(ev) else None,
                                        "up_mean_oc": float(ev[ev["d_night"] > 0]["oc"].mean()) if len(ev) else None,
                                        "down_mean_oc": float(ev[ev["d_night"] < 0]["oc"].mean()) if len(ev) else None}}
    adj = holm(pvals)
    for cls in CLASSES:
        for k in res[cls]["tests"]:
            res[cls]["tests"][k]["p_holm"] = adj.get(f"{cls}:{k}")
    lv = pd.DataFrame(levels)
    daily = lv.resample("D").last().ffill()
    daily.index = daily.index.date
    ix_close = days.set_index("date")["close"]
    dyn = daily.join(ix_close.rename("imoex"), how="left").ffill()
    dyn.tail(120).to_csv(os.path.join(OUT_DIR, "dynamics_daily.csv"))
    res["dynamics_last"] = {str(k): {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                            for k, row in dyn.tail(90).iloc[::7].iterrows()}
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 9,
                            "stage": "polymarket", "trials": 8, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
