"""
Геополитические новости (мир/война) → фьючерсы Brent и газ NG (ТЗ пользователя
23.09.2026, приоритет №1 из инвентаризации новостей/фьючерсов).

Механизм (предзарегистрирован ДО прогона): война РФ-Украина и санкции повышают
риск-премию по нефтяному/газовому предложению → рост Brent/NG; мир/деэскалация →
падение. Гипотеза НАПРАВЛЕННАЯ: ожидаем ОТРИЦАТЕЛЬНУЮ корреляцию баланса
(мир−война) с ходом фьючерса; торговое правило — война (balance<0) → лонг,
мир (balance>0) → шорт.

Отличие от прежнего: классификатор мир/война (research/political, словарь
заморожен там) гоняли ТОЛЬКО против IMOEX. Здесь та же разметка постов, но цель —
сами фьючерсы. Тексты постов из БД не выгружаются (классификация в SQL, в Python
только id/время/класс) — репозиторий публичный.

Выборки: фьючерсы в research_fut_5m с 2022-01. Отложенная 2022-01-03…2024-05-20
для новостей→фьючерсы ещё НЕ смотрена (политический прогон был только на dev и
только против IMOEX):
  dev     2024-05-21 … 2026-09-11
  holdout 2022-01-03 … 2024-05-20  (один прогон)

Ближний контракт — по объёму предыдущего дня; доходность считается ВНУТРИ одного
контракта (без скачка на ролле). Издержки фьючерса — фикс. круг 0,05 %
(консервативная оценка для ликвидного ближнего Brent; реальная ниже).

Запуск (на сервере): python -m research.futures.news_brent
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.political import political_news as pol         # noqa: E402

log = logging.getLogger("research.futures.news_brent")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "futures")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
ROOTS = {"BR": "Brent", "NG": "газ NG"}
DEV = (dt.date(2024, 5, 21), dt.date(2026, 9, 11))
HOLDOUT = (dt.date(2022, 1, 3), dt.date(2024, 5, 20))
COST_RT_PCT = 0.05                       # круг фьючерса, % (консервативно)
THRESHOLDS = {"any": 0.0, "q50": None, "q75": None}   # |balance| порог; q50/q75 — по dev

FUT_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
       (array_agg(close ORDER BY ts DESC))[1] AS last_close,
       sum(volume) AS vol
FROM research_fut_5m
WHERE ticker LIKE %(root)s AND close > 0 AND ts >= %(f)s AND ts < %(t)s
GROUP BY 1, 2
"""


def near_daily(conn, root: str, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    """Дневная доходность ближнего (по объёму вчера) контракта, внутри контракта."""
    raw = pd.read_sql(FUT_SQL, conn, params={"root": root + "%",
                      "f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"})
    if raw.empty:
        return pd.DataFrame()
    raw["d"] = pd.to_datetime(raw["d"]).dt.date
    raw["last_close"] = raw["last_close"].astype(float)
    raw["vol"] = raw["vol"].astype(float)
    close = raw.pivot(index="d", columns="ticker", values="last_close").sort_index()
    vol = raw.pivot(index="d", columns="ticker", values="vol").sort_index()
    prev_vol = vol.shift(1)
    days = list(close.index)
    rows = []
    for i in range(1, len(days)):
        d, dp = days[i], days[i - 1]
        pv = prev_vol.loc[d].dropna()
        if pv.empty:
            continue
        near = pv.idxmax()                                   # ближний = самый объёмный вчера
        c0, c1 = close.at[dp, near], close.at[d, near]
        if pd.isna(c0) or pd.isna(c1) or not c0:
            continue                                          # разные контракты/нет цены — ролл, пропуск
        rows.append({"d": d, "ret": (c1 / c0 - 1.0) * 100.0, "contract": near})
    return pd.DataFrame(rows).set_index("d")


def corr_block(balance: pd.Series, ret: pd.Series) -> dict:
    df = pd.concat([balance.rename("b"), ret.rename("r")], axis=1).dropna()
    df = df[df["b"] != 0]
    n = len(df)
    if n < 20:
        return {"n": n}
    r = float(df["b"].corr(df["r"]))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
    from scipy import stats
    return {"n": n, "r": r, "t": t, "p": float(2 * stats.t.sf(abs(t), n - 2))}


def rule_stats(balance: pd.Series, ret_next: pd.Series, thresh: float) -> dict:
    """Война(balance<0)→лонг, мир(balance>0)→шорт, |balance|>thresh. Нетто = -sign(b)*ret_next - издержки."""
    df = pd.concat([balance.rename("b"), ret_next.rename("r")], axis=1).dropna()
    df = df[df["b"].abs() > thresh]
    if len(df) < 15:
        return {"trades": len(df)}
    net = -np.sign(df["b"]) * df["r"] - COST_RT_PCT
    n = len(net)
    t = float(net.mean() / (net.std(ddof=1) / math.sqrt(n))) if net.std(ddof=1) else None
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), n - 1)) if t is not None else None
    return {"trades": n, "mean_pct": float(net.mean()), "t": t, "p": p,
            "win_rate": float((net > 0).mean()), "sum_pct": float(net.sum())}


def run_sample(conn, rules, d_from, d_to, thresholds=None) -> dict:
    posts = pol.load_posts(conn, rules, d_from, d_to)
    bal = pol.daily_balance(posts)["balance"]
    bal.index = pd.to_datetime(pd.Series(bal.index)).dt.date.values
    out = {"days_balance": int(len(bal))}
    pvals = {}
    for root, name in ROOTS.items():
        nd = near_daily(conn, root, d_from - dt.timedelta(days=5), d_to)
        if nd.empty:
            out[root] = {"note": "нет данных"}
            continue
        ret = nd["ret"].sort_index()
        rdays = list(ret.index)
        # ret следующего ТОРГОВОГО дня, привязанный к дню D (позиционный сдвиг, устойчив к выходным)
        ret_next = pd.Series({rdays[i]: float(ret.iloc[i + 1]) for i in range(len(rdays) - 1)})
        same = corr_block(bal, ret)
        nxt = corr_block(bal, ret_next)
        out[root] = {"name": name, "same_day": same, "next_day": nxt, "fut_days": int(len(ret))}
        if same.get("p") is not None:
            pvals[f"{root}:same"] = same["p"]
        if nxt.get("p") is not None:
            pvals[f"{root}:next"] = nxt["p"]
        # торговое правило на предсказательном (next-day) сигнале
        th = thresholds or {"any": 0.0}
        out[root]["rule"] = {k: rule_stats(bal, ret_next, v) for k, v in th.items() if v is not None or k == "any"}
    # Холм по всем корреляционным замерам выборки
    items = sorted((v, k) for k, v in pvals.items())
    run_p, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        r, tst = k.split(":")
        out[r].setdefault("holm", {})[tst] = run_p
    return out


def run(conn) -> dict:
    rules = pol.load_rules()
    # пороги правила берём по dev-балансу (квантили |balance|), затем применяем к обеим выборкам
    dev_posts = pol.load_posts(conn, rules, *DEV)
    dev_bal = pol.daily_balance(dev_posts)["balance"]
    ab = dev_bal[dev_bal != 0].abs()
    thr = {"any": 0.0, "q50": float(ab.quantile(0.50)), "q75": float(ab.quantile(0.75))}
    res = {"hypothesis": "война→фьючерс вверх, мир→вниз (ожидаем corr(balance,ret)<0)",
           "cost_rt_pct": COST_RT_PCT, "thresholds": thr,
           "dev": run_sample(conn, rules, *DEV, thresholds=thr),
           "holdout": run_sample(conn, rules, *HOLDOUT, thresholds=thr)}
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "news_brent_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 14,
                            "stage": "news_brent", "trials": 8, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
