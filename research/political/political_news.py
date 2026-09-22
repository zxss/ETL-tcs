"""
Влияние политических новостей markettwits на IMOEX: «мир» против «войны».

Вопрос пользователя 22.09.2026. Правила и словарь заморожены в
research/political/rules.json до прогона. Только выборка разработки
(2024-05-21…2026-09-14); отложенная 2022–2024 не трогается — если эффект
найдётся, её подтверждение будет отдельным одноразовым прогоном.

Классификация делается в PostgreSQL регулярными выражениями, в Python попадают
только id, время и класс поста — тексты из базы не выгружаются и в отчёт не
пишутся (репозиторий публичный).

Замеры:
  1. внутри дня — аномальная доходность IMOEX после поста (30 мин, 2 часа, до
     закрытия 18:30) против средней доходности того же окна в тот же час;
     плюс дрейф за 30 мин ДО поста — отыгран ли он уже к публикации;
  2. по дням — баланс (мир − война)/(мир + война) против доходности IMOEX
     того же и следующего дня;
  3. текущая динамика — понедельно за последние 16 недель.

Запуск (на сервере): python -m research.political.political_news
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

log = logging.getLogger("research.political")

RULES_PATH = os.path.join(ROOT, "research", "political", "rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "political")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
MSK = "Europe/Moscow"
CLASSES = ("PEACE", "WAR")

POSTS_SQL = """
SELECT id, posted_at AT TIME ZONE 'Europe/Moscow' AS tm,
       text ~* %(peace)s AS p, text ~* %(war)s AS w
FROM news.tg_posts
WHERE posted_at >= %(f)s AND posted_at < %(t)s AND text ~* %(anchor)s
  AND NOT text ~* %(exclude)s
"""
BARS_SQL = """
SELECT ts AT TIME ZONE 'Europe/Moscow' AS tm, close
FROM {table} WHERE ticker = 'IMOEX' AND close > 0 AND ts >= %(f)s AND ts < %(t)s
ORDER BY ts
"""
DAILY_SQL = "SELECT date, close FROM market_data WHERE ticker = 'IMOEX' AND close > 0 ORDER BY date"


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify(p: bool, w: bool) -> str:
    if p and w:
        return "MIXED"
    if p:
        return "PEACE"
    if w:
        return "WAR"
    return "OTHER_POLITICAL"


def load_posts(conn, rules: dict, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    c = rules["classify"]
    df = pd.read_sql(POSTS_SQL, conn, params={"anchor": c["anchor"], "exclude": c["exclude"], "peace": c["peace"], "war": c["war"],
                                              "f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"})
    df["tm"] = pd.to_datetime(df["tm"])
    df["cls"] = [classify(bool(p), bool(w)) for p, w in zip(df["p"], df["w"])]
    return df[["id", "tm", "cls"]]


# ── Внутри дня ───────────────────────────────────────────────────────────────

def bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    b["tm"] = pd.to_datetime(b["tm"])
    b["d"] = b["tm"].dt.date
    b["close"] = b["close"].astype(float)
    return b.set_index("tm").sort_index()


def close_at(day_bars: pd.Series, t: pd.Timestamp) -> float | None:
    """close последнего бара, начавшегося не позже t."""
    s = day_bars[:t]
    return float(s.iloc[-1]) if len(s) else None


def event_returns(posts: pd.DataFrame, bars: pd.DataFrame, rules: dict) -> pd.DataFrame:
    cfg = rules["intraday"]
    s0, s1 = (dt.time.fromisoformat(x) for x in cfg["session"])
    last_bar = dt.time(18, 25)
    by_day = {d: g["close"] for d, g in bars.groupby("d")}
    rows = []
    for r in posts.itertuples(index=False):
        t = r.tm
        if not (s0 <= t.time() <= s1) or t.date() not in by_day:
            continue
        c = by_day[t.date()]
        start = t.floor("5min")                       # бар, внутри которого вышел пост
        entry = close_at(c, start)                    # его close — первая цена после поста
        pre0 = close_at(c, start - pd.Timedelta(minutes=5 + cfg["pre_drift_min"]))
        pre1 = close_at(c, start - pd.Timedelta(minutes=5))
        if not entry:
            continue
        row = {"id": r.id, "tm": t, "d": t.date(), "hour": start.hour, "cls": r.cls,
               "pre_pct": (pre1 / pre0 - 1.0) * 100.0 if pre0 and pre1 else np.nan}
        for h in cfg["horizons_min"]:
            x = close_at(c, start + pd.Timedelta(minutes=h))
            row[f"r{h}"] = (x / entry - 1.0) * 100.0 if x and start + pd.Timedelta(minutes=h) <= \
                pd.Timestamp.combine(t.date(), last_bar) else np.nan
        x = close_at(c, pd.Timestamp.combine(t.date(), last_bar))
        row["rclose"] = (x / entry - 1.0) * 100.0 if x else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def baseline(bars: pd.DataFrame, rules: dict) -> dict:
    """Средняя доходность окна по часу старта — по всем барам сессии всех дней."""
    cfg = rules["intraday"]
    s0, s1 = (dt.time.fromisoformat(x) for x in cfg["session"])
    last_bar = dt.time(18, 25)
    out = {}
    for d, g in bars.groupby("d"):
        c = g["close"]
        times = [t for t in c.index if s0 <= t.time() <= s1]
        close_px = close_at(c, pd.Timestamp.combine(d, last_bar))
        for t in times:
            e = float(c[t])
            for h in cfg["horizons_min"]:
                tt = t + pd.Timedelta(minutes=h)
                if tt.time() > last_bar:
                    continue
                x = close_at(c, tt)
                if x:
                    out.setdefault((f"r{h}", t.hour), []).append((x / e - 1.0) * 100.0)
            if close_px:
                out.setdefault(("rclose", t.hour), []).append((close_px / e - 1.0) * 100.0)
            if t.minute == 0:
                pass
    return {k: float(np.mean(v)) for k, v in out.items()}


def t_by_day(values: pd.Series, days: pd.Series) -> tuple[float | None, int, float | None]:
    s = pd.Series(values.to_numpy(float), index=days.to_numpy()).dropna()
    if s.empty:
        return None, 0, None
    g = s.groupby(level=0).mean()
    if len(g) < 5 or not g.std(ddof=1):
        return None, len(g), float(g.mean())
    return float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))), len(g), float(g.mean())


def p_two_sided(t: float | None, n: int) -> float | None:
    if t is None or n < 3:
        return None
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ── По дням и динамика ───────────────────────────────────────────────────────

def news_day(tm: pd.Timestamp) -> dt.date:
    """Торговый «новостной день»: пост после 18:30 относится к следующему дню."""
    d = tm.date()
    if tm.time() > dt.time(18, 30):
        d += dt.timedelta(days=1)
    return d


def daily_balance(posts: pd.DataFrame, min_posts: int = 3) -> pd.DataFrame:
    p = posts[posts["cls"].isin(CLASSES)].copy()
    p["nd"] = p["tm"].map(news_day)
    g = p.groupby(["nd", "cls"]).size().unstack(fill_value=0)
    for c in CLASSES:
        if c not in g:
            g[c] = 0
    g = g[(g["PEACE"] + g["WAR"]) >= min_posts]
    g["balance"] = (g["PEACE"] - g["WAR"]) / (g["PEACE"] + g["WAR"])
    return g


def corr_t(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 10:
        return {"n": n}
    r = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
    return {"n": n, "r": r, "t": t, "p": p_two_sided(t, n)}


def weekly_dynamics(posts: pd.DataFrame, daily_ix: pd.Series, weeks: int = 16) -> pd.DataFrame:
    p = posts.copy()
    p["week"] = p["tm"].dt.to_period("W-SUN").dt.start_time.dt.date
    g = p.groupby(["week", "cls"]).size().unstack(fill_value=0)
    for c in ("PEACE", "WAR", "MIXED"):
        if c not in g:
            g[c] = 0
    g["balance"] = (g["PEACE"] - g["WAR"]) / (g["PEACE"] + g["WAR"]).replace(0, np.nan)
    ix = daily_ix.copy()
    ix.index = pd.to_datetime(ix.index)
    wk = ix.resample("W-SUN").last()
    ret = (wk / wk.shift(1) - 1.0) * 100.0
    ret.index = (ret.index - pd.Timedelta(days=6)).date
    g["imoex_week_pct"] = pd.Series(ret).reindex(g.index)
    return g[["PEACE", "WAR", "MIXED", "balance", "imoex_week_pct"]].tail(weeks)


# ── Прогон ───────────────────────────────────────────────────────────────────

def run(conn, rules: dict) -> dict:
    s = rules["sample"]
    d_from, d_to = dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])
    posts = load_posts(conn, rules, d_from, d_to)
    log.info("политических постов %d: %s", len(posts), posts["cls"].value_counts().to_dict())
    bars = bar_frame(pd.read_sql(BARS_SQL.format(table=s["bars"]), conn,
                                 params={"f": f"{d_from} 00:00+03", "t": f"{d_to + dt.timedelta(days=1)} 00:00+03"}))
    ev = event_returns(posts, bars, rules)
    base = baseline(bars, rules)
    horizons = [f"r{h}" for h in rules["intraday"]["horizons_min"]] + ["rclose"]
    for h in horizons:
        ev[f"ab_{h}"] = ev[h] - [base.get((h, hr), np.nan) for hr in ev["hour"]]
    intraday, pvals = {}, {}
    for c in CLASSES:
        e = ev[ev["cls"] == c]
        res = {"posts_in_session": int(len(e)), "days": int(e["d"].nunique())}
        t, n, m = t_by_day(e["pre_pct"], e["d"])
        res["pre_drift_30m"] = {"mean_pct": m, "t": t, "days": n}
        for h in horizons:
            t, n, m = t_by_day(e[f"ab_{h}"], e["d"])
            res[h] = {"abnormal_mean_pct": m, "t": t, "days": n, "p": p_two_sided(t, n)}
            pvals[f"{c}:{h}"] = res[h]["p"]
        intraday[c] = res
    # по дням
    daily_ix = pd.read_sql(DAILY_SQL, conn).set_index("date")["close"].astype(float)
    daily_ix.index = pd.to_datetime(daily_ix.index).date
    ret = (daily_ix / daily_ix.shift(1) - 1.0) * 100.0
    bal = daily_balance(posts)
    same = corr_t(bal["balance"], ret.reindex(bal.index))
    nxt = ret.shift(-1)
    nxt_c = corr_t(bal["balance"], nxt.reindex(bal.index))
    pvals["daily:same_day"] = same.get("p")
    pvals["daily:next_day"] = nxt_c.get("p")
    adj = holm(pvals)
    for c in CLASSES:
        for h in horizons:
            intraday[c][h]["p_holm"] = adj.get(f"{c}:{h}")
    same["p_holm"], nxt_c["p_holm"] = adj.get("daily:same_day"), adj.get("daily:next_day")
    # динамика до сегодняшнего дня
    recent = load_posts(conn, rules, dt.date.today() - dt.timedelta(weeks=17), dt.date.today())
    weekly = weekly_dynamics(recent, daily_ix)
    return {"rules": rules["version"], "sample": s,
            "posts": {k: int(v) for k, v in posts["cls"].value_counts().items()},
            "intraday": intraday, "daily": {"days": int(len(bal)), "same_day": same, "next_day": nxt_c,
                                             "mean_balance": float(bal["balance"].mean())},
            "weekly": weekly.reset_index().rename(columns={"week": "week_start"}).to_dict("records")}


def register_trials(n: int, revision: str) -> None:
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 8,
           "stage": "political-dev", "trials": n, "revision": revision}
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    rules = load_rules()
    conn = database.get_connection()
    try:
        res = run(conn, rules)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    register_trials(int(rules["trials"]), rev)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
