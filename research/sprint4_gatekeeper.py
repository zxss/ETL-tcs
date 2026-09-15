"""
Спринт 4 (ТЗ 50D, «Portfolio Gatekeeper»): фильтры допуска для intraday_short без
модели. Правила — research/sprint4_rules.json, фиксируются коммитом до прогона;
один прогон на обе выборки (решение пользователя 15.09.2026).

Базовый поток — условия прода из research/short_rule (ret1 < 0 на A′, рыночные
ворота по ATR и IMOEX/EMA50, нешортуемые и дорогой лот — пропуск), вход 10:05,
выход 18:15, без стопа. Фильтры F1–F4 — поверх него. Нетто сделки = валовой шорт −
издержки круга (base). Признаки — на A′ (последний будничный торговый день до
входа); дневки собираются из 5-минуток одинаково для обеих выборок.

Запуск (на сервере): python -m research.sprint4_gatekeeper
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import news_classify as ncl                # noqa: E402
from research import news_event_study as ns              # noqa: E402
from research import session_calendar as sc              # noqa: E402
from research import session_timing as st                # noqa: E402
from research import short_rule as sr                    # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402
from research import sprint3_panel as sp3                # noqa: E402

log = logging.getLogger("research.sprint4_gatekeeper")

RULES_PATH = os.path.join(ROOT, "research", "sprint4_rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "sprint4")
MSK = dt.timezone(dt.timedelta(hours=3))
FILTERS = ("F1", "F2", "F3", "F4")
QUANTUM = {"market_data_5m": 1e-4, "research_bars_5m": 1e-6}     # точность хранения цены

DAILY_SQL = """
SELECT ticker, d AS date,
  (array_agg(open ORDER BY ts))[1] AS open, max(high) AS high, min(low) AS low,
  (array_agg(close ORDER BY ts DESC))[1] AS close, sum(volume) AS volume
FROM (SELECT ticker, ts, open, high, low, close, volume, (ts AT TIME ZONE 'Europe/Moscow')::date AS d
      FROM {table} WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s) s
WHERE extract(isodow FROM d) < 6
GROUP BY ticker, d
"""

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, close
FROM {table}
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN TIME '09:15' AND TIME '18:15'
"""


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Данные ───────────────────────────────────────────────────────────────────

def _window(d_from: dt.date, d_to: dt.date) -> dict:
    return {"f": dt.datetime.combine(d_from, dt.time(), MSK),
            "t": dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), MSK)}


def daily_from_5m(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    df = pd.read_sql(DAILY_SQL.format(table=table), conn, params={"tk": tickers, **_window(d_from, d_to)})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_bars(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    df = pd.read_sql(BARS_SQL.format(table=table), conn, params={"tk": tickers, **_window(d_from, d_to)})
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    for c in ("open", "high", "close"):
        df[c] = df[c].astype(float)
    return df


def adjust_daily_splits(daily: pd.DataFrame) -> pd.DataFrame:
    """Скачок open/вчерашний close вне [0,5; 2] — сплит: цены до него переводятся в новые единицы."""
    out = []
    cols = ["open", "high", "low", "close"]
    for _, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        r = (g["open"] / g["close"].shift(1)).to_numpy(float)
        for i in np.nonzero((r < sp3.SPLIT_RANGE[0]) | (r > sp3.SPLIT_RANGE[1]))[0]:
            g.iloc[:i, [g.columns.get_loc(c) for c in cols]] = g.iloc[:i][cols].to_numpy(float) * r[i]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def filter_features(daily: pd.DataFrame, index: str = sr.INDEX) -> pd.DataFrame:
    """На каждую дату бумаги: RS30 к IMOEX, close < EMA50, спред Корвина–Шульца ниже медианы 20 дней."""
    ix = daily[daily["ticker"] == index].set_index("date")["close"].sort_index()
    out = []
    for tk, g in daily[daily["ticker"] != index].groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        c = g["close"]
        ic = ix.reindex(c.index)
        ema = c.ewm(span=sr.EMA_SPAN, adjust=False).mean()
        enough = pd.Series(np.arange(1, len(c) + 1) >= sr.EMA_SPAN, index=c.index)
        cs = sp3.corwin_schultz(g["high"], g["low"])
        med = cs.rolling(20, min_periods=10).median()
        out.append(pd.DataFrame({
            "ticker": tk, "date": c.index,
            "rs30": ((c / c.shift(30) - 1.0) - (ic / ic.shift(30) - 1.0)) * 100.0,
            "below_ema50": np.where(enough, c < ema, np.nan),
            "cs_narrow": np.where(cs.notna() & med.notna(), cs < med, np.nan)}))
    return pd.concat(out, ignore_index=True)


def negative_news(events: pd.DataFrame, categories: list[str]) -> dict:
    """ticker → отсортированные моменты публикации негативных событий нужных категорий."""
    if not len(events):
        return {}
    e = events[events["category"].isin(categories) & (events["sentiment"] < 0)]
    return {tk: np.sort(pd.to_datetime(g["posted"]).to_numpy(dtype="datetime64[ns]"))
            for tk, g in e.groupby("ticker")}


def had_news(times: np.ndarray | None, at: dt.datetime, hours: int) -> bool:
    if times is None or not len(times):
        return False
    t1 = np.datetime64(at, "ns")
    t0 = t1 - np.timedelta64(hours, "h")
    return bool(np.searchsorted(times, t1, "right") - np.searchsorted(times, t0, "right") > 0)


def apply_filters(c: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Флаги F1–F4 по кандидатам базового потока (NaN в условии — не допущен)."""
    f = rules["filters"]
    f1 = (c["rs30"] < 0) & (c["below_ema50"] == True)                              # noqa: E712
    oil = c["ticker"].isin(f["F3"]["oil"]) & (c["brent_day"] < f["F3"]["brent_max_pct"])
    met = c["ticker"].isin(f["F3"]["metals"]) & (c["cny_day"] < f["F3"]["cny_max_pct"])
    return c.assign(F1=f1, F2=f1 & (c["cs_narrow"] == True), F3=f1 & (oil | met),  # noqa: E712
                    F4=f1 & c["neg_news"].astype(bool))


# ── Прогон периода ───────────────────────────────────────────────────────────

def run_period(conn, smp: dict, rules: dict, news: dict, commodity_day, lots: dict, blocked: set,
               spreads: dict) -> tuple[pd.DataFrame, list[dt.date]]:
    table = smp["stock_table"]
    d_from, d_to = dt.date.fromisoformat(smp["from"]), dt.date.fromisoformat(smp["to"])
    tickers = list(sr.UNIVERSE) + [sr.INDEX]
    daily = adjust_daily_splits(daily_from_5m(conn, table, tickers, d_from - dt.timedelta(days=500), d_to))
    feats, mkt, above = sr.build_features(daily, lots)
    feats = feats.merge(filter_features(daily), on=["ticker", "date"], how="left")
    tdates = sr.trading_dates(daily)
    paths = sr.window_paths(load_bars(conn, table, tickers, d_from, d_to))
    exec_days = [d for d in sr.exec_days_from(paths) if d_from <= d <= d_to]
    by_date = {d: g for d, g in feats.groupby("date")}
    hours = rules["filters"]["F4"]["hours"]
    q = QUANTUM[table]
    rows = []
    for A in exec_days:
        Ap = sr.prev_date(tdates, A)
        if Ap is None:
            continue
        m = mkt.get(Ap)
        gate = sr.gates_open(None if m is None or (isinstance(m, float) and math.isnan(m)) else float(m), above.get(Ap))
        f = by_date.get(Ap)
        if gate is not True or f is None:
            continue
        c = f[f["ret1"].notna() & (f["ret1"] < 0) & ~f["ticker"].isin(blocked)].copy()
        if c.empty:
            continue
        cd = commodity_day(Ap)
        entry_at = dt.datetime.combine(A, sc.short_entry(A))
        c["brent_day"], c["cny_day"] = cd.get("brent", np.nan), cd.get("cny", np.nan)
        c["neg_news"] = [had_news(news.get(tk), entry_at, hours) for tk in c["ticker"]]
        c = apply_filters(c, rules)
        for r in c.itertuples(index=False):
            p = paths.get((r.ticker, A))
            if p is None:
                continue
            entry = float(p[0][0])
            if entry * lots.get(r.ticker, 1) > sr.POSITION_RUB or not (entry > 0 and q / entry * 100.0 <= sr.MAX_QUANTUM_PCT):
                continue
            gross = sr.short_outcome(*p)["gross"]
            rows.append({"date": A, "asof": Ap, "ticker": r.ticker, "gross": gross,
                         "net": gross - cm.round_trip(r.ticker, "base", spreads),
                         "imoex_short": -sr.index_move(paths, A), "rs30": r.rs30, "brent_day": r.brent_day,
                         "cny_day": r.cny_day, "neg_news": bool(r.neg_news),
                         **{k: bool(getattr(r, k)) for k in FILTERS}})
    return pd.DataFrame(rows), exec_days


# ── Статистика ───────────────────────────────────────────────────────────────

def stats(tr: pd.DataFrame, exec_days: list[dt.date], n_trials: int) -> dict:
    if not len(tr):
        return {"trades": 0, "dates": 0}
    per = tr.groupby("date")["net"].mean()
    daily = per.reindex(exec_days).fillna(0.0)
    bd = es.by_date(tr["net"], tr["date"])
    return {"trades": int(len(tr)), "dates": int(len(per)), "net": float(tr["net"].mean()),
            "median": float(tr["net"].median()), "win": float((tr["net"] > 0).mean()),
            "t": bd.get("t"), "imoex_short": float(tr["imoex_short"].mean()),
            "dsr": es.deflated_sharpe(daily, max(2, n_trials))}


def accepted(s: dict, acc: dict) -> bool:
    return bool(s.get("trades") and s["net"] >= acc["net_min_pct"] and s["win"] >= acc["win_rate_min"])


def report(res: dict, meta: dict, rules: dict) -> str:
    f = es._f
    acc = rules["acceptance"]
    L = ["# Спринт 4 — фильтры допуска для intraday_short (без модели), обе выборки", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правила `{rules['version']}`. "
         f"Один прогон. Испытаний в реестре: {meta['trials_total']}.", "",
         "Нетто сделки — шорт от 10:05 до 18:20 минус круг издержек (base), %. t — по датам. "
         "«Шорт IMOEX» — ход индекса в тех же окнах со знаком шорта (сколько дал бы рынок).", "",
         f"Приёмка (пользователь): нетто ≥ +{acc['net_min_pct']:.2f} % и доля плюсовых ≥ {acc['win_rate_min']:.0%} "
         "в каждой выборке.", "",
         "| фильтр | выборка | сделок | дат | нетто, % (t) | медиана, % | доля плюсовых | шорт IMOEX, % | DSR | приёмка |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"BASE": "базовый поток (справочно)", **{k: f"{k} {rules['filters'][k]['name']}" for k in FILTERS}}
    for k in ("BASE",) + FILTERS:
        for per in ("dev", "holdout"):
            s = res[per][k]
            if not s.get("trades"):
                L.append(f"| {names[k]} | {per} | 0 | 0 | — | — | — | — | — | — |")
                continue
            L.append(f"| {names[k]} | {per} | {s['trades']} | {s['dates']} | {f(s['net'])} ({f(s['t'], 1)}) | "
                     f"{f(s['median'])} | {f(s['win'], 2)} | {f(s['imoex_short'])} | {f(s['dsr'], 2)} | "
                     f"{'—' if k == 'BASE' else ('да' if accepted(s, acc) else 'нет')} |")
    L += ["", "| фильтр | итог (обе выборки) |", "|---|---|"]
    for k in FILTERS:
        ok = all(accepted(res[p][k], acc) for p in ("dev", "holdout"))
        L.append(f"| {names[k]} | {'ПРИНЯТ' if ok else 'не принят'} |")
    L += ["", f"Выборки: dev {meta['dev']} (market_data_5m), holdout {meta['holdout']} (research_bars_5m). "
          f"Дней исполнения: dev {meta['exec_days']['dev']}, holdout {meta['exec_days']['holdout']}.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 4: фильтры допуска intraday_short")
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rules = load_rules()
    if os.path.exists(os.path.join(OUT_DIR, "results.json")):
        raise SystemExit("замер Спринта 4 уже выполнен — повтор запрещён правилами")
    lots, blocked = sr.load_lots_and_blocked()
    spreads = cm.load_spreads()
    lo, hi = dt.date(2022, 8, 1), dt.date(2026, 9, 12)
    import database
    conn = database.get_connection()
    try:
        exp, root_of = st.read_contracts()
        roots = {"brent": "BR", "cny": "CR"}
        fd = st.load_daily(conn, "research_fut_5m", [t for t, r in root_of.items() if r in roots.values()], lo, hi)
        fd["root"] = fd["ticker"].map(root_of)
        front = st.front_contracts(fd, ll.eligible_until(exp, 2))
        fmap = {(r.root, r.d): r.front for r in front.itertuples()}
        fbars = es.load_bars(conn, "research_fut_5m", sorted(set(front["front"])), lo, hi)
        sig = {"from_time": "10:00", "to_bar": "18:25"}

        def commodity_day(d):
            out = {}
            for name, root in roots.items():
                b = fbars.get(fmap.get((root, d)))
                v = ll.signal_2b(b, d, sig) if b is not None else None
                out[name] = v if v is not None else np.nan
            return out

        posts = ns.load_posts(conn, "markettwits", lo)
        events, _ = es.build_events(posts[posts["msk"].dt.date <= hi], ncl.Classifier())
        news = negative_news(events, rules["filters"]["F4"]["categories"])
        rev = es._revision()
        total = ll.register("sprint4-filters", rules["trials"], rev, sprint=4)
        res, frames, ndays = {}, [], {}
        for per in ("dev", "holdout"):
            tr, days = run_period(conn, rules["samples"][per], rules, news, commodity_day, lots, blocked, spreads)
            ndays[per] = len(days)
            res[per] = {"BASE": stats(tr, days, total), **{k: stats(tr[tr[k]], days, total) for k in FILTERS}}
            frames.append(tr.assign(period=per))
    finally:
        conn.close()
    meta = {"created": dt.datetime.now().strftime("%d.%m.%Y %H:%M"), "revision": rev, "trials_total": total,
            "dev": f"{rules['samples']['dev']['from']} … {rules['samples']['dev']['to']}",
            "holdout": f"{rules['samples']['holdout']['from']} … {rules['samples']['holdout']['to']}",
            "exec_days": ndays}
    os.makedirs(OUT_DIR, exist_ok=True)
    text = report(res, meta, rules)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    verdict = {k: all(accepted(res[p][k], rules["acceptance"]) for p in ("dev", "holdout")) for k in FILTERS}
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": res, "accepted": verdict}, f, ensure_ascii=False, indent=1, default=str)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False, float_format="%.5f")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
