"""
Новости: проверка конфаундера на текущей разметке (ТЗ EDGE-R4 §5, блок C).
Только исследование: читает БД, в торговые таблицы не пишет.

МОДЕЛЬ, КОГОРТЫ И КРИТЕРИИ ЗАМОРОЖЕНЫ коммитом этого файла до прогона
(хеш — в отчёте, файл REVISION).

Вопрос. В первом прогоне (audit/news_research/20260914-2257) бумаги с новостью
к 18:35 проигрывали рынку ночью. Дни с новостью — это ещё и дни большого
внутридневного хода и объёма; ночной откат после такого дня дал бы тот же
знак без всякой новости. Контроль дневного движения отвечает, что осталось
от «эффекта новости».

Панель (C1): все бумаго-дни с 5-минутками, обе части периода, одна запись на
бумаго-день (не список событий: иначе News ≡ 1 и β₁ не оценивается).
  цель      AR_gap — аномальный ход 18:35 D → первая цена основной сессии D+1
            (close последнего бара 18:15…18:30 → open первого бара основной
            сессии не позже 10 минут от начала; research/session_calendar);
            аномальный = минус среднее остальных бумаг (leave-one-out);
  сырой     тот же ход без вычета рынка — экономика голого шорта;
  AR_day    аномальный ход от начала основной сессии D до 18:35 D;
  VolSpike  ln(объём основной сессии до 18:35 D / медиана того же объёма за
            20 предыдущих дней). Дневной бар брокера включает вечернюю сессию
            после 18:35 — его объём в 18:35 ещё не известен, поэтому окно
            объёма обрезано моментом решения;
  News      новость по бумаге известна к 18:35 D (публикация + 10 минут, после
            решения предыдущего дня); отчёты о движении цены исключены;
  Hold      новость пришла во время удержания — отдельный индикатор, в сигнал
            не входит, в отчёте отдельной моделью.
Бумаго-дни с шагом хранения цены грубее 0,05 % (NUMERIC(18,4): TGKA, FEES)
исключены до расчёта средних — по цене, не по доходности. Исход через разрыв
длиннее 5 дней — NaN.

Модель (C2):
  AR_gap = a + b1·News + b2·AR_day + b3·|AR_day| + b4·VolSpike + FE(дата) + e,
  ошибки кластеризованы по дате. Без контролей — только News + FE(дата).
Когорты (C3): K1 — к 18:35 был пост с хештегом бумаги; K2 — только упоминания
по названию, без хештега. Контроль — бумаго-дни без новости. K3 — после блока D.
Тестов 4 (2 когорты × 2 модели), поправка Холма (C5).

Точка остановки (C4) на K1, модель с контролями:
  закрыть ветку — |t(b1)| < 2 ИЛИ знак b1 разный в 05.2024–11.2025 и
  12.2025–09.2026;
  продолжать — только если b1 < 0, t ≤ −3, знак один в обеих частях И сырой
  ход шорта на K1 − издержки (комиссия 0,08 % + спред бумаги) − плата за
  перенос > 0.
Издержки и перенос — research/cost_model, тариф пользователя «Премиум»
(подтверждён 15.09.2026): комиссия 0,04 % за сделку (круг 0,08 %) + спред
бумаги; перенос непокрытой позиции до 5 000 ₽ бесплатно, свыше — от 45 ₽ в
календарный день (10 000 ₽ → 0,45 % за ночь, выходные ×3). До 15.09 здесь
стояли плоские 0,128 % и сетка тарифа «Инвестор» (40 ₽). Нешортуемые бумаги
(AKRN, CBOM, MVID) в экономику шорта не входят.

Запуск (на сервере):
    python -m research.news_confound --from 2024-05-21 --to 2026-09-11
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
from research import news_event_study as ns              # noqa: E402
from research import session_calendar as sc              # noqa: E402

log = logging.getLogger("research.news_confound")

SPLIT = dt.date(2025, 12, 1)                  # 05.2024–11.2025 | 12.2025–09.2026
MAX_GAP_DAYS = 5
VOL_WIN = 20
ENTRY_FROM = dt.time(18, 15)
EXIT_TOL_MIN = 10
PRICE_QUANTUM, MAX_QUANTUM_PCT = 1e-4, 0.05
MAIN_POSITION = 10_000.0
POSITIONS = (5_000.0, 10_000.0, 200_000.0)
NON_SHORTABLE = {"AKRN", "CBOM", "MVID"}
CONTROLS = ["ar_day", "abs_ar_day", "log_volspike"]


def carry_pct(position_rub: float, nights: int) -> float:
    """Плата за перенос непокрытой позиции за nights календарных дней, % позиции."""
    return cm.carry_pct(position_rub, nights)


# ── Панель ───────────────────────────────────────────────────────────────────

PANEL_SQL = """
SELECT ticker, d,
       (array_agg(open  ORDER BY ts)      FILTER (WHERE t >= mo AND t <= mo + %(tol)s::interval))[1] AS o_main,
       (array_agg(close ORDER BY ts DESC) FILTER (WHERE t >= %(e_from)s::time AND t <= %(e_to)s::time))[1] AS c_eve,
       sum(volume) FILTER (WHERE t >= mo AND t <= %(e_to)s::time) AS vol_day
FROM (
    SELECT ticker, ts, open, close, volume,
           (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
           (ts AT TIME ZONE 'Europe/Moscow')::time AS t,
           CASE WHEN (ts AT TIME ZONE 'Europe/Moscow')::date >= %(new)s
                THEN %(mo_new)s::time ELSE %(mo_old)s::time END AS mo
    FROM market_data_5m
    WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
      AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
) s
GROUP BY ticker, d
"""


def load_agg(conn, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    msk = dt.timezone(dt.timedelta(hours=3))
    old = sc.session(sc.NEW_SCHEDULE_FROM - dt.timedelta(days=1))
    new = sc.session(sc.NEW_SCHEDULE_FROM)
    df = pd.read_sql(PANEL_SQL, conn, params={
        "tk": list(ns.UNIVERSE), "tol": f"{EXIT_TOL_MIN} minutes",
        "e_from": ENTRY_FROM, "e_to": sc.EVENING_ENTRY_BAR,
        "new": sc.NEW_SCHEDULE_FROM, "mo_new": new.main_open, "mo_old": old.main_open,
        "f": dt.datetime.combine(d_from, dt.time(), msk),
        "t": dt.datetime.combine(d_to + dt.timedelta(days=2), dt.time(), msk)})
    df["d"] = pd.to_datetime(df["d"]).dt.date
    for c in ("o_main", "c_eve", "vol_day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _quantum_bad(p: pd.DataFrame) -> pd.DataFrame:
    return (PRICE_QUANTUM / p * 100.0) > MAX_QUANTUM_PCT


def build_panel(agg: pd.DataFrame, events: list[dict], d_to: dt.date | None = None,
                min_tickers: int = 20) -> pd.DataFrame:
    """Бумаго-дни с целями, контролями и флагами новостей."""
    o = agg.pivot(index="d", columns="ticker", values="o_main").sort_index()
    c = agg.pivot(index="d", columns="ticker", values="c_eve").reindex(o.index)
    v = agg.pivot(index="d", columns="ticker", values="vol_day").reindex(o.index)
    days = [d for d in o.index if d.weekday() < 5 and o.loc[d].notna().sum() >= min_tickers]
    o, c, v = o.reindex(days), c.reindex(days), v.reindex(days)
    o = o.mask(_quantum_bad(o))
    c = c.mask(_quantum_bad(c))
    nxt = pd.Series(days[1:] + [None], index=days)
    gap_days = pd.Series([(b - a).days if b else np.nan for a, b in zip(days, nxt)], index=days)
    raw = (o.shift(-1) / c - 1.0) * 100.0
    raw = raw.where(gap_days <= MAX_GAP_DAYS, axis=0)                   # разрыв > 5 дней → NaN
    day_move = (c / o - 1.0) * 100.0
    ar_gap, ar_day = ns.abnormal(raw), ns.abnormal(day_move)
    base = v.shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 2).median()
    vs = np.log((v / base).where((v > 0) & (base > 0)))

    known: dict[tuple, set] = {}
    hold: set = set()
    for e in events:
        if e["price_report"]:
            continue
        for kind, d in ns.exec_windows(e["avail"], days):
            key = (e["ticker"], d)
            if kind == "known":
                known.setdefault(key, set()).update(e["sources"])
            else:
                hold.add(key)

    rows = []
    for d in days[:-1]:
        if d_to and d > d_to:
            break
        for tk in ar_gap.columns:
            g = ar_gap.at[d, tk]
            if g != g:
                continue
            src = known.get((tk, d), set())
            k1 = "hashtag" in src
            rows.append({"date": d, "ticker": tk, "ar_gap": g, "raw_gap": raw.at[d, tk],
                         "ar_day": ar_day.at[d, tk], "abs_ar_day": abs(ar_day.at[d, tk]),
                         "log_volspike": vs.at[d, tk], "news": bool(src), "k1": k1,
                         "k2": bool(src) and not k1, "hold": (tk, d) in hold,
                         "nights": int(gap_days[d])})
    return pd.DataFrame(rows)


# ── Регрессия с FE по дате и кластером по дате ───────────────────────────────

def fe_cluster_ols(df: pd.DataFrame, y: str, xs: list[str], cluster: str = "date") -> dict:
    """OLS после вычитания средних по дате (FE), кластерные ошибки по дате.

    Поправка малой выборки как у Stata (FE вложены в кластеры и в K не входят):
    G/(G−1)·(N−1)/(N−K); p — по t с G−1 степенями свободы.
    """
    d = df.dropna(subset=[y] + xs).copy()
    for col in xs:
        d[col] = d[col].astype(float)
    g = d.groupby(cluster)
    Y = (d[y] - g[y].transform("mean")).to_numpy(float)
    X = (d[xs] - g[xs].transform("mean")).to_numpy(float)
    N, K = X.shape
    codes = pd.factorize(d[cluster])[0]
    G = int(codes.max()) + 1 if N else 0
    if N <= K or G < 3:
        return {"n": N, "clusters": G}
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ Y
    u = Y - X @ beta
    sc_ = np.zeros((G, K))
    np.add.at(sc_, codes, X * u[:, None])
    V = (G / (G - 1)) * ((N - 1) / (N - K)) * inv @ (sc_.T @ sc_) @ inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    from scipy import stats
    out = {"n": N, "clusters": G, "coef": {}}
    for i, name in enumerate(xs):
        t = beta[i] / se[i] if se[i] > 0 else float("nan")
        out["coef"][name] = {"b": float(beta[i]), "se": float(se[i]), "t": float(t),
                             "p": float(2 * stats.t.sf(abs(t), G - 1)) if t == t else None}
    return out


def cohort_sample(panel: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Когорта против бумаго-дней без новости; News = флаг когорты."""
    s = panel[panel[cohort] | ~panel["news"]].copy()
    s["News"] = s[cohort].astype(float)
    return s


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def short_economics(panel: pd.DataFrame, cohort: str, position: float,
                    scenario: str = cm.PRIMARY, spreads: dict | None = None) -> dict:
    """Голый ночной шорт бумаг когорты: −сырой ход − издержки бумаги − перенос; t по датам."""
    s = panel[panel[cohort] & ~panel["ticker"].isin(NON_SHORTABLE)].dropna(subset=["raw_gap"])
    if s.empty:
        return {"n": 0}
    spreads = spreads if spreads is not None else cm.load_spreads()
    carry = s["nights"].map(lambda n: carry_pct(position, int(n)))
    cost = s["ticker"].map(lambda t: cm.round_trip(t, scenario, spreads))
    net = -s["raw_gap"] - cost - carry
    per_day = net.groupby(s["date"]).mean()
    n = len(per_day)
    t = float(per_day.mean() / (per_day.std(ddof=1) / math.sqrt(n))) if n > 1 and per_day.std(ddof=1) else None
    return {"n": int(len(s)), "days": n, "gross_short": float((-s["raw_gap"]).groupby(s["date"]).mean().mean()),
            "carry_mean": float(carry.mean()), "net_day": float(per_day.mean()), "t": t,
            "median_trade": float(net.median())}


def c4_verdict(full: dict, halves: dict, econ: dict) -> dict:
    """Точка остановки C4 на K1, модель с контролями."""
    c = (full.get("coef") or {}).get("News", {})
    b, t = c.get("b"), c.get("t")
    signs = [((h.get("coef") or {}).get("News") or {}).get("b") for h in halves.values()]
    same_sign = all(s is not None for s in signs) and len({np.sign(s) for s in signs}) == 1
    close = (t is None or t != t or abs(t) < 2) or not same_sign
    go = (not close and b is not None and b < 0 and t <= -3 and same_sign
          and econ.get("net_day") is not None and econ["net_day"] > 0)
    return {"b1": b, "t": t, "half_signs": signs, "same_sign": same_sign,
            "short_net_day": econ.get("net_day"),
            "verdict": "ЗАКРЫТЬ ветку новостей" if close else
                       ("ПРОДОЛЖАТЬ (блок D)" if go else "НЕ ПРОДОЛЖАТЬ: критерии продолжения не выполнены")}


# ── События ──────────────────────────────────────────────────────────────────

def build_events(posts) -> list[dict]:
    out = []
    for r in posts.itertuples(index=False):
        src = ns.ticker_sources(r.text)
        if not src:
            continue
        rep = ns.is_price_report(r.text)
        avail = r.msk.to_pydatetime() + ns.LAG
        for tk, how in src.items():
            out.append({"message_id": r.message_id, "ticker": tk, "avail": avail,
                        "sources": set(how), "price_report": rep})
    return out


# ── Отчёт ────────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}".replace(".", ",")


def report(res: dict, meta: dict) -> str:
    L = ["# Новости: проверка конфаундера (ТЗ EDGE-R4, блок C)", "",
         f"Сформировано {meta['created']}. Код заморожен коммитом `{meta['revision']}`. "
         "Только исследование, r3 и кроны не затронуты.", "",
         f"Панель {meta['from']} … {meta['to']}: бумаго-дней {meta['rows']}, дат {meta['dates']}; "
         f"K1 (хештег) {meta['k1']}, K2 (только название) {meta['k2']}, без новости {meta['none']}. "
         f"Исключено по шагу цены: {meta['quantum_note']}.", "",
         "Цель — аномальный ход 18:35 D → первая цена основной сессии D+1, % "
         "(минус среднее остальных бумаг). FE по дате, ошибки кластеризованы по дате.", "",
         "## C2. Коэффициент b1 (News)", "",
         "| когорта | модель | период | N | дат | b1 | t | p | p Холма |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["models"].items():
        coh, mod, per = key
        c = (r.get("coef") or {}).get("News", {})
        L.append(f"| {coh} | {mod} | {per} | {r.get('n', 0)} | {r.get('clusters', 0)} | "
                 f"{_f(c.get('b'))} | {_f(c.get('t'), 2)} | {_f(c.get('p'), 4)} | "
                 f"{_f(res['holm'].get(key), 4)} |")
    L += ["", "## Контроли (K1, с контролями, весь период)", "",
          "| переменная | b | t |", "|---|---|---|"]
    for name, c in (res["models"][("K1", "с контролями", "весь")].get("coef") or {}).items():
        L.append(f"| {name} | {_f(c['b'])} | {_f(c['t'], 2)} |")
    hm = res.get("hold_model", {}).get("coef") or {}
    if hm:
        L += ["", "Новость во время удержания (отдельная модель K1 + контроли + Hold): "
              f"b_Hold {_f(hm['hold']['b'])} (t {_f(hm['hold']['t'], 2)}); "
              f"b1 {_f(hm['News']['b'])} (t {_f(hm['News']['t'], 2)})."]
    L += ["", "## Экономика голого ночного шорта (сырой ход, t по датам)", "",
          "| когорта | позиция, ₽ | сделок | дат | валовая шорта | перенос | нетто/день | t | медиана сделки |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (coh, pos), e in res["economics"].items():
        if not e.get("n"):
            continue
        L.append(f"| {coh} | {int(pos)} | {e['n']} | {e['days']} | {_f(e['gross_short'])} | "
                 f"{_f(e['carry_mean'])} | {_f(e['net_day'])} | {_f(e['t'], 2)} | {_f(e['median_trade'])} |")
    v = res["c4"]
    L += ["", "## C4. Точка остановки (K1, с контролями)", "",
          f"- b1 = {_f(v['b1'])}, t = {_f(v['t'], 2)}; знак по частям периода: "
          f"{', '.join(_f(s) for s in v['half_signs'])} — {'один' if v['same_sign'] else 'РАЗНЫЙ'}",
          f"- нетто голого шорта K1 при позиции 10 000 ₽: {_f(v['short_net_day'])} % в день",
          f"- **Вердикт: {v['verdict']}**", "",
          "## Как читать", "",
          "- Сравнение моделей без контролей и с контролями отвечает на вопрос C: сколько «эффекта новости» "
          "объясняется дневным ходом и объёмом.",
          "- Незначимость в K2 при малом n нулём не считается; вывод о шуме — только через K3 (блок D).",
          "- Экономика — по сырому ходу: голый шорт несёт и рынок; аномальная часть без хеджа не монетизируется.", ""]
    return "\n".join(L)


# ── Главное ──────────────────────────────────────────────────────────────────

def run(panel: pd.DataFrame) -> dict:
    """Модели — на одной выборке (строки, где есть все контроли), чтобы разница
    «без контролей / с контролями» была эффектом контролей, а не выборки."""
    res = {"models": {}, "economics": {}}
    pm = panel.dropna(subset=["ar_gap"] + CONTROLS)
    periods = {"весь": pm,
               "05.2024–11.2025": pm[pm["date"] < SPLIT],
               "12.2025–09.2026": pm[pm["date"] >= SPLIT]}
    for coh in ("K1", "K2"):
        for per, p in periods.items():
            s = cohort_sample(p, coh.lower())
            res["models"][(coh, "без контролей", per)] = fe_cluster_ols(s, "ar_gap", ["News"])
            res["models"][(coh, "с контролями", per)] = fe_cluster_ols(s, "ar_gap", ["News"] + CONTROLS)
    tests = {k: (v.get("coef") or {}).get("News", {}).get("p")
             for k, v in res["models"].items() if k[2] == "весь"}
    res["holm"] = holm(tests)
    s1 = cohort_sample(pm, "k1").assign(hold=lambda x: x["hold"].astype(float))
    res["hold_model"] = fe_cluster_ols(s1, "ar_gap", ["News"] + CONTROLS + ["hold"])
    for coh in ("k1", "k2"):
        for pos in POSITIONS:
            res["economics"][(coh.upper(), pos)] = short_economics(panel, coh, pos)
    halves = {k: res["models"][("K1", "с контролями", k)] for k in ("05.2024–11.2025", "12.2025–09.2026")}
    res["c4"] = c4_verdict(res["models"][("K1", "с контролями", "весь")], halves,
                           res["economics"][("K1", MAIN_POSITION)])
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Новости: проверка конфаундера (ТЗ EDGE-R4, блок C)")
    ap.add_argument("--channel", default="markettwits")
    ap.add_argument("--from", dest="date_from", default="2024-05-21")
    ap.add_argument("--to", dest="date_to", default="2026-09-11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    d_from, d_to = dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to)
    conn = database.get_connection()
    try:
        posts = ns.load_posts(conn, a.channel, d_from - dt.timedelta(days=7))
        agg = load_agg(conn, d_from, d_to)
    finally:
        conn.close()
    events = build_events(posts)
    panel = build_panel(agg, events, d_to)
    q = agg.assign(bad=PRICE_QUANTUM / agg["c_eve"] * 100.0 > MAX_QUANTUM_PCT)
    qn = q[q["bad"]].groupby("ticker").size().sort_values(ascending=False)
    res = run(panel)
    try:
        with open(os.path.join(ROOT, "REVISION"), encoding="utf-8") as f:
            rev = f.read().strip()[:12]
    except OSError:
        rev = "unknown"
    now = dt.datetime.now()
    meta = {"created": now.strftime("%d.%m.%Y %H:%M"), "revision": rev,
            "from": str(panel["date"].min()), "to": str(panel["date"].max()),
            "rows": int(len(panel)), "dates": int(panel["date"].nunique()),
            "k1": int(panel["k1"].sum()), "k2": int(panel["k2"].sum()),
            "none": int((~panel["news"]).sum()),
            "quantum_note": ", ".join(f"{t} {n} дн." for t, n in qn.head(5).items()) or "нет"}
    text = report(res, meta)
    out = a.out or os.path.join(ROOT, "audit", "r4_research", f"news_confound-{now:%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    panel.to_csv(os.path.join(out, "panel.csv"), index=False)
    flat = {"meta": meta, "c4": res["c4"], "holm": {" / ".join(k): v for k, v in res["holm"].items()},
            "models": {" / ".join(k): v for k, v in res["models"].items()},
            "hold_model": res["hold_model"],
            "economics": {f"{k[0]} / {int(k[1])}": v for k, v in res["economics"].items()}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=1, default=str)
    print(text)
    log.info("отчёт: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
