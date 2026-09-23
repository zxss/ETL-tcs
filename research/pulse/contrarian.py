"""
Контрариан-тест сигнала внимания Т-Банк Пульс (ТЗ пользователя 23.09.2026).

Гипотеза (предзарегистрирована ДО прогона): аномально высокое внимание
розницы Пульса к бумаге предсказывает ОТСТАВАНИЕ этой бумаги от универса
на следующий день — «толпа систематически неправа», поэтому фейдим её
любимцев. Проверяется, а не копируется популярность (см. память
tbank-pulse-data-source: наивный сигнал «за авторами» бесполезен).

Сигнал: attn_z[тикер, D] = z-оценка дневного числа постов (или уникальных
авторов) относительно собственной скользящей нормы бумаги (окно N дней,
только назад). Известен на закрытии D.
Цель: кросс-секционно демеанированная доходность close[D]→close[D+1]
(рыночно-нейтральная — эффект внимания против соседей, не бета к рынку).

Два теста:
  1) кросс-секционная корреляция attn_z ↔ fwd_ret_rel по дням (contrarian
     edge = устойчиво ОТРИЦАТЕЛЬНАЯ), t по дням, Холм по сетке;
  2) рыночно-нейтральный портфель: лонг нижнего дециля внимания, шорт
     верхнего, удержание 1 день, с издержками — dev + holdout, DoR-гейт.

Выборки: Пульс собран с 2026-04-01 (research/pulse/collect). Локальные цены
до 2026-09-09.
  dev     2026-04-01 … 2026-07-15
  holdout 2026-07-16 … 2026-09-08  (один прогон)

Запуск: python -m research.pulse.contrarian
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

from research import cost_model as cm                      # noqa: E402
from research import news_event_study as ns                # noqa: E402
from research.strategic import validation as va             # noqa: E402

log = logging.getLogger("research.pulse.contrarian")
COUNTS = os.path.join(ROOT, "audit", "r4_research", "pulse", "daily_counts.csv")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "pulse")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
POSITION_RUB = 10_000.0
# Данные Пульса собраны с 2026-06-15 (глубже тяжёлые тикеры через этот API
# тянуть непрактично, см. отчёт). Окна короткие — это ПЕРВЫЙ СРЕЗ, не полный
# holdout-протокол; z-окно короткое, чтобы осталось дней на разбиение.
DEV_FROM, DEV_TO = dt.date(2026, 4, 15), dt.date(2026, 7, 15)      # сбор рекомендуется --from 2026-03-01 (запас на разогрев z)
HO_FROM, HO_TO = dt.date(2026, 7, 16), dt.date(2026, 9, 19)
HEADLINE = {"measure": "posts", "window": 20, "n_names": 5}
GRID = [{"measure": m, "window": w, "n_names": 5}
        for m in ("posts", "authors") for w in (10, 20, 40)]


def load_prices(conn) -> pd.DataFrame:
    tickers = sorted(set(ns.UNIVERSE))
    q = ("SELECT ticker, date, close FROM market_data WHERE ticker = ANY(%(tk)s) "
         "AND close > 0 ORDER BY ticker, date")
    df = pd.read_sql(q, conn, params={"tk": tickers})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = df["close"].astype(float)
    df = df.sort_values(["ticker", "date"])
    df["fwd_ret"] = df.groupby("ticker")["close"].transform(lambda c: c.shift(-1) / c - 1.0) * 100.0
    return df


def build_panel(conn, measure: str, window: int) -> pd.DataFrame:
    counts = pd.read_csv(COUNTS)
    counts["date"] = pd.to_datetime(counts["date"]).dt.date
    prices = load_prices(conn)
    # только торговые дни (по наличию цены), полный универс на каждую дату
    panel = prices.merge(counts[["ticker", "date", measure]], on=["ticker", "date"], how="left")
    panel[measure] = panel[measure].fillna(0.0)
    panel = panel.sort_values(["ticker", "date"])
    g = panel.groupby("ticker")[measure]
    mu = g.transform(lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 2)).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 2)).std(ddof=1))
    panel["attn_z"] = (panel[measure] - mu) / sd.replace(0.0, np.nan)
    # кросс-секционно демеанированная форвард-доходность (рыночно-нейтрально)
    panel["fwd_ret_rel"] = panel["fwd_ret"] - panel.groupby("date")["fwd_ret"].transform("mean")
    return panel.dropna(subset=["attn_z", "fwd_ret_rel"])


def xsec_corr(panel: pd.DataFrame, d_from: dt.date, d_to: dt.date) -> dict:
    sub = panel[(panel["date"] >= d_from) & (panel["date"] <= d_to)]
    per_day = sub.groupby("date").apply(
        lambda x: float(np.corrcoef(x["attn_z"], x["fwd_ret_rel"])[0, 1])
        if len(x) >= 6 and x["attn_z"].std() > 0 else np.nan, include_groups=False).dropna()
    if len(per_day) < 10:
        return {"days": int(len(per_day))}
    t, _ = _t(per_day)
    return {"days": int(len(per_day)), "mean_corr": float(per_day.mean()), "t": t,
            "p": _p(t, len(per_day))}


def portfolio(panel: pd.DataFrame, d_from: dt.date, d_to: dt.date, n_names: int,
              spreads: dict) -> pd.DataFrame:
    """Лонг низ внимания, шорт верх; нетто на день = (ср. low − ср. high) − издержки."""
    sub = panel[(panel["date"] >= d_from) & (panel["date"] <= d_to)]
    rows = []
    for d, g in sub.groupby("date"):
        g = g.dropna(subset=["attn_z", "fwd_ret"])
        if len(g) < 2 * n_names:
            continue
        high = g.nlargest(n_names, "attn_z")           # любимцы толпы → шорт
        low = g.nsmallest(n_names, "attn_z")            # заброшенные → лонг
        gross = float(low["fwd_ret"].mean() - high["fwd_ret"].mean())
        cost = float(np.mean([cm.round_trip(t, "base", spreads) for t in
                              list(high["ticker"]) + list(low["ticker"])]))
        rows.append({"date": d, "gross": gross, "cost": cost, "net": gross - cost})
    return pd.DataFrame(rows)


def _t(x: pd.Series):
    n = len(x)
    sd = float(x.std(ddof=1))
    if n < 2 or not sd:
        return None, n
    return float(x.mean() / (sd / math.sqrt(n))), n


def _p(t, n):
    if t is None:
        return None
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), n - 1))


def stats_block(pf: pd.DataFrame) -> dict:
    if pf.empty:
        return {"days": 0}
    t, n = _t(pf["net"])
    top5 = pf["net"].nlargest(5).sum()
    total = float(pf["net"].sum())
    return {"days": int(len(pf)), "mean_day": float(pf["net"].mean()),
            "median_day": float(pf["net"].median()), "t": t, "p": _p(t, n),
            "win_rate": float((pf["net"] > 0).mean()), "sum": total,
            "sum_wo_top5": total - float(top5),
            "gross_mean_day": float(pf["gross"].mean())}


def dor(st: dict) -> dict:
    if not st.get("days"):
        return {}
    return {"t ≥ 2,5 и плюс/день": bool(st.get("t") and st["t"] >= 2.5 and st["mean_day"] > 0),
            "без 5 лучших дней > 0": bool(st["sum_wo_top5"] > 0),
            "медиана дня > 0": bool(st["median_day"] > 0)}


def run(conn) -> dict:
    spreads = cm.load_spreads()
    res = {"corr": {}, "portfolio": {}, "grid_corr": {}}

    # сетка: кросс-секционная корреляция на dev, Холм по вариантам
    pvals = {}
    for cfg in GRID:
        key = f"{cfg['measure']}_w{cfg['window']}"
        panel = build_panel(conn, cfg["measure"], cfg["window"])
        c = xsec_corr(panel, DEV_FROM, DEV_TO)
        res["grid_corr"][key] = c
        if c.get("p") is not None:
            pvals[key] = c["p"]
    items = sorted((v, k) for k, v in pvals.items())
    run_p, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        res["grid_corr"][k]["p_holm"] = run_p

    # головной вариант: корреляция dev+holdout и портфель dev+holdout
    panel = build_panel(conn, HEADLINE["measure"], HEADLINE["window"])
    res["corr"]["dev"] = xsec_corr(panel, DEV_FROM, DEV_TO)
    res["corr"]["holdout"] = xsec_corr(panel, HO_FROM, HO_TO)
    pf_dev = portfolio(panel, DEV_FROM, DEV_TO, HEADLINE["n_names"], spreads)
    pf_ho = portfolio(panel, HO_FROM, HO_TO, HEADLINE["n_names"], spreads)
    res["portfolio"]["dev"] = {**stats_block(pf_dev), "dor": dor(stats_block(pf_dev))}
    res["portfolio"]["holdout"] = {**stats_block(pf_ho), "dor": dor(stats_block(pf_ho))}
    res["headline"] = HEADLINE
    res["samples"] = {"dev": [str(DEV_FROM), str(DEV_TO)], "holdout": [str(HO_FROM), str(HO_TO)]}
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
    with open(os.path.join(OUT_DIR, "contrarian_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 13,
                            "stage": "pulse_contrarian", "trials": len(GRID) + 2,
                            "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
