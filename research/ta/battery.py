"""
Батарея технического анализа: есть ли кросс-секционное преимущество (ТЗ
пользователя 24.09.2026). Плюс комбинация ТА с текущей моделью (ночная корзина).

Дисциплина против data-snooping: батарея ~16 КЛАССИЧЕСКИХ индикаторов со
СТАНДАРТНЫМИ параметрами (не подбираются). Основной замер — кросс-секционный IC
(ранговая корреляция индикатора с завтрашней относительной доходностью) по дням,
t по дням, поправка Холма ПО ВСЕЙ батарее (двусторонне — направление не
навязываем). Правило строится только для индикаторов, переживших Холм на dev;
направление берётся по знаку IC на dev и проверяется на holdout (OOS и по знаку,
и по величине). После 217 прежних испытаний планка DSR очень высокая — это
учитываем в интерпретации.

Индикаторы (канон): RSI(14), MACD(12,26,9) гистограмма, Bollinger %b(20,2),
стохастик %K(14), Williams %R(14), CCI(20), ROC(10/20), дистанция до SMA(50/200),
режим SMA50><200, ATR%(14), позиция Дончиана(20), наклон OBV, всплеск объёма.

Выборки: dev 2024-05-21…2025-11-30, holdout 2025-12-01…2026-09-22.
Запуск (на сервере): python -m research.ta.battery
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
from research import news_event_study as ns               # noqa: E402

log = logging.getLogger("research.ta.battery")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "ta")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
DEV = (dt.date(2024, 5, 21), dt.date(2025, 11, 30))
HOLDOUT = (dt.date(2025, 12, 1), dt.date(2026, 9, 22))
QSPLIT = 0.2                        # квинтили для long-short


def _rsi(c: pd.Series, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Все индикаторы для одной бумаги (g отсортирован по дате)."""
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    out = pd.DataFrame(index=g.index)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    out["rsi14"] = _rsi(c, 14)
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    out["bb_pctb"] = (c - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = (c - ll14) / (hh14 - ll14).replace(0, np.nan) * 100
    out["williams_r"] = (hh14 - c) / (hh14 - ll14).replace(0, np.nan) * -100
    tp = (h + l + c) / 3
    out["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)).replace(0, np.nan)
    out["roc10"] = c.pct_change(10) * 100
    out["roc20"] = c.pct_change(20) * 100
    out["dist_sma50"] = (c / c.rolling(50).mean() - 1) * 100
    out["dist_sma200"] = (c / c.rolling(200).mean() - 1) * 100
    out["sma_regime"] = np.sign(c.rolling(50).mean() - c.rolling(200).mean())
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / c * 100
    ll20, hh20 = l.rolling(20).min(), h.rolling(20).max()
    out["donchian_pos"] = (c - ll20) / (hh20 - ll20).replace(0, np.nan)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    out["obv_slope"] = obv.diff(10) / v.rolling(20).mean().replace(0, np.nan)
    out["vol_surge"] = v / v.rolling(20).mean().replace(0, np.nan)
    return out


IND_COLS = ["macd_hist", "rsi14", "bb_pctb", "stoch_k", "williams_r", "cci20",
            "roc10", "roc20", "dist_sma50", "dist_sma200", "sma_regime",
            "atr_pct", "donchian_pos", "obv_slope", "vol_surge"]


def build_panel(conn) -> pd.DataFrame:
    tickers = sorted(set(ns.UNIVERSE))
    df = pd.read_sql("SELECT ticker,date,open,high,low,close,volume FROM market_data "
                     "WHERE ticker = ANY(%(tk)s) AND close>0 ORDER BY ticker,date",
                     conn, params={"tk": tickers})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for x in ("open", "high", "low", "close", "volume"):
        df[x] = df[x].astype(float)
    parts = []
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        ind = indicators(g)
        ind["ticker"], ind["date"], ind["close"] = tk, g["date"], g["close"]
        ind["fwd_ret"] = (g["close"].shift(-1) / g["close"] - 1) * 100
        ind["turn_prev"] = (g["close"] * g["volume"]).shift(1)
        parts.append(ind)
    panel = pd.concat(parts, ignore_index=True)
    panel["fwd_ret_rel"] = panel["fwd_ret"] - panel.groupby("date")["fwd_ret"].transform("mean")
    return panel.dropna(subset=["fwd_ret_rel"])


def daily_ic(panel: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    from scipy import stats
    vals, days = [], []
    for d, g in panel.groupby("date"):
        gg = g.dropna(subset=[col, "fwd_ret_rel"])
        if len(gg) >= 8 and gg[col].std() > 0:
            vals.append(float(stats.spearmanr(gg[col], gg["fwd_ret_rel"]).statistic))
            days.append(d)
    return pd.Series(vals), pd.Series(days)


def t_of(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 10 or s.std(ddof=1) == 0:
        return {"days": n}
    t = float(s.mean() / (s.std(ddof=1) / math.sqrt(n)))
    from scipy import stats
    return {"days": n, "mean_ic": float(s.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1))}


def ls_rule(panel: pd.DataFrame, col: str, direction: int, spreads: dict) -> dict:
    """Лонг верхнего квинтиля / шорт нижнего по direction*индикатору, ежедневно, нетто."""
    rows_net, rows_day = [], []
    for d, g in panel.groupby("date"):
        gg = g.dropna(subset=[col, "fwd_ret"])
        if len(gg) < 10:
            continue
        s = direction * gg[col]
        hi = gg[s >= s.quantile(1-QSPLIT)]
        lo = gg[s <= s.quantile(QSPLIT)]
        if len(hi) < 2 or len(lo) < 2:
            continue
        gross = hi["fwd_ret"].mean() - lo["fwd_ret"].mean()
        cost = np.mean([cm.round_trip(t, "base", spreads) for t in list(hi["ticker"]) + list(lo["ticker"])])
        rows_net.append(gross - cost)
        rows_day.append(d)
    net = pd.Series(rows_net, index=rows_day)
    r = t_of(net)
    r["mean_day_pct"] = float(net.mean()) if len(net) else None
    r["win_rate"] = float((net > 0).mean()) if len(net) else None
    return r


def run(conn) -> dict:
    spreads = cm.load_spreads()
    full = build_panel(conn)
    dev = full[(full["date"] >= DEV[0]) & (full["date"] <= DEV[1])]
    hold = full[(full["date"] >= HOLDOUT[0]) & (full["date"] <= HOLDOUT[1])]

    # 1) IC каждого индикатора на dev, Холм по батарее
    ic = {}
    pvals = {}
    dev_ic_sign = {}
    for col in IND_COLS:
        s, _ = daily_ic(dev, col)
        r = t_of(s)
        ic[col] = {"dev": r}
        if r.get("p") is not None:
            pvals[col] = r["p"]
            dev_ic_sign[col] = 1 if r["mean_ic"] >= 0 else -1
    items = sorted((v, k) for k, v in pvals.items())
    m = len(items); run_p = 0.0
    for i, (v, k) in enumerate(items):
        run_p = max(run_p, min(1.0, (m - i) * v))
        ic[k]["dev"]["p_holm"] = run_p

    # 2) для всех индикаторов — IC на holdout (OOS по знаку) + long-short правило dev/holdout
    for col in IND_COLS:
        s_h, _ = daily_ic(hold, col)
        ic[col]["holdout"] = t_of(s_h)
        d = dev_ic_sign.get(col, 1)
        ic[col]["rule_dir"] = d
        ic[col]["rule_dev"] = ls_rule(dev, col, d, spreads)
        ic[col]["rule_holdout"] = ls_rule(hold, col, d, spreads)

    # 3) композит: z-score по дню, знак по dev-IC, среднее
    def composite(panel):
        p = panel.copy()
        comp = pd.Series(0.0, index=p.index)
        cnt = pd.Series(0.0, index=p.index)
        for col in IND_COLS:
            z = p.groupby("date")[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else x*0)
            comp = comp.add((dev_ic_sign.get(col, 1) * z).fillna(0.0), fill_value=0.0)
            cnt = cnt.add(z.notna().astype(float), fill_value=0.0)
        p["composite"] = comp / cnt.replace(0, np.nan)
        return p
    dev_c, hold_c = composite(dev), composite(hold)
    comp_res = {"dev_ic": t_of(daily_ic(dev_c, "composite")[0]),
                "holdout_ic": t_of(daily_ic(hold_c, "composite")[0]),
                "rule_dev": ls_rule(dev_c, "composite", 1, spreads),
                "rule_holdout": ls_rule(hold_c, "composite", 1, spreads)}

    return {"battery": IND_COLS, "n_indicators": len(IND_COLS),
            "dev_sample": [str(x) for x in DEV], "holdout_sample": [str(x) for x in HOLDOUT],
            "indicators": ic, "composite": comp_res,
            "note": "IC двусторонний, Холм по батарее; правило — направление по знаку dev-IC, holdout OOS"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "battery_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 18,
                            "stage": "ta_battery", "trials": len(IND_COLS) + 1, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
