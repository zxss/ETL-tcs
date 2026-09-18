"""
H3. Двухэтапная фильтрация сделок (мета-разметка, López de Prado).

Первичная эвристика даёт сторону сделки, вторая модель отвечает только на вопрос
«брать ли этот сигнал». Это и есть та ниша ML, где он реально работает у Man AHL
и AQR: не генерация альфы, а отсев и сайзинг.

Первичный сигнал — пробой дневного диапазона: закрытие выше максимума прошлых
20 дней, вход на закрытии дня пробоя.
Метки — метод тройного барьера: тейк +1,5·ATR, стоп −1,0·ATR, горизонт 5
торговых дней; если оба барьера пробиты в один день, считаем стоп (консервативно,
внутридневного пути в дневках нет).
Мета-модель — случайный лес глубины 3 на признаках состояния рынка; сделка
берётся только при P(тейк) > 0,60, иначе капитал остаётся в фонде.

Мета-модель учится строго на обучающих блоках CPCV и применяется к тестовым.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.strategic import costs as sc                 # noqa: E402
from research.strategic import panel as pn                 # noqa: E402
from research.strategic import validation as va            # noqa: E402
from research.strategic.h1_factor_breadth import Fund, _perf_matrix, capacity_rub   # noqa: E402

log = logging.getLogger("research.strategic.h3")

MODULE = "h3_meta_labeling"
BREAKOUT_WINDOW = 20
ATR_WINDOW = 14
TP_ATR, SL_ATR, MAX_DAYS = 1.5, 1.0, 5
META_FEATURES = ["atr_pct", "dist_ema50", "garman_klass_vol", "amihud_illiq",
                 "resid_mom_30", "rel_volume", "imoex_ret_5"]
THRESHOLDS = {"p0.50": 0.50, "p0.55": 0.55, "p0.60": 0.60, "p0.65": 0.65}
# Барьеры несимметричны (тейк 1,5·ATR против стопа 1,0·ATR), поэтому базовая доля
# тейков заведомо ниже половины и абсолютный порог 0,60 может не достигаться ни
# разу. Чтобы гипотеза оставалась проверяемой, заранее объявлены и квантильные
# варианты: берём сделки с наибольшей оценкой P независимо от её абсолютной
# величины. Головным остаётся порог ТЗ.
QUANTILES = {"q80": 0.80, "q90": 0.90}
HEADLINE = "p0.60"


def atr_pct(g: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    h, l, c = (g[x].astype(float) for x in ("high", "low", "close"))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(window, min_periods=window // 2).mean() / c * 100.0)


def primary_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """Пробой: закрытие выше максимума прошлых 20 дней (сам день не участвует)."""
    out = []
    for tk, g in daily[daily["ticker"] != pn.INDEX].groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        g["atr_pct"] = atr_pct(g)
        g["hh"] = g["high"].astype(float).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max().shift(1)
        g["breakout"] = g["close"].astype(float) > g["hh"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def triple_barrier(g: pd.DataFrame, i: int, atr: float) -> dict | None:
    """Исход сделки, открытой на закрытии строки i: тейк, стоп или время."""
    entry = float(g["close"].iloc[i])
    if not np.isfinite(atr) or atr <= 0 or entry <= 0:
        return None
    tp, sl = entry * (1.0 + TP_ATR * atr / 100.0), entry * (1.0 - SL_ATR * atr / 100.0)
    for j in range(i + 1, min(i + 1 + MAX_DAYS, len(g))):
        low, high = float(g["low"].iloc[j]), float(g["high"].iloc[j])
        if low <= sl:                                   # стоп раньше тейка при спорном дне
            return {"label": 0, "exit_day": g["date"].iloc[j], "exit_price": sl, "reason": "стоп"}
        if high >= tp:
            return {"label": 1, "exit_day": g["date"].iloc[j], "exit_price": tp, "reason": "тейк"}
    j = min(i + MAX_DAYS, len(g) - 1)
    if j <= i:
        return None
    return {"label": 0, "exit_day": g["date"].iloc[j], "exit_price": float(g["close"].iloc[j]),
            "reason": "время"}


def build_events(daily: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Все сигналы первичной эвристики с метками тройного барьера и признаками."""
    sig = primary_signals(daily)
    fx = feat.set_index(["ticker", "date"])
    rows = []
    for tk, g in sig.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        for i in np.nonzero(g["breakout"].to_numpy())[0]:
            res = triple_barrier(g, int(i), float(g["atr_pct"].iloc[i]))
            if res is None:
                continue
            d = g["date"].iloc[i]
            try:
                f = fx.loc[(tk, d)]
            except KeyError:
                continue
            entry = float(g["close"].iloc[i])
            rows.append({"ticker": tk, "entry_day": d, "exit_day": res["exit_day"],
                         "gross_pct": (res["exit_price"] / entry - 1.0) * 100.0,
                         "label": res["label"], "reason": res["reason"],
                         "atr_pct": float(g["atr_pct"].iloc[i]),
                         **{c: float(f[c]) for c in META_FEATURES if c != "atr_pct"},
                         "sigma_pct": float(f["garman_klass_vol"]), "adv_rub": float(f["adv_rub"]),
                         "cs_spread_pct": float(f["cs_spread_pct"])})
    ev = pd.DataFrame(rows)
    return ev.dropna(subset=META_FEATURES + ["adv_rub"]).reset_index(drop=True)


def meta_probabilities(ev: pd.DataFrame, cpcv: dict) -> pd.Series:
    """P(тейк) вне обучения: лес учится на обучающих блоках CPCV."""
    from sklearn.ensemble import RandomForestClassifier
    splits = va.cpcv_splits(sorted(ev["entry_day"].unique()), cpcv["blocks"], cpcv["k_test"],
                            cpcv["embargo_days"], label_days=MAX_DAYS * 2)
    acc = {}
    for sp in splits:
        tr = ev[ev["entry_day"].isin(sp["train"])]
        te = ev[ev["entry_day"].isin(sp["test"])]
        if len(tr) < 100 or te.empty or tr["label"].nunique() < 2:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=20,
                                    random_state=42, n_jobs=2)
        rf.fit(tr[META_FEATURES].to_numpy(float), tr["label"].to_numpy(int))
        p = rf.predict_proba(te[META_FEATURES].to_numpy(float))[:, 1]
        for i, v in zip(te.index, p):
            acc.setdefault(i, []).append(float(v))
    return pd.Series({i: float(np.mean(v)) for i, v in acc.items()})


def to_trades(ev: pd.DataFrame, ctx, model: sc.CostModel, variant: str,
              position_rub: float = pn.POSITION_RUB) -> pd.DataFrame:
    rows = []
    for r in ev.itertuples(index=False):
        cost = model.round_trip_pct(r.ticker, r.sigma_pct, position_rub, r.adv_rub, r.cs_spread_pct)
        if not np.isfinite(cost):
            continue
        fund = ctx.fund_pct(r.entry_day, r.exit_day)
        net = r.gross_pct - cost - fund
        rows.append({"module": MODULE, "variant": variant, "ticker": r.ticker,
                     "entry_day": r.entry_day, "exit_day": r.exit_day, "notional": position_rub,
                     "gross_pct": r.gross_pct, "cost_pct": cost, "fund_pct": fund,
                     "net_excess_pct": net, "pnl_excess_rub": position_rub * net / 100.0,
                     "reason": r.reason, "sigma_pct": r.sigma_pct, "adv_rub": r.adv_rub})
    return pd.DataFrame(rows)


def run(conn, stage: str, rules: dict, ctx=None, n_trials: int | None = None) -> dict:
    s = rules["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])
    model = sc.CostModel(impact_y=rules["costs"]["impact_Y"])
    ctx = ctx or Fund()
    tickers = pn.universe(conn)
    daily = pn.with_turnover(pn.load_daily(conn, tickers, d_from - dt.timedelta(days=260), d_to),
                             pn.load_lots())
    feat = pn.cross_section_z(pn.features(daily))
    daily = daily[(daily["date"] >= d_from - dt.timedelta(days=40)) & (daily["date"] <= d_to)]
    ev = build_events(daily, feat)
    ev = ev[(ev["entry_day"] >= d_from) & (ev["entry_day"] <= d_to)].reset_index(drop=True)
    log.info("[H3] %s: сигналов пробоя %d, доля тейков %.2f", stage, len(ev), ev["label"].mean())
    raw = to_trades(ev, ctx, model, "raw")
    p = meta_probabilities(ev, rules["cpcv"])
    ev = ev.assign(p_meta=p.reindex(ev.index))
    per_config = {"raw": raw}
    for name, thr in THRESHOLDS.items():
        per_config[name] = to_trades(ev[ev["p_meta"] > thr], ctx, model, name)
    for name, q in QUANTILES.items():
        cut = ev["p_meta"].quantile(q) if ev["p_meta"].notna().any() else None
        per_config[name] = to_trades(ev[ev["p_meta"] >= cut], ctx, model, name) \
            if cut is not None else pd.DataFrame()
    tr = per_config[HEADLINE]
    summary = va.summarize(tr, d_from, d_to, 5_000_000.0, n_trials)
    grid_for_pbo = {k: v for k, v in per_config.items() if k != "raw" and len(v)}
    pbo = va.cscv_pbo(_perf_matrix(grid_for_pbo))
    cap = capacity_rub(tr, model, 5) if not tr.empty else None
    gate = va.dev_gate(summary, pbo.get("pbo"), cap, rules["gates"]["dev"]) if stage == "dev" \
        else va.holdout_gate(summary, rules["gates"]["holdout"])
    return {"module": MODULE, "stage": stage, "variant": HEADLINE, "summary": summary,
            "pbo": pbo, "capacity_rub": cap, "passed": gate[0], "failed": gate[1],
            "events": int(len(ev)), "take_rate": float(ev["label"].mean()) if len(ev) else None,
            "p_meta_median": float(ev["p_meta"].median()) if len(ev) else None,
            "grid": {k: {"trades": int(len(v)),
                         "net_excess_pct_trade": float(v["net_excess_pct"].mean()) if len(v) else None,
                         "t": va.t_by_date(v["net_excess_pct"], v["entry_day"]) if len(v) else None}
                     for k, v in per_config.items()},
            "trades": tr}
