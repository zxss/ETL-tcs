"""
H1. Кросс-секционный факторный охват (Gu, Kelly, Xiu 2020).

Идея не в том, чтобы точно предсказать одну бумагу (месячный R² ≈ 0,4 %), а в
широте охвата: слабый, но устойчивый ранжирующий сигнал по всей вселенной даёт
портфельный Шарп заметно выше, чем любая отдельная ставка.

Реализация: гребневая регрессия на кросс-секционных z-оценках шести факторов
(остаточный моментум 30/90, волатильность Гармана–Класса, неликвидность
Амихуда, расстояния до EMA50/EMA200) с целью «доходность вперёд на 5 дней
относительно среднего по вселенной». Ребалансировка раз в 5 торговых дней,
лонг верхнего дециля против фонда TMON@ (шорт-нога считается только справочно —
ночного непокрытого плеча в контуре нет).

Оценка — CPCV: модель учится только на обучающих блоках, предсказания
собираются по тестовым. Сетка (alpha × размер корзины) прогоняется целиком —
она нужна для PBO.
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

from research import event_study_news as es                # noqa: E402
from research.strategic import costs as sc                 # noqa: E402
from research.strategic import panel as pn                 # noqa: E402
from research.strategic import validation as va            # noqa: E402

log = logging.getLogger("research.strategic.h1")

MODULE = "h1_factor_breadth"
HORIZON = 5                     # торговых дней удержания = шаг ребалансировки
HEADLINE = {"alpha": 10.0, "n_names": 5}        # заморожено в rules.json до прогона
GRID = [{"alpha": a, "n_names": n} for a in (1.0, 10.0, 100.0) for n in (3, 5)]
ZCOLS = [c + "_z" for c in pn.FACTORS]


def prepare(conn, d_from: dt.date, d_to: dt.date, warmup_days: int = 260) -> pd.DataFrame:
    """Панель с факторами и целью. warmup — запас дней до начала выборки на окна."""
    tickers = pn.universe(conn)
    daily = pn.load_daily(conn, tickers, d_from - dt.timedelta(days=warmup_days), d_to)
    daily = pn.with_turnover(daily, pn.load_lots())
    feat = pn.features(daily)
    feat = pn.forward_return(pn.cross_section_z(feat), HORIZON)
    feat = feat[(feat["date"] >= d_from) & (feat["date"] <= d_to)]
    return feat.dropna(subset=ZCOLS + ["adv_rub", "garman_klass_vol", "close"]).reset_index(drop=True)


def rebalance_dates(feat: pd.DataFrame, step: int = HORIZON) -> list[dt.date]:
    days = sorted(feat["date"].unique())
    return days[::step]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> pd.Series:
    from sklearn.linear_model import Ridge
    tr = train.dropna(subset=["fwd_ret_rel"])
    if len(tr) < 200 or test.empty:
        return pd.Series(dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(tr[ZCOLS].to_numpy(float), tr["fwd_ret_rel"].to_numpy(float))
    return pd.Series(model.predict(test[ZCOLS].to_numpy(float)), index=test.index)


def oos_predictions(feat: pd.DataFrame, alpha: float, cpcv: dict) -> pd.DataFrame:
    """Предсказания вне обучения: каждая дата ребалансировки попадает в тест
    нескольких разбиений, оценки усредняются."""
    rdays = set(rebalance_dates(feat))
    splits = va.cpcv_splits(sorted(feat["date"].unique()), cpcv["blocks"], cpcv["k_test"],
                            cpcv["embargo_days"], label_days=HORIZON * 2)
    acc = {}
    for sp in splits:
        train = feat[feat["date"].isin(sp["train"])]
        test = feat[feat["date"].isin(sp["test"] & rdays)]
        pred = fit_predict(train, test, alpha)
        for i, p in pred.items():
            acc.setdefault(i, []).append(p)
    if not acc:
        return pd.DataFrame()
    idx = sorted(acc)
    out = feat.loc[idx].copy()
    out["pred"] = [float(np.mean(acc[i])) for i in idx]
    out["n_folds"] = [len(acc[i]) for i in idx]
    return out


def trades_from_predictions(pred: pd.DataFrame, ctx, model: sc.CostModel, n_names: int,
                            variant: str, position_rub: float = pn.POSITION_RUB,
                            side: str = "long") -> pd.DataFrame:
    """Верхний дециль (или нижний для справочной шорт-ноги) на каждую дату."""
    rows = []
    for d, g in pred.groupby("date"):
        g = g.dropna(subset=["fwd_ret", "fwd_date"])
        if len(g) < n_names:
            continue
        pick = g.nlargest(n_names, "pred") if side == "long" else g.nsmallest(n_names, "pred")
        for r in pick.itertuples(index=False):
            cost = model.round_trip_pct(r.ticker, r.garman_klass_vol, position_rub,
                                        r.adv_rub, r.cs_spread_pct)
            if not np.isfinite(cost):
                continue
            gross = r.fwd_ret if side == "long" else -r.fwd_ret
            fund = ctx.fund_pct(d, r.fwd_date)
            if side == "short":
                fund = -fund + model.carry_pct(position_rub, max(1, (r.fwd_date - d).days))
            net = gross - cost - fund
            rows.append({"module": MODULE, "variant": variant, "ticker": r.ticker,
                         "entry_day": d, "exit_day": r.fwd_date, "notional": position_rub,
                         "gross_pct": gross, "cost_pct": cost, "fund_pct": fund,
                         "net_excess_pct": net, "pnl_excess_rub": position_rub * net / 100.0,
                         "pred": r.pred, "sigma_pct": r.garman_klass_vol, "adv_rub": r.adv_rub})
    return pd.DataFrame(rows)


def capacity_rub(trades: pd.DataFrame, model: sc.CostModel, n_names: int) -> float | None:
    """Ёмкость стратегии: сумма Q_opt по одновременно удерживаемым бумагам."""
    if trades.empty:
        return None
    alpha = float(trades["net_excess_pct"].mean() + trades["cost_pct"].mean())   # валовой перевес
    caps = [model.capacity_rub(alpha, r.sigma_pct, r.adv_rub) for r in trades.itertuples(index=False)]
    per_name = float(np.median([c for c in caps if c > 0]) if any(c > 0 for c in caps) else 0.0)
    return per_name * n_names


def long_short_spread(pred: pd.DataFrame, n_names: int) -> dict:
    """Диагностика научной части гипотезы: верхний дециль минус нижний.

    Лонг против фонда на падающем рынке проигрывает независимо от качества
    факторов, поэтому сам ранжирующий сигнал проверяется спредом децилей
    (валовым, до издержек второй ноги) — так его меряют Gu, Kelly, Xiu.
    """
    rows = []
    for d, g in pred.groupby("date"):
        g = g.dropna(subset=["fwd_ret"])
        if len(g) < 2 * n_names:
            continue
        top = g.nlargest(n_names, "pred")["fwd_ret"].mean()
        bot = g.nsmallest(n_names, "pred")["fwd_ret"].mean()
        rows.append({"date": d, "spread_pct": float(top - bot),
                     "top_pct": float(top), "bottom_pct": float(bot)})
    if not rows:
        return {"dates": 0}
    df = pd.DataFrame(rows)
    return {"dates": int(len(df)), "spread_mean_pct": float(df["spread_pct"].mean()),
            "t": va.t_by_date(df["spread_pct"], df["date"]),
            "top_mean_pct": float(df["top_pct"].mean()),
            "bottom_mean_pct": float(df["bottom_pct"].mean()),
            "hit_rate": float((df["spread_pct"] > 0).mean())}


def run(conn, stage: str, rules: dict, ctx=None, n_trials: int | None = None) -> dict:
    """Полный прогон гипотезы на выборке stage."""
    s = rules["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])
    cpcv, gates = rules["cpcv"], rules["gates"]
    model = sc.CostModel(impact_y=rules["costs"]["impact_Y"])
    feat = prepare(conn, d_from, d_to)
    log.info("[H1] %s: строк %d, бумаг %d, дат %d", stage, len(feat),
             feat["ticker"].nunique(), feat["date"].nunique())
    ctx = ctx or Fund()
    per_config, headline = {}, None
    for cfg in GRID:
        name = f"ridge{cfg['alpha']:g}-n{cfg['n_names']}"
        pred = oos_predictions(feat, cfg["alpha"], cpcv)
        tr = trades_from_predictions(pred, ctx, model, cfg["n_names"], name)
        per_config[name] = tr
        if cfg == HEADLINE:
            headline = (name, tr, pred)
    name, tr, pred = headline
    summary = va.summarize(tr, d_from, d_to, 5_000_000.0, n_trials)
    perf = _perf_matrix(per_config)
    pbo = va.cscv_pbo(perf)
    cap = capacity_rub(tr, model, HEADLINE["n_names"])
    short_leg = trades_from_predictions(pred, ctx, model, HEADLINE["n_names"], name + "-short", side="short")
    gate = va.dev_gate(summary, pbo.get("pbo"), cap, gates["dev"]) if stage == "dev" \
        else va.holdout_gate(summary, gates["holdout"])
    return {"module": MODULE, "stage": stage, "variant": name, "summary": summary,
            "pbo": pbo, "capacity_rub": cap, "passed": gate[0], "failed": gate[1],
            "grid": {k: _mini(v) for k, v in per_config.items()},
            "short_leg_reference": _mini(short_leg),
            "long_short_diagnostic": long_short_spread(pred, HEADLINE["n_names"]),
            "universe_mean_fwd_pct": float(feat["fwd_ret"].mean()), "trades": tr}


def _mini(tr: pd.DataFrame) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0}
    return {"trades": int(len(tr)), "net_excess_pct_trade": float(tr["net_excess_pct"].mean()),
            "t": va.t_by_date(tr["net_excess_pct"], tr["entry_day"])}


def _perf_matrix(per_config: dict, slices: int = 10) -> pd.DataFrame:
    """Матрица «отрезок времени × конфигурация» для CSCV/PBO."""
    frames = {}
    for name, tr in per_config.items():
        if tr is None or tr.empty:
            continue
        g = tr.groupby("entry_day")["net_excess_pct"].mean().sort_index()
        frames[name] = g
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).dropna(how="all")
    if len(df) < slices:
        return df
    lab = np.repeat(np.arange(slices), int(np.ceil(len(df) / slices)))[:len(df)]
    return df.groupby(lab).mean()


class Fund:
    """Ставка фонда TMON@ (до 25.02.2025 — LQDT) как барьер доходности."""

    def __init__(self):
        self.h = es.Hurdle(es.HURDLE_PATH)

    def fund_pct(self, d0: dt.date, d1: dt.date) -> float:
        return float(self.h.growth(d0, d1))
