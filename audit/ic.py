"""
Блок 2.2 — прогностическая сила FinalScore и его компонент.

Rank IC (Information Coefficient) — корреляция Спирмена между рангом сигнала на
дату t и рангом фактической доходности на t+1, посчитанная ВНУТРИ кросс-секции
каждого дня и затем усреднённая по дням:

    IC_t   = spearman( rank(score_{i,t}), rank(R_{i,t+1}) )
    IC     = mean_t(IC_t)
    t-stat = IC / std(IC_t) * sqrt(T)

Порог из ТЗ: IC < 0,03 или |t| < 2,0 — сигнал неотличим от шума.

Считается по кросс-секции ОДНОГО дня, поэтому общерыночное движение (главный
источник ложной корреляции при пулинге) вычитается автоматически: в один день
все бумаги живут в одном рыночном режиме.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Веса FinalScore из tft_forecast/combined.py::_score_row
WEIGHTS = {
    "exp_score": 0.30,
    "prob_score": 0.15,
    "validation_score": 0.15,
    "liq_score_n": 0.15,
    "rs_score": 0.10,
    "regime_score": 0.10,
    "vol_score": 0.05,
}

LONG_STRATS = {"long_overnight", "intraday_long"}


def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def build_components(df: pd.DataFrame,
                     validation_score: float | pd.Series = 0.5) -> pd.DataFrame:
    """Восстанавливает суб-скоры FinalScore по формуле production combined.py.

    validation_score вынесен параметром: в реплее контур валидации на каждую
    историческую дату не пересчитывается (см. отчёт), поэтому по умолчанию он
    нейтрален и вклад остальных компонент измеряется чисто.
    """
    d = df.copy()
    d["is_long"] = d["strategy"].isin(LONG_STRATS)

    d["exp_score"] = _clamp01(0.5 + d["exp_pnl"].fillna(0.0) / 2.0)
    d["prob_score"] = d["prob_profit"].fillna(0.5)
    d["validation_score"] = validation_score
    d["liq_score_n"] = d["liq_score"].fillna(50.0) / 100.0

    rs_eff = np.where(d["is_long"], d["rs"], -d["rs"])
    rs_eff = pd.Series(rs_eff, index=d.index).fillna(0.0)
    d["rs_score"] = _clamp01(0.5 + rs_eff / 10.0)

    regime_long = {"BULL": 1.0, "NEUTRAL": 0.5, "BEAR": 0.2}
    regime_short = {"BEAR": 1.0, "NEUTRAL": 0.5, "BULL": 0.2}
    d["regime_score"] = [
        (regime_long if lg else regime_short).get(rg, 0.5)
        for lg, rg in zip(d["is_long"], d["regime"])
    ]

    vs = d["vol_spike"]
    d["vol_score"] = np.select(
        [vs.isna(), vs > 4.0, vs > 2.5, vs < 0.8],
        [0.5, 0.2, 0.4, 0.4], default=0.7)

    d["final_raw"] = sum(d[c] * w for c, w in WEIGHTS.items())

    # Мультипликативные риск-штрафы (те же, что в production).
    pen = np.ones(len(d))
    bear_long = d["is_long"] & (d["regime"] == "BEAR")
    bull_short = (~d["is_long"]) & (d["regime"] == "BULL")
    pen = np.where(bear_long | bull_short, pen * 0.70, pen)
    pen = np.where((~d["is_long"]) & (d["rs"] > 5.0), pen * 0.75, pen)
    pen = np.where(d["vol_spike"] > 4.0, pen * 0.70, pen)
    pen = np.where((d["strategy"] == "long_overnight") & (d["gap_down_prob"] > 0.50),
                   pen * 0.70, pen)
    d["final_score"] = d["final_raw"] * pen
    return d


def rank_ic(df: pd.DataFrame, score_col: str, ret_col: str = "realized_net",
            date_col: str = "asof_date", min_names: int = 8) -> dict:
    """Rank IC по кросс-секции каждого дня + сводная статистика."""
    ics, ns, dates = [], [], []
    for d, g in df.groupby(date_col, sort=True):
        g = g[[score_col, ret_col]].dropna()
        if len(g) < min_names:
            continue
        if g[score_col].nunique() < 3 or g[ret_col].nunique() < 3:
            continue
        rho = stats.spearmanr(g[score_col], g[ret_col]).statistic
        if np.isfinite(rho):
            ics.append(float(rho))
            ns.append(len(g))
            dates.append(d)

    if len(ics) < 5:
        return {"ic": np.nan, "t_stat": np.nan, "n_days": len(ics)}

    a = np.asarray(ics)
    mean, sd = a.mean(), a.std(ddof=1)
    t = mean / sd * np.sqrt(len(a)) if sd > 0 else np.nan
    return {
        "ic": float(mean),
        "ic_std": float(sd),
        "ir": float(mean / sd) if sd > 0 else np.nan,
        "t_stat": float(t),
        "p_value": float(2 * (1 - stats.t.cdf(abs(t), len(a) - 1))) if np.isfinite(t) else np.nan,
        "n_days": len(a),
        "avg_names": float(np.mean(ns)),
        "hit_rate": float((a > 0).mean()),
        "series": pd.Series(a, index=pd.DatetimeIndex(dates)),
    }


def ic_table(df: pd.DataFrame, cols: list[str], ret_col: str = "realized_net",
             by_strategy: bool = True) -> pd.DataFrame:
    """Таблица Rank IC по компонентам, опционально в разрезе стратегий."""
    rows = []
    groups = [("ВСЕ", df)]
    if by_strategy:
        groups += [(s, g) for s, g in df.groupby("strategy", sort=True)]
    for gname, g in groups:
        for c in cols:
            if c not in g.columns:
                continue
            r = rank_ic(g, c, ret_col)
            rows.append({
                "scope": gname, "signal": c,
                "rank_ic": r.get("ic"), "t_stat": r.get("t_stat"),
                "ir": r.get("ir"), "hit_rate": r.get("hit_rate"),
                "n_days": r.get("n_days"), "avg_names": r.get("avg_names"),
                "verdict": _verdict(r.get("ic"), r.get("t_stat")),
            })
    return pd.DataFrame(rows)


def _verdict(ic, t) -> str:
    """Порог ТЗ: IC >= 0,03 И |t| >= 2,0."""
    if ic is None or not np.isfinite(ic) or t is None or not np.isfinite(t):
        return "н/д"
    if abs(t) < 2.0:
        return "ШУМ (|t| < 2)"
    if ic >= 0.05:
        return "СИЛЬНЫЙ"
    if ic >= 0.03:
        return "СЛАБЫЙ, но значим"
    if ic <= -0.03:
        return "ИНВЕРСНЫЙ"
    return "ЗНАЧИМ, но мал"


def component_correlations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Матрица корреляций Спирмена между компонентами — проверка избыточности."""
    sub = df[cols].dropna()
    return sub.corr(method="spearman")


def decay_by_model_age(df: pd.DataFrame, score_col: str,
                       ret_col: str = "realized_net",
                       bins=(0, 15, 30, 45, 70)) -> pd.DataFrame:
    """Rank IC в разрезе «возраста» модели — насколько быстро стареет прогноз."""
    d = df.copy()
    d["age_bin"] = pd.cut(d["days_since_fit"], bins=bins, right=True)
    rows = []
    for b, g in d.groupby("age_bin", observed=True):
        r = rank_ic(g, score_col, ret_col)
        rows.append({"age_days": str(b), "rank_ic": r.get("ic"),
                     "t_stat": r.get("t_stat"), "n_days": r.get("n_days")})
    return pd.DataFrame(rows)
