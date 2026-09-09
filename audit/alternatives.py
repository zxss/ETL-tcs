"""
Раздел 3 ТЗ — альтернативные алгоритмы.

Альтернатива 1. Кросс-секционный ранжировщик на градиентном бустинге.
    Цель  y_i = Rank(R_{i,t→t+H}) - 0.5 внутри дня (кросс-секционный ранг, а не
    абсолютная доходность): модель учится сравнивать бумаги между собой, а не
    угадывать уровень рынка. Общерыночное движение уходит в ранжирование и
    перестаёт быть источником ложной «альфы».

Альтернатива 2. Режимы рынка через HMM с 3 состояниями вместо жёсткой
    эвристики EMA50/EMA200.

Альтернатива 3. Возврат к среднему по квантильным полосам (Ornstein-Uhlenbeck):
    оценка скорости возврата и сравнение целей тейк-профита q50 против q90.

Ограничение данных: в market_data есть только IMOEX. Признаков RGBI (облигации)
и RVI (вменённая волатильность), которых требует ТЗ для HMM, в базе нет — вместо
них используются волатильность и ширина рынка, посчитанные по самой вселенной.
Это указано в отчёте как разрыв данных.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("audit.alternatives")

TRADING_DAYS = 252


# ── Признаки для ранжировщика ────────────────────────────────────────────────

def parkinson_vol(h: pd.Series, l: pd.Series, win: int = 20) -> pd.Series:
    """Оценка волатильности Паркинсона по диапазону High/Low."""
    hl = np.log(h / l) ** 2
    return np.sqrt(hl.rolling(win).mean() / (4.0 * np.log(2.0))) * np.sqrt(TRADING_DAYS)


def garman_klass_vol(o, h, l, c, win: int = 20) -> pd.Series:
    """Оценка Гармана-Класса: использует все четыре цены бара."""
    term = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(term.rolling(win).mean().clip(lower=0)) * np.sqrt(TRADING_DAYS)


def build_ranker_features(daily: pd.DataFrame,
                          index_close: pd.Series,
                          session_ratio: pd.DataFrame | None = None,
                          horizon: int = 5) -> pd.DataFrame:
    """Панель признаков и целей для кросс-секционного ранжировщика.

    Все признаки строятся только по данным на дату t включительно.
    """
    out = []
    idx = index_close.sort_index()
    idx_ret = {w: idx.pct_change(w) for w in (3, 10, 30)}

    for tk, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).set_index("date")
        if len(g) < 260:
            continue
        c, o, h, l, v = g["close"], g["open"], g["high"], g["low"], g["volume"]
        rub = g["rub_volume"] if "rub_volume" in g else c * v

        d = pd.DataFrame(index=g.index)
        d["ticker"] = tk

        # — моментум и разворот, относительно рынка —
        for w in (3, 10, 30):
            own = c.pct_change(w)
            mkt = idx_ret[w].reindex(g.index)
            d[f"rs_{w}"] = (own - mkt) * 100.0
        d["ret_1"] = c.pct_change(1) * 100.0
        d["ret_5"] = c.pct_change(5) * 100.0

        # — расстояние до скользящих в единицах волатильности —
        ret = np.log(c / c.shift(1))
        sd20 = ret.rolling(20).std()
        for span in (50, 200):
            ema = c.ewm(span=span, adjust=False).mean()
            d[f"z_ema{span}"] = (np.log(c / ema) / sd20.replace(0, np.nan))

        # — волатильность: отношение диапазонных оценок к close-to-close —
        cc = sd20 * np.sqrt(TRADING_DAYS)
        pk = parkinson_vol(h, l, 20)
        gk = garman_klass_vol(o, h, l, c, 20)
        d["vol_cc"] = cc
        d["pk_over_cc"] = pk / cc.replace(0, np.nan)
        d["gk_over_cc"] = gk / cc.replace(0, np.nan)
        d["atr_pctl"] = _atr_pctl_series(h, l, c)

        # — микроструктура и ликвидность —
        d["amihud"] = (ret.abs() / rub.replace(0, np.nan)).rolling(60).median() * 1e9
        d["log_adv"] = np.log(rub.rolling(60).median().replace(0, np.nan))
        d["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
        # Дисбаланс объёма: доля дней роста в объёме за 10 дней.
        up = (c > c.shift(1)).astype(float)
        d["vol_imbalance"] = ((v * up).rolling(10).sum() /
                              v.rolling(10).sum().replace(0, np.nan))
        d["overnight_mean10"] = ((o / c.shift(1) - 1) * 100).rolling(10).mean()
        d["intraday_mean10"] = ((c / o - 1) * 100).rolling(10).mean()

        if session_ratio is not None:
            sr = session_ratio[session_ratio["ticker"] == tk].set_index("date")
            d["evening_share"] = sr["evening_share"].reindex(g.index).ffill(limit=5)

        # — цели —
        d["fwd_ret_1"] = (c.shift(-1) / c - 1.0) * 100.0
        d["fwd_ret_h"] = (c.shift(-horizon) / c - 1.0) * 100.0
        d["fwd_overnight"] = (o.shift(-1) / c - 1.0) * 100.0
        d["fwd_intraday"] = (c.shift(-1) / o.shift(-1) - 1.0) * 100.0
        out.append(d.reset_index())

    panel = pd.concat(out, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def _atr_pctl_series(h, l, c, win: int = 14, hist: int = 252) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(win).mean()
    return atr.rolling(hist).rank(pct=True) * 100.0


def add_cs_rank_target(panel: pd.DataFrame, col: str = "fwd_ret_h") -> pd.DataFrame:
    """Кросс-секционный ранг цели внутри дня: y = rank_pct - 0.5."""
    d = panel.copy()
    d["y_rank"] = d.groupby("date")[col].rank(pct=True) - 0.5
    return d


FEATURE_COLS = [
    "rs_3", "rs_10", "rs_30", "ret_1", "ret_5",
    "z_ema50", "z_ema200",
    "vol_cc", "pk_over_cc", "gk_over_cc", "atr_pctl",
    "amihud", "log_adv", "vol_z", "vol_imbalance",
    "overnight_mean10", "intraday_mean10",
]


# ── Альтернатива 1: GBDT-ранжировщик, walk-forward ───────────────────────────

def walkforward_ranker(panel: pd.DataFrame,
                       fit_dates: list[pd.Timestamp],
                       train_months: int = 12,
                       oos_months: int = 1,
                       features: list[str] | None = None,
                       target: str = "y_rank",
                       n_estimators: int = 400,
                       learning_rate: float = 0.03,
                       num_leaves: int = 15,
                       seed: int = 42) -> pd.DataFrame:
    """Обучает GBDT на скользящем окне и предсказывает следующий OOS-период."""
    features = features or [c for c in FEATURE_COLS if c in panel.columns]
    try:
        import lightgbm as lgb
        backend = "lightgbm"
    except ImportError:                                    # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor
        backend = "sklearn"

    preds = []
    for t0 in fit_dates:
        tr = panel[(panel["date"] <= t0) &
                   (panel["date"] > t0 - pd.DateOffset(months=train_months))]
        tr = tr.dropna(subset=features + [target])
        te = panel[(panel["date"] > t0) &
                   (panel["date"] <= t0 + pd.DateOffset(months=oos_months))]
        te = te.dropna(subset=features)
        if len(tr) < 500 or te.empty:
            continue

        if backend == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=seed, verbose=-1)
            model.fit(tr[features], tr[target])
        else:                                              # pragma: no cover
            model = HistGradientBoostingRegressor(
                max_iter=n_estimators, learning_rate=learning_rate,
                max_leaf_nodes=num_leaves, random_state=seed)
            model.fit(tr[features], tr[target])

        p = te.copy()
        p["pred"] = model.predict(te[features])
        p["fit_date"] = t0
        p["backend"] = backend
        preds.append(p)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def feature_importance(panel: pd.DataFrame, features: list[str] | None = None,
                       target: str = "y_rank", seed: int = 42) -> pd.DataFrame:
    """Важность признаков на всей истории — диагностика, не результат."""
    import lightgbm as lgb
    features = features or [c for c in FEATURE_COLS if c in panel.columns]
    d = panel.dropna(subset=features + [target])
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                          min_child_samples=40, random_state=seed, verbose=-1)
    m.fit(d[features], d[target])
    return (pd.DataFrame({"feature": features, "gain": m.booster_.feature_importance("gain")})
            .sort_values("gain", ascending=False).reset_index(drop=True))


# ── Альтернатива 2: режимы рынка на HMM ──────────────────────────────────────

def hmm_regimes(index_daily: pd.DataFrame, breadth: pd.Series | None = None,
                n_states: int = 3, seed: int = 42) -> pd.DataFrame:
    """3-режимная HMM по доходности и волатильности индекса.

    ТЗ требует связку [доходность IMOEX, изменение RGBI, волатильность RVI].
    RGBI и RVI в базе отсутствуют, поэтому используются доходность IMOEX,
    волатильность Паркинсона по IMOEX и (при наличии) ширина рынка.

    Состояния упорядочиваются по средней доходности, чтобы номер режима был
    интерпретируемым: 0 = худший, n-1 = лучший.
    """
    from hmmlearn import hmm

    d = index_daily.sort_values("date").reset_index(drop=True)
    ret = np.log(d["close"] / d["close"].shift(1)) * 100.0
    pk = parkinson_vol(d["high"], d["low"], 10)
    feats = pd.DataFrame({"ret": ret, "vol": pk})
    if breadth is not None:
        feats["breadth"] = breadth.reindex(d["date"]).to_numpy()
    feats["date"] = d["date"].to_numpy()
    feats = feats.dropna().reset_index(drop=True)
    if len(feats) < 200:
        return pd.DataFrame()

    cols = [c for c in ("ret", "vol", "breadth") if c in feats.columns]
    X = feats[cols].to_numpy(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=300, random_state=seed)
    model.fit(X)
    states = model.predict(X)

    # Переупорядочить состояния по средней доходности.
    order = pd.Series(feats["ret"].to_numpy()).groupby(states).mean().sort_values()
    remap = {old: new for new, old in enumerate(order.index)}
    feats["state"] = [remap[s] for s in states]
    feats["state_label"] = feats["state"].map(
        {0: "BEAR/CRASH", 1: "FLAT", 2: "BULL"} if n_states == 3
        else {i: f"S{i}" for i in range(n_states)})
    return feats[["date", "state", "state_label", "ret", "vol"]]


def ema_regimes(index_daily: pd.DataFrame) -> pd.DataFrame:
    """Текущая production-эвристика EMA50/EMA200 — для сравнения с HMM."""
    d = index_daily.sort_values("date").reset_index(drop=True)
    c = d["close"]
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    lab = np.where((c > e50) & (e50 > e200), "BULL",
                   np.where((c < e50) & (e50 < e200), "BEAR", "NEUTRAL"))
    return pd.DataFrame({"date": d["date"], "ema_regime": lab})


def regime_conditional_returns(panel: pd.DataFrame, regimes: pd.DataFrame,
                               regime_col: str,
                               ret_col: str = "fwd_ret_1") -> pd.DataFrame:
    """Средняя доходность бумаг в разрезе режима — проверка полезности фильтра."""
    d = panel.merge(regimes, on="date", how="inner")
    g = d.groupby(regime_col)[ret_col]
    out = g.agg(mean="mean", median="median", n="count", std="std").reset_index()
    out["t_stat"] = out["mean"] / out["std"] * np.sqrt(out["n"])
    return out


# ── Альтернатива 3: Ornstein-Uhlenbeck / возврат к среднему ──────────────────

def ou_halflife(series: pd.Series) -> float:
    """Период полураспада возврата к среднему из регрессии AR(1).

    dx_t = a + b * x_{t-1} + eps ;  halflife = -ln(2)/ln(1+b)
    Возвращает NaN, если ряд не возвращается к среднему (b >= 0).
    """
    x = series.dropna()
    if len(x) < 60:
        return np.nan
    lag = x.shift(1).dropna()
    dx = (x - x.shift(1)).dropna()
    n = min(len(lag), len(dx))
    if n < 60:
        return np.nan
    b = stats.linregress(lag.iloc[-n:], dx.iloc[-n:]).slope
    if b >= 0:
        return np.nan
    return float(-np.log(2) / np.log(1 + b))


def ou_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    """Проверяет, есть ли вообще возврат к среднему в дневных ценах бумаг."""
    rows = []
    for tk, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date")
        c = g["close"]
        if len(c) < 200:
            continue
        # Отклонение log-цены от EMA50 — стационарная величина, если есть возврат.
        dev = np.log(c / c.ewm(span=50, adjust=False).mean())
        hl = ou_halflife(dev)
        # Автокорреляция дневной доходности: < 0 = разворот, > 0 = моментум.
        r = c.pct_change().dropna()
        ac1 = r.autocorr(1)
        rows.append({"ticker": tk, "ou_halflife_days": hl, "ret_autocorr_1": ac1,
                     "dev_std": float(dev.std())})
    return pd.DataFrame(rows)
