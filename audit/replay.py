"""
Историческая реконструкция дашборда (walk-forward, без look-ahead).

Зачем нужен реплей: таблица forecasts пуста (0 строк), то есть истории того,
что система реально показывала, не существует. Rank IC и бэктест невозможно
посчитать по факту — их можно посчитать только заново, восстановив расчёт на
каждую историческую дату по данным, доступным на тот момент.

Гарантии отсутствия утечки:
  * признаки строятся production-кодом features._build_features на срезе
    raw[date <= t] — ни одна строка после t в расчёт не попадает;
  * масштабирование признаков (feat_mean/feat_std) фиксируется в момент
    обучения и переиспользуется на всех последующих днях, а не пересчитывается;
  * модель, обученная на данных < t0, применяется к дням t0..t0+H и затем
    переобучается — классический walk-forward с переобучением раз в период;
  * рыночный контекст (режим, RS, ATR-перцентиль, всплеск объёма, гэп-риск)
    считается теми же формулами, что в tft_forecast/market.py, но на срезе.

Архитектура модели и функция потерь берутся из production (tft_forecast.model),
поэтому аудит проверяет ту же сеть, что работает в бою. Цикл обучения повторён
здесь только потому, что production-функция _train_tft не возвращает объект
модели, а реплею нужно применять её к последующим дням без переобучения.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tft_forecast.dataset import LOOKBACK, _clean_window  # noqa: E402
from tft_forecast.features import (  # noqa: E402
    FEATURE_COLS, TARGET_COLS, MIN_ROWS, _build_features,
)

log = logging.getLogger("audit.replay")

# Индексы целей — должны совпадать с tft_forecast/forecast.py
T_LOW, T_HIGH, T_OVN, T_INTRA, T_TOTAL = 0, 1, 2, 3, 4
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
Q_LOW, Q_MED, Q_HIGH = 0, 2, 4

# Карта стратегия → (индекс цели, знак). Копия directional.STRATEGY_MAP.
STRATEGY_MAP = {
    "long_overnight": (T_OVN, +1.0),
    "intraday_long": (T_INTRA, +1.0),
    "intraday_short": (T_INTRA, -1.0),
    "short_hold": (T_TOTAL, -1.0),
}


# ── Хранилище сырых баров ────────────────────────────────────────────────────

class RawStore:
    """Дневные бары по тикерам, загруженные один раз."""

    def __init__(self, daily: pd.DataFrame):
        self.raw: dict[str, pd.DataFrame] = {}
        for tk, g in daily.groupby("ticker", sort=False):
            g = g[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
            self.raw[tk] = g.reset_index(drop=True)
        self._feat_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}

    @property
    def tickers(self) -> list[str]:
        return sorted(self.raw)

    def slice(self, tk: str, upto: pd.Timestamp) -> pd.DataFrame:
        d = self.raw[tk]
        return d[d["date"] <= upto]

    def features_asof(self, tk: str, upto: pd.Timestamp) -> pd.DataFrame | None:
        """Признаки по срезу date <= upto, production-кодом."""
        key = (tk, upto)
        if key in self._feat_cache:
            return self._feat_cache[key]
        raw = self.slice(tk, upto)
        if len(raw) < MIN_ROWS:
            return None
        feat = _build_features(raw.reset_index(drop=True))
        if len(self._feat_cache) > 4000:
            self._feat_cache.clear()
        self._feat_cache[key] = feat
        return feat


# ── Сборка обучающей выборки на дату ─────────────────────────────────────────

def build_training_set(store: RawStore, upto: pd.Timestamp,
                       tickers: list[str] | None = None,
                       lookback: int = LOOKBACK,
                       train_months: int | None = None):
    """Окна обучения из данных <= upto. Возвращает (X, T, Y, W, tk_index).

    train_months ограничивает обучение скользящим окном (ТЗ: 12 месяцев).
    None — расширяющееся окно на всю доступную историю.
    """
    tickers = tickers or store.tickers
    X, Tt, Y, W = [], [], [], []
    tk_index: dict[str, int] = {}
    since = (upto - pd.DateOffset(months=train_months)) if train_months else None

    for tk in tickers:
        feat = store.features_asof(tk, upto)
        if feat is None:
            continue
        d = feat
        feats = d[FEATURE_COLS].to_numpy(dtype=np.float64)
        tgts = d[TARGET_COLS].to_numpy(dtype=np.float64)
        dates = pd.to_datetime(d["date"])
        n = len(d)
        idx = tk_index.setdefault(tk, len(tk_index))
        for i in range(lookback - 1, n - 1):
            y = tgts[i]
            if not np.all(np.isfinite(y)):
                continue
            if since is not None and dates.iloc[i] < since:
                continue
            X.append(feats[i - lookback + 1: i + 1])
            Tt.append(idx)
            Y.append(y)
            W.append(dates.iloc[i].toordinal())

    if not X:
        return None
    X = _clean_window(np.asarray(X, dtype=np.float64))
    return (X, np.asarray(Tt, dtype=np.int64), np.asarray(Y, dtype=np.float64),
            np.asarray(W, dtype=np.int64), tk_index)


def inference_windows(store: RawStore, asof: pd.Timestamp,
                      tk_index: dict[str, int], lookback: int = LOOKBACK):
    """Окна инференса на дату asof: последние lookback дней <= asof.

    Прогноз относится к СЛЕДУЮЩЕМУ торговому дню после asof.
    """
    Xs, Ts, tks, closes = [], [], [], []
    for tk, idx in tk_index.items():
        feat = store.features_asof(tk, asof)
        if feat is None or len(feat) < lookback:
            continue
        if pd.Timestamp(feat["date"].iloc[-1]) != pd.Timestamp(asof):
            continue          # у тикера нет бара на эту дату — пропускаем
        feats = feat[FEATURE_COLS].to_numpy(dtype=np.float64)
        Xs.append(feats[-lookback:])
        Ts.append(idx)
        tks.append(tk)
        closes.append(float(feat["close"].iloc[-1]))
    if not Xs:
        return None
    X = _clean_window(np.asarray(Xs, dtype=np.float64))
    return X, np.asarray(Ts, dtype=np.int64), tks, np.asarray(closes)


# ── Обучение ─────────────────────────────────────────────────────────────────

class FittedModel:
    """Обученная сеть плюс зафиксированное масштабирование и индекс тикеров."""

    def __init__(self, model, feat_mean, feat_std, tk_index, fit_date):
        self.model = model
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.tk_index = tk_index
        self.fit_date = fit_date

    def predict(self, X_raw: np.ndarray, T: np.ndarray) -> np.ndarray:
        import torch
        Xs = (X_raw - self.feat_mean) / self.feat_std
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(Xs.astype(np.float32)),
                             torch.tensor(T))
        return out.cpu().numpy()


def fit(store: RawStore, upto: pd.Timestamp, *, epochs: int = 30,
        hidden: int = 32, tickers: list[str] | None = None,
        seed: int = 42, train_months: int | None = 12) -> FittedModel | None:
    """Обучает TFT на данных <= upto. Архитектура и лосс — production."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tft_forecast.model import TFT, quantile_loss

    built = build_training_set(store, upto, tickers, train_months=train_months)
    if built is None:
        return None
    X, T, Y, W, tk_index = built

    flat = X.reshape(-1, X.shape[-1])
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    Xs = ((X - feat_mean) / feat_std).astype(np.float32)

    # Хронологический сплит с эмбарго — как в production _train_tft.
    cutoff = np.quantile(W, 0.85)
    tr = np.where(W < (cutoff - LOOKBACK))[0]
    if len(tr) < 200:
        tr = np.arange(len(W))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TFT(X.shape[-1], len(tk_index), hidden=hidden, n_targets=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(Xs[tr]), torch.tensor(T[tr]),
                       torch.tensor(Y[tr].astype(np.float32)))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, tb, yb in loader:
            opt.zero_grad()
            loss = quantile_loss(model(xb, tb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return FittedModel(model, feat_mean, feat_std, tk_index, upto)


# ── Прогноз в формате дашборда ───────────────────────────────────────────────

def forecast_asof(fm: FittedModel, store: RawStore, asof: pd.Timestamp,
                  cost_rt: float | dict = 0.08) -> pd.DataFrame:
    """Прогноз на следующий торговый день после asof, по всем стратегиям.

    cost_rt — либо общее значение в %, либо {ticker: %}.
    Возвращает: asof_date, ticker, strategy, exp_pnl, prob_profit, q10/q50/q90
    (в процентах для примитива стратегии) и границы ценового коридора.
    """
    iw = inference_windows(store, asof, fm.tk_index)
    if iw is None:
        return pd.DataFrame()
    X, T, tks, closes = iw
    pred = fm.predict(X, T)          # (M, n_targets, n_quantiles)

    rows = []
    for j, tk in enumerate(tks):
        c = float(closes[j])
        cost = cost_rt.get(tk, 0.08) if isinstance(cost_rt, dict) else float(cost_rt)
        lo = c * (1.0 + pred[j, T_LOW, Q_LOW] / 100.0)
        hi = c * (1.0 + pred[j, T_HIGH, Q_HIGH] / 100.0)
        for strat, (t_idx, sign) in STRATEGY_MAP.items():
            q = np.sort(pred[j, t_idx, :] * sign)      # знак разворачивает квантили
            med = float(q[Q_MED])
            exp_pnl = med - cost
            # P(доходность > издержек) интерполяцией квантильной функции
            prob = float(np.interp(cost, q, QUANTILES))
            rows.append({
                "asof_date": asof, "ticker": tk, "strategy": strat,
                "anchor_price": c,
                "exp_pnl": exp_pnl,
                "prob_profit": 1.0 - prob,
                "q10": float(q[Q_LOW]), "q50": med, "q90": float(q[Q_HIGH]),
                "f_low": lo, "f_high": hi,
                "range_pct": (hi - lo) / c * 100.0,
                "cost_rt": cost,
            })
    return pd.DataFrame(rows)


# ── Рыночный контекст на дату (формулы market.py на срезе) ───────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def market_context_asof(store: RawStore, asof: pd.Timestamp,
                        tickers: list[str],
                        index_ticker: str = "IMOEX") -> dict:
    """Режим рынка, RS, всплеск объёма, ATR-перцентиль, гэп-риск на дату asof."""
    idx = store.raw.get(index_ticker)
    regime = "NEUTRAL"
    r10 = None
    if idx is not None:
        lvl = idx[idx["date"] <= asof]["close"].reset_index(drop=True)
        if len(lvl) >= 200:
            e50, e200 = _ema(lvl, 50).iloc[-1], _ema(lvl, 200).iloc[-1]
            cur = lvl.iloc[-1]
            if cur > e50 > e200:
                regime = "BULL"
            elif cur < e50 < e200:
                regime = "BEAR"
        if len(lvl) >= 11:
            r10 = float(lvl.iloc[-1] / lvl.iloc[-11] - 1.0)

    per_tk, above, counted = {}, 0, 0
    for tk in tickers:
        d = store.slice(tk, asof)
        if len(d) < 60:
            continue
        c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

        rs = None
        if r10 is not None and len(c) >= 11:
            rs = (float(c.iloc[-1] / c.iloc[-11] - 1.0) - r10) * 100.0

        vol_spike = None
        if len(v) >= 21:
            sma20 = float(v.iloc[-21:-1].mean())
            if sma20 > 0:
                vol_spike = float(v.iloc[-1] / sma20)

        atr_pctl = None
        if len(d) >= 19:
            prev = c.shift(1)
            tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()],
                           axis=1).max(axis=1)
            atr = tr.rolling(14).mean().dropna()
            if not atr.empty:
                hist = atr.tail(252)
                atr_pctl = float((hist <= float(atr.iloc[-1])).mean() * 100.0)

        gap = None
        if len(d) >= 30:
            sub = d.tail(127)
            g = (sub["open"] / sub["close"].shift(1) - 1.0).dropna()
            if not g.empty:
                gap = float((g < -0.005).mean())

        per_tk[tk] = {"rs": rs, "vol_spike": vol_spike,
                      "atr_pctl": atr_pctl, "gap_down_prob": gap}
        if len(c) >= 50:
            counted += 1
            if float(c.iloc[-1]) > float(_ema(c, 50).iloc[-1]):
                above += 1

    return {"regime": regime, "per_ticker": per_tk,
            "breadth": (above / counted) if counted else None}


def liquidity_asof(store: RawStore, asof: pd.Timestamp, tickers: list[str],
                   lots: dict[str, int], window: int = 60) -> dict:
    """Балл ликвидности 0..100 — перцентиль медианного оборота по рынку.

    В отличие от production, оборот считается с УЧЁТОМ размера лота
    (market_data.volume выражен в лотах).
    """
    adv = {}
    for tk in tickers:
        d = store.slice(tk, asof).tail(window)
        if len(d) < 20:
            continue
        rub = (d["close"] * d["volume"] * lots.get(tk, 1)).replace(0, np.nan)
        m = rub.median()
        if m and np.isfinite(m):
            adv[tk] = float(m)
    if not adv:
        return {}
    s = pd.Series(adv)
    pct = s.rank(pct=True) * 100.0
    return {tk: {"adv_rub": adv[tk], "liq_score": float(pct[tk])} for tk in adv}
