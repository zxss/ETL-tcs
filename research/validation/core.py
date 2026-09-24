"""
Каркас валидации для плана тестирования гипотез (04_plan, 24.09.2026) — T0.2.

Одна точка правды для метрик и тестов значимости, чтобы блоки V/C/R/U мерили
сравнимо. Всё, что здесь есть, взято из §6 обзора методов прогнозирования:
rolling-origin с эмбарго, Диболд–Мариано, Clark-West, Холм, Deflated Sharpe,
покрытие интервалов, pinball/CRPS.
"""
from __future__ import annotations

import math

import numpy as np

EULER = 0.5772156649015329


# ── метрики точечного прогноза ───────────────────────────────────────────────

def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE для дисперсии: RV/σ̂² − ln(RV/σ̂²) − 1 (робастна к шуму в RV)."""
    r = np.asarray(realized, float)
    f = np.asarray(forecast, float)
    ok = (r > 0) & (f > 0) & np.isfinite(r) & np.isfinite(f)
    out = np.full(r.shape, np.nan)
    x = r[ok] / f[ok]
    out[ok] = x - np.log(x) - 1.0
    return out


def mse(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    r, f = np.asarray(realized, float), np.asarray(forecast, float)
    return (r - f) ** 2


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return np.where(d >= 0, tau * d, (tau - 1.0) * d)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean()) if ok.any() else float("nan")


# ── значимость ───────────────────────────────────────────────────────────────

def _nw_var(x: np.ndarray, lag: int | None = None) -> float:
    """Ньюи–Уэст: дисперсия среднего с поправкой на автокорреляцию."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lag is None:
        lag = max(1, int(round(n ** (1.0 / 3.0))))
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(s, 1e-18) / n


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None) -> dict:
    """Диболд–Мариано на разности потерь (A − B). t<0 ⇒ модель A лучше B.

    HAC-дисперсия (Ньюи–Уэст), нормальная аппроксимация. Для ВЛОЖЕННЫХ моделей
    DM смещён — там использовать clark_west().
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "mean_diff": float("nan"), "t": float("nan"), "p": float("nan")}
    v = _nw_var(d, lag)
    t = float(d.mean() / math.sqrt(v))
    return {"n": n, "mean_diff": float(d.mean()), "t": t, "p": 2.0 * _norm_sf(abs(t))}


def clark_west(y: np.ndarray, f_small: np.ndarray, f_big: np.ndarray,
               lag: int | None = None) -> dict:
    """Clark-West для вложенных моделей: малая ⊂ большая. t>0 ⇒ большая лучше."""
    y = np.asarray(y, float)
    fs, fb = np.asarray(f_small, float), np.asarray(f_big, float)
    ok = np.isfinite(y) & np.isfinite(fs) & np.isfinite(fb)
    y, fs, fb = y[ok], fs[ok], fb[ok]
    if len(y) < 10:
        return {"n": len(y), "t": float("nan"), "p": float("nan")}
    adj = (y - fs) ** 2 - ((y - fb) ** 2 - (fs - fb) ** 2)
    v = _nw_var(adj, lag)
    t = float(adj.mean() / math.sqrt(v))
    return {"n": len(y), "mean": float(adj.mean()), "t": t, "p": _norm_sf(t)}


def tstat_clustered(values: np.ndarray, clusters: np.ndarray) -> dict:
    """t среднего с кластеризацией по датам: сначала среднее внутри дня."""
    import pandas as pd
    s = pd.Series(np.asarray(values, float))
    g = s.groupby(np.asarray(clusters)).mean().dropna()
    n = len(g)
    if n < 3:
        return {"n": n, "mean": float("nan"), "t": float("nan"), "p": float("nan")}
    se = g.std(ddof=1) / math.sqrt(n)
    t = float(g.mean() / se) if se > 0 else float("nan")
    return {"n": n, "mean": float(g.mean()), "t": t,
            "p": 2.0 * _norm_sf(abs(t)) if np.isfinite(t) else float("nan")}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Поправка Холма. Возвращает {имя: {p, p_adj, reject}}."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": float(p), "p_adj": float(adj), "reject": bool(adj < alpha)}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": float(v) if np.isfinite(v) else None, "p_adj": None, "reject": False}
    return out


# ── Deflated Sharpe Ratio (Bailey & López de Prado) ──────────────────────────

def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Ожидаемый максимум Шарпа при переборе n_trials независимых нулевых стратегий."""
    if n_trials < 2 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    from scipy.stats import norm
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    sr_benchmark: float | None = None) -> dict:
    """DSR: вероятность, что наблюдённый Шарп не артефакт перебора n_trials.

    returns — ряд доходностей за период ребаланса (доли, не проценты).
    sr_benchmark — порог; по умолчанию = ожидаемому максимуму под нулём.
    """
    from scipy.stats import skew, kurtosis
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 8:
        return {"n": t, "sr": float("nan"), "dsr": float("nan")}
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else float("nan")
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))
    # дисперсия оценки Шарпа при ненормальности (Mertens)
    var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (t - 1)
    sr_star = expected_max_sharpe(n_trials, var_sr) if sr_benchmark is None else sr_benchmark
    denom = math.sqrt(max(var_sr, 1e-18))
    dsr = _norm_cdf((sr - sr_star) / denom)
    return {"n": t, "sr": sr, "skew": g3, "kurtosis": g4, "var_sr": float(var_sr),
            "sr_star": float(sr_star), "dsr": float(dsr),
            "sr_annual": sr * math.sqrt(12.0) if t else float("nan")}


# ── разбиения ────────────────────────────────────────────────────────────────

def rolling_origin(index, train_min: int, step: int, embargo: int):
    """Итератор (train_idx, test_idx) с расширяющимся окном и эмбарго."""
    n = len(index)
    start = train_min
    while start + embargo < n:
        tr = np.arange(0, start)
        te = np.arange(start + embargo, min(start + embargo + step, n))
        if len(te) == 0:
            break
        yield tr, te
        start += step
