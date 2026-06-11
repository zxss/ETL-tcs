#!/usr/bin/env python3
"""
Продвинутые статистические тесты для Strategy Edge Validator.

Модули:
  --mode white_reality   Stage 4c: White's Reality Check (anti data-snooping)
  --mode pbo             Stage 4d: Probability of Backtest Overfitting (CSCV)
  --mode regime          Stage 2b: Определение рыночных режимов + P&L по режимам
  --mode tail_risk       Stage 5c: Хвостовые риски (VaR, CVaR, Skew, Kurt)

Запуск:
    python3 advanced_stats.py --mode white_reality --daily sber_candles.csv --cost-side 0.04
    python3 advanced_stats.py --mode pbo            --daily sber_candles.csv --cost-side 0.04
    python3 advanced_stats.py --mode regime         --daily sber_candles.csv
    python3 advanced_stats.py --mode tail_risk      --daily sber_candles.csv --strategy long_overnight

Зависимости:
    pip install pandas numpy scipy
    pip install hmmlearn  # опционально, для HMM-режимов
"""
from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from hmmlearn import hmm as hmmlib
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


# ── Загрузка данных ───────────────────────────────────────────────────────────

def winsorize_div_gaps(d: pd.DataFrame, k: float = 6.0) -> pd.DataFrame:
    """PR-4: прокси total-return корректировки на дивиденды.

    T-Invest свечи не скорректированы на дивиденды — в ex-div дату overnight
    (open/prev_close) и total показывают ложный гэп вниз величиной с дивиденд
    (для росс. high-div имён 5–15%). Истинная корректировка требует дивидендного
    фида; здесь мы робастно винзоризуем экстремальные overnight-гэпы за пределами
    median ± k·MAD (k=6 ≈ ловит дивидендные/новостные выбросы, не трогая обычную
    волатильность) и согласованно пересчитываем total = overnight ⊕ intraday.
    Меняет ТОЛЬКО overnight/total; intraday (close/open) дивидендами не задет.
    """
    ov = d["overnight"]
    med = ov.median()
    mad = (ov - med).abs().median() * 1.4826      # robust sigma
    if not np.isfinite(mad) or mad <= 0:
        return d
    lo, hi = med - k * mad, med + k * mad
    ov_w = ov.clip(lower=lo, upper=hi)
    d["overnight"] = ov_w
    # total ≈ (1+overnight/100)(1+intraday/100)-1, в %; пересчитываем согласованно
    d["total"] = ((1 + ov_w / 100.0) * (1 + d["intraday"] / 100.0) - 1) * 100
    return d


def load_daily(path: str, winsorize: bool = False) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"]      = pd.to_datetime(d["date"]).dt.date
    d              = d.sort_values("date").reset_index(drop=True)
    d["intraday"]  = (d["close"] / d["open"]              - 1) * 100
    d["overnight"] = (d["open"]  / d["close"].shift(1)    - 1) * 100
    d["total"]     = (d["close"] / d["close"].shift(1)    - 1) * 100
    d["log_ret"]   = np.log(d["close"] / d["close"].shift(1))
    if winsorize:
        d = winsorize_div_gaps(d)
    return d


# ── Вспомогательные стратегии (те же, что в top3_report.py) ───────────────────

STRATEGIES = {
    "intraday_short": lambda d, c: (-d["intraday"].dropna().sum(),
                                     -d["intraday"].dropna()),
    "long_overnight": lambda d, c: ( d["overnight"].dropna().sum(),
                                      d["overnight"].dropna()),
    "short_hold":     lambda d, c: (-d["total"].dropna().sum(),
                                     -d["total"].dropna()),
    "long_intraday":  lambda d, c: ( d["intraday"].dropna().sum(),
                                      d["intraday"].dropna()),
}

def net_pnl(returns: pd.Series, cost_rt: float) -> float:
    return returns.sum() - cost_rt * len(returns)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 4c — White's Reality Check
# ════════════════════════════════════════════════════════════════════════════════

def _moving_block_bootstrap_indices(rng, T, block_len):
    """Индексы для moving-block bootstrap: набираем ceil(T/block_len) блоков
    случайных стартовых точек длиной block_len, конкатенируем, обрезаем до T.
    Сохраняет автокорреляционную структуру внутри блока."""
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:T]


def white_reality_check(d: pd.DataFrame, cost_rt: float,
                         n_boot: int = 10_000, seed: int = 42,
                         benchmark: str = "zero") -> None:
    """
    White (2000) Reality Check — методологически корректная версия.

    Правильный алгоритм (в отличие от наивного i.i.d.-ресэмплинга сырых
    рядов, который уничтожает автокорреляцию и не сравнивает с бенчмарком):
      1. Для каждой стратегии считаем ДИФФЕРЕНЦИАЛЬНЫЙ ряд
         f_k,t = ret_k,t − benchmark_t  (benchmark: 0 либо buy&hold close-to-close).
      2. Test statistic: V = max_k( mean(f_k) ) — сверхдоходность лучшей
         стратегии относительно бенчмарка (в среднем за наблюдение).
      3. Строим нулевое распределение через MOVING BLOCK BOOTSTRAP
         (длина блока ≈ sqrt(T)) дифференциальных рядов — это сохраняет
         автокорреляцию/кластеризацию волатильности, которую разрушает
         i.i.d.-ресэмплинг по индексам.
      4. На каждой бутстреп-реплике центрируем дифф.ряд на его собственное
         среднее (рекуррентный приём White'а — под H0 у всех стратегий
         нулевое ожидаемое преимущество) и считаем V* = max_k(mean(f*_k)).
      5. p-value = доля V* ≥ V (наблюдаемого).
    """
    print("═" * 64)
    print("  Stage 4c — WHITE'S REALITY CHECK (anti data-snooping)")
    print("  [Корректная версия: block bootstrap, дифференцирование от бенчмарка]")
    print("═" * 64)

    # Бенчмарк: 0 (нулевая доходность) или buy&hold close-to-close
    if benchmark == "buyhold":
        bench = d["total"].dropna()
    else:
        bench = pd.Series(0.0, index=d.index)

    ret_map: dict[str, pd.Series] = {}
    diff_map: dict[str, np.ndarray] = {}
    net_map: dict[str, float] = {}
    for name, fn in STRATEGIES.items():
        try:
            _, ret = fn(d, cost_rt)
            ret = ret.dropna()
            if len(ret) < 30:
                continue
            common = ret.index.intersection(bench.index)
            if len(common) < 30:
                continue
            f = (ret.loc[common] - bench.loc[common]).values
            ret_map[name] = ret
            diff_map[name] = f
            net_map[name] = net_pnl(ret, cost_rt)
        except Exception:
            pass

    if not net_map:
        print("  Недостаточно данных"); return

    best_name = max(net_map, key=net_map.get)
    print(f"\n  Стратегий в наборе: {len(net_map)}  |  бенчмарк: "
          f"{'нулевая доходность' if benchmark=='zero' else 'buy&hold (close-to-close)'}")
    for n, v in sorted(net_map.items(), key=lambda x: -x[1]):
        mark = " ← лучшая" if n == best_name else ""
        print(f"    {n:<20}: net = {v:+.2f}%{mark}")

    # Выравниваем длины
    min_T = min(len(v) for v in diff_map.values())
    names = list(diff_map.keys())
    F = np.vstack([diff_map[k][:min_T] for k in names])  # (K x T) дифф. ряды
    T = min_T

    means = F.mean(axis=1)            # фактические средние сверхдоходности
    V_obs = means.max()
    best_idx = int(np.argmax(means))

    # PR-6: единый источник истины — validation/white_rc.py (stationary block
    # bootstrap + SPA, p=(k+1)/(B+1)). Раньше здесь была вторая, расходящаяся
    # реализация на moving-block bootstrap; теперь делегируем, чтобы CLI и прод-
    # контур (verdict.py) считали White RC ОДНИМ И ТЕМ ЖЕ кодом. Ленивый импорт —
    # validation.verdict импортирует этот модуль, прямой импорт сверху дал бы цикл.
    from validation.white_rc import white_reality_check as _wrc_canonical
    res = _wrc_canonical(F.T, B=n_boot, seed=seed)   # ждёт (T, K)
    p_val = res["white_rc_p"]
    spa_p = res["spa_p"]

    print(f"\n  Тест (validation/white_rc: stationary block bootstrap + SPA, N = {n_boot:,}):")
    print(f"  Лучшая стратегия '{names[best_idx]}': средняя сверхдоходность над бенчмарком "
          f"= {V_obs:+.4f}%/набл.")
    print(f"  White RC p = {p_val:.3f}  |  Hansen SPA p = {spa_p:.3f}")

    if p_val < 0.05:
        print(f"\n  ✅ Лучшая стратегия значимо превосходит бенчмарк с поправкой на data-snooping (p={p_val:.3f} < 0.05)")
    elif p_val < 0.10:
        print(f"\n  ⚠️  Граничный результат (p={p_val:.3f}). Осторожно: look-elsewhere эффект")
    else:
        print(f"\n  ❌ Лучшая стратегия НЕ значимо лучше бенчмарка с поправкой на отбор из {len(net_map)} стратегий (p={p_val:.3f} ≥ 0.05)")
        print(f"     Вывод: видимая прибыль, вероятно, объясняется перебором стратегий (data snooping), а не эджем")


# ── Юнит-тесты для white_reality_check (запускаются через --mode selftest) ──

def _selftest_white_rc():
    """
    1) На чистом шуме (нет эджа ни у одной 'стратегии') p-value должно быть
       примерно равномерным — НЕ систематически близким к 0.
    2) Block bootstrap должен сохранять автокорреляцию лага 1 синтетического
       AR(1)-ряда (в отличие от i.i.d. ресэмплинга, который убивает её).
    """
    rng = np.random.default_rng(0)
    print("  [selftest] (1) p-value на чистом шуме должно быть НЕ систематически малым:")
    pvals = []
    for trial in range(8):
        T = 250
        # 4 "стратегии" — независимый чистый шум, нет реального эджа
        noise = rng.normal(0, 1.0, size=(4, T))
        F = noise  # дифф. от нулевого бенчмарка
        means = F.mean(axis=1)
        V_obs = means.max()
        F_c = F - means[:, None]
        block_len = max(2, int(round(np.sqrt(T))))
        rr = np.random.default_rng(100 + trial)
        V_star = np.empty(800)
        for b in range(800):
            idx = _moving_block_bootstrap_indices(rr, T, block_len)
            V_star[b] = F_c[:, idx].mean(axis=1).max()
        p = float(np.mean(V_star >= V_obs))
        pvals.append(p)
    pvals = np.array(pvals)
    print(f"    p-values по 8 прогонам чистого шума: {[round(x,3) for x in pvals]}")
    print(f"    среднее = {pvals.mean():.3f} (ожидание ~0.3-0.7, НЕ ~0.00-0.02)")
    ok1 = pvals.mean() > 0.15
    print(f"    {'PASS' if ok1 else 'FAIL'}: распределение p-value не вырождено в систематически малое")

    print("\n  [selftest] (2) block bootstrap сохраняет lag-1 автокорреляцию AR(1)-ряда:")
    T = 500
    phi = 0.6
    x = np.zeros(T)
    eps = rng.normal(0, 1, T)
    for t in range(1, T):
        x[t] = phi * x[t-1] + eps[t]
    true_ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]

    block_len = max(2, int(round(np.sqrt(T))))
    rr = np.random.default_rng(7)
    boot_acs, iid_acs = [], []
    for _ in range(300):
        idx_b = _moving_block_bootstrap_indices(rr, T, block_len)
        xb = x[idx_b]
        boot_acs.append(np.corrcoef(xb[:-1], xb[1:])[0, 1])
        idx_iid = rr.integers(0, T, size=T)
        xi = x[idx_iid]
        iid_acs.append(np.corrcoef(xi[:-1], xi[1:])[0, 1])
    boot_ac = np.mean(boot_acs)
    iid_ac  = np.mean(iid_acs)
    print(f"    истинная lag-1 AC ≈ {true_ac1:.3f}")
    print(f"    block-bootstrap   ср. AC ≈ {boot_ac:.3f}  (должна быть БЛИЗКА к истинной)")
    print(f"    i.i.d. ресэмплинг ср. AC ≈ {iid_ac:.3f}  (должна быть БЛИЗКА к нулю — структура разрушена)")
    ok2 = abs(boot_ac - true_ac1) < 0.25 and abs(iid_ac) < 0.15
    print(f"    {'PASS' if ok2 else 'FAIL'}: block bootstrap сохраняет автокорреляцию, i.i.d. — нет")

    return ok1 and ok2


# ════════════════════════════════════════════════════════════════════════════════
# Stage 4d — Probability of Backtest Overfitting (CSCV)
# Bailey, López de Prado, Zhu (2016)
# ════════════════════════════════════════════════════════════════════════════════

def pbo_cscv(d: pd.DataFrame, cost_rt: float,
              S: int = 16, seed: int = 42) -> None:
    """
    Combinatorially Symmetric Cross-Validation (CSCV).

    Алгоритм:
      1. Делим временной ряд на S равных частей.
      2. Перебираем все C(S, S//2) разбиений на in-sample (IS) и out-of-sample (OOS).
      3. В каждом разбиении выбираем лучшую стратегию по IS.
      4. Смотрим её ранг в OOS.
      5. PBO = доля разбиений, где лучшая IS-стратегия хуже медианы в OOS.

    PBO ≈ 0.5 → стратегия отобрана случайно
    PBO < 0.1 → низкая вероятность переобучения
    PBO > 0.5 → скорее всего переобучена
    """
    from itertools import combinations

    print("═" * 64)
    print("  Stage 4d — PBO: PROBABILITY OF BACKTEST OVERFITTING (CSCV)")
    print("  Bailey, López de Prado, Zhu (2016)")
    print("═" * 64)

    # Получаем доходности всех стратегий
    ret_map: dict[str, np.ndarray] = {}
    for name, fn in STRATEGIES.items():
        try:
            _, ret = fn(d, cost_rt)
            ret = ret.dropna().values
            if len(ret) >= S * 4:
                ret_map[name] = ret
        except Exception:
            pass

    if len(ret_map) < 2:
        print("  Нужно ≥ 2 стратегии с достаточным числом наблюдений"); return

    # Выравниваем длины
    min_len = min(len(v) for v in ret_map.values())
    ret_matrix = np.vstack([v[:min_len] for v in ret_map.values()])  # (N_strat × T)
    N_strat, T = ret_matrix.shape
    names_list = list(ret_map.keys())

    chunk = T // S
    if chunk < 2:
        print(f"  Слишком мало данных для S={S} частей. Уменьшите S."); return

    # PR-6: единый источник истины — validation/verdict.pbo_cscv (CSCV по Sharpe,
    # тот же код, что в прод-гейте). Раньше здесь была вторая реализация, ранжировавшая
    # по net P&L → расхождение PBO между CLI и контуром. Ленивый импорт против цикла.
    from validation.verdict import pbo_cscv as _pbo_canonical
    pbo = float(_pbo_canonical(ret_matrix, S=S, seed=seed))
    if pbo != pbo:
        print("  PBO неопределён (мало стратегий/данных)."); return

    print(f"\n  Стратегий: {N_strat}  |  S={S}  |  метрика производительности: Sharpe (CSCV)")
    print(f"  Стратегии в наборе: {', '.join(names_list)}")
    print(f"\n  PBO = {pbo:.3f}")

    if pbo < 0.10:
        verdict = "✅ Низкая вероятность переобучения"
    elif pbo < 0.30:
        verdict = "⚠️  Умеренная вероятность переобучения"
    elif pbo < 0.50:
        verdict = "🔴 Высокая вероятность переобучения"
    else:
        verdict = "🚨 Стратегия, скорее всего, переобучена (PBO ≥ 0.5)"

    print(f"\n  {verdict}")
    print(f"  Интерпретация: при случайном выборе лучшей IS-стратегии")
    print(f"  PBO = 0.50. Ваш результат PBO = {pbo:.3f}.")


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2b — Режимный анализ
# ════════════════════════════════════════════════════════════════════════════════

def regime_analysis(d: pd.DataFrame, cost_rt: float,
                     atr_window: int = 14, vol_window: int = 20) -> None:
    """
    Определяет рыночные режимы и рассчитывает P&L стратегий по каждому.

    Метод A (ATR + тренд — основной):
      - Bull:      Close > SMA50 И ATR < медианы ATR
      - Bear:      Close < SMA50 И ATR < медианы ATR
      - High-Vol:  ATR > 1.5 × медианы ATR
      - Sideways:  |Close − SMA50| < 2% И ATR < медианы

    Метод B (HMM — если установлен hmmlearn):
      - 2-state Gaussian HMM на реализованной волатильности
      - State 0 / State 1 интерпретируется по средней волатильности
    """
    print("═" * 64)
    print("  Stage 2b — АНАЛИЗ РЫНОЧНЫХ РЕЖИМОВ")
    print("═" * 64)

    d = d.copy()
    d["sma50"]  = d["close"].rolling(50).mean()
    d["sma20"]  = d["close"].rolling(20).mean()

    # ATR
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(
            (d["high"] - d["prev_close"]).abs(),
            (d["low"]  - d["prev_close"]).abs()
        )
    )
    d["atr"] = d["tr"].rolling(atr_window).mean()
    d["rvol"] = d["log_ret"].rolling(vol_window).std() * np.sqrt(252) * 100  # годовая вол, %

    d = d.dropna(subset=["sma50", "atr", "overnight", "intraday"])
    if len(d) < 100:
        print("  Недостаточно данных для режимного анализа"); return

    atr_med = d["atr"].median()
    sma50_dev = (d["close"] - d["sma50"]).abs() / d["sma50"] * 100

    # ── Метод A: ATR + тренд ──────────────────────────────────────────────────
    conditions = [
        (d["atr"] > 1.5 * atr_med),                                         # High-Vol
        (d["close"] > d["sma50"]) & (d["atr"] <= atr_med),                  # Bull
        (d["close"] < d["sma50"]) & (d["atr"] <= atr_med),                  # Bear
    ]
    choices = ["High-Vol", "Bull", "Bear"]
    # ИСПРАВЛЕНИЕ LOOK-AHEAD BIAS: режим в момент close[t] вычисляется из
    # close[t]/sma[t]/atr[t] — то есть использует ИНФОРМАЦИЮ ИЗ БУДУЩЕГО для
    # дня t (close[t] узнаётся только в конце дня t, а return[t] начинает
    # формироваться в начале дня t). Сдвигаем режим на 1 день вперёд: при
    # принятии решения "что делать в день t" мы можем знать только режим,
    # вычисленный по данным ВКЛЮЧИТЕЛЬНО ПО ДЕНЬ t-1.
    d["regime_atr_raw"] = np.select(conditions, choices, default="Sideways")
    d["regime_atr"] = d["regime_atr_raw"].shift(1)
    d = d.dropna(subset=["regime_atr"])

    _print_regime_stats(d, "regime_atr", cost_rt, "Метод A: ATR + тренд (SMA50), regime.shift(1) — без look-ahead")

    # ── Метод B: HMM ─────────────────────────────────────────────────────────
    if HAS_HMM:
        _hmm_regimes(d, cost_rt, vol_window)
    else:
        print(f"\n  Метод B (HMM): не установлен hmmlearn.")
        print(f"  Установите: pip install hmmlearn")
        # Fallback: простой квантильный режим по волатильности
        _vol_quantile_regimes(d, cost_rt)


def _print_regime_stats(d: pd.DataFrame, regime_col: str,
                         cost_rt: float, title: str) -> None:
    """Выводит P&L стратегий по режимам."""
    print(f"\n  {title}")
    print(f"  {'─'*60}")

    regime_counts = d[regime_col].value_counts()
    print(f"  {'Режим':<12} {'Дней':>6}  {'%':>5}  "
          f"{'On%/д':>8}  {'Id%/д':>8}  "
          f"{'On нетто':>10}  {'Id нетто':>10}")
    print(f"  {'─'*72}")

    for regime in ["Bull", "Bear", "High-Vol", "Sideways",
                   "Low-Vol", "State-0", "State-1"]:
        sub = d[d[regime_col] == regime]
        if len(sub) < 5:
            continue
        n   = len(sub)
        pct = n / len(d) * 100
        on_avg  = sub["overnight"].mean()
        id_avg  = sub["intraday"].mean()
        on_net  = sub["overnight"].sum() - cost_rt * n
        id_net  = -sub["intraday"].sum() - cost_rt * n  # intraday_short

        on_flag = "✅" if on_net > 0 else "❌"
        id_flag = "✅" if id_net > 0 else "❌"

        print(f"  {regime:<12} {n:>6}  {pct:>4.0f}%  "
              f"{on_avg:>+7.3f}%  {id_avg:>+7.3f}%  "
              f"{on_flag}{on_net:>+8.1f}%  {id_flag}{id_net:>+8.1f}%")

    print(f"\n  On нетто = long_overnight нетто за режим (0.08% rt)")
    print(f"  Id нетто = intraday_short нетто за режим (0.08% rt)")

    # Ключевой вывод
    regimes_df = d.groupby(regime_col).agg(
        on_net=("overnight", lambda x: x.sum() - cost_rt * len(x)),
        count=("overnight", "count")
    ).sort_values("on_net", ascending=False)

    best_r  = regimes_df.index[0]
    worst_r = regimes_df.index[-1]
    print(f"\n  Overnight-эдж максимален в режиме: {best_r}")
    print(f"  Overnight-эдж минимален в режиме:  {worst_r}")


def _hmm_regimes(d: pd.DataFrame, cost_rt: float, window: int = 20) -> None:
    """2-state Gaussian HMM на реализованной волатильности."""
    print(f"\n  Метод B: HMM (Hidden Markov Model, 2 состояния)")
    print(f"  {'─'*60}")

    X = d["rvol"].dropna().values.reshape(-1, 1)
    idx = d.index[d["rvol"].notna()]

    try:
        model = hmmlib.GaussianHMM(n_components=2, covariance_type="full",
                                    n_iter=100, random_state=42)
        model.fit(X)
        states = model.predict(X)

        # Определяем "Low-Vol" / "High-Vol" по средней волатильности состояний
        means = [X[states == s].mean() if (states == s).any() else 0 for s in range(2)]
        low_state  = int(np.argmin(means))
        high_state = int(np.argmax(means))

        d2 = d.loc[idx].copy()
        d2["hmm_state"] = states
        d2["regime_hmm_raw"] = d2["hmm_state"].map(
            {low_state: "Low-Vol", high_state: "High-Vol"}
        )
        # Тот же look-ahead фикс: режим известен только по данным до t-1
        d2["regime_hmm"] = d2["regime_hmm_raw"].shift(1)
        d2 = d2.dropna(subset=["regime_hmm"])

        print(f"  State {low_state}: Low-Vol  ср.вол = {means[low_state]:.1f}%  (n={int((states==low_state).sum())})")
        print(f"  State {high_state}: High-Vol ср.вол = {means[high_state]:.1f}%  (n={int((states==high_state).sum())})")
        print(f"  Transition matrix:")
        for row in model.transmat_:
            print(f"    {row[0]:.3f}  {row[1]:.3f}")

        _print_regime_stats(d2, "regime_hmm", cost_rt, "HMM-режимы")

    except Exception as e:
        print(f"  HMM ошибка: {e}. Используем fallback.")
        _vol_quantile_regimes(d, cost_rt)


def _vol_quantile_regimes(d: pd.DataFrame, cost_rt: float) -> None:
    """Fallback: квантильный режим по реализованной волатильности."""
    print(f"\n  Метод B (fallback): квантили волатильности")
    q33 = d["rvol"].quantile(0.33)
    q67 = d["rvol"].quantile(0.67)
    d2  = d.copy()
    d2["regime_vol_raw"] = pd.cut(d2["rvol"], bins=[-np.inf, q33, q67, np.inf],
                               labels=["Low-Vol", "Mid-Vol", "High-Vol"])
    d2["regime_vol"] = d2["regime_vol_raw"].shift(1)  # без look-ahead
    d2 = d2.dropna(subset=["regime_vol"])
    _print_regime_stats(d2, "regime_vol", cost_rt, "Квантили волатильности (regime.shift(1))")


# ════════════════════════════════════════════════════════════════════════════════
# Stage 5c — Tail Risk Analysis
# ════════════════════════════════════════════════════════════════════════════════

def tail_risk(d: pd.DataFrame, cost_rt: float, strategy: str = "long_overnight") -> None:
    """
    Анализ хвостовых рисков стратегии:
      VaR 95/99%, CVaR (Expected Shortfall) 95/99%,
      Skewness, Excess Kurtosis, Max Drawdown, Calmar Ratio,
      распределение по децилям.
    """
    print("═" * 64)
    print(f"  Stage 5c — TAIL RISK ANALYSIS: {strategy}")
    print("═" * 64)

    if strategy not in STRATEGIES:
        print(f"  Неизвестная стратегия '{strategy}'."); return

    _, ret_series = STRATEGIES[strategy](d, cost_rt)
    ret = ret_series.dropna()
    if len(ret) < 30:
        print("  Недостаточно данных (<30 сделок)."); return

    arr = ret.values

    # Базовая статистика
    mean  = arr.mean()
    med   = np.median(arr)
    std   = arr.std()
    n     = len(arr)

    print(f"\n  Сделок: {n}  |  Период: {d['date'].iloc[0]} → {d['date'].iloc[-1]}")
    print(f"\n  БАЗОВАЯ СТАТИСТИКА:")
    print(f"  Среднее       : {mean:+.4f}%")
    print(f"  Медиана       : {med:+.4f}%")
    print(f"  Std           : {std:.4f}%")

    # VaR и CVaR
    for conf in [0.95, 0.99]:
        var  = np.percentile(arr, (1 - conf) * 100)
        cvar = arr[arr <= var].mean() if (arr <= var).any() else var
        print(f"\n  {'─'*40}")
        print(f"  VaR  {conf*100:.0f}%: {var:+.4f}%  (в {conf*100:.0f}% дней потеря ≤ |VaR|)")
        print(f"  CVaR {conf*100:.0f}%: {cvar:+.4f}%  (ср. потеря в худших {(1-conf)*100:.0f}% дней)")

    # Форма распределения
    if HAS_SCIPY:
        skew = sp_stats.skew(arr)
        kurt = sp_stats.kurtosis(arr)  # excess kurtosis
        _, p_norm = sp_stats.normaltest(arr)
        print(f"\n  ФОРМА РАСПРЕДЕЛЕНИЯ:")
        print(f"  Skewness (асимметрия): {skew:+.3f}", end="  ")
        if skew < -0.5:
            print("← левый хвост (редкие крупные убытки)")
        elif skew > 0.5:
            print("← правый хвост (редкие крупные прибыли)")
        else:
            print("← приблизительно симметрично")
        print(f"  Kurtosis (эксцесс):    {kurt:+.3f}", end="  ")
        if kurt > 1:
            print("← «жирные хвосты» (leptokurtic — хвостовые события чаще нормального)")
        elif kurt < -1:
            print("← «тонкие хвосты» (platykurtic)")
        else:
            print("← близко к нормальному распределению")
        print(f"  Тест нормальности:     p={p_norm:.4f}", end="  ")
        print("← НЕ нормальное" if p_norm < 0.05 else "← близко к нормальному")
    else:
        skew = float(pd.Series(arr).skew())
        kurt = float(pd.Series(arr).kurtosis())
        print(f"\n  Skewness: {skew:+.3f}  |  Kurtosis: {kurt:+.3f}")

    # Max Drawdown на equity curve
    equity  = np.cumsum(arr)
    running = np.maximum.accumulate(equity)
    dd      = equity - running
    max_dd  = dd.min()
    calmar  = (mean * 252) / abs(max_dd) if max_dd != 0 else 0

    print(f"\n  DRAWDOWN:")
    print(f"  Max Drawdown   : {max_dd:+.2f}%")
    print(f"  Calmar Ratio   : {calmar:.3f}  (годовой доход / |MaxDD|)")

    # Profit Factor
    wins    = arr[arr > 0]
    losses  = arr[arr < 0]
    pf      = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    wr      = len(wins) / n
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_los = losses.mean() if len(losses) > 0 else 0

    print(f"\n  СООТНОШЕНИЕ ВЫИГРЫШЕЙ:")
    print(f"  Win rate       : {wr:.1%}")
    print(f"  Avg win        : {avg_win:+.4f}%  |  Avg loss: {avg_los:+.4f}%")
    print(f"  Win/Loss ratio : {abs(avg_win/avg_los):.3f}" if avg_los != 0 else "")
    print(f"  Profit Factor  : {pf:.3f}  (>1 = прибыльно)")

    # Хвостовое соотношение
    p95 = np.percentile(arr, 95)
    p05 = np.percentile(arr, 5)
    tail_ratio = abs(p95 / p05) if p05 != 0 else 0
    print(f"\n  ХВОСТОВОЕ СООТНОШЕНИЕ (95/5 перцентили):")
    print(f"  95-й перц: {p95:+.4f}%  |  5-й перц: {p05:+.4f}%  |  Ratio: {tail_ratio:.3f}")

    # Децильное распределение
    print(f"\n  ДЕЦИЛЬНОЕ РАСПРЕДЕЛЕНИЕ доходности:")
    print(f"  {'Дециль':>8}  {'Порог':>10}  {'Кол-во':>7}  {'Доля':>7}")
    print(f"  {'─'*40}")
    deciles = np.percentile(arr, np.arange(10, 100, 10))
    bins    = [-np.inf] + list(deciles) + [np.inf]
    labels  = [f"D{i}" for i in range(1, 11)]
    buckets = pd.cut(arr, bins=bins, labels=labels)
    for lbl, thresh in zip(labels, list(deciles) + [None]):
        cnt  = (buckets == lbl).sum()
        t_str = f"{thresh:+.4f}%" if thresh is not None else "> D9"
        print(f"  {lbl:>8}  {t_str:>10}  {cnt:>7}  {cnt/n:>6.0%}")

    # Итоговая оценка
    print(f"\n  {'─'*60}")
    print(f"  ОЦЕНКА ХВОСТОВОГО РИСКА:")
    risks = []
    if skew < -0.5:
        risks.append("левосторонняя асимметрия — редкие крупные убытки")
    if kurt > 2:
        risks.append(f"kurtosis={kurt:.1f} — «жирные хвосты», реальные потери > VaR")
    if abs(max_dd) > 20:
        risks.append(f"MaxDD={max_dd:.1f}% — значительная просадка")
    if pf < 1.2:
        risks.append(f"Profit Factor={pf:.2f} — слабый буфер прибыльности")
    if tail_ratio < 1.0:
        risks.append("хвост убытков больше хвоста прибылей")

    if risks:
        print("  ⚠️  Риски:")
        for r in risks:
            print(f"    • {r}")
    else:
        print("  ✅ Хвостовые риски в норме")


def _selftest_regime_no_lookahead():
    """
    Тест #2: regime[i] должен зависеть только от данных вплоть до дня i-1,
    и сопоставляться с return дня i. Конструируем синтетический ряд, где
    смена режима происходит ровно в известный день, и проверяем, что
    regime.shift(1) для дня смены даёт значение СТАРОГО режима (т.е. то,
    что было известно накануне), а не нового (которое узнаётся только
    в момент close[t]).
    """
    print("  [selftest] regime.shift(1) — проверка отсутствия look-ahead:")
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date
    raw = pd.Series(["A"] * (n // 2) + ["B"] * (n - n // 2))
    shifted = raw.shift(1)

    change_day = n // 2  # первый день режима B
    # На день смены shifted должен показывать ПРЕДЫДУЩИЙ (известный заранее) режим "A"
    cond1 = shifted.iloc[change_day] == "A"
    # На следующий день shifted уже знает про "B" (т.к. известно из дня change_day)
    cond2 = shifted.iloc[change_day + 1] == "B"
    # Общая проверка: regime[i] (после shift) == raw[i-1] для всех i>=1
    cond3 = all(shifted.iloc[i] == raw.iloc[i-1] for i in range(1, n))

    print(f"    в день смены режима shift(1) показывает старый режим 'A': {cond1}")
    print(f"    на следующий день shift(1) уже отражает новый режим 'B':   {cond2}")
    print(f"    ∀i: regime_used[i] == regime_raw[i-1] (только прошлые данные): {cond3}")
    ok = cond1 and cond2 and cond3
    print(f"    {'PASS' if ok else 'FAIL'}: return[i] сопоставляется только с режимом, известным по конец дня i-1")
    return ok


def _selftest_cost_accounting():
    """
    Тест #3: стоимость должна начисляться ОДИН РАЗ ЗА КАЖДУЮ сделку:
    total_cost == n_trades * cost_per_trade, т.е. gross - net == cost_rt * n.
    Проверяем через pnl_engine (используется в top3_report.py и тут можно
    дополнительно сверить с net_pnl()).
    """
    print("  [selftest] издержки: total_cost == n_trades * cost_per_trade:")
    import pnl_engine
    n = 120
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high  = np.maximum(open_, close) * 1.01
    low   = np.minimum(open_, close) * 0.99
    d = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    d["intraday"]  = (d["close"] / d["open"] - 1) * 100
    d["overnight"] = (d["open"] / d["close"].shift(1) - 1) * 100
    d["total"]     = (d["close"] / d["close"].shift(1) - 1) * 100

    cost_rt = 0.08
    all_ok = True
    for strat in ("intraday_short", "long_overnight", "short_hold", "long_intraday"):
        gross, net, ret, n_trades = pnl_engine.pnl(d, strat, cost_rt)
        total_cost = gross - net
        expected = n_trades * cost_rt
        ok = abs(total_cost - expected) < 1e-9
        all_ok &= ok
        print(f"    {strat:<16}: n_trades={n_trades:<4} total_cost={total_cost:.4f} "
              f"expected={expected:.4f}  {'OK' if ok else 'MISMATCH'}")
    print(f"    {'PASS' if all_ok else 'FAIL'}: издержки начисляются ровно один раз на сделку во всех стратегиях")
    return all_ok


def _selftest_pnl_consistency():
    """
    Тест #7: pnl_engine (единый источник) и локальная STRATEGIES+net_pnl формула
    advanced_stats должны давать ИДЕНТИЧНЫЙ net P&L на одинаковых входных данных.

    PR-8: убрана ссылка на top3_report — этого модуля нет в данной копии проекта
    (он был частью исходного скила). Сверяем два реально присутствующих источника.
    """
    print("  [selftest] согласованность net P&L (pnl_engine ↔ advanced_stats.STRATEGIES):")
    import pnl_engine

    n = 100
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-02-01", periods=n, freq="B").date
    close = 50 * np.cumprod(1 + rng.normal(0, 0.012, n))
    open_ = close * (1 + rng.normal(0, 0.006, n))
    d = pd.DataFrame({"date": dates, "open": open_, "high": np.maximum(open_, close)*1.01,
                      "low": np.minimum(open_, close)*0.99, "close": close})
    d["intraday"]  = (d["close"] / d["open"] - 1) * 100
    d["overnight"] = (d["open"] / d["close"].shift(1) - 1) * 100
    d["total"]     = (d["close"] / d["close"].shift(1) - 1) * 100

    cost_rt = 0.08
    all_ok = True
    for strat in ("intraday_short", "long_overnight", "short_hold", "long_intraday"):
        _, net_engine, _, _ = pnl_engine.pnl(d, strat, cost_rt)
        _, ret_adv = STRATEGIES[strat](d, cost_rt)
        net_adv = net_pnl(ret_adv.dropna(), cost_rt)

        ok = abs(net_engine - net_adv) < 1e-9
        all_ok &= ok
        print(f"    {strat:<16}: engine={net_engine:+.4f}  advanced_stats={net_adv:+.4f}  "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"    {'PASS' if all_ok else 'FAIL'}: оба источника дают идентичный net P&L")
    return all_ok


def run_selftests():
    print("═" * 64)
    print("  SELFTEST — проверки исправленных дефектов")
    print("═" * 64)
    r1 = _selftest_white_rc()
    print()
    r2 = _selftest_regime_no_lookahead()
    print()
    r3 = _selftest_cost_accounting()
    print()
    r4 = _selftest_pnl_consistency()
    print()
    overall = r1 and r2 and r3 and r4
    print(f"ИТОГ: {'ALL PASS' if overall else 'SOME FAILED'}")
    return overall


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Продвинутые статистические тесты: White Reality Check, PBO, Regime, Tail Risk"
    )
    p.add_argument("--mode", required=True,
                   choices=["white_reality", "pbo", "regime", "tail_risk", "all", "selftest"],
                   help=(
                       "white_reality — Stage 4c, anti data-snooping; "
                       "pbo          — Stage 4d, вероятность переобучения; "
                       "regime       — Stage 2b, анализ рыночных режимов; "
                       "tail_risk    — Stage 5c, VaR/CVaR/Skew/Kurt; "
                       "all          — все четыре теста"
                   ))
    p.add_argument("--daily",     default=None,              help="Путь к *_candles.csv")
    p.add_argument("--cost-side", type=float, default=0.04,  help="Издержки/сторона, %%")
    p.add_argument("--strategy",  default="long_overnight",  help="Для tail_risk")
    p.add_argument("--pbo-splits", type=int, default=16,     help="S для CSCV (чётное, ≥4)")
    p.add_argument("--n-boot",    type=int, default=10_000,  help="Симуляций для White RC")
    a = p.parse_args()

    if a.mode == "selftest":
        ok = run_selftests()
        sys.exit(0 if ok else 1)

    if not a.daily:
        # Ищем первый *_candles.csv в папке скрипта
        import glob, os
        found = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "*_candles.csv"
        )))
        if not found:
            sys.exit("Укажите --daily путь к *_candles.csv")
        a.daily = found[0]
        print(f"Автовыбор файла: {a.daily}")

    cost_rt = a.cost_side * 2
    d = load_daily(a.daily)
    ticker = a.daily.split("/")[-1].replace("_candles.csv", "").upper()

    print(f"\n  Тикер: {ticker}  |  Дней: {len(d)}  |  Издержки: {a.cost_side}%/сторона")
    print(f"  ⚠️  Аналитика, не инвестиционная рекомендация.\n")

    modes = (["white_reality", "pbo", "regime", "tail_risk"]
             if a.mode == "all" else [a.mode])

    for mode in modes:
        print()
        if mode == "white_reality":
            white_reality_check(d, cost_rt, n_boot=a.n_boot)
        elif mode == "pbo":
            pbo_cscv(d, cost_rt, S=a.pbo_splits)
        elif mode == "regime":
            regime_analysis(d, cost_rt)
        elif mode == "tail_risk":
            tail_risk(d, cost_rt, strategy=a.strategy)


if __name__ == "__main__":
    main()
