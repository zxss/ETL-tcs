"""
Построение обучающих окон (sliding windows) и масштабирование признаков.

Возвращает numpy-массивы — их потребляет и torch-модель, и статистический
fallback. Масштабирование признаков — глобальное (по всему пулу тикеров),
т.к. модель одна на все инструменты. Цели НЕ масштабируем: это проценты,
интерпретируемые напрямую.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_COLS, TARGET_COLS, TickerFrame

LOOKBACK = 30  # длина окна истории, дней


@dataclass
class Panel:
    X: np.ndarray            # (N, LOOKBACK, F) признаки окна
    T: np.ndarray            # (N,) индекс тикера (static categorical)
    Y: np.ndarray            # (N, 2) цели [next_low_pct, next_high_pct]
    feat_mean: np.ndarray    # (F,) для масштабирования
    feat_std: np.ndarray     # (F,)
    tickers: list[str]       # индекс → имя тикера
    # Финальные окна (последний день каждого тикера) для инференса:
    infer_X: np.ndarray      # (M, LOOKBACK, F)
    infer_T: np.ndarray      # (M,)
    infer_tickers: list[str] # имена тикеров для каждого infer-окна


def _clean_window(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_panel(frames: list[TickerFrame], lookback: int = LOOKBACK) -> Panel | None:
    tickers = [f.ticker for f in frames]
    tk_index = {tk: i for i, tk in enumerate(tickers)}

    # --- сбор обучающих окон ---
    X_list, T_list, Y_list = [], [], []
    infer_X, infer_T, infer_tk = [], [], []

    for f in frames:
        d = f.df
        feats = d[FEATURE_COLS].to_numpy(dtype=np.float64)
        tgts = d[TARGET_COLS].to_numpy(dtype=np.float64)
        n = len(d)

        # обучающие окна: окно [i-lookback+1 .. i], цель в строке i (next_*),
        # пропускаем строки без цели (последняя строка) и неполные окна.
        for i in range(lookback - 1, n - 1):
            y = tgts[i]
            if not np.all(np.isfinite(y)):
                continue
            win = feats[i - lookback + 1 : i + 1]
            X_list.append(win)
            T_list.append(tk_index[f.ticker])
            Y_list.append(y)

        # инференс-окно: последние lookback дней (цель неизвестна — её предсказываем)
        if n >= lookback:
            infer_X.append(feats[n - lookback : n])
            infer_T.append(tk_index[f.ticker])
            infer_tk.append(f.ticker)

    if not X_list or not infer_X:
        return None

    X = _clean_window(np.asarray(X_list, dtype=np.float64))
    T = np.asarray(T_list, dtype=np.int64)
    Y = np.asarray(Y_list, dtype=np.float64)
    iX = _clean_window(np.asarray(infer_X, dtype=np.float64))
    iT = np.asarray(infer_T, dtype=np.int64)

    # глобальное масштабирование признаков (по обучающему пулу)
    flat = X.reshape(-1, X.shape[-1])
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    X = (X - feat_mean) / feat_std
    iX = (iX - feat_mean) / feat_std

    return Panel(
        X=X.astype(np.float32),
        T=T,
        Y=Y.astype(np.float32),
        feat_mean=feat_mean,
        feat_std=feat_std,
        tickers=tickers,
        infer_X=iX.astype(np.float32),
        infer_T=iT,
        infer_tickers=infer_tk,
    )
