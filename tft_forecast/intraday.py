"""
Обучаемая внутридневная поправка (Шаг 1): как ОСТАТОК дневного хода (сейчас→close)
зависит от времени запуска τ и уже реализованного движения r_so_far.

Заменяет арифметику `_adjust_for_realized` (remaining = predicted − realized,
допущение mean-reversion) на модель, обученную на market_data_5m:

    z_remaining ≈ β(τ) · z_so_far + ε(τ)          (в единицах дневной σ тикера)

где z = r / σ_daily (нормировка на волатильность тикера, чтобы честно пулить все
бумаги). β(τ) < 0 → в этом окне сессии ход РАЗВОРАЧИВАЕТСЯ (реверсия), β(τ) > 0 →
ПРОДОЛЖАЕТСЯ (моментум). Ширина остатка (квантили ε) естественно сужается к концу
сессии — меньше дня впереди.

Валидация — по времени суток и с разделением train/val ПО ДНЯМ (без утечки):
report показывает per-bucket β, OOS-корреляцию β·z_so_far ↔ z_remaining и
покрытие обученного интервала против безусловного (τ- и r-независимого) baseline.
Если β≈0 и OOS-corr≈0 — время запуска сигнала не несёт, поправка не помогает
(честный нулевой результат, а не молчаливое усложнение).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("tft.intraday")

# Основная сессия MOEX (МСК), минуты от полуночи. Аукцион открытия и вечёрку
# намеренно исключаем — там ликвидность/цены не репрезентативны.
SESSION_START_MIN = 10 * 60          # 10:00
SESSION_END_MIN = 18 * 60 + 40       # 18:40 (закрытие основной сессии)

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_Z_CLIP = 4.0                        # клип z для устойчивости к выбросам


@dataclass
class IntradayModel:
    n_buckets: int
    beta: dict[int, float]                    # bucket → наклон реверсии/моментума
    resid_q: dict[int, np.ndarray]            # bucket → квантили остатка (z)
    global_resid_q: np.ndarray                # фолбэк для тонких бакетов
    n: dict[int, int]                         # bucket → размер обучающей выборки
    apply: bool = False                       # прошла ли OOS-проверку полезности
    report: dict = field(default_factory=dict)

    def _bucket(self, tau: float) -> int:
        b = int(min(max(tau, 0.0), 0.999999) * self.n_buckets)
        return min(b, self.n_buckets - 1)

    def remaining_quantiles(self, tau: float, r_so_far_pct: float,
                            sigma_pct: float) -> np.ndarray | None:
        """Квантили ОСТАВШЕГОСЯ хода (сейчас→close), в % от текущей цены.
        None → бакет пуст/нет сигмы (вызывающий откатится на арифметику)."""
        if not sigma_pct or sigma_pct <= 0:
            return None
        b = self._bucket(tau)
        rq = self.resid_q.get(b)
        if rq is None or self.n.get(b, 0) < 1:
            rq = self.global_resid_q
        beta = self.beta.get(b, 0.0)
        z_so = float(np.clip(r_so_far_pct / sigma_pct, -_Z_CLIP, _Z_CLIP))
        z_rem = beta * z_so + rq                  # квантили остатка в z-единицах
        return z_rem * sigma_pct                  # обратно в проценты


# ── Загрузка 5M и построение выборки ──────────────────────────────────────────

_SQL_5M = """
    SELECT ticker,
           (ts AT TIME ZONE 'Europe/Moscow') AS tsm,
           close
    FROM market_data_5m
    WHERE ticker = ANY(%(tks)s)
      AND ts >= now() - (%(days)s || ' days')::interval
    ORDER BY ticker, ts;
"""


def _load_intraday_samples(conn, tickers: list[str], lookback_days: int,
                           sigma_by_tk: dict[str, float]) -> pd.DataFrame:
    """Строит по 5M-барам основной сессии наблюдения (ticker, date, tau,
    z_so_far, z_remaining). open = первый бар сессии, close = последний."""
    with conn.cursor() as cur:
        cur.execute(_SQL_5M, {"tks": list(tickers), "days": str(lookback_days)})
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ticker", "tsm", "close"])
    df["tsm"] = pd.to_datetime(df["tsm"])
    df["close"] = df["close"].astype(float)
    df["date"] = df["tsm"].dt.normalize()
    mins = df["tsm"].dt.hour * 60 + df["tsm"].dt.minute
    df = df[(mins >= SESSION_START_MIN) & (mins <= SESSION_END_MIN)].copy()
    df["min"] = mins[df.index]
    if df.empty:
        return df

    # open/close дня по (ticker, date)
    g = df.groupby(["ticker", "date"])
    day_open = g["close"].transform("first")     # ≈ цена открытия сессии
    day_close = g["close"].transform("last")
    span = max(1, SESSION_END_MIN - SESSION_START_MIN)
    df["tau"] = (df["min"] - SESSION_START_MIN) / span
    df["r_so_far"] = (df["close"] / day_open - 1.0) * 100.0
    df["remaining"] = (day_close / df["close"] - 1.0) * 100.0

    sig = df["ticker"].map(sigma_by_tk)
    df = df[sig.notna() & (sig > 0)].copy()
    if df.empty:
        return df
    df["z_so"] = np.clip(df["r_so_far"] / sig[df.index], -_Z_CLIP, _Z_CLIP)
    df["z_rem"] = df["remaining"] / sig[df.index]
    # отбрасываем крайние бары (tau≈0: remaining=почти весь день, шум открытия;
    # tau≈1: remaining≈0 тривиально) — оставляем информативную середину
    df = df[(df["tau"] > 0.02) & (df["tau"] < 0.98)]
    return df[["ticker", "date", "tau", "z_so", "z_rem"]].reset_index(drop=True)


def _slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 1e-9 else 0.0


def fit(conn, tickers: list[str], sigma_by_tk: dict[str, float], *,
        lookback_days: int = 60, n_buckets: int = 6,
        min_bucket: int = 200, val_days: int = 15,
        min_oos_corr: float = 0.05) -> IntradayModel | None:
    """Обучает поправку. Возвращает None, если данных мало (вызывающий
    останется на арифметике). report — метрики валидации по времени суток.

    apply выставляется True ТОЛЬКО если поправка доказала пользу на OOS:
    средняя OOS-корреляция ≥ min_oos_corr И покрытие не хуже безусловного.
    Иначе apply=False → вызывающий останется на арифметике (не ухудшаем
    прогноз не-сигналом)."""
    df = _load_intraday_samples(conn, tickers, lookback_days, sigma_by_tk)
    if df.empty or len(df) < min_bucket * 2:
        log.info("Внутридневная поправка: мало данных (%d набл.) — пропуск.", len(df))
        return None

    # хронологический сплит по ДНЯМ (без утечки внутри дня)
    uniq_days = np.sort(df["date"].unique())
    if len(uniq_days) <= val_days + 3:
        val_days = max(1, len(uniq_days) // 4)
    cutoff = uniq_days[-val_days]
    tr = df[df["date"] < cutoff]
    va = df[df["date"] >= cutoff]
    if len(tr) < min_bucket or len(va) < 20:
        log.info("Внутридневная поправка: короткий ряд для сплита — пропуск.")
        return None

    tr_b = (tr["tau"] * n_buckets).astype(int).clip(0, n_buckets - 1)
    va_b = (va["tau"] * n_buckets).astype(int).clip(0, n_buckets - 1)

    beta: dict[int, float] = {}
    resid_q: dict[int, np.ndarray] = {}
    n: dict[int, int] = {}
    global_resid_q = np.quantile(tr["z_rem"].to_numpy(), QUANTILES)

    per_bucket = []
    base_cov_all, learn_cov_all, learn_w, base_w = [], [], [], []
    for b in range(n_buckets):
        xt = tr.loc[tr_b == b, "z_so"].to_numpy()
        yt = tr.loc[tr_b == b, "z_rem"].to_numpy()
        n[b] = len(xt)
        if len(xt) < min_bucket:
            beta[b] = 0.0
            resid_q[b] = global_resid_q
        else:
            beta[b] = _slope_through_origin(xt, yt)
            resid_q[b] = np.quantile(yt - beta[b] * xt, QUANTILES)

        # OOS-оценка на валидации этого бакета
        xv = va.loc[va_b == b, "z_so"].to_numpy()
        yv = va.loc[va_b == b, "z_rem"].to_numpy()
        if len(yv) >= 15:
            pred = beta[b] * xv
            corr = float(np.corrcoef(pred, yv)[0, 1]) if np.std(pred) > 1e-9 else 0.0
            lo, hi = resid_q[b][0], resid_q[b][-1]
            learn_cov = float(np.mean((yv >= pred + lo) & (yv <= pred + hi)))
            blo, bhi = global_resid_q[0], global_resid_q[-1]
            base_cov = float(np.mean((yv >= blo) & (yv <= bhi)))
            per_bucket.append({
                "bucket": b, "tau_mid": (b + 0.5) / n_buckets, "n": n[b],
                "beta": beta[b], "oos_corr": corr,
                "learn_cov": learn_cov, "base_cov": base_cov,
                "learn_width": float(hi - lo), "base_width": float(bhi - blo),
            })
            learn_cov_all.append(learn_cov); base_cov_all.append(base_cov)
            learn_w.append(hi - lo); base_w.append(bhi - blo)

    report = {
        "n_train": len(tr), "n_val": len(va), "n_buckets": n_buckets,
        "per_bucket": per_bucket,
        "oos_corr_mean": float(np.nanmean([p["oos_corr"] for p in per_bucket])) if per_bucket else 0.0,
        "learn_cov_mean": float(np.mean(learn_cov_all)) if learn_cov_all else float("nan"),
        "base_cov_mean": float(np.mean(base_cov_all)) if base_cov_all else float("nan"),
        "width_ratio": (float(np.mean(learn_w) / np.mean(base_w))
                        if base_w and np.mean(base_w) > 0 else float("nan")),
    }
    # авто-гард: применяем только при доказанной OOS-пользе
    apply = (report["oos_corr_mean"] >= min_oos_corr
             and not np.isnan(report["learn_cov_mean"])
             and report["learn_cov_mean"] >= report["base_cov_mean"] - 0.02)
    report["apply"] = bool(apply)
    report["min_oos_corr"] = min_oos_corr
    return IntradayModel(n_buckets=n_buckets, beta=beta, resid_q=resid_q,
                         global_resid_q=global_resid_q, n=n, apply=bool(apply),
                         report=report)


def log_report(model: IntradayModel) -> None:
    r = model.report
    verdict = ("ПРИМЕНЯЕТСЯ (OOS-польза подтверждена)" if model.apply
               else "НЕ применяется — нет OOS-сигнала, остаёмся на арифметике")
    log.info("Внутридневная поправка: train=%d val=%d, OOS-corr(β·z↔остаток)=%.3f, "
             "покрытие обуч=%.0f%% vs безусл=%.0f%%, ширина обуч/безусл=%.2f → %s",
             r.get("n_train", 0), r.get("n_val", 0), r.get("oos_corr_mean", 0.0),
             r.get("learn_cov_mean", float("nan")) * 100,
             r.get("base_cov_mean", float("nan")) * 100,
             r.get("width_ratio", float("nan")), verdict)
    for p in r.get("per_bucket", []):
        log.info("    τ≈%.2f  n=%-6d β=%+.3f  OOS-corr=%+.3f  cov(обуч/безусл)=%.0f%%/%.0f%%",
                 p["tau_mid"], p["n"], p["beta"], p["oos_corr"],
                 p["learn_cov"] * 100, p["base_cov"] * 100)
