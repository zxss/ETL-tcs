"""
Оркестрация Multi-Asset TFT Next-Day Range Forecast.

run(conn):
  1. грузит дневные свечи по всем тикерам (features.load_panel)
  2. строит обучающие окна (dataset.build_panel)
  3. обучает ОДНУ модель TFT на всём пуле (если есть torch) или
     калиброванный статистический baseline (если torch нет)
  4. калибрует CoverageProb на held-out валидации
  5. печатает итоговую таблицу прогнозов диапазона на следующий день
     и таблицу, привязанную к стратегиям из контура валидации.

Управление через config: TFT_FORECAST, TFT_EPOCHS, TFT_HIDDEN, TFT_TICKERS.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from .dataset import LOOKBACK, build_panel
from .features import load_panel

log = logging.getLogger("tft.forecast")

# Должны совпадать с model.QUANTILES
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
Q_LOW, Q_MED, Q_HIGH = 0, 2, 4   # индексы 0.1 / 0.5 / 0.9 в сетке выше

# Индексы целей (порядок — features.TARGET_COLS)
T_LOW, T_HIGH, T_OVN, T_INTRA, T_TOTAL = 0, 1, 2, 3, 4
T_WLOW, T_WHIGH, T_WTOTAL = 5, 6, 7   # недельный горизонт (WEEK_H дней)


# ── Обучение TFT (torch) ──────────────────────────────────────────────────────

def _train_tft(panel, epochs: int, hidden: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .model import TFT, quantile_loss

    torch.manual_seed(42)
    np.random.seed(42)

    N = panel.X.shape[0]
    n_feat = panel.X.shape[-1]
    n_tk = len(panel.tickers)

    # PR-7: ХРОНОЛОГИЧЕСКИ ЧЕСТНЫЙ сплит по общей календарной оси (panel.W —
    # ordinal даты цели окна). Валидация = последние 15% наблюдений ПО ВРЕМЕНИ
    # (а не случайные), train = всё, что строго раньше, с ЭМБАРГО в LOOKBACK дней,
    # чтобы обучающие окна не перекрывались по входам с валидационными целями
    # (иначе утечка). Это даёт честную OOS-оценку покрытия вместо оптимистичной.
    from .dataset import LOOKBACK
    W = panel.W
    cutoff = np.quantile(W, 0.85)
    val_mask = W >= cutoff
    tr_mask = W < (cutoff - LOOKBACK)            # эмбарго против перекрытия окон
    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(tr_mask)[0]
    if len(val_idx) < 1 or len(tr_idx) < 10:     # деградация на коротком ряду
        rng = np.random.default_rng(42)
        perm = rng.permutation(N)
        n_val = max(1, int(N * 0.15))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        log.warning("    TFT: ряд слишком короткий для хронологического сплита — fallback на случайный")

    dev = torch.device("cpu")
    Xtr = torch.tensor(panel.X[tr_idx], device=dev)
    Ttr = torch.tensor(panel.T[tr_idx], device=dev)
    Ytr = torch.tensor(panel.Y[tr_idx], device=dev)
    Xva = torch.tensor(panel.X[val_idx], device=dev)
    Tva = torch.tensor(panel.T[val_idx], device=dev)
    Yva = panel.Y[val_idx]

    model = TFT(n_feat, n_tk, hidden=hidden, n_targets=panel.Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = TensorDataset(Xtr, Ttr, Ytr)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, tb, yb in loader:
            opt.zero_grad()
            pred = model(xb, tb)
            loss = quantile_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            log.info("    epoch %2d/%d  train pinball=%.4f", ep + 1, epochs, tot / len(tr_idx))

    # прогноз на инференс-окнах
    model.eval()
    with torch.no_grad():
        iX = torch.tensor(panel.infer_X, device=dev)
        iT = torch.tensor(panel.infer_T, device=dev)
        infer_pred = model(iX, iT).cpu().numpy()      # (M, 2, Q)
        val_pred = model(Xva, Tva).cpu().numpy()      # (n_val, 2, Q)

    return infer_pred, val_pred, Tva.cpu().numpy(), Yva


# ── Статистический fallback (без torch) ───────────────────────────────────────

def _predict_fallback(panel):
    """
    Квантильный baseline без torch: для каждого тикера — эмпирические квантили
    завтрашних целей из его окон. Возвращает (M, 2, Q) как и TFT-голова.

    PR-7: причинно-честная версия. Раньше квантили считались по ВСЕЙ истории
    тикера (включая будущее относительно точки), а покрытие оценивалось in-sample
    по всем окнам → завышенный CoverageProb. Теперь квантили берутся ТОЛЬКО из
    обучающего периода (W < cutoff), а покрытие меряется на отложенном хвосте
    (W >= cutoff). Для инференс-окна (последний день) вся история — прошлое.

    Сезонность по дню недели: квантили считаются с учётом day-of-week целевого
    дня (бакет (тикер, dow)). Так прогноз на понедельник (overnight через
    выходные) опирается на распределение именно понедельников, а не всех дней.
    Если бакет тонкий (<MIN_DOW наблюдений) — откат на (тикер, все дни), затем
    на глобальное распределение.
    """
    from .dataset import LOOKBACK
    n_tgt = panel.Y.shape[1]
    nq = len(QUANTILES)
    M = panel.infer_X.shape[0]
    MIN_DOW = 30   # минимум наблюдений в бакете дня недели, иначе откат

    W = panel.W
    Wd = getattr(panel, "Wd", np.full(len(W), -1, dtype=np.int64))
    cutoff = np.quantile(W, 0.85)
    train_mask = W < (cutoff - LOOKBACK)
    val_idx = np.where(W >= cutoff)[0]
    if len(val_idx) < 1 or int(train_mask.sum()) < 10:
        train_mask = np.ones(len(W), bool)              # короткий ряд — без сплита
        val_idx = np.arange(len(W))

    # квантильный источник: только обучающий период.
    # by_tk[ti] — все дни тикера; by_tk_dow[(ti, dow)] — отдельно по дню недели.
    by_tk_train: dict[int, list[np.ndarray]] = {}
    by_tk_dow_train: dict[tuple[int, int], list[np.ndarray]] = {}
    for t, dwd, y in zip(panel.T[train_mask], Wd[train_mask], panel.Y[train_mask]):
        by_tk_train.setdefault(int(t), []).append(y)
        if int(dwd) >= 0:
            by_tk_dow_train.setdefault((int(t), int(dwd)), []).append(y)
    global_train_Y = panel.Y[train_mask]

    def _q(arr):
        return np.stack([np.quantile(arr[:, k], QUANTILES) for k in range(n_tgt)], axis=0)

    def _q_for(tk_i, dow):
        if int(dow) >= 0:
            arr = np.asarray(by_tk_dow_train.get((int(tk_i), int(dow)), []), dtype=np.float64)
            if len(arr) >= MIN_DOW:
                return _q(arr)
        arr = np.asarray(by_tk_train.get(int(tk_i), []), dtype=np.float64)
        if len(arr) < 20:
            arr = global_train_Y
        return _q(arr)

    infer_pred = np.zeros((M, n_tgt, nq), dtype=np.float64)
    for m, tk_i in enumerate(panel.infer_T):
        infer_pred[m] = _q_for(tk_i, panel.infer_Wd[m])

    val_pred = np.zeros((len(val_idx), n_tgt, nq), dtype=np.float64)
    for j, idx in enumerate(val_idx):
        val_pred[j] = _q_for(panel.T[idx], Wd[idx])
    return infer_pred, val_pred, panel.T[val_idx], panel.Y[val_idx]


# ── Калибровка покрытия ────────────────────────────────────────────────────────

def _conformal_week_margin(val_pred, val_Y, target: float = 0.80) -> float:
    """Конформная поправка ширины недельного коридора (split-conformal).

    Сырые q0.1/q0.9 на недельном горизонте недокрывают (59% при номинале 80%):
    экстремумы за 5 дней рассеяны шире, чем выучили квантильные головы. Считаем
    на held-out хвосте нормированный на ширину коридора «промах»:

        score = max(pred_low − actual_low, actual_high − pred_high, 0) / width

    и берём его target-квантиль (с поправкой (n+1)/n малой выборки) → λ.
    Расширение обеих границ на λ·width доводит покрытие до ≈target по
    построению; гарантия на будущее — в предположении обменности наблюдений.
    Нормировка на width адаптивна: волатильные бумаги получают больший запас
    в абсолюте, тихие — меньший.
    """
    lo = val_pred[:, T_WLOW, Q_LOW]
    hi = val_pred[:, T_WHIGH, Q_HIGH]
    width = np.maximum(hi - lo, 1e-6)
    viol = np.maximum(lo - val_Y[:, T_WLOW], val_Y[:, T_WHIGH] - hi)
    scores = np.maximum(viol / width, 0.0)
    n = len(scores)
    if n == 0:
        return 0.0
    q = min(1.0, np.ceil((n + 1) * target) / n)
    return float(np.quantile(scores, q))


def _widen_week(pred: np.ndarray, lam: float) -> np.ndarray:
    """Копия pred с недельным коридором, расширенным на λ·width с обеих сторон."""
    out = pred.copy()
    width = np.maximum(out[:, T_WHIGH, Q_HIGH] - out[:, T_WLOW, Q_LOW], 0.0)
    out[:, T_WLOW, Q_LOW] -= lam * width
    out[:, T_WHIGH, Q_HIGH] += lam * width
    return out


def _coverage(val_pred, val_T, val_Y, lo_idx: int = T_LOW, hi_idx: int = T_HIGH):
    """
    Эмпирическое покрытие интервала [low_q0.1, high_q0.9] для пары целей
    (lo_idx, hi_idx): доля наблюдений, где реальные low/high попали внутрь
    прогноза. По умолчанию — дневной коридор; (T_WLOW, T_WHIGH) — недельный.
    Возвращает (per_ticker: dict idx→prob, global_prob).
    """
    low_pred = val_pred[:, lo_idx, Q_LOW]
    high_pred = val_pred[:, hi_idx, Q_HIGH]
    actual_low = val_Y[:, lo_idx]
    actual_high = val_Y[:, hi_idx]
    hit = (actual_low >= low_pred) & (actual_high <= high_pred)

    global_prob = float(hit.mean()) if len(hit) else float("nan")
    per_tk: dict[int, float] = {}
    for t in np.unique(val_T):
        mask = val_T == t
        if mask.sum() >= 15:
            per_tk[int(t)] = float(hit[mask].mean())
    return per_tk, global_prob


# ── Сборка прогнозов ───────────────────────────────────────────────────────────

def _assemble(frames, panel, infer_pred, per_tk_cov, global_cov, quotes=None,
              week_cov=None):
    """
    Собирает прогноз диапазона. Якорная цена — актуальная Last Price (если есть
    котировка), иначе последнее закрытие. ForecastLow/High/RangePct считаются
    ОТНОСИТЕЛЬНО якорной цены, а не вчерашнего закрытия.

    Недельный коридор (WeekLow/High) — НАСТОЯЩИЕ квантили модели по целям
    week_low_pct/week_high_pct (min/max за WEEK_H дней), с отдельной калибровкой
    покрытия week_cov=(per_tk, global) на held-out хвосте.
    """
    quotes = quotes or {}
    week_per_tk, week_global = week_cov if week_cov else ({}, float("nan"))
    fr_by_tk = {f.ticker: f for f in frames}
    out = {}
    tk_index = {tk: i for i, tk in enumerate(panel.tickers)}
    h = max(1, int(getattr(config, "WEEK_HORIZON_DAYS", 5)))
    for m, tk in enumerate(panel.infer_tickers):
        f = fr_by_tk[tk]
        low_pct = float(infer_pred[m, T_LOW, Q_LOW])    # консервативная нижняя граница
        high_pct = float(infer_pred[m, T_HIGH, Q_HIGH]) # консервативная верхняя граница
        # защита от вырожденного интервала
        if high_pct <= low_pct:
            mid = (high_pct + low_pct) / 2.0
            low_pct, high_pct = mid - 0.5, mid + 0.5

        q = quotes.get(tk)
        anchor = q.price if q else f.last_close      # текущая цена либо закрытие
        forecast_low = anchor * (1 + low_pct / 100.0)
        forecast_high = anchor * (1 + high_pct / 100.0)
        range_pct = (forecast_high - forecast_low) / anchor * 100.0
        cov = per_tk_cov.get(tk_index[tk], global_cov)

        # Недельный коридор из предсказанных квантилей week_*-целей.
        wlow_pct = float(infer_pred[m, T_WLOW, Q_LOW])
        whigh_pct = float(infer_pred[m, T_WHIGH, Q_HIGH])
        # консистентность горизонтов: экстремум за 5 дней не может быть уже
        # дневного коридора — мягкий клэмп на случай шума квантильных голов
        wlow_pct = min(wlow_pct, low_pct)
        whigh_pct = max(whigh_pct, high_pct)
        week_low = anchor * (1 + wlow_pct / 100.0)
        week_high = anchor * (1 + whigh_pct / 100.0)
        week_range_pct = (week_high - week_low) / anchor * 100.0
        week_med_pct = float(infer_pred[m, T_WTOTAL, Q_MED])  # медиана close за H дней

        out[tk] = {
            "ForecastLow": forecast_low,
            "ForecastHigh": forecast_high,
            "RangePct": range_pct,
            "WeekLow": week_low,
            "WeekHigh": week_high,
            "WeekRangePct": week_range_pct,
            "WeekMedPct": week_med_pct,
            "WeekCoverage": week_per_tk.get(tk_index[tk], week_global),
            "WeekHorizon": h,
            "CoverageProb": cov,
            "last_close": f.last_close,
            "last_date": f.last_date,
            "anchor_price": anchor,
            "price": q.price if q else None,
            "price_ts": q.ts if q else None,
            "price_age_sec": q.age_sec if q else None,
        }
    return out


# ── Реализованное движение → пересчёт оставшейся доходности ────────────────────

def _adjust_for_realized(infer_pred, panel, frames, ctx):
    """
    Когда рынок открыт и часть дневного хода уже отработана, направленный PnL
    должен отражать ОСТАВШУЮСЯ доходность. Для каждого примитива вычитаем уже
    реализованную часть из предсказанных квантилей:

        overnight_remaining = predicted_overnight - (today_open/yest_close - 1)
        intraday_remaining  = predicted_intraday  - (last_price/today_open - 1)
        total_remaining     = predicted_total     - (last_price/yest_close - 1)

    Так после сильного движения лучшая дневная стратегия (long/short) может
    автоматически смениться. Возвращает новый массив (копию) той же формы.
    """
    adj = infer_pred.copy()
    if not ctx or not ctx.market_open or not ctx.quotes:
        return adj
    fr_by_tk = {f.ticker: f for f in frames}
    for m, tk in enumerate(panel.infer_tickers):
        q = ctx.quotes.get(tk)
        f = fr_by_tk.get(tk)
        if q is None or f is None:
            continue
        yest_close = f.last_close
        last_price = q.price
        to = q.today_open
        r_ovn = (to / yest_close - 1.0) * 100.0 if to else 0.0
        r_intra = (last_price / to - 1.0) * 100.0 if to else 0.0
        r_total = (last_price / yest_close - 1.0) * 100.0
        adj[m, T_OVN, :] -= r_ovn
        adj[m, T_INTRA, :] -= r_intra
        adj[m, T_TOTAL, :] -= r_total
    return adj


# ── Печать ─────────────────────────────────────────────────────────────────────

def _print_full(forecasts: dict):
    print("\n" + "=" * 78)
    print("  MULTI-ASSET TFT — ПРОГНОЗ ДИАПАЗОНА НА СЛЕДУЮЩИЙ ТОРГОВЫЙ ДЕНЬ")
    print("=" * 78)
    hdr = f"  {'ticker':<8}{'last':>10}{'ForecastLow':>13}{'ForecastHigh':>14}{'RangePct':>10}{'Coverage':>10}"
    print(hdr)
    print("  " + "-" * 74)
    for tk in sorted(k for k in forecasts if k != "__meta__"):
        r = forecasts[tk]
        print(f"  {tk:<8}{r['last_close']:>10.2f}{r['ForecastLow']:>13.2f}"
              f"{r['ForecastHigh']:>14.2f}{r['RangePct']:>9.2f}%{r['CoverageProb'] * 100:>9.0f}%")
    print("=" * 78)


def print_weekly(forecasts: dict) -> None:
    """Таблица недельного прогноза диапазона по акциям (по аналогии с дневной).

    Настоящий multi-horizon прогноз: модель обучена на недельных целях
    week_low/high/total (min/max/close за WEEK_H торговых дней), Coverage —
    отдельно калиброванное НЕДЕЛЬНОЕ покрытие на held-out хвосте."""
    rows = [k for k in forecasts if k != "__meta__"]
    if not rows:
        return
    h = forecasts[rows[0]].get("WeekHorizon", 5)
    print("\n" + "=" * 86)
    print(f"  MULTI-ASSET TFT — ПРОГНОЗ ДИАПАЗОНА НА НЕДЕЛЮ ({h} торг. дн., обученные цели)")
    print("=" * 86)
    hdr = (f"  {'ticker':<8}{'last':>10}{'WeekLow':>13}{'WeekHigh':>14}"
           f"{'RangePct':>10}{'Med5d%':>8}{'Coverage':>10}")
    print(hdr)
    print("  " + "-" * 82)
    for tk in sorted(rows):
        r = forecasts[tk]
        if r.get("WeekLow") is None:
            continue
        med = r.get("WeekMedPct")
        med_s = f"{med:>+7.2f}%" if med is not None else f"{'—':>8}"
        wcov = r.get("WeekCoverage")
        wcov_s = f"{wcov * 100:>9.0f}%" if wcov is not None and wcov == wcov else f"{'—':>10}"
        print(f"  {tk:<8}{r['last_close']:>10.2f}{r['WeekLow']:>13.2f}"
              f"{r['WeekHigh']:>14.2f}{r['WeekRangePct']:>9.2f}%{med_s}{wcov_s}")
    print("  " + "-" * 82)
    print("  WeekLow/High — q0.1/q0.9 модели по целям min(low)/max(high) за неделю,")
    print("  расширенные конформной поправкой до целевого покрытия;")
    print("  Med5d% — медиана предсказанного изменения close за горизонт;")
    print("  Coverage — эмпирическое НЕДЕЛЬНОЕ покрытие коридора на held-out валидации.")
    print("=" * 86)


def _print_strategy_table(forecasts: dict, tickers: list[str], strats: list[str]):
    print("\n  ПРОГНОЗ ДИАПАЗОНА, ПРИВЯЗАННЫЙ К СТРАТЕГИЯМ")
    print("  " + "-" * 74)
    hdr = (f"  {'ticker':<8}{'strategy':<17}{'ForecastLow':>13}{'ForecastHigh':>14}"
           f"{'RangePct':>10}{'Coverage':>10}")
    print(hdr)
    print("  " + "-" * 74)
    for tk in tickers:
        r = forecasts.get(tk.upper()) or forecasts.get(tk)
        if not r:
            continue
        for st in strats:
            print(f"  {tk.upper():<8}{st:<17}{r['ForecastLow']:>13.2f}{r['ForecastHigh']:>14.2f}"
                  f"{r['RangePct']:>9.2f}%{r['CoverageProb'] * 100:>9.0f}%")
    print("  " + "-" * 74)


# ── Точка входа ────────────────────────────────────────────────────────────────

def run(conn, quiet: bool = False) -> dict | None:
    """
    Обучает единый TFT на всех тикерах и печатает прогноз диапазона на
    следующий день. Возвращает dict {ticker: {...}} или None если пропущено.
    Не бросает исключений наружу — main.py не должен падать из-за прогноза.

    quiet=True — не печатать промежуточные таблицы (диапазон по тикерам,
    диапазон по стратегиям, направленный PnL); только посчитать и вернуть
    данные для сводной итоговой таблицы (combined.print_combined).
    """
    if not getattr(config, "TFT_FORECAST", True):
        log.info("TFT_FORECAST=0 — модуль прогноза диапазона пропущен.")
        return None

    tickers = getattr(config, "TFT_TICKERS", None) or config.TICKERS
    log.info("Multi-Asset TFT: загрузка панели по %d тикерам...", len(tickers))
    frames = load_panel(conn, tickers)
    if len(frames) < 2:
        log.warning("Недостаточно тикеров для обучения общей модели (%d) — пропуск.", len(frames))
        return None

    panel = build_panel(frames, lookback=LOOKBACK)
    if panel is None or panel.X.shape[0] < 200:
        log.warning("Недостаточно обучающих окон — пропуск TFT.")
        return None

    log.info("Обучающих окон: %d, признаков: %d, тикеров: %d",
             panel.X.shape[0], panel.X.shape[-1], len(panel.tickers))

    epochs = int(getattr(config, "TFT_EPOCHS", 30))
    hidden = int(getattr(config, "TFT_HIDDEN", 32))

    try:
        import torch  # noqa: F401
        log.info("PyTorch найден — обучение Temporal Fusion Transformer (epochs=%d, hidden=%d)...",
                 epochs, hidden)
        infer_pred, val_pred, val_T, val_Y = _train_tft(panel, epochs, hidden)
        backend = "TFT"
    except ImportError:
        log.warning("PyTorch не установлен — используется калиброванный статистический baseline "
                    "(pip install torch для полноценного TFT).")
        infer_pred, val_pred, val_T, val_Y = _predict_fallback(panel)
        backend = "baseline"
    except Exception as e:  # noqa: BLE001
        log.warning("Ошибка обучения TFT (%s) — fallback на baseline.", e)
        infer_pred, val_pred, val_T, val_Y = _predict_fallback(panel)
        backend = "baseline"

    # Актуальные котировки на момент запуска: якорим прогноз на текущей цене.
    from . import quotes as quotes_mod
    ctx = quotes_mod.get_market_context(conn, tickers)
    log.info("Котировки: source=%s, рынок %s, тикеров с ценой=%d%s",
             ctx.source, "открыт" if ctx.market_open else "закрыт",
             len(ctx.quotes),
             f", max age={ctx.max_age_sec / 60:.0f} мин" if ctx.quotes else "")

    per_tk_cov, global_cov = _coverage(val_pred, val_T, val_Y)

    # Недельный коридор: конформная поправка до целевого покрытия, затем
    # калибровка покрытия уже НА РАСШИРЕННЫХ границах (сырое — для лога).
    week_target = float(getattr(config, "WEEK_TARGET_COVERAGE", 0.80))
    _, week_cov_raw = _coverage(val_pred, val_T, val_Y, lo_idx=T_WLOW, hi_idx=T_WHIGH)
    week_lam = _conformal_week_margin(val_pred, val_Y, target=week_target)
    week_cov = _coverage(_widen_week(val_pred, week_lam), val_T, val_Y,
                         lo_idx=T_WLOW, hi_idx=T_WHIGH)
    log.info("Недельный коридор: сырое покрытие=%.0f%%, конформная λ=%.3f → "
             "покрытие=%.0f%% (цель %.0f%%).",
             week_cov_raw * 100, week_lam, week_cov[1] * 100, week_target * 100)

    forecasts = _assemble(frames, panel, _widen_week(infer_pred, week_lam),
                          per_tk_cov, global_cov,
                          quotes=ctx.quotes, week_cov=week_cov)

    # Оценка максимально допустимого размера позиции по ликвидности (Max Pos ₽).
    from . import liquidity as liquidity_mod
    liq = liquidity_mod.compute(conn, list(forecasts.keys()))
    # Кросс-секционный балл ликвидности 0..100 (перцентиль ADV₽ по вселенной).
    advs = {tk: m["adv_rub"] for tk, m in liq.items() if m.get("adv_rub")}
    for tk, m in liq.items():
        if tk in forecasts:
            forecasts[tk]["MaxPos"] = m["max_pos"]
            if advs:
                rank = sum(1 for a in advs.values() if a <= advs.get(tk, 0)) / len(advs)
                forecasts[tk]["LiqScore"] = round(rank * 100.0)

    # Слой рыночного контекста: режим, RS, объёмные/волатильностные/гэп-фильтры.
    from . import market as market_mod
    mctx = market_mod.compute(conn, list(forecasts.keys()))
    for tk in list(forecasts.keys()):
        tm = mctx.per.get(tk)
        forecasts[tk]["Regime"] = mctx.regime
        if tm:
            forecasts[tk]["RS"] = tm.rs
            forecasts[tk]["VolSpike"] = tm.vol_spike
            forecasts[tk]["ATRpctl"] = tm.atr_pctl
            forecasts[tk]["GapDownProb"] = tm.gap_down_prob
            # Риск-фильтр волатильности: сжимаем максимальную позицию.
            mp = forecasts[tk].get("MaxPos")
            if mp is not None and tm.atr_pctl is not None:
                if tm.atr_pctl > 95:
                    forecasts[tk]["MaxPos"] = mp / 4.0
                elif tm.atr_pctl > 85:
                    forecasts[tk]["MaxPos"] = mp / 2.0

    # Пересчёт оставшейся доходности относительно текущей цены.
    infer_pred_adj = _adjust_for_realized(infer_pred, panel, frames, ctx)

    forecasts["__meta__"] = {
        "as_of": ctx.as_of,
        "source": ctx.source,
        "market_open": ctx.market_open,
        "stale": ctx.stale,
        "max_age_sec": ctx.max_age_sec,
        "stale_threshold_sec": quotes_mod.STALE_SECONDS,
        "market": {
            "regime": mctx.regime,
            "breadth": mctx.breadth,
            "atr_pctl": mctx.atr_pctl_market,
            "risk_level": mctx.risk_level,
            "imoex_ret10": mctx.imoex_ret10,
            "index_source": mctx.source,
        },
    }

    log.info("Прогноз готов (backend=%s, дневное покрытие=%.0f%%, недельное=%.0f%%).",
             backend, global_cov * 100, week_cov[1] * 100)
    if not quiet:
        _print_full(forecasts)

    val_tickers = getattr(config, "VALIDATION_TICKERS", [])
    val_strats = getattr(config, "VALIDATION_STRATS", [])
    if val_strats:
        # направленный прогноз: ожидаемый PnL стратегии с учётом коридора
        # (на пересчитанных под текущую цену квантилях оставшейся доходности).
        # Считаем по ВСЕЙ вселенной прогноза (это лёгкая арифметика квантилей),
        # чтобы сводный дашборд мог ранжировать топ-N по всем бумагам, а не
        # только по VALIDATION_TICKERS (у остальных не будет лишь стат-метрик
        # валидации — White RC / PBO / FDR — они останутся «—»).
        from . import directional
        cost_rt = float(getattr(config, "TFT_COST_RT", 0.08))
        all_fc_tickers = list(panel.infer_tickers)
        dir_rows = directional.build_rows(
            infer_pred_adj, panel.infer_tickers, all_fc_tickers, val_strats,
            cost_rt=cost_rt, coridor=forecasts,
        )
        for r in dir_rows:
            forecasts.setdefault(r["ticker"], {}).setdefault("directional", {})[
                r["strategy"]
            ] = {k: r[k] for k in ("ExpPnL", "Downside", "Upside", "ProbProfit")}

        # Отдельные таблицы (не combined-режим) печатаем по VALIDATION_TICKERS.
        if not quiet and val_tickers:
            _print_strategy_table(forecasts, val_tickers, val_strats)
            val_set = {t.upper() for t in val_tickers}
            val_dir = [r for r in dir_rows if r["ticker"] in val_set]
            if val_dir:
                directional.print_table(val_dir, cost_rt)

    return forecasts
