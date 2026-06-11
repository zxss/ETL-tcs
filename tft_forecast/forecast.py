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

    model = TFT(n_feat, n_tk, hidden=hidden)
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
    """
    from .dataset import LOOKBACK
    n_tgt = panel.Y.shape[1]
    nq = len(QUANTILES)
    M = panel.infer_X.shape[0]

    W = panel.W
    cutoff = np.quantile(W, 0.85)
    train_mask = W < (cutoff - LOOKBACK)
    val_idx = np.where(W >= cutoff)[0]
    if len(val_idx) < 1 or int(train_mask.sum()) < 10:
        train_mask = np.ones(len(W), bool)              # короткий ряд — без сплита
        val_idx = np.arange(len(W))

    # квантильный источник: только обучающий период
    by_tk_train: dict[int, list[np.ndarray]] = {}
    for t, y in zip(panel.T[train_mask], panel.Y[train_mask]):
        by_tk_train.setdefault(int(t), []).append(y)
    global_train_Y = panel.Y[train_mask]

    def _q(arr):
        return np.stack([np.quantile(arr[:, k], QUANTILES) for k in range(n_tgt)], axis=0)

    def _q_for(tk_i):
        arr = np.asarray(by_tk_train.get(int(tk_i), []), dtype=np.float64)
        if len(arr) < 20:
            arr = global_train_Y
        return _q(arr)

    infer_pred = np.zeros((M, n_tgt, nq), dtype=np.float64)
    for m, tk_i in enumerate(panel.infer_T):
        infer_pred[m] = _q_for(tk_i)

    val_pred = np.zeros((len(val_idx), n_tgt, nq), dtype=np.float64)
    for j, idx in enumerate(val_idx):
        val_pred[j] = _q_for(panel.T[idx])
    return infer_pred, val_pred, panel.T[val_idx], panel.Y[val_idx]


# ── Калибровка покрытия ────────────────────────────────────────────────────────

def _coverage(val_pred, val_T, val_Y):
    """
    Эмпирическое покрытие интервала [low_q0.1, high_q0.9]:
    доля дней, где реальные next_low/high попали внутрь прогноза.
    Возвращает (per_ticker: dict idx→prob, global_prob).
    """
    low_pred = val_pred[:, 0, Q_LOW]
    high_pred = val_pred[:, 1, Q_HIGH]
    actual_low = val_Y[:, 0]
    actual_high = val_Y[:, 1]
    hit = (actual_low >= low_pred) & (actual_high <= high_pred)

    global_prob = float(hit.mean()) if len(hit) else float("nan")
    per_tk: dict[int, float] = {}
    for t in np.unique(val_T):
        mask = val_T == t
        if mask.sum() >= 15:
            per_tk[int(t)] = float(hit[mask].mean())
    return per_tk, global_prob


# ── Сборка прогнозов ───────────────────────────────────────────────────────────

def _assemble(frames, panel, infer_pred, per_tk_cov, global_cov, quotes=None):
    """
    Собирает прогноз диапазона. Якорная цена — актуальная Last Price (если есть
    котировка), иначе последнее закрытие. ForecastLow/High/RangePct считаются
    ОТНОСИТЕЛЬНО якорной цены, а не вчерашнего закрытия.
    """
    quotes = quotes or {}
    fr_by_tk = {f.ticker: f for f in frames}
    out = {}
    tk_index = {tk: i for i, tk in enumerate(panel.tickers)}
    for m, tk in enumerate(panel.infer_tickers):
        f = fr_by_tk[tk]
        low_pct = float(infer_pred[m, 0, Q_LOW])    # консервативная нижняя граница
        high_pct = float(infer_pred[m, 1, Q_HIGH])  # консервативная верхняя граница
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
        out[tk] = {
            "ForecastLow": forecast_low,
            "ForecastHigh": forecast_high,
            "RangePct": range_pct,
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
    forecasts = _assemble(frames, panel, infer_pred, per_tk_cov, global_cov,
                          quotes=ctx.quotes)

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

    log.info("Прогноз готов (backend=%s, global coverage=%.0f%%).", backend, global_cov * 100)
    if not quiet:
        _print_full(forecasts)

    val_tickers = getattr(config, "VALIDATION_TICKERS", [])
    val_strats = getattr(config, "VALIDATION_STRATS", [])
    if val_tickers and val_strats:
        if not quiet:
            _print_strategy_table(forecasts, val_tickers, val_strats)

        # направленный прогноз: ожидаемый PnL стратегии с учётом коридора
        # (на пересчитанных под текущую цену квантилях оставшейся доходности)
        from . import directional
        cost_rt = float(getattr(config, "TFT_COST_RT", 0.08))
        dir_rows = directional.build_rows(
            infer_pred_adj, panel.infer_tickers, val_tickers, val_strats,
            cost_rt=cost_rt, coridor=forecasts,
        )
        if dir_rows:
            if not quiet:
                directional.print_table(dir_rows, cost_rt)
            for r in dir_rows:
                forecasts.setdefault(r["ticker"], {}).setdefault("directional", {})[
                    r["strategy"]
                ] = {k: r[k] for k in ("ExpPnL", "Downside", "Upside", "ProbProfit")}

    return forecasts
