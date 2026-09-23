"""
Этап 1 ТЗ — событийный walk-forward бэктест.

Два уровня, потому что они отвечают на разные вопросы.

  Уровень 1 (сигнальный, дневные бары). Есть ли у сигнала преимущество в
  принципе — при исполнении «по рынку», без моделирования очереди в стакане.
  Покрывает всю историю. Если преимущества нет здесь, моделировать исполнение
  бессмысленно.

  Уровень 2 (исполнительный, 5-минутные бары). Что остаётся от преимущества
  после реального исполнения лимитками: доля заливок, отбор в свою сторону
  (adverse selection), фактическое проскальзывание. Правило заливки жёсткое,
  как требует ТЗ: касание цены НЕ считается исполнением, нужен выход цены
  за уровень лимитки.

Метрики портфеля считаются на дневной сетке; годовые величины — при 252
торговых днях.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("audit.backtest")

TRADING_DAYS = 252
LONG_STRATS = {"long_overnight", "intraday_long"}


# ── Метрики ──────────────────────────────────────────────────────────────────

def performance(returns: pd.Series, freq: int = TRADING_DAYS) -> dict:
    """Sharpe, Profit Factor, максимальная просадка, win rate по ряду доходностей (%)."""
    r = returns.dropna()
    if len(r) < 20:
        return {"n": len(r)}
    mean, sd = r.mean(), r.std(ddof=1)
    equity = (1.0 + r / 100.0).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    years = len(r) / freq
    total = float(equity.iloc[-1] - 1.0)
    return {
        "n": int(len(r)),
        "total_return_pct": total * 100.0,
        "cagr_pct": ((1 + total) ** (1 / years) - 1) * 100.0 if years > 0 else np.nan,
        "mean_daily_pct": float(mean),
        "vol_annual_pct": float(sd * np.sqrt(freq)),
        "sharpe": float(mean / sd * np.sqrt(freq)) if sd > 0 else np.nan,
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "max_drawdown_pct": float(dd.min() * 100.0),
        "win_rate": float((r > 0).mean()),
        "best_day_pct": float(r.max()),
        "worst_day_pct": float(r.min()),
        "equity": equity,
        "drawdown": dd,
    }


# ── Уровень 1: сигнальный бэктест ────────────────────────────────────────────

def signal_backtest(df: pd.DataFrame, score_col: str = "final_score",
                    top_n: int = 10, ret_col: str = "realized_net",
                    weighting: str = "equal",
                    min_score: float | None = None,
                    long_only: bool = False,
                    allowed_strats: set[str] | None = None,
                    short_blocked: set[str] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Портфель «топ-N по рейтингу, держим один день».

    weighting:
      equal        — равные веса
      risk_parity  — вес ∝ 1/волатильность прогнозного коридора (прокси стопа)

    short_blocked — тикеры, по которым шорт физически недоступен у брокера;
    короткие сигналы по ним отбрасываются (иначе бэктест торгует невозможное).
    Возвращает (дневная доходность портфеля %, журнал сделок).
    """
    d = df.dropna(subset=[score_col, ret_col]).copy()
    if allowed_strats:
        d = d[d["strategy"].isin(allowed_strats)]
    if long_only:
        d = d[d["strategy"].isin(LONG_STRATS)]
    if short_blocked:
        is_short = ~d["strategy"].isin(LONG_STRATS)
        d = d[~(is_short & d["ticker"].isin(short_blocked))]
    if min_score is not None:
        d = d[d[score_col] >= min_score]
    if d.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    # По каждому тикеру и дню оставляем одну лучшую стратегию — как в дашборде.
    d = (d.sort_values([score_col], ascending=False)
           .groupby(["asof_date", "ticker"], as_index=False).head(1))

    picks = (d.sort_values([ "asof_date", score_col], ascending=[True, False])
               .groupby("asof_date", as_index=False).head(top_n).copy())

    if weighting == "risk_parity":
        # Вес обратно пропорционален ширине прогнозного коридора (прокси риска).
        picks["risk"] = picks["range_pct"].clip(lower=0.5)
        picks["w"] = 1.0 / picks["risk"]
    else:
        picks["w"] = 1.0
    picks["w"] = picks["w"] / picks.groupby("asof_date")["w"].transform("sum")
    picks["contrib"] = picks["w"] * picks[ret_col]

    daily = picks.groupby("asof_date")["contrib"].sum().sort_index()
    return daily, picks


def benchmark_returns(daily_bars: pd.DataFrame, ticker: str = "IMOEX",
                      forward: bool = True) -> pd.Series:
    """Buy & hold индекса — эталон для сравнения альфы.

    forward=True приводит эталон к соглашению стратегии: значение с индексом t —
    доходность t → t+1, то есть та же, которую зарабатывает сигнал, выданный на
    дату t. Без этого сдвига ряды разъезжаются на день, и бета получается
    околонулевой даже у полностью длинного портфеля.
    """
    d = daily_bars[daily_bars["ticker"] == ticker].sort_values("date")
    r = (d["close"].pct_change() * 100.0)
    r.index = pd.DatetimeIndex(d["date"])
    if forward:
        r = r.shift(-1)
    return r.dropna()


def alpha_vs_benchmark(strategy_daily: pd.Series, bench_daily: pd.Series) -> dict:
    """Годовая альфа и бета относительно эталона на общих датах."""
    idx = strategy_daily.index.intersection(bench_daily.index)
    if len(idx) < 30:
        return {}
    s, b = strategy_daily.reindex(idx), bench_daily.reindex(idx)
    var = b.var(ddof=1)
    beta = float(s.cov(b) / var) if var > 0 else np.nan
    alpha_daily = float(s.mean() - beta * b.mean())
    return {
        "beta": beta,
        "alpha_daily_pct": alpha_daily,
        "alpha_annual_pct": alpha_daily * TRADING_DAYS,
        "strategy_annual_pct": float(s.mean() * TRADING_DAYS),
        "bench_annual_pct": float(b.mean() * TRADING_DAYS),
        "excess_annual_pct": float((s.mean() - b.mean()) * TRADING_DAYS),
        "corr": float(s.corr(b)),
        "n_days": len(idx),
    }


# ── Уровень 2: исполнительный бэктест на 5-минутках ──────────────────────────

def simulate_limit_fills(signals: pd.DataFrame, bars5m: pd.DataFrame,
                         entry_frac: float = 0.2, tp_frac: float = 1.0,
                         stop_frac: float = 1.0,
                         wait_minutes: int | None = None) -> pd.DataFrame:
    """Моделирует лимитный вход и выход по 5-минутному пути цены.

    Правило заливки (ТЗ): касание не считается. Для лонга нужен бар, у которого
    close СТРОГО ниже цены лимитки; для шорта — строго выше. Это консервативное
    предположение о позиции в очереди заявок.

    signals: asof_date, ticker, strategy, anchor_price, f_low, f_high.
    Вход ищется в торговый день, СЛЕДУЮЩИЙ за asof_date.
    """
    bars5m = bars5m.sort_values(["ticker", "ts_msk"])
    by_tk_day = {k: v for k, v in bars5m.groupby(["ticker", "date"], sort=False)}
    all_days = np.sort(bars5m["date"].unique())

    out = []
    for s in signals.itertuples():
        nxt = all_days[all_days > np.datetime64(s.asof_date)]
        if len(nxt) == 0:
            continue
        day = nxt[0]
        g = by_tk_day.get((s.ticker, pd.Timestamp(day)))
        if g is None or len(g) < 10:
            continue

        is_long = s.strategy in LONG_STRATS
        anchor = float(s.anchor_price)
        lo, hi = float(s.f_low), float(s.f_high)

        # Цена входа: лимитка внутри коридора, entry_frac — доля пути к краю.
        if is_long:
            entry = anchor - entry_frac * (anchor - lo)
            target = entry + tp_frac * (hi - entry)
            stop = entry - stop_frac * (entry - lo)
        else:
            entry = anchor + entry_frac * (hi - anchor)
            target = entry - tp_frac * (entry - lo)
            stop = entry + stop_frac * (hi - entry)

        g = g.reset_index(drop=True)
        if wait_minutes:
            g = g.head(max(1, wait_minutes // 5))

        closes = g["close"].to_numpy(float)
        highs = g["high"].to_numpy(float)
        lows = g["low"].to_numpy(float)
        ts = g["ts_msk"].to_numpy()

        fill_i = None
        for i, c in enumerate(closes):
            if (is_long and c < entry) or ((not is_long) and c > entry):
                fill_i = i
                break

        rec = {
            "asof_date": s.asof_date, "trade_date": pd.Timestamp(day),
            "ticker": s.ticker, "strategy": s.strategy, "is_long": is_long,
            "anchor": anchor, "entry": entry, "target": target, "stop": stop,
            "filled": fill_i is not None,
        }

        if fill_i is None:
            out.append(rec)
            continue

        # Стоящая в стакане лимитка исполняется ПО СВОЕЙ ЦЕНЕ, а не по цене
        # закрытия бара: брать closes[fill_i] значило бы дарить себе улучшение
        # цены, которого в реальности нет. Спред учтён отдельно в cost_rt.
        fill_price = entry
        rec["fill_price"] = fill_price
        rec["fill_time"] = ts[fill_i]
        rec["fill_bar"] = int(fill_i)
        # Насколько глубоко цена прошла за лимитку — мера того, что заявку
        # «переехали»: положительное значение означает, что рынок ушёл дальше
        # в нашу сторону входа, то есть вход был против движения.
        rec["overshoot_pct"] = abs(closes[fill_i] / entry - 1.0) * 100.0

        # Отбор в свою сторону: куда пошла цена сразу после заливки (6 баров = 30 мин).
        j = min(fill_i + 6, len(closes) - 1)
        move = (closes[j] / fill_price - 1.0) * 100.0
        rec["move_30m_pct"] = move if is_long else -move
        rec["adverse"] = rec["move_30m_pct"] < 0

        # Выход: что случится раньше — цель или стоп; иначе закрытие сессии.
        exit_price, exit_reason = closes[-1], "session_close"
        for i in range(fill_i + 1, len(closes)):
            if is_long:
                if highs[i] >= target:
                    exit_price, exit_reason = target, "target"
                    break
                if lows[i] <= stop:
                    exit_price, exit_reason = stop, "stop"
                    break
            else:
                if lows[i] <= target:
                    exit_price, exit_reason = target, "target"
                    break
                if highs[i] >= stop:
                    exit_price, exit_reason = stop, "stop"
                    break
        gross = (exit_price / fill_price - 1.0) * 100.0
        rec["exit_price"] = exit_price
        rec["exit_reason"] = exit_reason
        rec["gross_pct"] = gross if is_long else -gross
        out.append(rec)

    return pd.DataFrame(out)


def execution_report(fills: pd.DataFrame, cost_map: dict | None = None) -> dict:
    """Сводка исполнения: доля заливок, adverse selection, PnL нетто."""
    if fills.empty:
        return {}
    f = fills
    done = f[f["filled"] == True]  # noqa: E712
    res = {
        "signals": int(len(f)),
        "filled": int(len(done)),
        "fill_rate": float(len(done) / len(f)),
    }
    if done.empty:
        return res
    cost = done["ticker"].map(lambda t: (cost_map or {}).get(t, 0.08)) if cost_map \
        else pd.Series(0.08, index=done.index)
    net = done["gross_pct"] - cost

    # Отбор в свою сторону меряем только там, где после заливки реально осталось
    # 30 минут сессии: у заливок в последних барах окно вырождается в ноль и
    # движение тождественно равно нулю, что занижало бы долю неблагоприятных.
    adv = done[done["fill_bar"] <= done["fill_bar"].max() - 6]
    adv_share = float((adv["move_30m_pct"] < 0).mean()) if len(adv) else float("nan")

    res.update({
        "adverse_share": adv_share,
        "adverse_n": int(len(adv)),
        "mean_move_30m_pct": float(done["move_30m_pct"].mean()),
        "gross_mean_pct": float(done["gross_pct"].mean()),
        "net_mean_pct": float(net.mean()),
        "net_total_pct": float(net.sum()),
        "win_rate": float((net > 0).mean()),
        "exit_mix": done["exit_reason"].value_counts(normalize=True).to_dict(),
        "mean_overshoot_pct": float(done["overshoot_pct"].mean()),
        "median_fill_bar": float(done["fill_bar"].median()),
    })
    return res
