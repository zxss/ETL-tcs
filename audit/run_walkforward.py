"""
Драйвер walk-forward реплея (ТЗ, Этап 1).

Схема: скользящее окно 12 месяцев обучения → 2 месяца OOS → сдвиг на 1 месяц.
Модель, обученная на данных строго до t0, применяется к каждому торговому дню
OOS-периода без переобучения; поле days_since_fit позволяет отдельно посмотреть,
как быстро деградирует прогноз по мере старения модели.

Для непересекающейся кривой доходности используется только ПЕРВЫЙ месяц каждого
OOS-окна (fold_primary=True); второй месяц нужен для замера деградации.

Результат: audit/out/walkforward.csv — по строке на (дата, тикер, стратегия) с
прогнозом, рыночным контекстом, ликвидностью и фактическим исходом.

Запуск:
    python3 -m audit.run_walkforward --epochs 30 --start 2025-05-01
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import costs, data, replay  # noqa: E402

log = logging.getLogger("audit.walkforward")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def realized_outcomes(daily: pd.DataFrame) -> pd.DataFrame:
    """Фактические примитивные доходности СЛЕДУЮЩЕГО дня для каждой даты."""
    rows = []
    for tk, g in daily.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        nxt_o = g["open"].shift(-1)
        nxt_c = g["close"].shift(-1)
        rows.append(pd.DataFrame({
            "ticker": tk,
            "asof_date": g["date"],
            "r_overnight": (nxt_o / g["close"] - 1.0) * 100.0,
            "r_intraday": (nxt_c / nxt_o - 1.0) * 100.0,
            "r_total": (nxt_c / g["close"] - 1.0) * 100.0,
            "next_low": g["low"].shift(-1),
            "next_high": g["high"].shift(-1),
            "next_open": nxt_o,
            "next_close": nxt_c,
        }))
    return pd.concat(rows, ignore_index=True)


STRAT_SIGN = {"long_overnight": ("r_overnight", +1.0),
              "intraday_long": ("r_intraday", +1.0),
              "intraday_short": ("r_intraday", -1.0),
              "short_hold": ("r_total", -1.0)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward реплей дашборда")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--oos-months", type=int, default=2)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--start", default=None,
                   help="Первая дата обучения (по умолчанию: начало данных + train_months)")
    p.add_argument("--out", default=os.path.join(OUT, "walkforward.csv"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)

    log.info("Загрузка баров...")
    daily_all = data.load_daily()
    daily = daily_all[daily_all.ticker != "IMOEX"].copy()
    store = replay.RawStore(daily_all)
    instr = costs.load_instruments()
    lots = instr.set_index("ticker")["lot"].to_dict()

    # Реалистичные издержки по тикерам (Блок 2.1); если матрицы нет — плоские 0.08.
    cm_path = os.path.join(OUT, "cost_matrix.csv")
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path)
        cost_map = dict(zip(cm["ticker"], cm["cost_rt_base"]))
        log.info("Матрица издержек: медиана %.3f%%", np.nanmedian(list(cost_map.values())))
    else:
        cost_map = {}

    tdays = data.trading_days(daily)
    outcomes = realized_outcomes(daily)
    tickers = sorted(daily["ticker"].unique())

    first = pd.Timestamp(args.start) if args.start else (
        tdays[0] + pd.DateOffset(months=args.train_months))
    last = tdays[-1]

    fit_dates = []
    t = first
    while t <= last:
        fit_dates.append(t)
        t = t + pd.DateOffset(months=args.step_months)
    log.info("Окон обучения: %d (%s → %s)", len(fit_dates),
             fit_dates[0].date(), fit_dates[-1].date())

    frames = []
    for k, t0 in enumerate(fit_dates, 1):
        train_end = tdays[tdays <= t0]
        if len(train_end) == 0:
            continue
        train_end = train_end[-1]

        s = time.time()
        fm = replay.fit(store, train_end, epochs=args.epochs, hidden=args.hidden,
                        tickers=tickers, train_months=args.train_months)
        if fm is None:
            log.warning("[%d/%d] %s — обучение не удалось", k, len(fit_dates), train_end.date())
            continue

        oos_end = t0 + pd.DateOffset(months=args.oos_months)
        primary_end = t0 + pd.DateOffset(months=args.step_months)
        oos_days = tdays[(tdays > train_end) & (tdays <= oos_end)]
        log.info("[%d/%d] fit@%s  %.0fs  OOS=%d дней",
                 k, len(fit_dates), train_end.date(), time.time() - s, len(oos_days))

        for d in oos_days:
            fc = replay.forecast_asof(fm, store, d, cost_rt=cost_map or 0.08)
            if fc.empty:
                continue
            mc = replay.market_context_asof(store, d, tickers)
            lq = replay.liquidity_asof(store, d, tickers, lots)

            fc["regime"] = mc["regime"]
            fc["breadth"] = mc["breadth"]
            for col in ("rs", "vol_spike", "atr_pctl", "gap_down_prob"):
                fc[col] = fc["ticker"].map(
                    lambda tk, c=col: (mc["per_ticker"].get(tk) or {}).get(c))
            fc["liq_score"] = fc["ticker"].map(
                lambda tk: (lq.get(tk) or {}).get("liq_score"))
            fc["adv_rub"] = fc["ticker"].map(
                lambda tk: (lq.get(tk) or {}).get("adv_rub"))
            fc["fit_date"] = train_end
            fc["days_since_fit"] = (d - train_end).days
            fc["fold"] = k
            fc["fold_primary"] = d <= primary_end
            frames.append(fc)

    if not frames:
        log.error("Нет результатов реплея.")
        return 1

    df = pd.concat(frames, ignore_index=True)

    # Фактический исход стратегии.
    df = df.merge(outcomes, on=["ticker", "asof_date"], how="left")
    df["realized"] = np.nan
    for strat, (col, sign) in STRAT_SIGN.items():
        m = df["strategy"] == strat
        df.loc[m, "realized"] = df.loc[m, col] * sign
    df["realized_net"] = df["realized"] - df["cost_rt"]

    df.to_csv(args.out, index=False)
    log.info("Сохранено %d строк → %s", len(df), args.out)
    log.info("Дат: %d, тикеров: %d, стратегий: %d",
             df["asof_date"].nunique(), df["ticker"].nunique(),
             df["strategy"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
