"""
H2. Агрегированный дисбаланс потока заявок — ПРОКСИ.

ВАЖНОЕ ОГРАНИЧЕНИЕ ДАННЫХ. Настоящий OFI (Cont, Kukanov, Stoikov 2014) считается
по изменениям первого уровня СТАКАНА (bid/ask, цена и объём). В базе проекта
стакана нет, и исторический стакан T-Invest API не отдаёт — только свечи. Поэтому
проверяется не OFI, а его бар-прокси: знаковый оборот по правилу тика
(Lee–Ready без котировок) за основную сессию 10:00–16:30, нормированный на общий
оборот, плюс подтверждение ценой выше VWAP сессии.

Прокси слабее оригинала: правило тика путает инициатора сделки на 10–20 % сделок
и полностью теряет отменённые заявки, которые и составляют половину OFI. Любой
результат этого модуля нельзя выдавать за проверку статьи Cont и соавторов.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.strategic import costs as sc                 # noqa: E402
from research.strategic import panel as pn                 # noqa: E402
from research.strategic import validation as va            # noqa: E402
from research.strategic.h1_factor_breadth import Fund      # noqa: E402

log = logging.getLogger("research.strategic.h2")

MODULE = "h2_aggregated_ofi"
MSK = dt.timezone(dt.timedelta(hours=3))
Z_WINDOW = 60
VARIANTS = {"z2.0-h1": (2.0, 1), "z2.0-h2": (2.0, 2), "z2.5-h1": (2.5, 1), "z1.5-h1": (1.5, 1)}
HEADLINE = "z2.0-h1"

SESSION_SQL = """
WITH b AS (
  SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow')::date AS d, ts, close, volume,
         lag(close) OVER (PARTITION BY ticker, (ts AT TIME ZONE 'Europe/Moscow')::date
                          ORDER BY ts) AS pc
  FROM {table}
  WHERE ticker = ANY(%(tk)s) AND close > 0 AND volume > 0 AND ts >= %(f)s AND ts < %(t)s
    AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN TIME '10:00' AND TIME '16:30'
    AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
)
SELECT ticker, d AS date,
       sum(volume * close) AS turnover,
       sum(sign(close - pc) * volume * close) AS signed_turnover,
       sum(volume * close) / NULLIF(sum(volume), 0) AS vwap,
       (array_agg(close ORDER BY ts DESC))[1] AS session_close
FROM b WHERE pc IS NOT NULL
GROUP BY ticker, d
"""


def session_flow(conn, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    """Прокси-OFI по сессии: знаковый оборот / общий оборот, VWAP, закрытие."""
    parts = []
    for table, lo, hi in pn.TABLES:
        a, b = max(lo, d_from), min(hi, d_to)
        if a > b:
            continue
        parts.append(pd.read_sql(SESSION_SQL.format(table=table), conn, params={
            "tk": list(tickers),
            "f": dt.datetime.combine(a, dt.time(), MSK),
            "t": dt.datetime.combine(b + dt.timedelta(days=1), dt.time(), MSK)}))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("turnover", "signed_turnover", "vwap", "session_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ofi"] = df["signed_turnover"] / df["turnover"].replace(0.0, np.nan)
    out = []
    for tk, g in df.sort_values("date").groupby("ticker", sort=False):
        g = g.copy()
        m = g["ofi"].rolling(Z_WINDOW, min_periods=40).mean().shift(1)
        s = g["ofi"].rolling(Z_WINDOW, min_periods=40).std().shift(1)
        g["ofi_z"] = (g["ofi"] - m) / s.replace(0.0, np.nan)
        g["above_vwap"] = g["session_close"] > g["vwap"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def trades(flow: pd.DataFrame, feat: pd.DataFrame, ctx, model: sc.CostModel,
           z_min: float, hold: int, variant: str,
           position_rub: float = pn.POSITION_RUB) -> pd.DataFrame:
    """Вход на закрытии дня сигнала, выход через hold торговых дней."""
    px = feat.set_index(["ticker", "date"])
    days_by_ticker = {tk: sorted(g["date"]) for tk, g in feat.groupby("ticker")}
    rows = []
    sig = flow[(flow["ofi_z"] >= z_min) & flow["above_vwap"]]
    for r in sig.itertuples(index=False):
        days = days_by_ticker.get(r.ticker)
        if not days or r.date not in days:
            continue
        i = days.index(r.date)
        if i + hold >= len(days):
            continue
        d1 = days[i + hold]
        try:
            a, b = px.loc[(r.ticker, r.date)], px.loc[(r.ticker, d1)]
        except KeyError:
            continue
        if not np.isfinite(a["adv_rub"]) or not np.isfinite(a["garman_klass_vol"]):
            continue
        cost = model.round_trip_pct(r.ticker, a["garman_klass_vol"], position_rub,
                                    a["adv_rub"], a["cs_spread_pct"])
        if not np.isfinite(cost):
            continue
        gross = (b["close"] / a["close"] - 1.0) * 100.0
        fund = ctx.fund_pct(r.date, d1)
        net = gross - cost - fund
        rows.append({"module": MODULE, "variant": variant, "ticker": r.ticker,
                     "entry_day": r.date, "exit_day": d1, "notional": position_rub,
                     "gross_pct": gross, "cost_pct": cost, "fund_pct": fund,
                     "net_excess_pct": net, "pnl_excess_rub": position_rub * net / 100.0,
                     "ofi_z": r.ofi_z, "sigma_pct": float(a["garman_klass_vol"]),
                     "adv_rub": float(a["adv_rub"])})
    return pd.DataFrame(rows)


def run(conn, stage: str, rules: dict, ctx=None, n_trials: int | None = None) -> dict:
    from research.strategic.h1_factor_breadth import capacity_rub, prepare, _perf_matrix
    s = rules["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])
    model = sc.CostModel(impact_y=rules["costs"]["impact_Y"])
    ctx = ctx or Fund()
    feat = prepare(conn, d_from, d_to)
    flow = session_flow(conn, sorted(feat["ticker"].unique()), d_from, d_to)
    log.info("[H2] %s: дней потока %d, бумаг %d", stage, flow["date"].nunique(), flow["ticker"].nunique())
    per_config = {}
    for name, (z, hold) in VARIANTS.items():
        per_config[name] = trades(flow, feat, ctx, model, z, hold, name)
    tr = per_config[HEADLINE]
    summary = va.summarize(tr, d_from, d_to, 5_000_000.0, n_trials)
    pbo = va.cscv_pbo(_perf_matrix(per_config))
    cap = capacity_rub(tr, model, 5) if not tr.empty else None
    gate = va.dev_gate(summary, pbo.get("pbo"), cap, rules["gates"]["dev"]) if stage == "dev" \
        else va.holdout_gate(summary, rules["gates"]["holdout"])
    return {"module": MODULE, "stage": stage, "variant": HEADLINE, "summary": summary,
            "pbo": pbo, "capacity_rub": cap, "passed": gate[0], "failed": gate[1],
            "grid": {k: {"trades": int(len(v)),
                         "net_excess_pct_trade": float(v["net_excess_pct"].mean()) if len(v) else None,
                         "t": va.t_by_date(v["net_excess_pct"], v["entry_day"]) if len(v) else None}
                     for k, v in per_config.items()},
            "data_limitation": rules["hypotheses"][MODULE]["data_limitation"], "trades": tr}
