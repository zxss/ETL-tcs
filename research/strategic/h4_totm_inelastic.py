"""
H4. Календарный макро-приток (McConnell & Xu 2008; неэластичный спрос).

Капитал 90 % месяца лежит в TMON@. Вход в лонг топ-5 ликвидных бумаг строго в
18:35 последнего торгового дня месяца (T−1), выход в 18:20 третьего торгового дня
нового месяца (T+3). Цены берутся с тех же баров, что и в боевом контуре: close
бара 18:30 — цена фазы 18:35, close бара 18:15 — цена фазы 18:20.

Состав корзины определяется оборотом за 60 дней ДО дня входа (сдвиг на день,
чтобы оборот самого дня входа не подглядывал).

Родственный вариант уже проверялся 17.09.2026 (структурный модуль 1Б,
удержание 4 сессии): на выборке разработки −50,04 % годовых, t −1,5. Здесь
проверяется точная параметризация ТЗ — это новое испытание на РАЗРАБОТКЕ.
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

from research import session_calendar as sc                # noqa: E402
from research import sprint2_leadlag as ll                 # noqa: E402
from research.strategic import costs as cst                # noqa: E402
from research.strategic import panel as pn                 # noqa: E402
from research.strategic import validation as va            # noqa: E402
from research.strategic.h1_factor_breadth import Fund, _perf_matrix, capacity_rub   # noqa: E402
from research.structural import common as cmn             # noqa: E402

log = logging.getLogger("research.strategic.h4")

MODULE = "h4_totm_inelastic"
ADV_WINDOW = 60
CONFIGS = {f"n{n}-T+{k}": (n, k) for n in (3, 5, 10) for k in (1, 3)}
HEADLINE = "n5-T+3"


def month_windows(days: list[dt.date], depth: int = 3) -> list[dict]:
    """T−1 — последний торговый день месяца, T+1…T+depth — первые дни следующего."""
    by: dict[tuple, list] = {}
    for d in days:
        by.setdefault((d.year, d.month), []).append(d)
    keys = sorted(by)
    out = []
    for cur, nxt in zip(keys, keys[1:]):
        if (nxt[0] * 12 + nxt[1]) - (cur[0] * 12 + cur[1]) != 1:
            continue
        if not by[cur] or len(by[nxt]) < depth:
            continue
        w = {"month": f"{cur[0]}-{cur[1]:02d}", "t_m1": by[cur][-1]}
        for k in range(1, depth + 1):
            w[f"t_p{k}"] = by[nxt][k - 1]
        out.append(w)
    return out


def basket(feat: pd.DataFrame, day: dt.date, n: int) -> list[str]:
    """Топ-n по среднему обороту за ADV_WINDOW дней до дня входа."""
    hist = feat[(feat["date"] < day) & (feat["date"] >= day - dt.timedelta(days=ADV_WINDOW * 2))]
    if hist.empty:
        return []
    adv = hist.groupby("ticker")["turnover_rub"].mean().sort_values(ascending=False)
    return list(adv.head(n).index)


def run(conn, stage: str, rules: dict, ctx=None, n_trials: int | None = None) -> dict:
    s = rules["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(s["from"]), dt.date.fromisoformat(s["to"])
    model = cst.CostModel(impact_y=rules["costs"]["impact_Y"])
    ctx = ctx or Fund()
    tickers = pn.universe(conn)
    daily = pn.with_turnover(pn.load_daily(conn, tickers, d_from - dt.timedelta(days=260), d_to),
                             pn.load_lots())
    feat = pn.features(daily)
    days = sorted(d for d in daily[daily["ticker"] == pn.INDEX]["date"] if d.weekday() < 5)
    windows = [w for w in month_windows(days) if d_from <= w["t_m1"] <= d_to]
    bars = cmn.union_bars(conn, tickers, d_from - dt.timedelta(days=10), d_to + dt.timedelta(days=15),
                          t0="18:10", t1="18:40")
    fx = feat.set_index(["ticker", "date"])
    log.info("[H4] %s: окон конца месяца %d", stage, len(windows))
    per_config, skipped = {}, {}
    for name, (n_names, hold) in CONFIGS.items():
        rows = []
        for w in windows:
            d0, d1 = w["t_m1"], w[f"t_p{hold}"]
            for tk in basket(feat, d0, n_names):
                b = bars.get(tk)
                p0 = ll.close_on_day(b, d0, sc.EVENING_ENTRY_BAR) if b is not None else None
                p1 = ll.close_on_day(b, d1, sc.SHORT_EXIT_BAR) if b is not None else None
                if not p0 or not p1:
                    skipped["нет цены на баре"] = skipped.get("нет цены на баре", 0) + 1
                    continue
                try:
                    f = fx.loc[(tk, d0)]
                except KeyError:
                    skipped["нет признаков"] = skipped.get("нет признаков", 0) + 1
                    continue
                cost = model.round_trip_pct(tk, float(f["garman_klass_vol"]), pn.POSITION_RUB,
                                            float(f["adv_rub"]), float(f["cs_spread_pct"]))
                if not np.isfinite(cost):
                    skipped["нет оборота"] = skipped.get("нет оборота", 0) + 1
                    continue
                gross = (p1 / p0 - 1.0) * 100.0
                fund = ctx.fund_pct(d0, d1)
                net = gross - cost - fund
                rows.append({"module": MODULE, "variant": name, "ticker": tk, "entry_day": d0,
                             "exit_day": d1, "notional": pn.POSITION_RUB, "gross_pct": gross,
                             "cost_pct": cost, "fund_pct": fund, "net_excess_pct": net,
                             "pnl_excess_rub": pn.POSITION_RUB * net / 100.0,
                             "sigma_pct": float(f["garman_klass_vol"]), "adv_rub": float(f["adv_rub"])})
        per_config[name] = pd.DataFrame(rows)
    tr = per_config[HEADLINE]
    summary = va.summarize(tr, d_from, d_to, 5_000_000.0, n_trials)
    pbo = va.cscv_pbo(_perf_matrix(per_config), s=6)
    cap = capacity_rub(tr, model, 5) if not tr.empty else None
    gate = va.dev_gate(summary, pbo.get("pbo"), cap, rules["gates"]["dev"]) if stage == "dev" \
        else va.holdout_gate(summary, rules["gates"]["holdout"])
    return {"module": MODULE, "stage": stage, "variant": HEADLINE, "summary": summary,
            "pbo": pbo, "capacity_rub": cap, "passed": gate[0], "failed": gate[1],
            "windows": len(windows), "skipped": skipped,
            "grid": {k: {"trades": int(len(v)),
                         "net_excess_pct_trade": float(v["net_excess_pct"].mean()) if len(v) else None,
                         "t": va.t_by_date(v["net_excess_pct"], v["entry_day"]) if len(v) else None}
                     for k, v in per_config.items()},
            "trades": tr}
