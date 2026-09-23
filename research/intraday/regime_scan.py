"""
Автоматический поиск внутридневных режимов ускорения (ТЗ пользователя
23.09.2026, «Анализ внутридневной динамики и поиск точек входа»).

Отличие от `research/intraday_hypotheses.py` (15.09.2026, 4 гипотезы с
заранее прописанными правилами — все провалили DoR и на holdout, и на dev):
здесь правило НЕ прописывается руками, а ищется автоматически по сетке
«часовое окно × порог ускорения», и лишь то, что переживает поправку Холма
и CPCV, превращается в проверяемое торговое правило. Ключевой результат —
не сигнал, а сетка признаков (какие окна дня вообще статистически
отличаются от нормы) — сигнал строится из неё вторым шагом.

Определение «ускорения» в часовом окне [t, t+1ч): |доходность бара| относительно
СОБСТВЕННОГО исторического разброса той же бумаги в ТОМ ЖЕ часовом окне
(скользящее, 40 предыдущих сессий, без забегания вперёд) — z-оценка.
«Что происходит после» — доходность от конца этого часа до конца сессии
(18:15, тот же LAST_BAR, что в intraday_hypotheses).

Локальные 5-минутки есть только с 30.11.2025 (см. память
local-db-research-data-gaps) — вся история здесь умещается в этот период,
целиком ДО смены расписания 14.09.2026 (session_calendar.NEW_SCHEDULE_FROM),
так что старое/новое расписание сессии смешивать не нужно.

Запуск: python -m research.intraday.regime_scan
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                     # noqa: E402
from research import intraday_hypotheses as ih            # noqa: E402
from research import news_event_study as ns               # noqa: E402
from research import session_calendar as sc                # noqa: E402
from research import short_rule as sr                      # noqa: E402
from research.strategic import validation as va             # noqa: E402

log = logging.getLogger("research.intraday.regime_scan")

OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "intraday_regime")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
POSITION_RUB = 10_000.0
BUCKET_HOURS = [(dt.time(h, 0), dt.time(h + 1, 0)) for h in range(10, 18)]   # 10:00..18:00, 8 окон
Z_WINDOW, Z_MIN_PERIODS = 40, 20
Z_THRESH = 2.0                              # порог заморожен до прогона
CPCV = {"blocks": 5, "k_test": 2, "embargo_days": 5}


def bucket_returns(bars: pd.DataFrame, days: list[dt.date]) -> pd.DataFrame:
    """(ticker, date, bucket) → доходность часового окна (close/open первого-последнего бара, %)."""
    u = bars[bars["ticker"].isin(ns.UNIVERSE)].copy()
    rows = []
    for bi, (t0, t1) in enumerate(BUCKET_HOURS):
        m = (u["tm"].dt.time >= t0) & (u["tm"].dt.time < t1)
        g = u[m]
        if g.empty:
            continue
        agg = g.groupby(["ticker", "d"]).agg(open=("open", "first"), close=("close", "last"),
                                             n=("close", "size"))
        agg = agg[agg["n"] >= 8]                            # окно почти полное (из 12 баров)
        agg["ret"] = (agg["close"] / agg["open"] - 1.0) * 100.0
        agg["bucket"] = bi
        rows.append(agg.reset_index()[["ticker", "d", "bucket", "ret"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aftermath(bars: pd.DataFrame, days: list[dt.date]) -> pd.Series:
    """(ticker, date, bucket) → доходность от конца бакета до конца сессии (18:15), %."""
    u = bars[bars["ticker"].isin(ns.UNIVERSE)].copy()
    by = {(tk, d): g.set_index("tm") for (tk, d), g in u.groupby(["ticker", "d"])}
    out = {}
    for bi, (_, t1) in enumerate(BUCKET_HOURS):
        for (tk, d), g in by.items():
            after = g[g.index.time >= t1]
            before = g[g.index.time < t1]
            if after.empty or before.empty:
                continue
            out[(tk, d, bi)] = (float(after["close"].iloc[-1]) / float(before["close"].iloc[-1]) - 1.0) * 100.0
    return pd.Series(out)


def zscore(ret: pd.DataFrame) -> pd.DataFrame:
    """z бакетной доходности относительно собственной истории (той же бумаги, того же бакета)."""
    ret = ret.sort_values("d").copy()
    out = []
    for (tk, bi), g in ret.groupby(["ticker", "bucket"]):
        g = g.sort_values("d")
        roll_mean = g["ret"].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
        roll_std = g["ret"].shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std(ddof=1)
        z = (g["ret"] - roll_mean) / roll_std.replace(0.0, np.nan)
        out.append(pd.DataFrame({"ticker": tk, "bucket": bi, "d": g["d"], "ret": g["ret"], "z": z}))
    return pd.concat(out, ignore_index=True)


def scan(panel: pd.DataFrame) -> dict:
    """Тест по каждому бакету: корр(z в бакете, доходность до конца сессии) — Холм по 8 бакетам."""
    res, pvals = {}, {}
    for bi in sorted(panel["bucket"].unique()):
        g = panel[panel["bucket"] == bi].dropna(subset=["z", "aftermath"])
        per_day = g.groupby("d").apply(lambda x: float(np.corrcoef(x["z"], x["aftermath"])[0, 1])
                                       if len(x) >= 5 and x["z"].std() > 0 else np.nan)
        per_day = per_day.dropna()
        t, p = sr._t_p(per_day) if len(per_day) >= 10 else (None, None)
        res[bi] = {"days": int(len(per_day)), "mean_corr": float(per_day.mean()) if len(per_day) else None,
                   "t": t, "p": p, "n_obs": int(len(g))}
        if p is not None:
            pvals[bi] = p
    items = sorted((v, k) for k, v in pvals.items())
    run, m = 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        res[k]["p_holm"] = run
    return res


def rule_trades(panel: pd.DataFrame, bucket: int, thresh: float, direction: int,
                spreads: dict) -> pd.DataFrame:
    """direction=+1 momentum (по знаку z), -1 contrarian. Вход на следующем баре после
    сигнала (открытие бакета+1ч), выход — конец сессии (без забегания, честная задержка)."""
    g = panel[(panel["bucket"] == bucket) & (panel["z"].abs() >= thresh)].dropna(subset=["aftermath"])
    if g.empty:
        return pd.DataFrame()
    rows = []
    for r in g.itertuples(index=False):
        side = direction * (1 if r.z > 0 else -1)
        gross = side * r.aftermath
        cost = cm.round_trip(r.ticker, "base", spreads)
        net = gross - cost
        rows.append({"entry_day": r.d, "exit_day": r.d, "ticker": r.ticker, "notional": POSITION_RUB,
                     "gross_pct": gross, "cost_pct": cost, "net_excess_pct": net,
                     "pnl_excess_rub": POSITION_RUB * net / 100.0})
    return pd.DataFrame(rows)


def oos_best_rule(panel: pd.DataFrame, bucket: int, spreads: dict) -> dict:
    """CPCV по датам: направление (momentum/contrarian) выбирается на train по net-доходу,
    сделки — на test. Порог фиксирован (Z_THRESH) — заморожен до прогона."""
    dates = sorted(panel["d"].unique())
    if len(dates) < 30:
        return {"oos": pd.DataFrame(), "note": "мало дат для CPCV"}
    splits = va.cpcv_splits(dates, CPCV["blocks"], CPCV["k_test"], CPCV["embargo_days"], label_days=1)
    oos_rows, perf = [], []
    for sp in splits:
        train = panel[panel["d"].isin(sp["train"])]
        test = panel[panel["d"].isin(sp["test"])]
        cand = {}
        for d in (1, -1):
            tr_tr = rule_trades(train, bucket, Z_THRESH, d, spreads)
            cand[d] = tr_tr["net_excess_pct"].mean() if not tr_tr.empty else -np.inf
        best_dir = max(cand, key=cand.get)
        te_tr = rule_trades(test, bucket, Z_THRESH, best_dir, spreads)
        if not te_tr.empty:
            oos_rows.append(te_tr)
        perf.append({"momentum": cand[1], "contrarian": cand[-1]})
    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    return {"oos": oos, "perf": pd.DataFrame(perf)}


def run(conn) -> dict:
    d_from, d_to = dt.date(2025, 12, 1), dt.date(2026, 9, 9)
    bars, _last = ih.load(conn, d_from, d_to)
    days = ih.exec_days(bars)
    log.info("торговых дней %d, тикеров %d", len(days), bars["ticker"].nunique())
    ret = bucket_returns(bars, days)
    z = zscore(ret)
    aft = aftermath(bars, days).rename("aftermath")
    panel = z.merge(aft.reset_index().rename(columns={"level_0": "ticker", "level_1": "d", "level_2": "bucket"}),
                    on=["ticker", "d", "bucket"], how="inner")
    panel = panel[panel["d"] >= d_from]
    log.info("панель: %d наблюдений", len(panel))

    scan_res = scan(panel)
    spreads = cm.load_spreads() if os.path.exists(cm.MATRIX_PATH) else {}
    survivors = [bi for bi, r in scan_res.items() if r.get("p_holm") is not None and r["p_holm"] < 0.05]

    rule_results = {}
    for bi in survivors:
        oos = oos_best_rule(panel, bi, spreads)
        tr = oos["oos"]
        if tr.empty:
            rule_results[bi] = {"trades": 0}
            continue
        stats_df = tr.rename(columns={"net_excess_pct": "net", "entry_day": "date"})
        per_day = stats_df.groupby("date")["net"].mean()
        t, p = sr._t_p(per_day)
        rule_results[bi] = {"trades": int(len(tr)), "days": int(len(per_day)),
                            "mean_trade_pct": float(tr["net_excess_pct"].mean()),
                            "mean_day_pct": float(per_day.mean()), "t_day": t, "p_day": p,
                            "win_rate": float((tr["net_excess_pct"] > 0).mean())}

    return {"period": {"from": str(d_from), "to": str(d_to)}, "days": len(days),
           "buckets": {f"{BUCKET_HOURS[bi][0]}-{BUCKET_HOURS[bi][1]}": v for bi, v in scan_res.items()},
           "holm_survivors": survivors, "z_thresh": Z_THRESH,
           "rule_oos_by_bucket": {BUCKET_HOURS[bi][0].strftime("%H:%M"): v for bi, v in rule_results.items()}}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    conn = database.get_connection()
    try:
        res = run(conn)
    finally:
        conn.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    rev = open(os.path.join(ROOT, "REVISION")).read().strip() if os.path.exists(os.path.join(ROOT, "REVISION")) else "?"
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 11,
                            "stage": "intraday_regime_scan", "trials": 8, "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
