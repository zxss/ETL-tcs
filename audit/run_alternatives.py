"""
Раздел 3 ТЗ — прогон альтернативных алгоритмов на той же walk-forward схеме,
что и production-модель, чтобы сравнение было честным.

Запуск:
    python3 -m audit.run_alternatives
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import alternatives as alt  # noqa: E402
from audit import costs, data, ic  # noqa: E402

log = logging.getLogger("audit.alternatives")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def session_ratio(d5_all: pd.DataFrame) -> pd.DataFrame:
    """Доля вечерней сессии в дневном объёме — микроструктурный признак."""
    d = d5_all.copy()
    t = d["ts_msk"].dt.time
    d["evening"] = (t >= pd.Timestamp("19:00").time())
    g = d.groupby(["ticker", "date"]).apply(
        lambda x: pd.Series({
            "evening_share": (x.loc[x["evening"], "volume"].sum() /
                              max(x["volume"].sum(), 1.0)),
        }), include_groups=False).reset_index()
    return g


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)
    results = {}

    log.info("Загрузка данных...")
    daily_all = data.add_returns(data.load_daily())
    instr = costs.load_instruments()
    daily = costs.apply_lot_sizes(daily_all[daily_all.ticker != "IMOEX"], instr)
    index_daily = daily_all[daily_all.ticker == "IMOEX"].sort_values("date")
    idx_close = index_daily.set_index("date")["close"]

    log.info("5-минутки для микроструктуры...")
    d5_all = data.load_5m(main_session_only=False)
    sr = session_ratio(d5_all)

    # ── Альтернатива 1: кросс-секционный GBDT-ранжировщик ────────────────────
    log.info("Альт.1: признаки ранжировщика...")
    panel = alt.build_ranker_features(daily, idx_close, sr, horizon=5)
    panel = alt.add_cs_rank_target(panel, "fwd_ret_h")
    panel.to_csv(os.path.join(OUT, "ranker_panel.csv"), index=False)
    log.info("   панель: %s строк, %d признаков", f"{len(panel):,}",
             len([c for c in alt.FEATURE_COLS if c in panel.columns]))

    tdays = pd.DatetimeIndex(sorted(panel["date"].unique()))
    first = tdays[0] + pd.DateOffset(months=12)
    fit_dates = pd.date_range(first, tdays[-1], freq="MS")
    log.info("Альт.1: walk-forward, окон=%d", len(fit_dates))

    for horizon_name, target_col in (("H5", "fwd_ret_h"), ("H1", "fwd_ret_1")):
        p = alt.add_cs_rank_target(panel, target_col)
        preds = alt.walkforward_ranker(p, list(fit_dates))
        if preds.empty:
            continue
        preds["asof_date"] = preds["date"]
        preds["realized_net"] = preds[target_col]
        r = ic.rank_ic(preds, "pred", "realized_net")
        results[f"ranker_{horizon_name}"] = {
            "rank_ic": r.get("ic"), "t_stat": r.get("t_stat"),
            "ir": r.get("ir"), "hit_rate": r.get("hit_rate"),
            "n_days": r.get("n_days"), "avg_names": r.get("avg_names"),
            "backend": str(preds["backend"].iloc[0]),
        }
        preds[["date", "ticker", "pred", target_col]].to_csv(
            os.path.join(OUT, f"ranker_preds_{horizon_name}.csv"), index=False)
        log.info("   %s: Rank IC=%.4f  t=%.2f  дней=%d",
                 horizon_name, r.get("ic", np.nan), r.get("t_stat", np.nan),
                 r.get("n_days", 0))

    imp = alt.feature_importance(alt.add_cs_rank_target(panel, "fwd_ret_h"))
    imp.to_csv(os.path.join(OUT, "ranker_importance.csv"), index=False)
    results["ranker_top_features"] = imp.head(8).to_dict("records")

    # ── Альтернатива 2: режимы рынка на HMM ─────────────────────────────────
    log.info("Альт.2: HMM-режимы...")
    breadth = None
    hmm_df = alt.hmm_regimes(index_daily, breadth, n_states=3)
    ema_df = alt.ema_regimes(index_daily)
    if not hmm_df.empty:
        hmm_df.to_csv(os.path.join(OUT, "regimes_hmm.csv"), index=False)
        reg = hmm_df.merge(ema_df, on="date", how="inner")
        reg.to_csv(os.path.join(OUT, "regimes_compare.csv"), index=False)

        cond_hmm = alt.regime_conditional_returns(panel, hmm_df, "state_label")
        cond_ema = alt.regime_conditional_returns(panel, ema_df, "ema_regime")
        cond_hmm.to_csv(os.path.join(OUT, "regime_returns_hmm.csv"), index=False)
        cond_ema.to_csv(os.path.join(OUT, "regime_returns_ema.csv"), index=False)
        results["regime_hmm"] = cond_hmm.to_dict("records")
        results["regime_ema"] = cond_ema.to_dict("records")
        results["regime_agreement"] = float(
            (reg["state_label"].map({"BULL": "BULL", "FLAT": "NEUTRAL",
                                     "BEAR/CRASH": "BEAR"}) ==
             reg["ema_regime"]).mean())
        # Разделяющая сила: разброс средней доходности между режимами.
        results["regime_spread"] = {
            "hmm_pct": float(cond_hmm["mean"].max() - cond_hmm["mean"].min()),
            "ema_pct": float(cond_ema["mean"].max() - cond_ema["mean"].min()),
        }
        log.info("   HMM состояний: %s", hmm_df["state_label"].value_counts().to_dict())

    # ── Альтернатива 3: возврат к среднему (OU) ─────────────────────────────
    log.info("Альт.3: диагностика возврата к среднему...")
    ou = alt.ou_diagnostics(daily)
    ou.to_csv(os.path.join(OUT, "ou_diagnostics.csv"), index=False)
    results["ou"] = {
        "median_halflife_days": float(ou["ou_halflife_days"].median()),
        "tickers_with_reversion": int(ou["ou_halflife_days"].notna().sum()),
        "tickers_total": int(len(ou)),
        "median_ret_autocorr": float(ou["ret_autocorr_1"].median()),
        "share_negative_autocorr": float((ou["ret_autocorr_1"] < 0).mean()),
    }
    log.info("   медианный период полураспада: %.1f дн., автокорр: %.4f",
             results["ou"]["median_halflife_days"],
             results["ou"]["median_ret_autocorr"])

    with open(os.path.join(OUT, "alternatives_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in results.items()
                      if k != "ranker_top_features"},
                     ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
