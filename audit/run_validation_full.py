"""
Блок 2.3 — честная мультитестовая вселенная и переобучение.

Прогоняет production-контур валидации (strategy_validation/validation/verdict.py)
в двух режимах на одних и тех же данных:

  A. «как в бою»   — отчёт по 6 отобранным тикерам, семейство FDR = те же 6.
                     Это текущий дефолт (VALIDATION_FULL_UNIVERSE=0).
  B. «честный»     — отчёт по всем 46 тикерам, семейство FDR = все 46.
                     Поправка на выбор тикеров применяется в полном объёме.

Разница между A и B и есть цена data snooping при отборе «любимчиков».

Издержки берутся из матрицы Блока 2.1 (audit/out/cost_matrix.csv), а не из
плоских 0,08%.

Запуск:
    python3 -m audit.run_validation_full --boot 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import data  # noqa: E402

log = logging.getLogger("audit.validation")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CSV_DIR = os.path.join(OUT, "candles")

STRATS = ["long_overnight", "intraday_short", "intraday_long"]
# Текущая отчётная выборка из config.VALIDATION_TICKERS.
SELECTED = ["ETLN", "SELG", "SMLT", "MGNT", "ALRS", "UPRO"]


def export_csvs(daily: pd.DataFrame) -> list[str]:
    """Выгружает дневные свечи в формат, который ждёт advanced_stats.load_daily."""
    os.makedirs(CSV_DIR, exist_ok=True)
    written = []
    for tk, g in daily.groupby("ticker", sort=False):
        if tk == "IMOEX":
            continue
        g = g.sort_values("date")
        path = os.path.join(CSV_DIR, f"{tk.lower()}_candles.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for r in g.itertuples():
                w.writerow([str(r.date)[:10], float(r.open), float(r.high),
                            float(r.low), float(r.close), int(r.volume or 0)])
        written.append(tk)
    return sorted(written)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--oos-fraction", type=float, default=0.30)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)

    daily = data.load_daily()
    tickers = export_csvs(daily)
    log.info("Выгружено тикеров: %d", len(tickers))

    # Реалистичные издержки из Блока 2.1.
    cm = pd.read_csv(os.path.join(OUT, "cost_matrix.csv"))
    cost_map = {r.ticker: float(r.cost_rt_base) for r in cm.itertuples()
                if np.isfinite(r.cost_rt_base)}
    log.info("Издержки: медиана %.3f%% (плоский дефолт в config — 0,080%%)",
             float(np.median(list(cost_map.values()))))

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "strategy_validation"))
    from validation import verdict  # noqa: E402

    results = {}

    for name, report_tks, universe_tks, cmap in [
        ("A_selected_flat_cost", SELECTED, SELECTED, None),
        ("B_selected_real_cost", SELECTED, SELECTED, cost_map),
        ("C_full_universe_real_cost", tickers, tickers, cost_map),
    ]:
        log.info("── Режим %s: отчёт по %d, семейство FDR по %d ──",
                 name, len(report_tks), len(universe_tks))
        rows = verdict.run(
            [t.lower() for t in report_tks], STRATS, CSV_DIR,
            B=args.boot,
            cost_rt=0.08,
            cost_map=cmap,
            winsorize=False,
            universe_tickers=[t.lower() for t in universe_tks],
            oos_fraction=args.oos_fraction,
        )
        df = pd.DataFrame(rows)
        df["mode"] = name
        df.to_csv(os.path.join(OUT, f"validation_{name}.csv"), index=False)
        results[name] = df
        log.info("   строк: %d, вердикты: %s", len(df),
                 df["verdict"].value_counts().to_dict() if "verdict" in df else "n/a")

    summary = {}
    for name, df in results.items():
        if df.empty:
            continue
        summary[name] = {
            "rows": int(len(df)),
            "verdicts": df["verdict"].value_counts().to_dict() if "verdict" in df else {},
            "fdr_pass": int(df["fdr_pass"].sum()) if "fdr_pass" in df else None,
            "edge_above_cost": int(df["edge_above_cost"].sum()) if "edge_above_cost" in df else None,
            "pbo_median": float(df["pbo"].median()) if "pbo" in df else None,
            "pbo_gt_030": int((df["pbo"] > 0.30).sum()) if "pbo" in df else None,
            "white_rc_p_median": float(df["white_rc_p"].median()) if "white_rc_p" in df else None,
            "spa_p_median": float(df["spa_p"].median()) if "spa_p" in df else None,
        }
    with open(os.path.join(OUT, "validation_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
