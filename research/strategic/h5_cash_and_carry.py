"""
H5. Структурный дельта-нейтральный Cash-and-Carry.

Гипотеза уже проверена в программе структурных моделей 17.09.2026 как модуль 2′
(C&C-2, без заглядывания вперёд по дивидендам):
  • разработка 2024-05-21…2026-09-07: 73 сделки, +0,056 %/сделку, t +2,5,
    +1,07 % годовых на задействованный капитал, IR 1,50;
  • ОТЛОЖЕННАЯ ВЫБОРКА ИЗРАСХОДОВАНА 17.09.2026: 32 сделки, t 1,9,
    +1,44 % годовых.

По протоколу анти-data-snooping отложенная выборка одноразовая, поэтому здесь
она НЕ ПЕРЕЗАПУСКАЕТСЯ. Модуль только пересчитывает вердикт по новым — более
жёстким — гейтам ТЗ 18.09.2026 (t > 3,0 на разработке вместо t ≥ 2,0).

Прогон заново на разработке доступен флагом --rerun-h5: новая модель издержек
(комиссия круга 0,14 % вместо 0,08 % плюс корневое воздействие на обеих ногах)
делает результат заведомо хуже прежнего, поэтому по умолчанию он не тратится.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.strategic import validation as va            # noqa: E402

log = logging.getLogger("research.strategic.h5")

MODULE = "h5_cash_and_carry"
STRUCTURAL_DIR = os.path.join(ROOT, "audit", "r4_research", "structural")
PRIOR = {
    "dev": {"source": "audit/r4_research/structural/dev_cc2", "trades": 73, "t": 2.5,
            "excess_annual_pct": 1.07, "ir": 1.50, "net_excess_pct_trade": 0.056},
    "holdout": {"source": "audit/r4_research/structural/holdout", "trades": 32, "t": 1.9,
                "excess_annual_pct": 1.44, "ir": 1.36, "net_excess_pct_trade": 0.072,
                "spent_on": "2026-09-17"},
}


def load_prior(stage: str) -> dict:
    """Метрики прошлого прогона: из результатов на диске, иначе из зафиксированных."""
    sub = "dev_cc2" if stage == "dev" else "holdout"
    path = os.path.join(STRUCTURAL_DIR, sub, "results.json")
    prior = dict(PRIOR[stage])
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            prior["results_json"] = path
            prior["raw"] = data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("[H5] не прочитал %s: %s", path, e)
    return prior


def run(conn, stage: str, rules: dict, ctx=None, n_trials: int | None = None) -> dict:
    """Вердикт по ранее полученным метрикам — без нового прогона."""
    prior = load_prior(stage)
    summary = {"trades": prior["trades"], "t": prior["t"],
               "excess_annual_pct": prior["excess_annual_pct"], "ir": prior["ir"],
               "net_excess_pct_trade": prior["net_excess_pct_trade"]}
    if stage == "dev":
        passed, failed = va.dev_gate(summary, pbo=None, capacity_rub=None, gates=rules["gates"]["dev"])
        failed = [f for f in failed if not f.startswith(("PBO", "ёмкость"))]
        failed.append("PBO и ёмкость не считались: прогон не повторялся, отложенная выборка израсходована")
        passed = False
    else:
        passed, failed = va.holdout_gate({**summary, "dsr": None}, rules["gates"]["holdout"])
    return {"module": MODULE, "stage": stage, "variant": "C&C-2", "summary": summary,
            "pbo": {"pbo": None}, "capacity_rub": None, "passed": passed, "failed": failed,
            "reused": True, "prior": {k: v for k, v in prior.items() if k != "raw"},
            "trades": pd.DataFrame()}
