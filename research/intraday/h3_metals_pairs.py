"""
H3-металлурги: полный тест парной внутридневной торговли металлургами как
ОТДЕЛЬНОЙ предзарегистрированной гипотезы (ТЗ пользователя 23.09.2026).

Контекст: в `research/intraday_hypotheses.py` (15.09.2026) H3 тестировался
СРАЗУ по трём кластерам (нефть-газ, банки, металлурги); общий вердикт —
провал DoR. Но в описательном разрезе `cuts` кластер металлургов (CHMF/NLMK/
MAGN) показал устойчивый плюс на ОБОИХ периодах того прогона: holdout
(2024-05-21…2025-11-30) excess_t +7,6, dev-повтор (2025-12-01…2026-09-11)
excess_t +3,3 — единственный разрез во всей программе с воспроизведённым
знаком на двух периодах. Он никогда не был проверен как САМОСТОЯТЕЛЬНАЯ
гипотеза со своим DoR — это и есть задача здесь.

ВАЖНО про выборки. Локальные 5-минутки существуют только с 30.11.2025 (см.
память local-db-research-data-gaps) — окно 2024-05-21…2025-11-30, на котором
эффект был впервые замечен, здесь физически недоступно и уже было
просмотрено в прошлом прогоне (его результат процитирован выше). Поэтому
«отложенная выборка» здесь — это НЕ то же самое старое окно (это было бы
повторным подглядыванием), а честно отделённый, ранее НИКЕМ не смотренный
хвост локально доступных данных:
    dev     2025-12-01 … 2026-06-15  (~140 торговых дней, выбор параметров)
    holdout 2026-06-16 … 2026-09-09  (~60 торговых дней, один прогон)

Метод торговли — БЕЗ ИЗМЕНЕНИЙ идентичен прежнему H3 (пары внутри кластера,
z по логарифмическому спреду close, вход по |z|≥z_in, выход по пересечению
нуля / стопу z≥z_stop / закрытию сессии) — параметризован, чтобы можно было
прогнать несколько вариантов гипотезы (сетка). ГОЛОВНОЙ вариант — ближайшая
копия уже наблюдавшегося эффекта (тот же порог, то же окно, тот же состав
CHMF/NLMK/MAGN). Остальные варианты сетки — проверка устойчивости
(другой порог входа, другое окно z, расширенный состав металлургов) — они
считаются и на dev, и на holdout для полноты картины, НО формальным
решающим прогоном отложенной выборки считается только головной: остальные
holdout-числа — диагностика, не повод задним числом выбирать «победителя»
(тот же принцип анти-снупинга, что и `cuts` в прежнем файле).

Запуск: python -m research.intraday.h3_metals_pairs
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import cost_model as cm                      # noqa: E402
from research import intraday_hypotheses as ih             # noqa: E402
from research import short_rule as sr                       # noqa: E402

log = logging.getLogger("research.intraday.h3_metals")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "intraday_h3_metals")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
INDEX = ih.INDEX

DEV_FROM, DEV_TO = dt.date(2025, 12, 1), dt.date(2026, 6, 15)
HOLDOUT_FROM, HOLDOUT_TO = dt.date(2026, 6, 16), dt.date(2026, 9, 9)
LAST_SIGNAL = dt.time(17, 45)

CORE = ("CHMF", "NLMK", "MAGN")
EXTENDED = ("CHMF", "NLMK", "MAGN", "RUAL", "GMKN")           # чёрная + цветная металлургия;
                                                               # золото/серебро (PLZL/SELG) сознательно
                                                               # исключены заранее — другой макро-драйвер
HEADLINE_NAME = "headline"
GRID = {
    "headline":        {"tickers": CORE, "window": 520, "min_periods": 260, "z_in": 2.0, "z_stop": 3.5},
    "z_in_2.5":        {"tickers": CORE, "window": 520, "min_periods": 260, "z_in": 2.5, "z_stop": 3.5},
    "z_in_1.5":        {"tickers": CORE, "window": 520, "min_periods": 260, "z_in": 1.5, "z_stop": 3.5},
    "window_260":      {"tickers": CORE, "window": 260, "min_periods": 130, "z_in": 2.0, "z_stop": 3.5},
    "window_1040":     {"tickers": CORE, "window": 1040, "min_periods": 520, "z_in": 2.0, "z_stop": 3.5},
    "extended_universe": {"tickers": EXTENDED, "window": 520, "min_periods": 260, "z_in": 2.0, "z_stop": 3.5},
}

# DoR — тот же гейт, что и в прежнем H3 (research/intraday_hypotheses.dor):
#   t по дням ≥ 2,5 и ср./день > 0; выживает при стресс-спреде; не держится
#   на 5 лучших днях; медиана сделки ≥ +0,20 %.


def pair_frame(bars: pd.DataFrame, a: str, b: str, window: int, min_periods: int) -> pd.DataFrame:
    ga, gb = ih._series(bars, a), ih._series(bars, b)
    m = ga[["open", "close", "d"]].join(gb[["open", "close"]], lsuffix="_a", rsuffix="_b", how="inner")
    if m.empty:
        return m
    s = np.log(m["close_a"] / m["close_b"])
    mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    m["z"] = (s - mu) / sd
    return m


def pair_trades(m: pd.DataFrame, a: str, b: str, idx: pd.DataFrame, blocked: set,
                z_in: float, z_stop: float, d_from: dt.date, d_to: dt.date) -> list[dict]:
    trades = []
    for d, day in m.groupby("d"):
        if not (d_from <= d <= d_to):
            continue
        day = day.sort_index()
        z = day["z"].to_numpy()
        tms = day.index
        i, n = 0, len(day)
        while i < n - 1:
            if z[i] == z[i] and abs(z[i]) >= z_in and tms[i].time() <= LAST_SIGNAL:
                side = 1 if z[i] > 0 else -1
                short_leg = a if side > 0 else b
                if short_leg in blocked:
                    break
                e = i + 1
                ea, eb = day["open_a"].iloc[e], day["open_b"].iloc[e]
                if not (ih._ok_price(ea) and ih._ok_price(eb)):
                    break
                x, how = n - 1, "eod"
                for j in range(e, n):
                    zj = z[j]
                    if zj != zj:
                        continue
                    if side * zj <= 0:
                        x, how = j, "zero"
                        break
                    if side * zj >= z_stop:
                        x, how = j, "stop"
                        break
                if how != "eod" and x + 1 < n:
                    xa, xb, t_out = day["open_a"].iloc[x + 1], day["open_b"].iloc[x + 1], tms[x + 1]
                else:
                    xa, xb, t_out = day["close_a"].iloc[x], day["close_b"].iloc[x], tms[x]
                ra, rb = (xa / ea - 1.0) * 100.0, (xb / eb - 1.0) * 100.0
                gross = (rb - ra) if side > 0 else (ra - rb)
                trades.append({"date": d, "ticker": f"{a}/{b}", "dir": 0, "t_in": tms[e], "t_out": t_out,
                               "gross": gross, "exit": how,
                               "idx_move": ih._window_idx_move(idx, tms[e], t_out)})
                break
            i += 1
    return trades


def variant_trades(bars: pd.DataFrame, idx: pd.DataFrame, blocked: set, cfg: dict,
                   d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    tks = cfg["tickers"]
    out = []
    for i, a in enumerate(tks):
        for b in tks[i + 1:]:
            m = pair_frame(bars, a, b, cfg["window"], cfg["min_periods"])
            if m.empty:
                continue
            out += pair_trades(m, a, b, idx, blocked, cfg["z_in"], cfg["z_stop"], d_from, d_to)
    return pd.DataFrame(out)


def with_costs(tr: pd.DataFrame, spreads: dict) -> pd.DataFrame:
    if tr.empty:
        return tr
    tr = tr.copy()
    for s in cm.SCENARIOS:
        tr[f"cost_{s}"] = [cm.trade_cost(t, s, spreads) for t in tr["ticker"]]
    return tr


def run(conn) -> dict:
    bars, _last = ih.load(conn, DEV_FROM, HOLDOUT_TO)
    idx = ih._series(bars, INDEX)
    lots, blocked = sr.load_lots_and_blocked()
    spreads = cm.load_spreads()
    log.info("баров загружено: %d, тикеров %d", len(bars), bars["ticker"].nunique())

    res = {"dev": {}, "holdout": {}}
    perf_dev = {}
    for name, cfg in GRID.items():
        tr_dev = with_costs(variant_trades(bars, idx, blocked, cfg, DEV_FROM, DEV_TO), spreads)
        st_base = ih.stats(tr_dev, cm.PRIMARY)
        st_stress = ih.stats(tr_dev, cm.SENSITIVITY)
        res["dev"][name] = {"base": st_base, "stress": st_stress,
                            "dor": ih.dor(st_base, st_stress) if st_base.get("n") else {}}
        if st_base.get("n"):
            per_day = (tr_dev.assign(net=tr_dev["gross"] - tr_dev[f"cost_{cm.PRIMARY}"])
                      .groupby("date")["net"].mean())
            perf_dev[name] = per_day

        tr_ho = with_costs(variant_trades(bars, idx, blocked, cfg, HOLDOUT_FROM, HOLDOUT_TO), spreads)
        st_base_ho = ih.stats(tr_ho, cm.PRIMARY)
        st_stress_ho = ih.stats(tr_ho, cm.SENSITIVITY)
        res["holdout"][name] = {"base": st_base_ho, "stress": st_stress_ho,
                                "dor": ih.dor(st_base_ho, st_stress_ho) if st_base_ho.get("n") else {},
                                "note": "решающий прогон holdout" if name == HEADLINE_NAME
                                        else "диагностика, НЕ решающий прогон (см. заголовок файла)"}

    if perf_dev:
        perf = pd.DataFrame(perf_dev).dropna(how="all")
        n_cfg = perf.shape[1]
        if n_cfg >= 2 and len(perf.dropna(axis=1, how="any")) >= 8:
            from research.strategic import validation as va
            res["pbo_dev"] = va.cscv_pbo(perf, s=8)
        else:
            res["pbo_dev"] = {"note": "недостаточно конфигураций/дней для PBO"}

    res["headline_verdict"] = {
        "dev_passed": all(res["dev"][HEADLINE_NAME]["dor"].values()) if res["dev"][HEADLINE_NAME]["dor"] else False,
        "holdout_passed": all(res["holdout"][HEADLINE_NAME]["dor"].values()) if res["holdout"][HEADLINE_NAME]["dor"] else False,
    }
    return res


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
    n_trials = len(GRID) * 2                     # dev + holdout на каждый вариант сетки
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 12,
                            "stage": "h3_metals_pairs_standalone", "trials": n_trials,
                            "revision": rev}) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
