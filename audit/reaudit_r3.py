"""
Третий квант-аудит — против конфигурации r3 (stage2-demo-30d-r3, старт 15.09).

Что изменилось после переаудита 09.09 (audit/reaudit.py):
  * выход из овернайта — утренняя фаза CLOSE (09:10), а не «открытие»;
  * расписание фаз под Мосбиржу с 14.09: ORDER 09:15, CLEANUP 18:20,
    OVERNIGHT 18:35;
  * пол стопа MIN_STOP_PCT = 1 %;
  * казначейство: свободный кэш паркуется в фонд денежного рынка;
  * капитал песочницы 2 000 000 ₽ при позиции 10 000 ₽.

Модельный слой (features / dataset / model) не менялся — реплей
walkforward.csv валиден и переиспользуется. Переизмеряется то, чего реплей
не видел: ОКНА ИСПОЛНЕНИЯ.

Реплей меряет исход по дневному бару брокера. Дневной бар — весь торговый
день: open — первая сделка (с февраля 2025 — 06:50), close — конец вечерней
сессии (23:50). Значит, r_overnight реплея = 23:50 → 06:50, а r_intraday =
06:50 → 23:50. Бой торгует другое:

  long_overnight  вход 18:35 дня A → выход 09:10 следующего торгового дня;
  intraday_short  вход 09:15 дня A → выход 18:20 того же дня.

И ещё одно расхождение (дефект 4): в 18:35 дневной бар дня A не закрыт, и
вечерняя фаза считает прогноз на барах ПО ВЧЕРА — ровно тот же рейтинг, что
утренний PREP. Поэтому реалистичная пара для обеих стратегий: рейтинг даты
A′ (последний закрытый бар до A) → окна исполнения дня A.

Упрощения (одинаковы для всех сравниваемых окон, поэтому сравнение честное):
вход и выход по цене бара в момент фазы, без лимитного улучшения и без
риска незаливки; издержки — та же матрица cost_rt, что в реплее.

Запуск (на сервере, где БД; только чтение):
    cd /opt/etl-tcs && python3 /var/tmp/reaudit-r3/reaudit_r3.py \
        --wf /var/tmp/reaudit-r3/walkforward.csv --out /var/tmp/reaudit-r3/out
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("ETL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from audit import backtest, costs, data, ic, reaudit  # noqa: E402

log = logging.getLogger("audit.reaudit_r3")

TOP_N = 5
POSITION_RUB = 10_000.0
CAPITAL_RUB = 2_000_000.0
STOP_PCT = 1.0                 # MIN_STOP_PCT: для уверенных прогнозов действует пол


# ── 5-минутки: цены в моменты фаз и экстремумы окон ──────────────────────────

EXEC_SQL = """
WITH b AS (
    SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, low, close
    FROM market_data_5m
)
SELECT ticker, tm::date AS d,
    MAX(close) FILTER (WHERE tm::time = TIME '18:30')                          AS c1830,
    MAX(close) FILTER (WHERE tm::time = TIME '18:15')                          AS c1815,
    MAX(close) FILTER (WHERE tm::time = TIME '23:45')                          AS c2345,
    MAX(open)  FILTER (WHERE tm::time = TIME '09:10')                          AS o0910,
    MAX(open)  FILTER (WHERE tm::time = TIME '09:15')                          AS o0915,
    MAX(open)  FILTER (WHERE tm::time = TIME '10:00')                          AS o1000,
    MAX(open)  FILTER (WHERE tm::time = TIME '10:05')                          AS o1005,
    (ARRAY_AGG(open ORDER BY tm) FILTER (WHERE tm::time >= TIME '06:00'))[1]  AS o_first,
    MAX(high) FILTER (WHERE tm::time >= TIME '09:15' AND tm::time < TIME '18:20') AS hi_a,
    MAX(high) FILTER (WHERE tm::time >= TIME '10:05' AND tm::time < TIME '18:20') AS hi_b,
    MIN(low)  FILTER (WHERE tm::time >= TIME '18:35')                          AS lo_eve,
    MIN(low)  FILTER (WHERE tm::time >= TIME '06:00' AND tm::time < TIME '09:10') AS lo_m0910
FROM b
GROUP BY 1, 2
"""


def load_exec_bars() -> pd.DataFrame:
    conn = data._conn()
    try:
        df = pd.read_sql(EXEC_SQL, conn)
    finally:
        conn.close()
    df["d"] = pd.to_datetime(df["d"])
    for c in df.columns:
        if c not in ("ticker", "d"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def exec_days(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Будни с основной сессией (бар 10:00) не меньше чем у 20 бумаг.

    Фазы r3 идут по будням; выходные сессии Мосбиржи в окно позиции входят
    неявно — через цену, но торговыми днями бота не являются.
    """
    ok = bars[bars["o1000"].notna() & (bars["d"].dt.dayofweek < 5)]
    cnt = ok.groupby("d")["ticker"].nunique()
    return pd.DatetimeIndex(cnt[cnt >= 20].index).sort_values()


# ── Исходы в окнах исполнения ────────────────────────────────────────────────

def _g(bar: dict, key: str) -> float:
    """Цена из бара или NaN: у бумаги в этот день бара может не быть."""
    v = bar.get(key)
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _ret(a, b):
    return (b / a - 1.0) * 100.0


def short_exec(bar: dict, entry_col: str, hi_col: str,
               stop_pct: float = STOP_PCT) -> tuple[float, float, bool]:
    """(валовая без стопа, валовая со стопом, сработал ли стоп) для шорта."""
    entry, exitp, hi = _g(bar, entry_col), _g(bar, "c1815"), _g(bar, hi_col)
    if not all(np.isfinite(x) for x in (entry, exitp, hi)) or entry <= 0:
        return np.nan, np.nan, False
    gross = -_ret(entry, exitp)
    stop = entry * (1 + stop_pct / 100.0)
    if hi >= stop:
        return gross, -stop_pct, True
    return gross, gross, False


def overnight_exec(bar_a: dict, bar_b: dict, exit_col: str,
                   stop_pct: float = STOP_PCT) -> dict:
    """Лонг 18:35 дня A → exit_col следующего торгового дня, со стопом и сегментами."""
    entry = _g(bar_a, "c1830")
    exitp = _g(bar_b, exit_col)
    out = {"gross": np.nan, "gross_stop": np.nan, "stopped": False,
           "seg_evening": np.nan, "seg_gap": np.nan, "seg_morning": np.nan}
    if not (np.isfinite(entry) and np.isfinite(exitp)) or entry <= 0:
        return out
    gross = _ret(entry, exitp)
    out["gross"] = gross
    c2345, o_first = _g(bar_a, "c2345"), _g(bar_b, "o_first")
    if np.isfinite(c2345):
        out["seg_evening"] = _ret(entry, c2345)
        if np.isfinite(o_first):
            out["seg_gap"] = _ret(c2345, o_first)
            out["seg_morning"] = _ret(o_first, exitp)
    stop = entry * (1 - stop_pct / 100.0)
    lo_eve, lo_m = _g(bar_a, "lo_eve"), _g(bar_b, "lo_m0910")
    if np.isfinite(lo_eve) and lo_eve <= stop:
        out["gross_stop"], out["stopped"] = -stop_pct, True
    elif np.isfinite(o_first) and o_first <= stop:          # гэп сквозь стоп
        out["gross_stop"], out["stopped"] = _ret(entry, o_first), True
    elif exit_col == "o0910" and np.isfinite(lo_m) and lo_m <= stop:
        out["gross_stop"], out["stopped"] = -stop_pct, True
    else:
        out["gross_stop"] = gross
    return out


# ── Статистика ───────────────────────────────────────────────────────────────

def by_date_t(df: pd.DataFrame, col: str, date_col: str = "exec_day") -> dict:
    """Среднее на сделку и t по датам (сделки одного дня связаны общим шоком)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return {"n": 0}
    per_day = d.groupby(date_col)[col].mean()
    n_days = len(per_day)
    sd = per_day.std(ddof=1) if n_days > 1 else np.nan
    t = per_day.mean() / sd * np.sqrt(n_days) if sd and sd > 0 else np.nan
    return {"n": int(len(d)), "days": int(n_days),
            "mean_trade": round(float(d[col].mean()), 4),
            "median_trade": round(float(d[col].median()), 4),
            "sd_trade": round(float(d[col].std(ddof=1)), 4) if len(d) > 1 else None,
            "hit": round(float((d[col] > 0).mean()), 3),
            "mean_day": round(float(per_day.mean()), 4),
            "t_by_date": round(float(t), 2) if np.isfinite(t) else None}


def ic_of(df: pd.DataFrame, score: str, ret: str, date_col: str) -> dict:
    frame = pd.DataFrame({"d": df[date_col].to_numpy(), "s": df[score].to_numpy(),
                          "r": df[ret].to_numpy()})
    r = ic.rank_ic(frame, "s", "r", date_col="d", min_names=8)
    return {"ic": None if not np.isfinite(r.get("ic", np.nan)) else round(r["ic"], 4),
            "t": None if not np.isfinite(r.get("t_stat", np.nan)) else round(r["t_stat"], 2),
            "days": r.get("n_days")}


def curve_stats(s: pd.Series, bench: pd.Series, label: str) -> dict:
    s = s.sort_index()
    p = backtest.performance(s)
    a = backtest.alpha_vs_benchmark(s, bench)
    mid = s.index[len(s) // 2]
    h1 = backtest.performance(s[s.index <= mid])
    h2 = backtest.performance(s[s.index > mid])
    return {"label": label, "days": int(len(s)), "active_days": int((s != 0).sum()),
            "CAGR_%": round(p.get("cagr_pct", np.nan), 3),
            "Sharpe": round(p.get("sharpe", np.nan), 3),
            "PF": round(p.get("profit_factor", np.nan), 3),
            "MDD_%": round(p.get("max_drawdown_pct", np.nan), 3),
            "alpha_%год": round(a.get("alpha_annual_pct", np.nan), 3) if a else None,
            "Sharpe_H1": round(h1.get("sharpe", np.nan), 3),
            "Sharpe_H2": round(h2.get("sharpe", np.nan), 3),
            "gates": reaudit.stage1_gates(p, a or {}),
            "dsr": reaudit.deflated_sharpe(s[s != 0], reaudit.N_TRIALS)}


# ── Главное ──────────────────────────────────────────────────────────────────

def picks_topn(comp: pd.DataFrame, sb: set[str], top_n: int = TOP_N) -> pd.DataFrame:
    """Отбор как в select_top_rows: специализация, одна стратегия на тикер, топ-N."""
    d = comp[comp["specialised"] & comp["allowed"] & comp["score"].notna()
             & comp["exp_pnl"].notna()].copy()
    d = d[~((d["strategy"] == "intraday_short") & d["ticker"].isin(sb))]
    d = (d.sort_values("score", ascending=False)
           .groupby(["asof_date", "ticker"], as_index=False).head(1))
    return (d.sort_values(["asof_date", "score"], ascending=[True, False])
             .groupby("asof_date", as_index=False).head(top_n).copy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True, help="путь к walkforward.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(args.out, exist_ok=True)
    import config

    # 0. Текущий конвейер r3 на реплее (конфигурация — из .env сервера)
    reaudit.OUT = os.path.dirname(os.path.abspath(args.wf))
    log.info("Сборка конвейера r3 на реплее...")
    comp = reaudit.build_current_pipeline()
    daily = data.load_daily()
    bench = backtest.benchmark_returns(daily, "IMOEX", forward=True)
    instr = costs.load_instruments()
    sb = set(instr.loc[instr["short_enabled"] != True, "ticker"])  # noqa: E712

    res: dict = {"config": {k: getattr(config, k, None) for k in (
        "STAGE2_TEST_ID", "TRADING_STRATEGIES", "SCORE_MODE", "APPLY_RISK_PENALTIES",
        "SELLER_MOMENTUM_SHORT_ENABLED", "SHORT_IMOEX_MAX_TREND",
        "OVERNIGHT_MAX_MARKET_ATR_PCTL", "OVERNIGHT_MIN_EDGE_X_COST",
        "BEST_TRADES_TOP_N", "BEST_TRADES_POSITION_RUB", "MIN_STOP_PCT",
        "STAGE2_ORDER_TIME", "STAGE2_CLEANUP_TIME", "STAGE2_OVERNIGHT_TIME",
        "STAGE2_CLOSE_TIME", "TREASURY_ENABLED", "VALIDATION_COST_RT", "TFT_COST_RT",
        "VALIDATION_FULL_UNIVERSE", "ORDER_FILL_WAIT_SEC")}}
    res["config"]["TRADING_STRATEGIES"] = list(res["config"]["TRADING_STRATEGIES"] or [])

    rdates = pd.DatetimeIndex(sorted(comp["asof_date"].unique()))
    res["replay"] = {"from": str(rdates[0].date()), "to": str(rdates[-1].date()),
                     "dates": len(rdates),
                     "weekday_counts": {int(k): int(v) for k, v in
                                        pd.Series(rdates.dayofweek).value_counts().sort_index().items()}}

    # 0a. Контроль воспроизводимости: кривая «ТЕКУЩАЯ» 09.09 на тех же окнах реплея
    picks = picks_topn(comp, sb)
    s_rep = picks.groupby("asof_date")["realized_net"].mean().reindex(rdates).fillna(0.0)
    res["reproduce_0909"] = curve_stats(s_rep, bench, "реплей, окна дневного бара (как 09.09)")
    res["reproduce_weekdays"] = curve_stats(s_rep[s_rep.index.dayofweek < 5], bench,
                                            "реплей, только будни (кроны r3 — пн–пт)")

    # 1. Окна исполнения
    log.info("Загрузка 5-минуток: цены фаз и экстремумы окон...")
    bars = load_exec_bars()
    edays = exec_days(bars)
    bar_of = {(r.ticker, r.d): r._asdict() for r in bars.itertuples(index=False)}
    log.info("Дней исполнения: %d (%s … %s)", len(edays), edays[0].date(), edays[-1].date())

    # A → A′ (последняя дата реплея строго до A) и A → следующий день исполнения
    # Прод: INCLUDE_WEEKEND_TRADING=0 — выходных баров в признаках нет, и в
    # понедельник рейтинг строится по пятнице. A′ берётся только из будней.
    rd = rdates[rdates.dayofweek < 5].to_numpy()
    ed = edays.to_numpy()
    pairs = []
    for i, A in enumerate(ed[:-1]):
        j = np.searchsorted(rd, A, side="left") - 1
        if j < 0:
            continue
        A_prev = rd[j]
        if A_prev < ed[0] - np.timedelta64(7, "D"):
            continue
        pairs.append((pd.Timestamp(A), pd.Timestamp(A_prev), pd.Timestamp(ed[i + 1]),
                      A in set(rd)))
    pdf = pd.DataFrame(pairs, columns=["exec_day", "asof_prev", "next_day", "A_in_replay"])
    last_ok = rdates[-1]
    pdf = pdf[pdf["asof_prev"] <= last_ok]
    res["exec_window"] = {"days": int(len(pdf)),
                          "from": str(pdf.exec_day.min().date()),
                          "to": str(pdf.exec_day.max().date())}

    # Строки реплея, продублированные на дни исполнения
    def outcomes(rows: pd.DataFrame, key_col: str) -> pd.DataFrame:
        """rows: строки реплея с asof_date; key_col — какой дате реплея они
        соответствуют в pdf (asof_prev — реалистично, exec_day — «окно без дефекта 4»)."""
        m = rows.merge(pdf, left_on="asof_date", right_on=key_col, how="inner")
        recs = []
        for r in m.itertuples(index=False):
            ba = bar_of.get((r.ticker, r.exec_day), {})
            bb = bar_of.get((r.ticker, r.next_day), {})
            rec = {"asof_date": r.asof_date, "exec_day": r.exec_day, "ticker": r.ticker,
                   "strategy": r.strategy, "score": r.score, "cost_rt": r.cost_rt,
                   "realized_net_replay": r.realized_net}
            # Дистанция стопа как в бою: stop_distance_pct(Downside), где
            # Downside = q10 знаковой доходности − издержки (directional.py).
            down = float(r.q10) - float(r.cost_rt) if pd.notna(r.q10) else np.nan
            stop_real = max(max(-down, 0.0), STOP_PCT) if np.isfinite(down) else STOP_PCT
            rec["stop_pct"] = stop_real
            if r.strategy == "intraday_short":
                g, gs, st = short_exec(ba, "o0915", "hi_a", stop_real)
                _, gs1, st1 = short_exec(ba, "o0915", "hi_a", STOP_PCT)
                g_b, _, _ = short_exec(ba, "o1005", "hi_b", stop_real)
                rec.update(net_exec=g - r.cost_rt, net_exec_stop=gs - r.cost_rt, stopped=st,
                           net_exec_stop1=gs1 - r.cost_rt, stopped1=st1,
                           net_exec_alt=g_b - r.cost_rt)
            elif r.strategy == "long_overnight":
                o = overnight_exec(ba, bb, "o0910", stop_real)
                o1 = overnight_exec(ba, bb, "o0910", STOP_PCT)
                o2 = overnight_exec(ba, bb, "o1000", stop_real)
                rec.update(net_exec=o["gross"] - r.cost_rt,
                           net_exec_stop=o["gross_stop"] - r.cost_rt, stopped=o["stopped"],
                           net_exec_stop1=o1["gross_stop"] - r.cost_rt, stopped1=o1["stopped"],
                           net_exec_alt=o2["gross"] - r.cost_rt,
                           seg_evening=o["seg_evening"], seg_gap=o["seg_gap"],
                           seg_morning=o["seg_morning"])
            recs.append(rec)
        return pd.DataFrame(recs)

    strat_rows = comp[comp["strategy"].isin(["intraday_short", "long_overnight"])]
    log.info("Исходы: вся вселенная стратегий (для Rank IC)...")
    uni_real = outcomes(strat_rows, "asof_prev")        # рейтинг A′ → окно дня A
    uni_nodef = outcomes(strat_rows[strat_rows.strategy == "long_overnight"], "exec_day")
    uni_real.to_csv(os.path.join(args.out, "universe_r3.csv"), index=False)

    # 1a. Rank IC по стратегиям: окно реплея против окна исполнения, одни и те же дни
    res["rank_ic"] = {}
    for st in ("intraday_short", "long_overnight"):
        u = uni_real[uni_real.strategy == st]
        ok = u.dropna(subset=["net_exec", "realized_net_replay"])
        blk = {
            "окно реплея (дневной бар), рейтинг A′": ic_of(ok, "score", "realized_net_replay", "asof_date"),
            "окно исполнения r3, рейтинг A′": ic_of(ok, "score", "net_exec", "exec_day"),
            "окно исполнения (альт. время), рейтинг A′": ic_of(ok, "score", "net_exec_alt", "exec_day"),
        }
        if st == "long_overnight":
            v = uni_nodef.dropna(subset=["net_exec"])
            blk["окно исполнения, рейтинг A (без дефекта 4)"] = ic_of(v, "score", "net_exec", "exec_day")
            # Проверка стенда: окно реплея при рейтинге A — тот же ряд строк, что
            # у сегмента «ночь»; расхождение означало бы разные цены или окна.
            blk["окно реплея, рейтинг A (те же строки, что сегменты)"] = ic_of(
                v.dropna(subset=["realized_net_replay"]), "score", "realized_net_replay", "exec_day")
            for lab, dmask in (("пн–чт", lambda x: x.dt.dayofweek < 4), ("пт", lambda x: x.dt.dayofweek == 4)):
                vv = v[dmask(v["exec_day"])]
                blk[f"окно реплея, рейтинг A, день A {lab}"] = ic_of(
                    vv.dropna(subset=["realized_net_replay"]), "score", "realized_net_replay", "exec_day")
                blk[f"сегмент «ночь», рейтинг A, день A {lab}"] = ic_of(
                    vv.dropna(subset=["seg_gap"]), "score", "seg_gap", "exec_day")
                uu = ok[dmask(ok["asof_date"])]
                blk[f"окно реплея, рейтинг A′, день A′ {lab}"] = ic_of(
                    uu, "score", "realized_net_replay", "asof_date")
            uni_nodef.to_csv(os.path.join(args.out, "universe_nodef_r3.csv"), index=False)
            for seg, nm in (("seg_evening", "вечер 18:35→23:50"),
                            ("seg_gap", "ночь 23:50→первая сделка"),
                            ("seg_morning", "утро: первая сделка→09:10")):
                blk[f"сегмент «{nm}», рейтинг A′"] = ic_of(u.dropna(subset=[seg]), "score", seg, "exec_day")
                blk[f"сегмент «{nm}», рейтинг A"] = ic_of(v.dropna(subset=[seg]), "score", seg, "exec_day")
        res["rank_ic"][st] = blk

    # 1b. Книги r3: те же сделки, три оценки исхода
    log.info("Книги r3...")
    book = outcomes(picks, "asof_prev")
    book_nodef = outcomes(picks[picks.strategy == "long_overnight"], "exec_day")
    book.to_csv(os.path.join(args.out, "book_r3.csv"), index=False)
    res["books"] = {}
    for st in ("intraday_short", "long_overnight"):
        b = book[book.strategy == st].dropna(subset=["net_exec", "realized_net_replay"])
        blk = {
            "реплей (дневной бар)": by_date_t(b, "realized_net_replay"),
            "исполнение r3": by_date_t(b, "net_exec"),
            "исполнение r3 + стоп (дистанция как в бою)": by_date_t(b, "net_exec_stop"),
            "исполнение r3 + стоп 1 % всем": by_date_t(b, "net_exec_stop1"),
            "исполнение, альт. время": by_date_t(b, "net_exec_alt"),
            "доля стопов (как в бою)": round(float(b["stopped"].mean()), 3) if len(b) else None,
            "доля стопов (1 % всем)": round(float(b["stopped1"].mean()), 3) if len(b) else None,
            "дистанция стопа, медиана %": round(float(b["stop_pct"].median()), 3) if len(b) else None,
            "дистанция = пол 1 %, доля": round(float((b["stop_pct"] <= STOP_PCT + 1e-9).mean()), 3)
            if len(b) else None,
        }
        if st == "long_overnight":
            for seg in ("seg_evening", "seg_gap", "seg_morning"):
                blk[seg] = by_date_t(b, seg)
            nb = book_nodef.dropna(subset=["net_exec"])
            blk["исполнение, рейтинг A (без дефекта 4)"] = by_date_t(nb, "net_exec")
        res["books"][st] = blk

    # 1b'. Шорт-книга: тайминг рынка против отбора бумаг
    us = uni_real[(uni_real.strategy == "intraday_short")
                  & ~uni_real.ticker.isin(sb)].dropna(subset=["net_exec"])
    umean = us.groupby("exec_day")["net_exec"].mean().rename("u")
    bs = book[book.strategy == "intraday_short"].dropna(subset=["net_exec"]).join(umean, on="exec_day")
    bs["excess"] = bs["net_exec"] - bs["u"]
    res["short_decomposition"] = {
        "книга": by_date_t(bs, "net_exec"),
        "вся шортуемая вселенная в те же дни": by_date_t(bs, "u"),
        "отбор (книга − вселенная)": by_date_t(bs, "excess"),
        "вселенная во все дни": {"mean_day": round(float(umean.mean()), 4), "days": int(len(umean))},
    }

    # 1c. Кривая r3 и гейты Этапа 1 на окнах исполнения (общие дни)
    common = book.dropna(subset=["net_exec", "realized_net_replay"])
    days = pd.DatetimeIndex(sorted(pdf.exec_day.unique()))

    def curve(col):
        # равный вес на ЗАНЯТЫЙ слот: как signal_backtest (mean по сделкам дня)
        return common.groupby("exec_day")[col].mean().reindex(days).fillna(0.0)

    # эталон на окне исполнения: доходность IMOEX того же дня (close→close) — приблизительно
    bench_exec = bench.copy()
    bench_exec.index = bench_exec.index + pd.Timedelta(days=0)
    res["curves"] = {
        "реплей": curve_stats(curve("realized_net_replay"), bench, "окна реплея, дни исполнения"),
        "исполнение r3": curve_stats(curve("net_exec"), bench, "окна r3"),
        "исполнение r3 + стоп": curve_stats(curve("net_exec_stop"), bench, "окна r3 + стоп как в бою"),
        "исполнение r3 + стоп 1 % всем": curve_stats(curve("net_exec_stop1"), bench, "окна r3 + стоп 1 % всем"),
    }

    # 2. Частота сделок и мощность 30-дневного теста
    per_day = (picks[picks.asof_date.dt.dayofweek < 5].assign(one=1).pivot_table(index="asof_date", columns="strategy",
                                               values="one", aggfunc="sum")
               .reindex(rdates[rdates.dayofweek < 5]).fillna(0))
    last60 = per_day.tail(60)
    freq = {}
    for st in ("intraday_short", "long_overnight"):
        col = per_day.get(st, pd.Series(0, index=per_day.index))
        c60 = last60.get(st, pd.Series(0, index=last60.index))
        sd = book[book.strategy == st]["net_exec"].std(ddof=1)
        exp30 = float(col.mean() * 30)
        freq[st] = {"сделок_в_день": round(float(col.mean()), 2),
                    "дней_без_сделок_%": round(float((col == 0).mean() * 100), 1),
                    "за_60_дней_сделок_в_день": round(float(c60.mean()), 2),
                    "ожидаемо_за_30_дней": round(exp30, 1),
                    "sd_сделки_п.п.": round(float(sd), 3) if np.isfinite(sd) else None,
                    "MDE_за_30_дней_п.п.": round(float(2.8 * sd / np.sqrt(exp30)), 3)
                    if exp30 > 1 and np.isfinite(sd) else None}
    res["frequency"] = freq

    # 3. Капитал: вклад стратегии в счёт 2 млн против казначейства
    trades_day = sum(v["сделок_в_день"] for v in freq.values())
    mean_exec = float(common["net_exec"].mean()) if len(common) else np.nan
    mean_exec_stop = float(common["net_exec_stop"].mean()) if len(common) else np.nan
    rub_day = trades_day * POSITION_RUB * mean_exec / 100.0
    res["capital"] = {
        "капитал_₽": CAPITAL_RUB, "позиция_₽": POSITION_RUB, "макс_в_рынке_₽": TOP_N * POSITION_RUB,
        "доля_в_рынке_макс_%": round(TOP_N * POSITION_RUB / CAPITAL_RUB * 100, 2),
        "средняя_нетто_сделки_исполнение_%": round(mean_exec, 4),
        "средняя_нетто_сделки_со_стопом_%": round(mean_exec_stop, 4),
        "стратегия_со_стопом_%_капитала_в_год": round(trades_day * POSITION_RUB * mean_exec_stop
                                                     / 100.0 * 252 / CAPITAL_RUB * 100, 3),
        "сделок_в_день": round(trades_day, 2),
        "₽_в_день_стратегия": round(rub_day, 2),
        "стратегия_%_капитала_в_год": round(rub_day * 252 / CAPITAL_RUB * 100, 3),
        "казначейство_%_в_год_при_ставке": {"ставка_допущение_%": costs.DEFAULT_KEY_RATE_PCT,
                                            "≈": round(costs.DEFAULT_KEY_RATE_PCT
                                                       * (1 - TOP_N * POSITION_RUB / CAPITAL_RUB), 2)},
    }

    # 4. Фактическое исполнение в песочнице (попытка 1, r2)
    conn = data._conn()
    try:
        ea = pd.read_sql("SELECT asof_date, ticker, strategy, side, filled, requested_price, "
                         "filled_price, slippage_pct, expected_slippage_pct, exit_reason, "
                         "pnl_net_rub, run_id FROM execution_audit ORDER BY created_at", conn)
    finally:
        conn.close()
    res["execution_audit"] = json.loads(ea.to_json(orient="records", date_format="iso",
                                                   force_ascii=False))

    with open(os.path.join(args.out, "reaudit_r3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
