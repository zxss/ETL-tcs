"""
Шорт-правило без TFT (ТЗ EDGE-R4 §4, блок B). Только исследование.

ПРАВИЛО, ВАРИАНТЫ И СТОПЫ ЗАМОРОЖЕНЫ коммитом этого файла до прогона на
отложенной выборке (B1, B7). Хеш коммита пишется в отчёт (файл REVISION).

Условия — из кода прода, не из пересказа
(tft_forecast/combined.apply_strategy_specialisation, tft_forecast/market):
  * ret1 < 0 — доходность последнего закрытого дня бумаги (market._ret1);
  * market_atr_pctl > 50 — медиана по вселенной перцентиля ATR(14) за 252 бара
    (market._atr_pctl → atr_pctl_market);
  * запрет, когда IMOEX выше EMA50 (market._above_ema, окно 320 баров);
  * нешортуемые: non_shortable_tickers() и short_enabled=False в instruments.json;
  * лот дороже позиции 10 000 ₽ — пропуск ПОСЛЕ отбора, слот не замещается
    (combined.compute_orders).
Признаки — по будничным дневкам (INCLUDE_WEEKEND_TRADING=0) на дату A′:
последний будничный торговый день строго до дня исполнения A, как утренний
PREP. «Фильтры ликвидности прода» в режиме heuristic сделку не запрещают
(_score_row → allowed=True): ликвидность входит только в рейтинг TFT, которого
здесь нет.

Отличия от прода, заданные заранее:
  * бумага без бара A′ (не торговалась накануне) не рассматривается — прод взял
    бы ret1 по старым барам;
  * дни, когда у IMOEX меньше 50 будничных баров (история БД с 28.05.2024), не
    оцениваются: прод пропустил бы предохранитель (None), то есть торговал бы
    другое правило;
  * перцентиль ATR в начале истории — по неполным 252 барам, как в проде на
    короткой базе.

Окна (research/session_calendar): вход — open первого бара не раньше «начало
основной + 5 минут» (10:05 до 14.09.2026, 09:15 после) и не позже ещё через
10 минут; выход — close последнего бара не позже 18:15 (фаза 18:20). Через
ночь не держим, плата за перенос не возникает.

Варианты, других нет:
  отбор (B2): all   — все прошедшие, равный вес;
              drop5 — топ-5 по наибольшему падению ret1;
              liq5  — топ-5 по среднему рублёвому обороту за 20 баров;
  стоп (B4):  none  — выход 18:20;
              atr   — max(1,5 × ATR14 %, 3,5 %); исполнение по max(стоп,
                      open бара пересечения) — гэп сквозь стоп не прощается;
              боевой стоп — только справка по книге r3 (audit/out/book_r3.csv):
                      без TFT нет q10.
  издержки:   research/cost_model — комиссия тарифа «Премиум» 0,04 % за сделку
              (круг 0,08 %) + спред бумаги (base), стресс-спред, только
              комиссия (fee). Пересчёт 15.09.2026: плоские 0,128/0,20 %
              заменены тарифом пользователя, правило не менялось. P&L без TMON.

Статистика по дням: среднее сделок дня → t по активным дням. Бета — регрессия
дня книги на ход IMOEX в том же окне; альфа над шортом индекса в те же дни.

Периоды: dev 01.12.2025 – 11.09.2026 (просмотрен тремя аудитами), holdout — с
начала истории 5-минуток по 30.11.2025. Вердикт — по holdout.

Запуск (на сервере, где БД; только чтение):
    python -m research.short_rule --from 2024-05-21 --to 2026-09-11
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import session_calendar as sc              # noqa: E402
from research.news_event_study import UNIVERSE           # noqa: E402

log = logging.getLogger("research.short_rule")

INDEX = "IMOEX"
POSITION_RUB = 10_000.0
TOP_N = 5
ATR_WIN, ATR_HIST, LOOKBACK, MIN_ROWS = 14, 252, 320, 30
EMA_SPAN = 50
MKT_ATR_MIN = 50.0
STOP_ATR_K, STOP_FLOOR_PCT = 1.5, 3.5
LIQ_WIN = 20
ENTRY_TOL = dt.timedelta(minutes=10)
COSTS = cm.SCENARIOS                     # сценарии издержек research/cost_model
DEV_FROM = dt.date(2025, 12, 1)
PRICE_QUANTUM = 1e-4           # точность хранения цены в market_data / market_data_5m
MAX_QUANTUM_PCT = 0.05         # шаг записи не грубее 0,05 % цены
SELECTIONS = ("all", "drop5", "liq5")
STOPS = ("none", "atr")
_BASE_NON_SHORTABLE = {"AKRN", "CBOM", "MVID"}


# ── Признаки (как в tft_forecast/market) ─────────────────────────────────────

def ticker_features(g: pd.DataFrame, lot: int = 1) -> pd.DataFrame:
    """Будничные дневки одной бумаги → признаки на каждую дату.

    Совпадает с market._ret1/_atr_pctl/_atr_pct на окне 320 баров: ATR(14) —
    скользящее среднее TR, перцентиль — доля последних ≤252 значений ATR не
    выше текущего. Прод берёт бумагу в контекст, только если у неё ≥ 30 баров.
    """
    g = g.sort_values("date").reset_index(drop=True)
    h, lo, c = g["high"].astype(float), g["low"].astype(float), g["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - lo), (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN).mean()
    valid = atr.dropna()
    pctl = valid.rolling(ATR_HIST, min_periods=1).apply(
        lambda w: float((w <= w[-1]).mean() * 100.0), raw=True).reindex(atr.index)
    ok = pd.Series(np.arange(1, len(g) + 1) >= MIN_ROWS, index=g.index)
    out = pd.DataFrame({"date": g["date"],
                        "ret1": (c / prev - 1.0) * 100.0,
                        "atr_pctl": pctl,
                        "atr_pct": atr / c * 100.0,
                        "adv_rub": (c * g["volume"].astype(float) * lot)
                        .rolling(LIQ_WIN, min_periods=LIQ_WIN).mean()})
    for col in ("ret1", "atr_pctl", "atr_pct"):
        out.loc[~ok, col] = np.nan
    return out


def index_above_ema(close: pd.Series) -> pd.Series:
    """market._above_ema на окне 320 баров; None при < 50 баров."""
    c = close.astype(float).to_numpy()
    out = []
    for i in range(len(c)):
        w = c[max(0, i - LOOKBACK + 1): i + 1]
        if len(w) < EMA_SPAN:
            out.append(None)
            continue
        ema = pd.Series(w).ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1]
        out.append(bool(w[-1] > ema))
    return pd.Series(out, index=close.index, dtype=object)


def build_features(daily: pd.DataFrame, lots: dict[str, int]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(признаки бумаг long, market_atr_pctl по дате, IMOEX выше EMA50 по дате)."""
    wd = daily[pd.to_datetime(daily["date"]).dt.dayofweek < 5].copy()
    wd["date"] = pd.to_datetime(wd["date"]).dt.date
    feats = []
    for tk, g in wd[wd["ticker"].isin(UNIVERSE)].groupby("ticker"):
        f = ticker_features(g, lots.get(tk, 1))
        f["ticker"] = tk
        feats.append(f)
    feats = pd.concat(feats, ignore_index=True)
    # atr_pctl_market: медиана по бумагам их последнего значения на дату
    piv = feats.pivot(index="date", columns="ticker", values="atr_pctl").sort_index().ffill()
    mkt = piv.median(axis=1, skipna=True)
    idx = wd[wd["ticker"] == INDEX].sort_values("date")
    above = pd.Series(index_above_ema(idx["close"]).to_numpy(), index=idx["date"].to_numpy())
    return feats, mkt, above


def trading_dates(daily: pd.DataFrame, min_tickers: int = 20) -> list[dt.date]:
    wd = daily[pd.to_datetime(daily["date"]).dt.dayofweek < 5]
    cnt = wd[wd["ticker"].isin(UNIVERSE)].groupby("date")["ticker"].nunique()
    return sorted(pd.to_datetime(cnt[cnt >= min_tickers].index).date)


def prev_date(dates: list[dt.date], day: dt.date) -> dt.date | None:
    import bisect
    i = bisect.bisect_left(dates, day)
    return dates[i - 1] if i > 0 else None


# ── Окна исполнения из 5-минуток ─────────────────────────────────────────────

def window_paths(bars: pd.DataFrame) -> dict:
    """(ticker, day) → (opens, highs, closes) от бара входа до бара 18:15."""
    out = {}
    bars = bars.sort_values(["ticker", "tm"])
    for (tk, d), g in bars.groupby(["ticker", "d"], sort=False):
        start = np.datetime64(dt.datetime.combine(d, sc.short_entry(d)), "ns")
        stop = np.datetime64(dt.datetime.combine(d, sc.SHORT_EXIT_BAR), "ns")
        tm = g["tm"].to_numpy(dtype="datetime64[ns]")
        m = (tm >= start) & (tm <= stop)
        if not m.any():
            continue
        if tm[m][0] - start > np.timedelta64(ENTRY_TOL):
            continue
        gg = g[m]
        out[(tk, d)] = (gg["open"].to_numpy(float), gg["high"].to_numpy(float),
                        gg["close"].to_numpy(float))
    return out


def short_outcome(opens, highs, closes, stop_pct: float | None = None) -> dict:
    """Шорт от open первого бара до close последнего; валовая доходность, %."""
    entry, exitp = float(opens[0]), float(closes[-1])
    mae = (float(np.max(highs)) / entry - 1.0) * 100.0
    gross = -(exitp / entry - 1.0) * 100.0
    if stop_pct is None:
        return {"gross": gross, "mae": mae, "stopped": False}
    level = entry * (1.0 + stop_pct / 100.0)
    hit = np.nonzero(np.asarray(highs) >= level)[0]
    if len(hit):
        fill = max(level, float(opens[hit[0]]))
        loss = -(fill / entry - 1.0) * 100.0
        return {"gross": loss, "mae": -loss, "stopped": True}
    return {"gross": gross, "mae": mae, "stopped": False}


def index_move(paths: dict, day: dt.date) -> float:
    p = paths.get((INDEX, day))
    if p is None:
        return np.nan
    return (float(p[2][-1]) / float(p[0][0]) - 1.0) * 100.0


# ── Правило и отбор ──────────────────────────────────────────────────────────

def quantum_ok(price: float) -> bool:
    """Шаг хранения цены (4 знака) не больше MAX_QUANTUM_PCT от цены.

    market_data_5m хранит цены как NUMERIC(18,4): у TGKA (≈ 0,006 ₽) шаг
    записи ≈ 1,6 % цены — доходность такой бумаги измеряется шумом округления.
    Исключение задано по цене, до просмотра доходностей.
    """
    return price > 0 and PRICE_QUANTUM / price * 100.0 <= MAX_QUANTUM_PCT


def gates_open(mkt_atr: float | None, above: bool | None) -> bool | None:
    """Рыночные ворота правила на дату A′. None — день не оценивается."""
    if above is None:
        return None
    if mkt_atr is not None and not (isinstance(mkt_atr, float) and math.isnan(mkt_atr)):
        if not mkt_atr > MKT_ATR_MIN:
            return False
    return not above


def select(cands: pd.DataFrame, how: str) -> pd.DataFrame:
    """Отбор B2 по кандидатам одного дня (ticker, ret1, adv_rub)."""
    if how == "all":
        return cands
    if how == "drop5":
        return cands.sort_values(["ret1", "ticker"]).head(TOP_N)
    if how == "liq5":
        return cands.sort_values(["adv_rub", "ticker"], ascending=[False, True]).head(TOP_N)
    raise ValueError(how)


def run_rule(feats, mkt, above, tdates, paths, lots, non_shortable,
             exec_days) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сделки всех вариантов и журнал дней (ворота, кандидаты)."""
    by_date = {d: g for d, g in feats.groupby("date")}
    trades, days = [], []
    for A in exec_days:
        Ap = prev_date(tdates, A)
        if Ap is None:
            continue
        m = mkt.get(Ap)
        ab = above.get(Ap)
        gate = gates_open(None if m is None or (isinstance(m, float) and math.isnan(m)) else float(m), ab)
        rec = {"date": A, "asof": Ap, "market_atr_pctl": m, "index_above_ema50": ab,
               "gate": gate, "candidates": 0, "imoex_move": index_move(paths, A)}
        if gate is not True:
            days.append(rec)
            continue
        f = by_date.get(Ap)
        if f is None:
            days.append(rec)
            continue
        c = f[f["ret1"].notna() & (f["ret1"] < 0) & ~f["ticker"].isin(non_shortable)]
        rec["candidates"] = int(len(c))
        days.append(rec)
        for how in SELECTIONS:
            for r in select(c, how).itertuples(index=False):
                p = paths.get((r.ticker, A))
                t = {"date": A, "asof": Ap, "ticker": r.ticker, "selection": how,
                     "ret1": r.ret1, "atr_pct": r.atr_pct, "adv_rub": r.adv_rub}
                if p is None:
                    t["status"] = "no_data"
                    trades.append(t)
                    continue
                entry = float(p[0][0])
                if entry * lots.get(r.ticker, 1) > POSITION_RUB:
                    t["status"] = "lot_over_position"
                    trades.append(t)
                    continue
                if not quantum_ok(entry):
                    t["status"] = "price_quantum"
                    trades.append(t)
                    continue
                t["status"] = "ok"
                t["entry"] = entry
                o = short_outcome(*p)
                t.update(gross_none=o["gross"], mae_none=o["mae"])
                sp = max(STOP_ATR_K * r.atr_pct, STOP_FLOOR_PCT) if r.atr_pct == r.atr_pct else STOP_FLOOR_PCT
                s = short_outcome(*p, stop_pct=sp)
                t.update(stop_pct=sp, gross_atr=s["gross"], mae_atr=s["mae"], stopped_atr=s["stopped"])
                trades.append(t)
    return pd.DataFrame(trades), pd.DataFrame(days)


# ── Статистика ───────────────────────────────────────────────────────────────

def _t_p(x: pd.Series) -> tuple[float | None, float | None]:
    n = len(x)
    if n < 2:
        return None, None
    sd = float(x.std(ddof=1))
    if not sd:
        return None, None
    t = float(x.mean()) / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except ImportError:                                   # pragma: no cover
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return t, p


def variant_stats(tr: pd.DataFrame, days: pd.DataFrame, stop: str, cost: str) -> dict:
    """Сделки одного варианта (selection уже отфильтрован) → метрики §4.

    cost — сценарий издержек (колонка cost_<сценарий> у сделки)."""
    ok = tr[tr["status"] == "ok"].copy()
    if ok.empty:
        return {"n": 0}
    ok["net"] = ok[f"gross_{stop}"] - ok[f"cost_{cost}"]
    per_day = ok.groupby("date")["net"].mean().sort_index()
    t, p = _t_p(per_day)
    total = float(per_day.sum())
    top5 = float(per_day.nlargest(5).sum())
    worst_day = per_day.idxmin()
    mae = ok[f"mae_{stop}"]
    res = {
        "n": int(len(ok)), "days": int(len(per_day)),
        "mean_trade": float(ok["net"].mean()), "median_trade": float(ok["net"].median()),
        "hit": float((ok["net"] > 0).mean()),
        "mean_day": float(per_day.mean()), "median_day": float(per_day.median()),
        "t_day": t, "p_day": p,
        "stopped_share": float(ok["stopped_atr"].mean()) if stop == "atr" else None,
        "mae_p95": float(mae.quantile(0.95)), "mae_p99": float(mae.quantile(0.99)),
        "mae_max": float(mae.max()),
        "worst_trade": float(ok["net"].min()),
        "worst_trade_at": f"{ok.loc[ok['net'].idxmin(), 'ticker']} {ok.loc[ok['net'].idxmin(), 'date']}",
        "worst_day": float(per_day.min()), "worst_day_at": str(worst_day),
        "sum_days": total,
        "top5_share": top5 / total if total > 0 else None,
        "sum_wo_top5": total - top5,
        "sum_wo_top10": total - float(per_day.nlargest(10).sum()),
        "no_data": int((tr["status"] == "no_data").sum()),
        "lot_skipped": int((tr["status"] == "lot_over_position").sum()),
        "quantum_skipped": int((tr["status"] == "price_quantum").sum()),
    }
    # B5: бета к IMOEX в том же окне и превышение над шортом индекса
    ix = days.set_index("date")["imoex_move"].reindex(per_day.index)
    gross_day = ok.groupby("date")[f"gross_{stop}"].mean().reindex(per_day.index)
    both = pd.DataFrame({"y": gross_day, "x": ix}).dropna()
    if len(both) >= 10:
        b, a = np.polyfit(both["x"], both["y"], 1)
        resid = both["y"] - (a + b * both["x"])
        n = len(both)
        sxx = float(((both["x"] - both["x"].mean()) ** 2).sum())
        s2 = float((resid ** 2).sum()) / (n - 2)
        se_a = math.sqrt(s2 * (1.0 / n + both["x"].mean() ** 2 / sxx)) if sxx else float("nan")
        excess = both["y"] + both["x"]            # книга − шорт индекса, одинаковые издержки
        te, pe = _t_p(excess)
        res.update(beta=float(b), alpha_day=float(a), alpha_t=float(a / se_a) if se_a else None,
                   index_days=int(n), index_short_day=float((-both["x"]).mean()),
                   excess_vs_index=float(excess.mean()), excess_t=te, excess_p=pe)
    return res


def holm(pvals: dict) -> dict:
    """Поправка Холма: {имя: p} → {имя: скорректированное p}."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def dor(st: dict, st_hi: dict) -> dict:
    """Definition of Ready r4 (§9, пп. 1–3) по варианту на holdout."""
    if not st.get("n"):
        return {}
    return {
        "1. среднее/день > 0 и t ≥ 2,5": bool(st["mean_day"] > 0 and (st["t_day"] or 0) >= 2.5),
        "1'. при стресс-спреде среднее/день > 0": bool(st_hi.get("mean_day", -1) > 0),
        "2. без 5 лучших дней > 0": bool(st["sum_wo_top5"] > 0),
        "2'. доля 5 лучших ≤ 50 %": bool(st["top5_share"] is not None and st["top5_share"] <= 0.5),
        "3. медиана сделки ≥ +0,20 %": bool(st["median_trade"] >= 0.20),
    }


def r3_combat_reference(path: str) -> dict:
    """Справка B4(в): книга r3, боевой стоп — только там, где был q10 TFT."""
    if not os.path.exists(path):
        return {}
    b = pd.read_csv(path, parse_dates=["exec_day"])
    b = b[b["strategy"] == "intraday_short"]
    out = {}
    for col, name in (("net_exec", "без стопа"), ("net_exec_stop", "боевой стоп")):
        d = b.dropna(subset=[col])
        per_day = d.groupby("exec_day")[col].mean()
        t, _ = _t_p(per_day)
        out[name] = {"n": int(len(d)), "days": int(len(per_day)), "mean": float(d[col].mean()),
                     "median": float(d[col].median()), "t_day": t}
    out["доля стопов"] = float(b["stopped"].mean()) if "stopped" in b else None
    out["период"] = f"{b['exec_day'].min().date()} … {b['exec_day'].max().date()}"
    return out


# ── Загрузка ─────────────────────────────────────────────────────────────────

def load_daily(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT ticker, date, open, high, low, close, volume FROM market_data "
                     "WHERE ticker = ANY(%s) AND open > 0 AND high > 0 AND low > 0 AND close > 0 "
                     "ORDER BY ticker, date", conn, params=(list(UNIVERSE) + [INDEX],))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, close
FROM market_data_5m
WHERE ticker = ANY(%(tk)s) AND close > 0
  AND ts >= %(f)s AND ts < %(t)s
  AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN TIME '09:15' AND TIME '18:15'
"""


def load_bars(conn, d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    msk = dt.timezone(dt.timedelta(hours=3))
    df = pd.read_sql(BARS_SQL, conn, params={
        "tk": list(UNIVERSE) + [INDEX],
        "f": dt.datetime.combine(d_from, dt.time(), msk),
        "t": dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), msk)})
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    for c in ("open", "high", "close"):
        df[c] = df[c].astype(float)
    return df


def load_lots_and_blocked() -> tuple[dict, set]:
    path = os.path.join(ROOT, "audit", "out", "instruments.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    lots = {r["ticker"]: int(r.get("lot") or 1) for r in rows}
    blocked = {r["ticker"] for r in rows if r.get("short_enabled") is False}
    extra = set(os.getenv("NON_SHORTABLE_TICKERS", "").upper().split())
    return lots, blocked | _BASE_NON_SHORTABLE | extra


# ── Отчёт ────────────────────────────────────────────────────────────────────

def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    s = f"{x * 100:.1f} %" if pct else f"{x:+.{nd}f}"
    return s.replace(".", ",")


def report(res: dict, meta: dict) -> str:
    L = [f"# Шорт-правило без TFT (ТЗ EDGE-R4, блок B)", "",
         f"Сформировано {meta['created']}. Код заморожен коммитом `{meta['revision']}`. "
         "Режим: только исследование, r3 и кроны не затронуты.", "",
         "Нетто в % на сделку/день после издержек, t — по активным дням. "
         "Отбор: all — все прошедшие; drop5 — топ-5 по падению вчера; liq5 — топ-5 по обороту. "
         "Стоп: none — выход 18:20; atr — max(1,5×ATR14, 3,5 %).", ""]
    for per in ("holdout", "dev"):
        pm = meta["periods"][per]
        L += [f"## {'Отложенная выборка (вердикт)' if per == 'holdout' else 'Dev (просмотрен аудитами)'}: "
              f"{pm['from']} … {pm['to']}", "",
              f"- дней исполнения {pm['exec_days']}, из них с открытыми воротами {pm['gate_days']} "
              f"({_f(pm['gate_share'], pct=True)}), не оценивались (предохранитель IMOEX без данных) "
              f"{pm['skipped']}; кандидатов в день ворот — медиана {pm['cand_median']}", ""]
        for cost in COSTS:
            L += [f"### Издержки: {cm.LABELS[cost]}", "",
                  "| вариант | сделок | дней | ср. сделка | медиана | ср. день | t | p Холма | "
                  "стопов | MAE p95/p99/max | худшая сделка | худший день | доля 5 лучших | "
                  "без топ-5 | без топ-10 | бета | альфа/день (t) | книга − шорт IMOEX (t) |",
                  "|" + "---|" * 18]
            for sel in SELECTIONS:
                for stop in STOPS:
                    st = res[per][cost][(sel, stop)]
                    if not st.get("n"):
                        L.append(f"| {sel}/{stop} | 0 |" + " — |" * 16)
                        continue
                    L.append(
                        f"| {sel}/{stop} | {st['n']} | {st['days']} | {_f(st['mean_trade'])} | "
                        f"{_f(st['median_trade'])} | {_f(st['mean_day'])} | {_f(st['t_day'], 2)} | "
                        f"{_f(st.get('p_holm'), 3)} | "
                        f"{_f(st['stopped_share'], pct=True) if st['stopped_share'] is not None else '—'} | "
                        f"{_f(st['mae_p95'], 2)}/{_f(st['mae_p99'], 2)}/{_f(st['mae_max'], 2)} | "
                        f"{_f(st['worst_trade'], 2)} ({st['worst_trade_at']}) | "
                        f"{_f(st['worst_day'], 2)} ({st['worst_day_at']}) | "
                        f"{_f(st['top5_share'], pct=True) if st['top5_share'] is not None else '—'} | "
                        f"{_f(st['sum_wo_top5'], 2)} | {_f(st['sum_wo_top10'], 2)} | "
                        f"{_f(st.get('beta'), 2)} | {_f(st.get('alpha_day'))} ({_f(st.get('alpha_t'), 2)}) | "
                        f"{_f(st.get('excess_vs_index'))} ({_f(st.get('excess_t'), 2)}) |")
            L.append("")
        if per == "holdout":
            L += ["### Definition of Ready r4 (§9, пп. 1–3) на отложенной выборке", "",
                  "| вариант | " + " | ".join(next(iter(res["dor"].values())).keys()) + " |",
                  "|---|" + "---|" * len(next(iter(res["dor"].values())))]
            for k, v in res["dor"].items():
                L.append(f"| {k} | " + " | ".join("✓" if x else "✗" for x in v.values()) + " |")
            L.append("")
    ref = res.get("r3_ref") or {}
    if ref:
        L += ["## Справка B4(в): боевой стоп на книге r3 (окна r3, издержки — матрица cost_rt)", "",
              f"Период {ref['период']}, доля стопов {_f(ref['доля стопов'], pct=True)}.", "",
              "| исход | сделок | дней | среднее | медиана | t |", "|---|---|---|---|---|---|"]
        for k in ("без стопа", "боевой стоп"):
            v = ref[k]
            L.append(f"| {k} | {v['n']} | {v['days']} | {_f(v['mean'])} | {_f(v['median'])} | {_f(v['t_day'], 2)} |")
        L.append("")
    L += ["## Как читать", "",
          "- Вердикт — только по отложенной выборке; dev показан для сравнения, лучший вариант задним числом не выбирается.",
          "- «Бета» и «книга − шорт IMOEX»: если превышение над шортом индекса незначимо, книга — тайминг рынка, "
          "и r4 проектируется как шорт индекса (фьючерс), а не отбор бумаг (§4, B5).",
          "- p Холма — поправка на 6 вариантов (3 отбора × 2 стопа) внутри периода и уровня издержек.",
          "- Выборка отложена частично: дневные сигналы 2024–2025 видели аудиты 1–2, 5-минутные окна — никто.", ""]
    return "\n".join(L)


# ── Главное ──────────────────────────────────────────────────────────────────

def exec_days_from(paths: dict, min_tickers: int = 20) -> list[dt.date]:
    cnt: dict[dt.date, int] = {}
    for tk, d in paths:
        if tk != INDEX:
            cnt[d] = cnt.get(d, 0) + 1
    return sorted(d for d, n in cnt.items() if n >= min_tickers and d.weekday() < 5)


def evaluate(trades: pd.DataFrame, days: pd.DataFrame, d_from, d_to) -> tuple[dict, dict]:
    tr = trades[(trades["date"] >= d_from) & (trades["date"] <= d_to)] if len(trades) else trades
    dd = days[(days["date"] >= d_from) & (days["date"] <= d_to)]
    out = {}
    for cost in COSTS:
        out[cost] = {}
        for sel in SELECTIONS:
            for stop in STOPS:
                sub = tr[tr["selection"] == sel] if len(tr) else tr
                out[cost][(sel, stop)] = variant_stats(sub, dd, stop, cost) if len(sub) else {"n": 0}
        adj = holm({k: v.get("p_day") for k, v in out[cost].items()})
        for k, v in out[cost].items():
            v["p_holm"] = adj.get(k)
    gd = dd[dd["gate"] == True]                            # noqa: E712
    meta = {"from": str(d_from), "to": str(d_to), "exec_days": int(len(dd)),
            "gate_days": int(len(gd)), "gate_share": len(gd) / len(dd) if len(dd) else None,
            "skipped": int(dd["gate"].isna().sum()),
            "cand_median": float(gd["candidates"].median()) if len(gd) else None}
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Шорт-правило без TFT (ТЗ EDGE-R4, блок B)")
    ap.add_argument("--from", dest="date_from", default="2024-05-21")
    ap.add_argument("--to", dest="date_to", default="2026-09-11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    d_from, d_to = dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to)
    conn = database.get_connection()
    try:
        daily = load_daily(conn)
        log.info("5-минутки окна шорта…")
        bars = load_bars(conn, d_from, d_to)
    finally:
        conn.close()
    lots, blocked = load_lots_and_blocked()
    feats, mkt, above = build_features(daily, lots)
    tdates = trading_dates(daily)
    paths = window_paths(bars)
    del bars
    edays = exec_days_from(paths)
    log.info("дней исполнения %d (%s … %s)", len(edays), edays[0], edays[-1])
    trades, days = run_rule(feats, mkt, above, tdates, paths, lots, blocked, edays)
    spreads = cm.load_spreads()
    for s in COSTS:
        trades[f"cost_{s}"] = [cm.round_trip(t, s, spreads) for t in trades["ticker"]]

    res, meta = {}, {"periods": {}}
    hold_to = DEV_FROM - dt.timedelta(days=1)
    for per, lo, hi in (("holdout", d_from, hold_to), ("dev", DEV_FROM, d_to)):
        res[per], meta["periods"][per] = evaluate(trades, days, lo, hi)
    res["dor"] = {f"{s}/{p}": dor(res["holdout"][cm.PRIMARY][(s, p)], res["holdout"][cm.SENSITIVITY][(s, p)])
                  for s in SELECTIONS for p in STOPS if res["holdout"][cm.PRIMARY][(s, p)].get("n")}
    res["r3_ref"] = r3_combat_reference(os.path.join(ROOT, "audit", "out", "book_r3.csv"))
    try:
        with open(os.path.join(ROOT, "REVISION"), encoding="utf-8") as f:
            rev = f.read().strip()[:12]
    except OSError:
        rev = "unknown"
    now = dt.datetime.now()
    meta.update(created=now.strftime("%d.%m.%Y %H:%M"), revision=rev)
    text = report(res, meta)
    out = a.out or os.path.join(ROOT, "audit", "r4_research", f"short_rule-{now:%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    trades.to_csv(os.path.join(out, "trades.csv"), index=False)
    days.to_csv(os.path.join(out, "days.csv"), index=False)
    flat = {per: {c: {f"{s}/{p}": v for (s, p), v in res[per][c].items()} for c in COSTS}
            for per in ("holdout", "dev")}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": flat, "dor": res["dor"], "r3_ref": res["r3_ref"]},
                  f, ensure_ascii=False, indent=1, default=str)
    print(text)
    log.info("отчёт: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
