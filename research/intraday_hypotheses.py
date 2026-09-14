"""
Внутридневные гипотезы при правиле капитала «90 % депозита в TMON» (задание
пользователя 15.09.2026). Только исследование: читает БД, в торговые таблицы не
пишет, торговый контур и тест r3 не затрагивает.

ГИПОТЕЗЫ, ПАРАМЕТРЫ И СПОСОБ ОЦЕНКИ ЗАМОРОЖЕНЫ коммитом этого файла до прогона
(хеш — в отчёте, файл REVISION).

Общее для всех гипотез
  * Только основная сессия; позиция закрывается внутри дня, не позже close бара
    18:15 (фаза 18:20). Плата за перенос непокрытой позиции не возникает: по
    тарифу позиция, закрытая до конца торгового дня, бесплатна.
  * Сигнал известен на close бара, вход — open следующего бара.
  * Издержки — research/cost_model: комиссия тарифа «Премиум» 0,04 % за сделку
    (круг 0,08 %) + спред бумаги; основной сценарий base, чувствительность
    stress, нижняя граница fee (только комиссия). Пересчёт 15.09.2026: плоские
    0,128/0,20 % заменены тарифом пользователя, правила гипотез не менялись.
  * Бумаги с шагом хранения цены грубее 0,05 % (NUMERIC(18,4): TGKA, FEES)
    исключены по цене. Шорт — только по шортуемым (не AKRN, CBOM, MVID).
  * t — по дням (сделки одного дня связаны общим шоком); бета к IMOEX в окне
    сделки; «превышение над индексом» = доход сделки − направление × ход IMOEX
    в том же окне. Поправка Холма на 4 основных теста. DoR r4 (§9 EDGE-R4-TZ)
    по отложенной выборке 21.05.2024–30.11.2025; 01.12.2025–11.09.2026 —
    повтор. Лучший вариант задним числом не выбирается: у каждой гипотезы один
    основной вариант, разрезы — описательные.
  * История до 14.09.2026: основная сессия с 10:00 после утренней сессии
    (06:50–09:50). Режима «аукцион 09:00 → торги 09:10 без утренней сессии»
    в истории нет — время гипотез задано событиями сессии
    (research/session_calendar), перенос результата на новый режим — с
    оговоркой.

H1 Fade Retail Panics (лонг)
  Вселенная дня — топ-15 по среднему рублёвому обороту основной сессии за 20
  предыдущих дней. Сигнал на close 5-минутного бара t (t ≥ начало основной +
  30 мин, t ≤ 16:40): ход бумаги за 30 минут < −2,0 %, ход IMOEX за те же
  30 минут > −0,5 %, объём бара > 3 × среднего объёма 20 предыдущих баров.
  Вход — open следующего бара. Выход — тейк при касании VWAP дня (VWAP по
  барам до предыдущего включительно; исполнение по max(open, VWAP)) или close
  через 90 минут от входа. Одна сделка на бумагу в день. Стопа в постановке нет.

H2 Fade Opening Gap
  Гэп = open первого бара основной сессии / последняя цена до дня D − 1.
  |гэп| > 1,5 % и нет поста о бумаге (хештег или название, без отчётов о
  движении цены) с 18:50 предыдущего торгового дня до входа → вход против гэпа
  по open бара «начало основной + 5 мин» (10:05; с 14.09.2026 — 09:15), выход —
  close бара «начало основной + 75 мин» (цена через 80 минут: 11:20; с
  14.09.2026 — 10:30). Гэп вверх — шорт, вниз — лонг.

H3 Cross-Sectional Pairs (рыночно-нейтрально, внутри дня)
  Кластеры: нефть и газ LKOH, ROSN, SNGS, TATN; банки SBER, VTBR, T; металлурги
  CHMF, NLMK, MAGN — все пары внутри кластера. Спред s = ln(Pa/Pb) по close
  5-минуток; z = (s − среднее) / σ за 520 предыдущих общих баров (≈5 дней,
  минимум 260). |z| ≥ 2 на close бара (не позже 17:45) → на open следующего
  бара шорт дорогой ноги и лонг дешёвой, по 10 000 ₽ на ногу. Выход на open
  бара после того, как z пересёк 0, стоп — |z| ≥ 3,5 (на open следующего бара),
  иначе close бара 18:15. Одна сделка на пару в день. Горизонт «1–3 дня» из
  постановки противоречит правилу «только внутри дня» и платит перенос шорта
  (40 ₽ в календарный день на 10 000 ₽) — проверяется внутридневной вариант.
  Доход — на ногу: ход лонга − ход шорта; издержки — две ноги.

H4 Regime Breakdown (шорт)
  В 14:00 (close бара 13:55): IMOEX ниже минимума предыдущего торгового дня
  более чем на 0,5 % И рублёвый оборот вселенной с начала основной сессии до
  14:00 выше медианы того же окна за 20 предыдущих дней → шорт 3 самых слабых
  бумаг (ход от последней цены до дня D до close бара 13:55) по open бара
  14:00, выход — close бара 18:15. Стопа в постановке нет.

Запуск (на сервере, где БД):
    python -m research.intraday_hypotheses --from 2024-05-21 --to 2026-09-11
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
from research import news_event_study as ns              # noqa: E402
from research import session_calendar as sc              # noqa: E402
from research import short_rule as sr                    # noqa: E402

log = logging.getLogger("research.intraday")

INDEX = "IMOEX"
SCENARIOS = cm.SCENARIOS                   # издержки — research/cost_model (тариф «Премиум»)
POSITION_RUB = 10_000.0
DEV_FROM = dt.date(2025, 12, 1)
LAST_BAR = dt.time(18, 15)
BAR = pd.Timedelta(minutes=5)

H1_TOP, H1_R30, H1_IDX30, H1_VOLX, H1_VOLN = 15, -2.0, -0.5, 3.0, 20
H1_FROM_OPEN, H1_LAST_SIGNAL, H1_HOLD = pd.Timedelta(minutes=30), dt.time(16, 40), pd.Timedelta(minutes=90)
H1_LIQ_DAYS = 20

H2_GAP = 1.5
H2_ENTRY, H2_EXIT = pd.Timedelta(minutes=5), pd.Timedelta(minutes=75)   # бар входа / бар выхода
H2_NEWS_FROM = dt.time(18, 50)
H2_TOL = pd.Timedelta(minutes=10)

H3_CLUSTERS = {"нефть и газ": ("LKOH", "ROSN", "SNGS", "TATN"),
               "банки": ("SBER", "VTBR", "T"),
               "металлурги": ("CHMF", "NLMK", "MAGN")}
H3_WIN, H3_MIN, H3_Z_IN, H3_Z_STOP = 520, 260, 2.0, 3.5
H3_LAST_SIGNAL = dt.time(17, 45)
H3_TICKERS = tuple(sorted({t for c in H3_CLUSTERS.values() for t in c}))

H4_SIGNAL_BAR, H4_ENTRY_BAR = dt.time(13, 55), dt.time(14, 0)
H4_BREAK, H4_TOP, H4_VOL_DAYS = 0.5, 3, 20

HYPOTHESES = ("H1", "H2", "H3", "H4")
NAMES = {"H1": "Fade Retail Panics (лонг)", "H2": "Fade Opening Gap",
         "H3": "Pairs внутри кластеров (внутри дня)", "H4": "Regime Breakdown (шорт)"}


# ── Данные ───────────────────────────────────────────────────────────────────

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, low, close, volume
FROM market_data_5m
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
  AND (ts AT TIME ZONE 'Europe/Moscow')::time >= CASE
        WHEN (ts AT TIME ZONE 'Europe/Moscow')::date >= %(new)s THEN %(mo_new)s::time
        ELSE %(mo_old)s::time END
  AND (ts AT TIME ZONE 'Europe/Moscow')::time <= %(last)s::time
"""

LASTPX_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
       (array_agg(close ORDER BY ts DESC))[1] AS last_close,
       min(low) AS day_low
FROM market_data_5m
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
GROUP BY 1, 2
"""


def load(conn, d_from: dt.date, d_to: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    msk = dt.timezone(dt.timedelta(hours=3))
    old = sc.session(sc.NEW_SCHEDULE_FROM - dt.timedelta(days=1))
    new = sc.session(sc.NEW_SCHEDULE_FROM)
    tks = sorted(set(ns.UNIVERSE) | set(H3_TICKERS) | {INDEX})
    p = {"tk": tks, "new": sc.NEW_SCHEDULE_FROM, "mo_new": new.main_open,
         "mo_old": old.main_open, "last": LAST_BAR,
         "f": dt.datetime.combine(d_from - dt.timedelta(days=45), dt.time(), msk),
         "t": dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), msk)}
    bars = pd.read_sql(BARS_SQL, conn, params=p)
    bars["tm"] = pd.to_datetime(bars["tm"])
    bars["d"] = bars["tm"].dt.date
    for c in ("open", "high", "low", "close", "volume"):
        bars[c] = bars[c].astype(float)
    last = pd.read_sql(LASTPX_SQL, conn, params=p)
    last["d"] = pd.to_datetime(last["d"]).dt.date
    for c in ("last_close", "day_low"):
        last[c] = last[c].astype(float)
    return bars.sort_values(["ticker", "tm"]).reset_index(drop=True), last


def exec_days(bars: pd.DataFrame, min_tickers: int = 20) -> list[dt.date]:
    first = bars[bars["ticker"].isin(ns.UNIVERSE)].groupby(["d", "ticker"])["tm"].min().reset_index()
    first["ok"] = [t.time() <= (dt.datetime.combine(d, sc.main_open(d)) + H2_TOL).time()
                   for d, t in zip(first["d"], first["tm"])]
    cnt = first[first["ok"]].groupby("d")["ticker"].nunique()
    return sorted(d for d, n in cnt.items() if n >= min_tickers and d.weekday() < 5)


def prev_close_map(last: pd.DataFrame) -> dict:
    """(ticker, D) → последняя цена строго до дня D (любая сессия, включая вечернюю)."""
    out = {}
    for tk, g in last.sort_values("d").groupby("ticker"):
        ds, px = g["d"].tolist(), g["last_close"].tolist()
        for i in range(1, len(ds)):
            out[(tk, ds[i])] = px[i - 1]
        if ds:
            out[(tk, ds[-1] + dt.timedelta(days=1))] = px[-1]
    return out


def _ok_price(p: float) -> bool:
    return sr.quantum_ok(p)


def _series(bars: pd.DataFrame, tk: str) -> pd.DataFrame:
    return bars[bars["ticker"] == tk].set_index("tm").sort_index()


def _window_idx_move(idx: pd.DataFrame, t_in: pd.Timestamp, t_out: pd.Timestamp) -> float:
    """Ход IMOEX от open бара входа до close бара выхода, %.

    Нет бара индекса ровно в момент входа — берётся последний close до него.
    """
    if idx.empty:
        return np.nan
    a = idx.at[t_in, "open"] if t_in in idx.index else idx["close"].asof(t_in)
    b = idx["close"].asof(t_out)
    if a is None or b is None or a != a or b != b or not a:
        return np.nan
    return (float(b) / float(a) - 1.0) * 100.0


# ── H1 ───────────────────────────────────────────────────────────────────────

def top_liquid(bars: pd.DataFrame, lots: dict, days: list[dt.date]) -> dict:
    """D → топ-15 по среднему обороту основной сессии за 20 предыдущих дней."""
    u = bars[bars["ticker"].isin(ns.UNIVERSE)]
    rub = (u["close"] * u["volume"] * u["ticker"].map(lots).fillna(1)).groupby([u["d"], u["ticker"]]).sum()
    piv = rub.unstack().sort_index()
    avg = piv.shift(1).rolling(H1_LIQ_DAYS, min_periods=H1_LIQ_DAYS // 2).mean()
    out = {}
    for d in days:
        if d in avg.index:
            row = avg.loc[d].dropna()
            out[d] = set(row.nlargest(H1_TOP).index)
    return out


def h1_trades(bars: pd.DataFrame, idx: pd.DataFrame, top: dict) -> list[dict]:
    trades = []
    idx_c = idx["close"]
    for tk in ns.UNIVERSE:
        g = _series(bars, tk)
        if g.empty:
            continue
        c = g["close"]
        r30 = (c / c.reindex(g.index - H1_FROM_OPEN).to_numpy() - 1.0) * 100.0
        ir30 = (idx_c.reindex(g.index).to_numpy() / idx_c.reindex(g.index - H1_FROM_OPEN).to_numpy() - 1.0) * 100.0
        vol20 = g["volume"].shift(1).rolling(H1_VOLN).mean()
        sig = (r30 < H1_R30) & (ir30 > H1_IDX30) & (g["volume"] > H1_VOLX * vol20)
        cand = g.index[sig.fillna(False).to_numpy()]
        seen = set()
        for t in cand:
            d = t.date()
            if d in seen or tk not in top.get(d, ()):
                continue
            mo = pd.Timestamp(dt.datetime.combine(d, sc.main_open(d)))
            if t < mo + H1_FROM_OPEN or t.time() > H1_LAST_SIGNAL:
                continue
            seen.add(d)
            day = g[g["d"] == d]
            after = day[day.index > t]
            if after.empty:
                continue
            t_in = after.index[0]
            entry = float(after["open"].iloc[0])
            if not _ok_price(entry):
                continue
            # VWAP дня до предыдущего бара включительно
            tp = (day["high"] + day["low"] + day["close"]) / 3.0
            vwap = ((tp * day["volume"]).cumsum() / day["volume"].cumsum().replace(0, np.nan)).shift(1)
            t_end = t_in + H1_HOLD - BAR
            win = after[after.index <= t_end]
            exitp, t_out, how = float(win["close"].iloc[-1]), win.index[-1], "time"
            for tt, row in win.iterrows():
                lvl = vwap.get(tt)
                if lvl == lvl and lvl is not None and row["high"] >= lvl:
                    exitp, t_out, how = max(float(row["open"]), float(lvl)), tt, "vwap"
                    break
            trades.append({"hyp": "H1", "date": d, "ticker": tk, "dir": 1, "t_in": t_in,
                           "t_out": t_out, "gross": (exitp / entry - 1.0) * 100.0, "exit": how,
                           "idx_move": _window_idx_move(idx, t_in, t_out), "legs": 1})
    return trades


# ── H2 ───────────────────────────────────────────────────────────────────────

def news_times(events: list[dict]) -> dict:
    """ticker → отсортированный список моментов доступности новостей."""
    out: dict[str, list] = {}
    for e in events:
        if not e["price_report"]:
            out.setdefault(e["ticker"], []).append(e["avail"])
    return {k: sorted(v) for k, v in out.items()}


def has_news(times: list, a: dt.datetime, b: dt.datetime) -> bool:
    import bisect
    i = bisect.bisect_right(times, a)
    return i < len(times) and times[i] <= b


def h2_trades(bars: pd.DataFrame, idx: pd.DataFrame, prev_close: dict, days: list[dt.date],
              news: dict, blocked: set) -> list[dict]:
    trades = []
    u = bars[bars["ticker"].isin(ns.UNIVERSE)]
    frames = {k: g.set_index("tm") for k, g in u.groupby(["ticker", "d"])}
    tickers = sorted(u["ticker"].unique())
    for i, d in enumerate(days):
        if i == 0:
            continue
        mo = pd.Timestamp(dt.datetime.combine(d, sc.main_open(d)))
        news_from = dt.datetime.combine(days[i - 1], H2_NEWS_FROM)
        for tk in tickers:
            day = frames.get((tk, d))
            if day is None:
                continue
            first = day[(day.index >= mo) & (day.index <= mo + H2_TOL)]
            pc = prev_close.get((tk, d))
            if first.empty or not pc:
                continue
            gap = (float(first["open"].iloc[0]) / pc - 1.0) * 100.0
            if abs(gap) <= H2_GAP:
                continue
            ent = day[(day.index >= mo + H2_ENTRY) & (day.index <= mo + H2_ENTRY + H2_TOL)]
            ext = day[(day.index > mo + H2_ENTRY) & (day.index <= mo + H2_EXIT)]
            if ent.empty or ext.empty:
                continue
            t_in, entry = ent.index[0], float(ent["open"].iloc[0])
            if not _ok_price(entry):
                continue
            if has_news(news.get(tk, []), news_from, t_in.to_pydatetime()):
                continue
            direction = -1 if gap > 0 else 1
            if direction < 0 and tk in blocked:
                continue
            t_out, exitp = ext.index[-1], float(ext["close"].iloc[-1])
            trades.append({"hyp": "H2", "date": d, "ticker": tk, "dir": direction, "gap": gap,
                           "t_in": t_in, "t_out": t_out,
                           "gross": direction * (exitp / entry - 1.0) * 100.0,
                           "idx_move": _window_idx_move(idx, t_in, t_out), "legs": 1})
    return trades


# ── H3 ───────────────────────────────────────────────────────────────────────

def pair_frame(bars: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    ga, gb = _series(bars, a), _series(bars, b)
    m = ga[["open", "close", "d"]].join(gb[["open", "close"]], lsuffix="_a", rsuffix="_b", how="inner")
    if m.empty:
        return m
    s = np.log(m["close_a"] / m["close_b"])
    mu = s.shift(1).rolling(H3_WIN, min_periods=H3_MIN).mean()
    sd = s.shift(1).rolling(H3_WIN, min_periods=H3_MIN).std()
    m["z"] = (s - mu) / sd
    return m


def h3_pair_trades(m: pd.DataFrame, a: str, b: str, idx: pd.DataFrame, blocked: set) -> list[dict]:
    trades = []
    for d, day in m.groupby("d"):
        day = day.sort_index()
        z = day["z"].to_numpy()
        tms = day.index
        i, n = 0, len(day)
        while i < n - 1:
            if z[i] == z[i] and abs(z[i]) >= H3_Z_IN and tms[i].time() <= H3_LAST_SIGNAL:
                side = 1 if z[i] > 0 else -1            # +1: a дорогая → шорт a, лонг b
                short_leg = a if side > 0 else b
                if short_leg in blocked:
                    break
                e = i + 1
                ea, eb = day["open_a"].iloc[e], day["open_b"].iloc[e]
                if not (_ok_price(ea) and _ok_price(eb)):
                    break
                x, how = n - 1, "eod"
                for j in range(e, n):
                    zj = z[j]
                    if zj != zj:
                        continue
                    if side * zj <= 0:
                        x, how = j, "zero"
                        break
                    if side * zj >= H3_Z_STOP:
                        x, how = j, "stop"
                        break
                if how != "eod" and x + 1 < n:
                    xa, xb, t_out = day["open_a"].iloc[x + 1], day["open_b"].iloc[x + 1], tms[x + 1]
                else:
                    xa, xb, t_out = day["close_a"].iloc[x], day["close_b"].iloc[x], tms[x]
                ra, rb = (xa / ea - 1.0) * 100.0, (xb / eb - 1.0) * 100.0
                gross = (rb - ra) if side > 0 else (ra - rb)
                trades.append({"hyp": "H3", "date": d, "ticker": f"{a}/{b}", "dir": 0,
                               "t_in": tms[e], "t_out": t_out, "gross": gross, "exit": how,
                               "idx_move": _window_idx_move(idx, tms[e], t_out), "legs": 2})
                break                                   # одна сделка на пару в день
            i += 1
    return trades


def h3_trades(bars: pd.DataFrame, idx: pd.DataFrame, blocked: set) -> list[dict]:
    out = []
    for cl, tks in H3_CLUSTERS.items():
        for i, a in enumerate(tks):
            for b in tks[i + 1:]:
                m = pair_frame(bars, a, b)
                if m.empty:
                    log.warning("H3: нет общих баров %s/%s", a, b)
                    continue
                for t in h3_pair_trades(m, a, b, idx, blocked):
                    t["cluster"] = cl
                    out.append(t)
    return out


# ── H4 ───────────────────────────────────────────────────────────────────────

def h4_trades(bars: pd.DataFrame, idx: pd.DataFrame, last: pd.DataFrame, prev_close: dict,
              days: list[dt.date], lots: dict, blocked: set) -> tuple[list[dict], list[dict]]:
    trades, log_days = [], []
    ilow = last[last["ticker"] == INDEX].set_index("d")["day_low"]
    u = bars[bars["ticker"].isin(ns.UNIVERSE)]
    early = u[u["tm"].dt.time <= H4_SIGNAL_BAR]
    turn = (early["close"] * early["volume"] * early["ticker"].map(lots).fillna(1)).groupby(early["d"]).sum()
    turn = turn.reindex(days)
    med = turn.shift(1).rolling(H4_VOL_DAYS, min_periods=H4_VOL_DAYS // 2).median()
    by_tk = {tk: g.set_index("tm") for tk, g in u.groupby("ticker")}
    for i, d in enumerate(days):
        if i == 0:
            continue
        t_sig = pd.Timestamp(dt.datetime.combine(d, H4_SIGNAL_BAR))
        low_prev = ilow.get(days[i - 1])
        ix = idx["close"].get(t_sig)
        rec = {"date": d, "imoex_1355": ix, "prev_low": low_prev,
               "turnover": turn.get(d), "turnover_med": med.get(d), "signal": False}
        if ix is None or low_prev is None or ix != ix or low_prev != low_prev:
            log_days.append(rec)
            continue
        brk = (ix / low_prev - 1.0) * 100.0
        rec["break_pct"] = brk
        vol_ok = rec["turnover"] == rec["turnover"] and rec["turnover_med"] == rec["turnover_med"] \
            and rec["turnover"] > rec["turnover_med"]
        rec["signal"] = bool(brk < -H4_BREAK and vol_ok)
        log_days.append(rec)
        if not rec["signal"]:
            continue
        perf = []
        for tk, g in by_tk.items():
            if tk in blocked:
                continue
            c = g["close"].get(t_sig)
            pc = prev_close.get((tk, d))
            if c is None or not pc or c != c:
                continue
            perf.append((c / pc - 1.0, tk))
        t_in0 = pd.Timestamp(dt.datetime.combine(d, H4_ENTRY_BAR))
        t_last = pd.Timestamp(dt.datetime.combine(d, LAST_BAR))
        for _, tk in sorted(perf)[:H4_TOP]:
            g = by_tk[tk]
            ent = g[(g.index >= t_in0) & (g.index <= t_in0 + H2_TOL)]
            ext = g[(g.index > t_in0) & (g.index <= t_last)]
            rec_t = {"hyp": "H4", "date": d, "ticker": tk, "dir": -1, "legs": 1}
            if ent.empty or ext.empty or not _ok_price(float(ent["open"].iloc[0])):
                rec_t["status"] = "no_data"
                trades.append(rec_t)
                continue
            entry, exitp = float(ent["open"].iloc[0]), float(ext["close"].iloc[-1])
            rec_t.update(t_in=ent.index[0], t_out=ext.index[-1], status="ok",
                         gross=-(exitp / entry - 1.0) * 100.0,
                         idx_move=_window_idx_move(idx, ent.index[0], ext.index[-1]))
            trades.append(rec_t)
    return trades, log_days


# ── Статистика ───────────────────────────────────────────────────────────────

def stats(tr: pd.DataFrame, scenario: str) -> dict:
    """Метрики гипотезы; нетто = валовой доход − издержки сценария (колонка cost_<сценарий>)."""
    if tr.empty or "gross" not in tr:
        return {"n": 0}
    ok = tr[tr["status"].fillna("ok") == "ok"] if "status" in tr else tr
    ok = ok.dropna(subset=["gross"]).copy()
    if ok.empty:
        return {"n": 0}
    ok["net"] = ok["gross"] - ok[f"cost_{scenario}"]
    per_day = ok.groupby("date")["net"].mean().sort_index()
    t, p = sr._t_p(per_day)
    total = float(per_day.sum())
    top5 = float(per_day.nlargest(5).sum())
    res = {"n": int(len(ok)), "days": int(len(per_day)),
           "mean_trade": float(ok["net"].mean()), "median_trade": float(ok["net"].median()),
           "hit": float((ok["net"] > 0).mean()), "mean_day": float(per_day.mean()),
           "t_day": t, "p_day": p, "sum_days": total,
           "top5_share": top5 / total if total > 0 else None,
           "sum_wo_top5": total - top5, "sum_wo_top10": total - float(per_day.nlargest(10).sum()),
           "worst_trade": float(ok["net"].min()), "worst_day": float(per_day.min()),
           "worst_day_at": str(per_day.idxmin())}
    ex = ok["gross"] - ok["dir"] * ok["idx_move"]
    exd = ex.groupby(ok["date"]).mean().dropna()
    te, _ = sr._t_p(exd)
    res.update(excess_idx=float(exd.mean()) if len(exd) else None, excess_t=te)
    both = pd.DataFrame({"y": ok.groupby("date")["gross"].mean(),
                         "x": ok.groupby("date")["idx_move"].mean()}).dropna()
    if len(both) >= 10 and both["x"].std() > 0:
        res["beta"] = float(np.polyfit(both["x"], both["y"], 1)[0])
    return res


def dor(st: dict, st_hi: dict) -> dict:
    if not st.get("n"):
        return {}
    return {"1. ср./день > 0 и t ≥ 2,5": bool(st["mean_day"] > 0 and (st["t_day"] or 0) >= 2.5),
            "1'. при стресс-спреде ср./день > 0": bool(st_hi.get("mean_day", -1) > 0),
            "2. без 5 лучших дней > 0": bool(st["sum_wo_top5"] > 0),
            "2'. доля 5 лучших ≤ 50 %": bool(st["top5_share"] is not None and st["top5_share"] <= 0.5),
            "3. медиана сделки ≥ +0,20 %": bool(st["median_trade"] >= 0.20)}


def rub_per_year(st: dict, n_days_period: int) -> float | None:
    """₽ в год при 10 000 ₽ на ногу: сделок в год × нетто на сделку (легов учтено в net)."""
    if not st.get("n") or not n_days_period:
        return None
    per_year = st["n"] / n_days_period * 252
    return per_year * st["mean_trade"] / 100.0 * POSITION_RUB


def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return (f"{x * 100:.1f} %" if pct else f"{x:+.{nd}f}").replace(".", ",")


def report(res: dict, meta: dict) -> str:
    L = ["# Внутридневные гипотезы при «90 % в TMON»", "",
         f"Сформировано {meta['created']}. Код заморожен коммитом `{meta['revision']}`. "
         "Только исследование: r3 и торговый контур не затронуты.", "",
         "Нетто в % на сделку после издержек тарифа «Премиум» (комиссия 0,04 % за сделку) и спреда "
         "бумаги; на ногу 10 000 ₽, у пар — две ноги; t — по дням. Холм и DoR — по сценарию "
         f"«{cm.LABELS[cm.PRIMARY]}». "
         "«Превышение над индексом» — доход сделки минус направление × ход IMOEX в том же окне. "
         "Перенос не платится: всё закрывается до 18:20.", ""]
    for per in ("holdout", "dev"):
        pm = meta["periods"][per]
        L += [f"## {'Отложенная выборка (вердикт)' if per == 'holdout' else 'Повтор'}: "
              f"{pm['from']} … {pm['to']} ({pm['days']} дней)", ""]
        for cost in SCENARIOS:
            L += [f"### Издержки: {cm.LABELS[cost]}", "",
                  "| гипотеза | сделок | дней | ср. сделка | медиана | доля + | ср. день | t | p Холма | "
                  "доля 5 лучших | без топ-5 | худший день | бета | превышение над индексом (t) | ₽/год на 10 000 ₽ |",
                  "|" + "---|" * 15]
            for h in HYPOTHESES:
                st = res[per][cost][h]
                if not st.get("n"):
                    L.append(f"| {h} {NAMES[h]} | 0 |" + " — |" * 13)
                    continue
                L.append(f"| {h} {NAMES[h]} | {st['n']} | {st['days']} | {_f(st['mean_trade'])} | "
                         f"{_f(st['median_trade'])} | {_f(st['hit'], pct=True)} | {_f(st['mean_day'])} | "
                         f"{_f(st['t_day'], 2)} | {_f(st.get('p_holm'), 3)} | "
                         f"{_f(st['top5_share'], pct=True) if st['top5_share'] is not None else '—'} | "
                         f"{_f(st['sum_wo_top5'], 2)} | {_f(st['worst_day'], 2)} ({st['worst_day_at']}) | "
                         f"{_f(st.get('beta'), 2)} | {_f(st.get('excess_idx'))} ({_f(st.get('excess_t'), 2)}) | "
                         f"{_f(st.get('rub_year'), 0)} |")
            L.append("")
        L += ["### Разрезы (описательно, не тесты)", ""]
        for name, st in res["cuts"][per].items():
            if st.get("n"):
                L.append(f"- {name}: сделок {st['n']}, дней {st['days']}, ср. сделка {_f(st['mean_trade'])}, "
                         f"медиана {_f(st['median_trade'])}, t {_f(st['t_day'], 2)}")
        L.append("")
        if per == "holdout":
            first = next((v for v in res["dor"].values() if v), None)
            if first:
                L += ["### Definition of Ready r4 (§9, пп. 1–3) на отложенной выборке", "",
                      "| гипотеза | " + " | ".join(first.keys()) + " |", "|---|" + "---|" * len(first)]
                for k, v in res["dor"].items():
                    if v:
                        L.append(f"| {k} | " + " | ".join("✓" if x else "✗" for x in v.values()) + " |")
                L.append("")
    L += ["## H4: дни сигнала", "", f"- {meta['h4_signal_days']} ({meta['h4_per_month']})", "",
          "## Как читать", "",
          "- Вердикт — по отложенной выборке и одному основному варианту каждой гипотезы; разрезы не тесты.",
          "- Бета ≈ +1 у лонга и ≈ −1 у шорта означает, что доход в основном от рынка; «превышение над индексом» "
          "показывает, что остаётся после вычета хода IMOEX в том же окне.",
          "- История — режим с утренней сессией и открытием основной в 10:00; с 14.09.2026 торги с 09:10 "
          "без утренней сессии, истории у этого режима нет.", ""]
    return "\n".join(L)


# ── Главное ──────────────────────────────────────────────────────────────────

def evaluate(trades: pd.DataFrame, d_from, d_to, n_days) -> tuple[dict, dict]:
    tr = trades[(trades["date"] >= d_from) & (trades["date"] <= d_to)]
    out = {}
    for cost in SCENARIOS:
        out[cost] = {h: stats(tr[tr["hyp"] == h], cost) for h in HYPOTHESES}
        for h in HYPOTHESES:
            out[cost][h]["rub_year"] = rub_per_year(out[cost][h], n_days)
        adj = sr.holm({h: v.get("p_day") for h, v in out[cost].items()})
        for h, v in out[cost].items():
            v["p_holm"] = adj.get(h)
    cuts = {}
    h1 = tr[tr["hyp"] == "H1"]
    if len(h1):
        for how in ("vwap", "time"):
            cuts[f"H1 выход по {'VWAP' if how == 'vwap' else 'времени'}"] = stats(h1[h1["exit"] == how], cm.PRIMARY)
    h2 = tr[tr["hyp"] == "H2"]
    if len(h2):
        cuts["H2 гэп вниз → лонг"] = stats(h2[h2["dir"] > 0], cm.PRIMARY)
        cuts["H2 гэп вверх → шорт"] = stats(h2[h2["dir"] < 0], cm.PRIMARY)
    h3 = tr[tr["hyp"] == "H3"]
    for cl in sorted(h3["cluster"].dropna().unique()) if len(h3) else []:
        cuts[f"H3 {cl}"] = stats(h3[h3["cluster"] == cl], cm.PRIMARY)
    return out, cuts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Внутридневные гипотезы H1–H4")
    ap.add_argument("--channel", default="markettwits")
    ap.add_argument("--from", dest="date_from", default="2024-05-21")
    ap.add_argument("--to", dest="date_to", default="2026-09-11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import database
    d_from, d_to = dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to)
    conn = database.get_connection()
    try:
        bars, last = load(conn, d_from, d_to)
        posts = ns.load_posts(conn, a.channel, d_from - dt.timedelta(days=7))
    finally:
        conn.close()
    lots, blocked = sr.load_lots_and_blocked()
    days_all = exec_days(bars)
    days = [d for d in days_all if d_from <= d <= d_to]
    idx = _series(bars, INDEX)
    pcm = prev_close_map(last)
    log.info("дней %d; H1…", len(days))
    trades = h1_trades(bars, idx, top_liquid(bars, lots, days_all))
    log.info("H2…")
    trades += h2_trades(bars, idx, pcm, days_all, news_times(ns.build_events(posts)), blocked)
    log.info("H3…")
    trades += h3_trades(bars, idx, blocked)
    log.info("H4…")
    t4, d4 = h4_trades(bars, idx, last, pcm, days_all, lots, blocked)
    trades += t4
    tr = pd.DataFrame(trades)
    tr = tr[(tr["date"] >= d_from) & (tr["date"] <= d_to)]
    if "status" not in tr:
        tr["status"] = "ok"
    tr["status"] = tr["status"].fillna("ok")
    spreads = cm.load_spreads()
    for s in SCENARIOS:
        tr[f"cost_{s}"] = [cm.trade_cost(t, s, spreads) for t in tr["ticker"]]

    res, meta = {"cuts": {}}, {"periods": {}}
    hold_to = DEV_FROM - dt.timedelta(days=1)
    for per, lo, hi in (("holdout", d_from, hold_to), ("dev", DEV_FROM, d_to)):
        n = sum(1 for d in days if lo <= d <= hi)
        res[per], res["cuts"][per] = evaluate(tr, lo, hi, n)
        meta["periods"][per] = {"from": str(lo), "to": str(hi), "days": n}
    res["dor"] = {f"{h} {NAMES[h]}": dor(res["holdout"][cm.PRIMARY][h], res["holdout"][cm.SENSITIVITY][h])
                  for h in HYPOTHESES}
    d4 = pd.DataFrame(d4)
    sig = d4[d4["signal"] == True] if len(d4) else d4                # noqa: E712
    months = max(1.0, len(days) / 21.0)
    try:
        with open(os.path.join(ROOT, "REVISION"), encoding="utf-8") as f:
            rev = f.read().strip()[:12]
    except OSError:
        rev = "unknown"
    now = dt.datetime.now()
    meta.update(created=now.strftime("%d.%m.%Y %H:%M"), revision=rev,
                h4_signal_days=f"{len(sig)} дней из {len(days)}",
                h4_per_month=f"{len(sig) / months:.1f} в месяц")
    text = report(res, meta)
    out = a.out or os.path.join(ROOT, "audit", "r4_research", f"intraday-{now:%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    tr.to_csv(os.path.join(out, "trades.csv"), index=False)
    d4.to_csv(os.path.join(out, "h4_days.csv"), index=False)
    flat = {per: {c: res[per][c] for c in SCENARIOS} for per in ("holdout", "dev")}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": flat, "cuts": res["cuts"], "dor": res["dor"]},
                  f, ensure_ascii=False, indent=1, default=str)
    print(text)
    log.info("отчёт: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
