"""
Event study новостей markettwits по категориям и тональности (Спринт 1.2, ТЗ 50D).
Только исследование: читает БД, в торговые таблицы не пишет, торговым контуром
не импортируется.

ДИЗАЙН ЗАМОРОЖЕН коммитом этого файла до прогона на dev (21.05.2024–11.09.2026,
market_data_5m). Правила, отобранные на dev, замораживаются отдельным коммитом
(rules.json) до ЕДИНСТВЕННОГО прогона на отложенной выборке (01.01.2022–
20.05.2024, research_bars_5m), которую никто не видел.

События. Пост × бумага по классификатору research/news_classify (словари
заморожены): связь «объект», не отчёт о цене. На (бумага, категория, день)
— первый пост; повторы в тот же день новое окно не открывают.

Момент входа t0 — open первого 5-минутного бара бумаги, начавшегося не раньше
публикации + LAG. Реакция рынка: LAG = 0. Торгуемые правила: LAG = 10 минут
(период сборщика news_tg). Нет бара в течение 30 минут — события без цены
входа, их число в отчёте. Бары любой сессии: канал пишет круглосуточно.

Окна CAR (накопленная аномальная доходность, %):
  5m   — close бара t0;
  30m  — close бара, начавшегося в t0 + 25 минут;
  eod  — close основной сессии дня входа (последний бар до 18:50, с 14.09.2026
         до 18:59); если t0 после неё — следующего торгового дня;
  1d   — close основной сессии следующего торгового дня;
  2d   — через два торговых дня.
Аномальная = доходность бумаги − доходность IMOEX на тех же моментах (бета 1).

Экономика (торгуемая версия, LAG = 10 мин, направление d):
  лонг  net = сырая − издержки бумаги − порог фонда за удержание;
  шорт  net = −сырая − издержки бумаги − плата за перенос (ночи, 10 000 ₽);
издержки — research/cost_model (base: комиссия «Премиум» 0,08 % + спред бумаги);
порог фонда — рост паёв TMON@ (с 25.02.2025) или LQDT (раньше) между датой
входа и датой выхода (audit/r4_research/sprint1/hurdle_funds.csv). Внутри дня
пай не дорожает — порог 0.

Статистика по датам (среднее событий даты → t по датам). Когорта =
категория × корзина тональности (neg ≤ −0,5, neu, pos ≥ +0,5).
Дрейф или разворот — знак CAR(2d) против CAR(30m).

Отбор правил на dev (задан здесь, до прогона):
  когорта (категория × neg/pos) × окно ∈ {30m, eod, 1d, 2d};
  направление = знак средней CAR реакции на dev;
  кандидат, если |t| ≥ 3 по датам, дат ≥ 30 и среднее нетто торгуемой версии > 0.
  Плюс правило из ТЗ, заданное явно: DIVIDEND и тональность ≥ 0,8 → лонг 2 дня.
Отложенная выборка — только правила из rules.json; вердикт: тот же знак CAR,
p Холма < 0,05 и среднее нетто > 0. IR к фонду и DSR (число испытаний — из
реестра audit/r4_research/trials.jsonl) — в отчёте.

Запуск (на сервере):
    python -m research.event_study_news dev
    python -m research.event_study_news holdout --rules <rules.json>
"""
from __future__ import annotations

import argparse
import bisect
import csv
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
from research import news_classify as ncl                # noqa: E402
from research import news_event_study as ns              # noqa: E402
from research import session_calendar as sc              # noqa: E402
from research import short_rule as sr                    # noqa: E402

log = logging.getLogger("research.event_study_news")

INDEX = "IMOEX"
LAG_REACTION = dt.timedelta(0)
LAG_TRADE = dt.timedelta(minutes=10)
MAX_WAIT = dt.timedelta(minutes=30)
WINDOWS = ("5m", "30m", "eod", "1d", "2d")
RULE_WINDOWS = ("30m", "eod", "1d", "2d")
BUCKETS = ("neg", "neu", "pos")
PERIODS = {"dev": (dt.date(2024, 5, 21), dt.date(2026, 9, 11), "market_data_5m"),
           "holdout": (dt.date(2022, 1, 1), dt.date(2024, 5, 20), "research_bars_5m")}
MIN_DATES, MIN_T = 30, 3.0
POSITION_RUB = 10_000.0
TZ_RULE = {"category": "DIVIDEND", "sentiment_min": 0.8, "window": "2d", "direction": 1,
           "source": "ТЗ 50D, Спринт 1.2"}
HURDLE_PATH = os.path.join(ROOT, "audit", "r4_research", "sprint1", "hurdle_funds.csv")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")


def bucket(s: float) -> str:
    return "neg" if s <= -0.5 else "pos" if s >= 0.5 else "neu"


def main_close_cut(day: dt.date) -> dt.time:
    return dt.time(18, 59) if day >= sc.NEW_SCHEDULE_FROM else dt.time(18, 50)


def _np(t: dt.datetime) -> np.datetime64:
    return np.datetime64(t, "ns")


# ── Бары ─────────────────────────────────────────────────────────────────────

class Bars:
    """5-минутки одной бумаги (время начала бара, МСК) и close основной сессии по дням."""

    def __init__(self, tm, opens, closes):
        tm = pd.to_datetime(pd.Series(tm)).reset_index(drop=True)
        order = np.argsort(tm.to_numpy(dtype="datetime64[ns]"), kind="stable")
        self.t = tm.to_numpy(dtype="datetime64[ns]")[order]
        self.o = np.asarray(opens, float)[order]
        self.c = np.asarray(closes, float)[order]
        ts = pd.Series(self.t)
        days, times = ts.dt.date.to_numpy(), ts.dt.time.to_numpy()
        keep = np.array([tt < main_close_cut(d) for d, tt in zip(days, times)], dtype=bool)
        self.day_close: dict[dt.date, float] = {}
        for d, c in zip(days[keep], self.c[keep]):
            self.day_close[d] = c                            # последний бар до закрытия основной

    def entry(self, t_start: dt.datetime) -> int | None:
        i = int(np.searchsorted(self.t, _np(t_start), "left"))
        if i >= len(self.t) or self.t[i] - _np(t_start) > np.timedelta64(MAX_WAIT):
            return None
        return i

    def close_at(self, s: dt.datetime) -> float:
        """close последнего бара, начавшегося не позже s."""
        j = int(np.searchsorted(self.t, _np(s), "right")) - 1
        return float(self.c[j]) if j >= 0 else float("nan")

    def price_at(self, s: dt.datetime) -> float:
        """open бара, начавшегося ровно в s, иначе close предыдущего."""
        i = int(np.searchsorted(self.t, _np(s), "left"))
        if i < len(self.t) and self.t[i] == _np(s):
            return float(self.o[i])
        return float(self.c[i - 1]) if i > 0 else float("nan")


def next_day(days: list[dt.date], d: dt.date | None) -> dt.date | None:
    if d is None:
        return None
    i = bisect.bisect_right(days, d)
    return days[i] if i < len(days) else None


def outcome(posted: dt.datetime, lag: dt.timedelta, bars: Bars, idx: Bars,
            tdays: list[dt.date]) -> dict | None:
    """Цены входа и выхода по окнам; сырая и аномальная доходность, %."""
    i0 = bars.entry(posted + lag)
    if i0 is None:
        return None
    t0 = pd.Timestamp(bars.t[i0]).to_pydatetime()
    p0, x0 = float(bars.o[i0]), idx.price_at(t0)
    d0 = t0.date()
    tset = set(tdays)
    base = d0 if (d0 in tset and t0.time() < main_close_cut(d0)) else next_day(tdays, d0)
    ends = {"5m": ("bar", t0), "30m": ("bar", t0 + dt.timedelta(minutes=25)),
            "eod": ("day", base), "1d": ("day", next_day(tdays, base)),
            "2d": ("day", next_day(tdays, next_day(tdays, base)))}
    out = {"t0": t0, "p0": p0}
    for w, (kind, v) in ends.items():
        if v is None:
            p1 = x1 = float("nan")
        elif kind == "bar":
            p1, x1 = bars.close_at(v), idx.close_at(v)
        else:
            p1, x1 = bars.day_close.get(v, float("nan")), idx.day_close.get(v, float("nan"))
        raw = (p1 / p0 - 1.0) * 100.0 if p0 > 0 else float("nan")
        ix = (x1 / x0 - 1.0) * 100.0 if x0 and x0 == x0 else float("nan")
        out[f"raw_{w}"], out[f"ar_{w}"] = raw, raw - ix
        out[f"end_{w}"] = (v.date() if isinstance(v, dt.datetime) else v)
    return out


# ── Порог фонда и издержки ───────────────────────────────────────────────────

class Hurdle:
    """Рост паёв фонда денежного рынка между датами, %: TMON@, до него LQDT."""

    def __init__(self, path: str = HURDLE_PATH):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["fund"], []).append((dt.date.fromisoformat(r["date"]), float(r["close"])))
        self.series = {k: sorted(v) for k, v in rows.items()}

    def _px(self, fund: str, d: dt.date) -> float | None:
        s = self.series.get(fund) or []
        i = bisect.bisect_right([x[0] for x in s], d) - 1
        return s[i][1] if i >= 0 else None

    def growth(self, d0: dt.date, d1: dt.date) -> float:
        if d1 <= d0:
            return 0.0
        for fund in ("TMON@", "LQDT"):
            s = self.series.get(fund)
            if s and s[0][0] <= d0:
                a, b = self._px(fund, d0), self._px(fund, d1)
                if a and b:
                    return (b / a - 1.0) * 100.0
        return 0.0


def economics(o: dict, ticker: str, hurdle: Hurdle, spreads: dict) -> dict:
    """Нетто лонга и шорта по окнам торгуемой версии, %."""
    cost = cm.round_trip(ticker, "base", spreads)
    out = {"cost": cost}
    d0 = o["t0"].date()
    for w in WINDOWS:
        raw = o.get(f"raw_{w}")
        end = o.get(f"end_{w}")
        if raw is None or raw != raw or end is None:
            out[f"net_long_{w}"] = out[f"net_short_{w}"] = float("nan")
            continue
        nights = max(0, (end - d0).days)
        out[f"net_long_{w}"] = raw - cost - hurdle.growth(d0, end)
        out[f"net_short_{w}"] = -raw - cost - (cm.carry_pct(POSITION_RUB, nights) if nights else 0.0)
    return out


# ── События ──────────────────────────────────────────────────────────────────

def build_events(posts: pd.DataFrame, clf: ncl.Classifier) -> pd.DataFrame:
    """Пост × бумага: объект, не отчёт о цене; первый пост на (бумага, категория, день)."""
    rows, seen = [], set()
    for r in posts.sort_values("msk").itertuples(index=False):
        for c in clf.classify_post(r.text):
            if c["relation"] != "объект" or c["price_report"]:
                continue
            key = (c["ticker"], c["category"], r.msk.date())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"message_id": r.message_id, "posted": r.msk.to_pydatetime(),
                         "ticker": c["ticker"], "category": c["category"],
                         "category2": c["category2"], "sentiment": c["sentiment"],
                         "bucket": bucket(c["sentiment"])})
    return pd.DataFrame(rows)


def attach_outcomes(events: pd.DataFrame, bars_of: dict, idx: Bars, tdays: list[dt.date],
                    hurdle: Hurdle, spreads: dict) -> pd.DataFrame:
    rows = []
    for e in events.itertuples(index=False):
        b = bars_of.get(e.ticker)
        rec = e._asdict()
        if b is None:
            rec["status"] = "нет баров"
            rows.append(rec)
            continue
        react = outcome(e.posted, LAG_REACTION, b, idx, tdays)
        trade = outcome(e.posted, LAG_TRADE, b, idx, tdays)
        if react is None or trade is None or not sr.quantum_ok(react["p0"]):
            rec["status"] = "нет цены входа" if (react is None or trade is None) else "шаг цены"
            rows.append(rec)
            continue
        rec["status"] = "ok"
        rec["date"] = react["t0"].date()
        rec.update({f"ar_{w}": react[f"ar_{w}"] for w in WINDOWS})
        rec.update({f"raw_{w}": react[f"raw_{w}"] for w in WINDOWS})
        rec.update({f"t_raw_{w}": trade[f"raw_{w}"] for w in WINDOWS})
        rec.update(economics(trade, e.ticker, hurdle, spreads))
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Статистика ───────────────────────────────────────────────────────────────

def by_date(values: pd.Series, dates: pd.Series) -> dict:
    s = pd.Series(values.to_numpy(float), index=dates.to_numpy()).dropna()
    if s.empty:
        return {"n": 0, "dates": 0}
    g = s.groupby(level=0).mean()
    t, p = sr._t_p(g)
    return {"n": int(len(s)), "dates": int(len(g)), "mean": float(g.mean()),
            "median": float(s.median()), "t": t, "p": p}


def cohorts(ev: pd.DataFrame):
    """(категория, корзина) → строки; плюс «ВСЕ» по корзинам."""
    for cat in ncl.CATEGORIES + ("ВСЕ",):
        for b in BUCKETS:
            m = (ev["bucket"] == b) & ((ev["category"] == cat) if cat != "ВСЕ" else True)
            yield cat, b, ev[m]


def reaction_table(ev: pd.DataFrame) -> list[dict]:
    out = []
    for cat, b, sub in cohorts(ev):
        row = {"category": cat, "bucket": b}
        for w in WINDOWS:
            row[w] = by_date(sub[f"ar_{w}"], sub["date"])
        a30, a2 = row["30m"].get("mean"), row["2d"].get("mean")
        row["pattern"] = ("—" if a30 is None or a2 is None else
                          "дрейф" if a30 * a2 > 0 and abs(a2) > abs(a30) else
                          "затухание" if a30 * a2 > 0 else "разворот")
        out.append(row)
    return out


def select_rules(ev: pd.DataFrame) -> tuple[list[dict], int]:
    """Кандидаты по правилу отбора из докстринга; число испытаний — для DSR."""
    rules, trials = [], 0
    for cat in ncl.CATEGORIES:
        for b in ("neg", "pos"):
            sub = ev[(ev["category"] == cat) & (ev["bucket"] == b)]
            for w in RULE_WINDOWS:
                trials += 1
                st = by_date(sub[f"ar_{w}"], sub["date"])
                if not st.get("dates") or st["t"] is None:
                    continue
                d = 1 if st["mean"] > 0 else -1
                net = by_date(sub[f"net_{'long' if d > 0 else 'short'}_{w}"], sub["date"])
                ok = abs(st["t"]) >= MIN_T and st["dates"] >= MIN_DATES and (net.get("mean") or -1) > 0
                if ok:
                    rules.append({"category": cat, "bucket": b, "window": w, "direction": d,
                                  "dev_ar": st, "dev_net": net, "source": "отбор на dev"})
    sub = ev[(ev["category"] == TZ_RULE["category"]) & (ev["sentiment"] >= TZ_RULE["sentiment_min"])]
    rules.append({**TZ_RULE, "dev_ar": by_date(sub["ar_2d"], sub["date"]),
                  "dev_net": by_date(sub["net_long_2d"], sub["date"])})
    trials += 1
    return rules, trials


def rule_mask(ev: pd.DataFrame, rule: dict) -> pd.Series:
    m = ev["category"] == rule["category"]
    if "sentiment_min" in rule:
        return m & (ev["sentiment"] >= rule["sentiment_min"])
    return m & (ev["bucket"] == rule["bucket"])


def rule_daily_series(ev: pd.DataFrame, rule: dict, days: list[dt.date]) -> pd.Series:
    """Нетто правила по торговым дням (0 в дни без событий), % на позицию."""
    col = f"net_{'long' if rule['direction'] > 0 else 'short'}_{rule['window']}"
    sub = ev[rule_mask(ev, rule)].dropna(subset=[col])
    per = sub.groupby("date")[col].mean()
    return per.reindex(days).fillna(0.0)


def information_ratio(daily: pd.Series) -> float | None:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * math.sqrt(252)) if sd and sd > 0 else None


def deflated_sharpe(daily: pd.Series, n_trials: int) -> float | None:
    """DSR (Lopez de Prado): вероятность, что истинный Sharpe > 0 с учётом числа
    испытаний и негауссовости."""
    from scipy import stats
    r = daily.dropna()
    T = len(r)
    if T < 30 or n_trials < 2 or not r.std(ddof=1):
        return None
    sr_ = float(r.mean() / r.std(ddof=1))
    var = (1 + 0.5 * sr_ ** 2) / T
    g = 0.5772156649015329
    sr0 = math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                            + g * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    den = math.sqrt(max(1e-12, 1 - sk * sr_ + (ku - 1) / 4 * sr_ ** 2))
    return float(stats.norm.cdf((sr_ - sr0) * math.sqrt(T - 1) / den))


def register_trials(stage: str, n: int, revision: str, path: str = TRIALS_PATH) -> int:
    """Дописывает испытания в реестр; возвращает накопленную сумму."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"),
                            "sprint": 1, "stage": stage, "trials": n, "revision": revision},
                           ensure_ascii=False) + "\n")
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            total += int(json.loads(line).get("trials", 0))
    return total


# ── Загрузка ─────────────────────────────────────────────────────────────────

def load_bars(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> dict:
    msk = dt.timezone(dt.timedelta(hours=3))
    f = dt.datetime.combine(d_from, dt.time(), msk)
    t = dt.datetime.combine(d_to + dt.timedelta(days=6), dt.time(), msk)
    out = {}
    for tk in tickers:
        df = pd.read_sql(f"SELECT (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, close FROM {table} "
                         "WHERE ticker = %s AND ts >= %s AND ts < %s AND close > 0 ORDER BY ts",
                         conn, params=(tk, f, t))
        if not df.empty:
            out[tk] = Bars(df["tm"], df["open"].astype(float), df["close"].astype(float))
    return out


# ── Отчёт ────────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}".replace(".", ",")


def report_dev(ev: pd.DataFrame, table: list[dict], rules: list[dict], meta: dict) -> str:
    L = ["# Event study новостей markettwits — dev (Спринт 1.2)", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, словари `{meta['dicts']}`. "
         "Только исследование.", "",
         f"Период {meta['from']} … {meta['to']} ({meta['table']}). Событий «объект»: {meta['events']}; "
         f"с ценами {meta['ok']}; без баров {meta['no_bars']}; без цены входа {meta['no_entry']}; "
         f"шаг цены {meta['quantum']}.", "",
         "CAR — аномальная доходность против IMOEX, %, реакция (вход — первый бар после "
         "публикации). t — по датам. Корзины тональности: neg ≤ −0,5, neu, pos ≥ +0,5.", "",
         "| категория | тональность | " + " | ".join(f"CAR {w} (t)" for w in WINDOWS) + " | дат | картина |",
         "|---|---|" + "---|" * (len(WINDOWS) + 2)]
    for r in table:
        dates = r["2d"].get("dates") or r["5m"].get("dates") or 0
        if not dates:
            continue
        L.append(f"| {r['category']} | {r['bucket']} | "
                 + " | ".join(f"{_f(r[w].get('mean'))} ({_f(r[w].get('t'), 1)})" for w in WINDOWS)
                 + f" | {dates} | {r['pattern']} |")
    L += ["", f"## Правила-кандидаты (|t| ≥ {MIN_T:g}, дат ≥ {MIN_DATES}, нетто > 0) и правило из ТЗ", "",
          f"Испытаний на dev: {meta['trials']} (накоплено в реестре: {meta['trials_total']}).", "",
          "| правило | направление | окно | CAR dev (t) | дат | нетто dev, % (t) |", "|---|---|---|---|---|---|"]
    for r in rules:
        name = (f"{r['category']} & тональность ≥ {r['sentiment_min']}" if "sentiment_min" in r
                else f"{r['category']} / {r['bucket']}")
        L.append(f"| {name} ({r['source']}) | {'лонг' if r['direction'] > 0 else 'шорт'} | {r['window']} | "
                 f"{_f(r['dev_ar'].get('mean'))} ({_f(r['dev_ar'].get('t'), 1)}) | {r['dev_ar'].get('dates', 0)} | "
                 f"{_f(r['dev_net'].get('mean'))} ({_f(r['dev_net'].get('t'), 1)}) |")
    L += ["", "## Как читать", "",
          "- dev просмотрен; вывод только по отложенной выборке 2022–2024, куда идут лишь правила из rules.json.",
          "- Нетто — торгуемая версия: вход через 10 минут после публикации, издержки «Премиум» + спред, "
          "порог фонда за удержание (лонг) или плата за перенос (шорт через ночь).", ""]
    return "\n".join(L)


def report_holdout(results: list[dict], meta: dict) -> str:
    L = ["# Event study новостей — отложенная выборка (Спринт 1.2)", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правила `{meta['rules']}`. "
         f"Период {meta['from']} … {meta['to']} ({meta['table']}), единственный прогон.", "",
         f"Событий «объект»: {meta['events']}, с ценами {meta['ok']}. Испытаний в реестре: {meta['trials_total']}.", "",
         "| правило | направление | окно | CAR (t) | дат | p Холма | нетто, % (t) | IR к фонду | DSR | вердикт |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['name']} | {r['dir']} | {r['window']} | {_f(r['ar'].get('mean'))} ({_f(r['ar'].get('t'), 1)}) | "
                 f"{r['ar'].get('dates', 0)} | {_f(r.get('p_holm'), 3)} | {_f(r['net'].get('mean'))} "
                 f"({_f(r['net'].get('t'), 1)}) | {_f(r.get('ir'), 2)} | {_f(r.get('dsr'), 2)} | {r['verdict']} |")
    L += ["", "Вердикт «подтверждено» — тот же знак CAR, что на dev, p Холма < 0,05 и среднее нетто > 0.", ""]
    return "\n".join(L)


def _revision() -> str:
    try:
        with open(os.path.join(ROOT, "REVISION"), encoding="utf-8") as f:
            return f.read().strip()[:12]
    except OSError:
        return "unknown"


def run(stage: str, rules_path: str | None, out: str | None, channel: str = "markettwits") -> int:
    import database
    d_from, d_to, table = PERIODS[stage]
    clf = ncl.Classifier()
    conn = database.get_connection()
    try:
        posts = ns.load_posts(conn, channel, d_from)
        posts = posts[posts["msk"].dt.date <= d_to]
        events = build_events(posts, clf)
        bars_of = load_bars(conn, table, sorted(set(events["ticker"])), d_from, d_to)
        idx = load_bars(conn, table, [INDEX], d_from, d_to).get(INDEX)
    finally:
        conn.close()
    if idx is None:
        raise SystemExit(f"нет баров {INDEX} в {table}")
    tdays = sorted(d for d in idx.day_close if d_from <= d <= d_to + dt.timedelta(days=6) and d.weekday() < 5)
    ev = attach_outcomes(events, bars_of, idx, tdays, Hurdle(), cm.load_spreads())
    ok = ev[ev["status"] == "ok"].copy()
    now = dt.datetime.now()
    rev = _revision()
    meta = {"created": now.strftime("%d.%m.%Y %H:%M"), "revision": rev, "dicts": clf.version,
            "from": str(d_from), "to": str(d_to), "table": table, "events": int(len(ev)),
            "ok": int(len(ok)), "no_bars": int((ev["status"] == "нет баров").sum()),
            "no_entry": int((ev["status"] == "нет цены входа").sum()),
            "quantum": int((ev["status"] == "шаг цены").sum())}
    out = out or os.path.join(ROOT, "audit", "r4_research", f"sprint1-{stage}-{now:%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    keep = [c for c in ok.columns if c not in ("posted",)]
    ok[keep].to_csv(os.path.join(out, "events.csv"), index=False)
    if stage == "dev":
        table_rows = reaction_table(ok)
        rules, trials = select_rules(ok)
        meta["trials"] = trials
        meta["trials_total"] = register_trials("dev", trials, rev)
        with open(os.path.join(out, "rules.json"), "w", encoding="utf-8") as f:
            json.dump({"revision": rev, "dicts": clf.version, "created": meta["created"],
                       "trials_dev": trials, "rules": rules}, f, ensure_ascii=False, indent=1, default=str)
        text = report_dev(ok, table_rows, rules, meta)
    else:
        with open(rules_path, encoding="utf-8") as f:
            spec = json.load(f)
        meta["rules"] = f"{os.path.basename(rules_path)} ({spec.get('revision')})"
        meta["trials_total"] = register_trials("holdout", 0, rev)
        days = [d for d in tdays if d <= d_to]
        results = []
        for r in spec["rules"]:
            sub = ok[rule_mask(ok, r)]
            col = f"net_{'long' if r['direction'] > 0 else 'short'}_{r['window']}"
            ar = by_date(sub[f"ar_{r['window']}"], sub["date"])
            net = by_date(sub[col], sub["date"])
            daily = rule_daily_series(ok, r, days)
            name = (f"{r['category']} & тональность ≥ {r['sentiment_min']}" if "sentiment_min" in r
                    else f"{r['category']} / {r['bucket']}")
            results.append({"name": name, "dir": "лонг" if r["direction"] > 0 else "шорт",
                            "window": r["window"], "direction": r["direction"], "ar": ar, "net": net,
                            "ir": information_ratio(daily),
                            "dsr": deflated_sharpe(daily, max(2, meta["trials_total"]))})
        adj = sr.holm({i: (x["ar"].get("p")) for i, x in enumerate(results)})
        for i, x in enumerate(results):
            x["p_holm"] = adj.get(i)
            same = x["ar"].get("mean") is not None and np.sign(x["ar"]["mean"]) == x["direction"]
            x["verdict"] = ("подтверждено" if same and (x["p_holm"] or 1) < 0.05
                            and (x["net"].get("mean") or -1) > 0 else "не подтверждено")
        with open(os.path.join(out, "holdout_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1, default=str)
        text = report_holdout(results, meta)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    log.info("отчёт: %s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Event study новостей markettwits (Спринт 1.2)")
    ap.add_argument("stage", choices=sorted(PERIODS))
    ap.add_argument("--rules", default=None, help="rules.json с dev (обязательно для holdout)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.stage == "holdout" and not a.rules:
        ap.error("для holdout нужен --rules (замороженные правила с dev)")
    return run(a.stage, a.rules, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
