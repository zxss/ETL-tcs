"""
Исследование: влияют ли новости markettwits на гэпы и избыточную доходность
наших бумаг. Базовая проверка без ML.

РЕЖИМ «ТОЛЬКО ИССЛЕДОВАНИЕ» (решение пользователя 14.09): модуль только читает
БД, в торговые таблицы не пишет, торговым контуром не импортируется и на тест
r3 не влияет. Результат — отчёт в audit/news_research/<метка>/.

Две схемы измерения
-------------------
A. Дневные бары (с 21.05.2024). Дневной бар брокера — весь торговый день:
   open — первая сделка, close — конец вечерней сессии (23:50; сверено 14.09 на
   SBER за 11.09: close дневки = close 5-минутки 23:45). Гэп = open(D)/close(D−1) − 1
   — ровно цель next_overnight, на которой валидирован long_overnight.
     A1. новость в окне гэпа (после 23:50 D−1 до открытия D) → аномальный гэп D;
     A2. новость днём D, известная к 18:35 → аномальный гэп D+1 и доходность D+1.
   Открытие дня менялось: до осени 2024 торги начинались около 09:50, с октября
   2024 — редкие сделки с 07:00, с февраля 2025 — утренняя сессия с 06:50
   (часовые свечи SBER из T-Invest, 14.09). Граница окна гэпа берётся по дате.

B. 5-минутки (с декабря 2025) — реальное окно исполнения long_overnight:
   вход 18:35 (close бара 18:30), выход 10:00 следующего торгового дня (open
   бара 10:00 — фаза CLOSE). На дневках это окно не измерить.

Аномальная доходность — минус равновзвешенное среднее остальных бумаг
вселенной на том же окне. IMOEX в вечернюю сессию не считается и с барами
акций несопоставим.

Доступность новости = время публикации + 10 минут (период крона сборщика).

Статистика — по датам: сначала среднее по бумагам внутри даты, затем
t-статистика по датам. Наблюдения одной даты связаны общим рыночным шоком, и
t по отдельным бумагам завышал бы значимость в разы.

Издержки 0,128% round-trip (TFT_COST_RT): правило «торгуемо», только если его
средняя аномальная доходность больше издержек.

Тональность — фиксированный словарь, заданный ДО просмотра результатов и по
ним не подбиравшийся. Это не модель, а грубая проверка: есть ли вообще
направленность, которую имеет смысл уточнять LLM.

Посты-отчёты о движении цены («#SMLT = -10%») из событий исключаются: они
описывают уже случившееся движение, а не его причину.

Запуск (на сервере, где БД):
    python3 -m research.news_event_study --channel markettwits
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import logging
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("research.news")

COST_RT_PCT = 0.128
LAG = dt.timedelta(minutes=10)
CLOSE_T = dt.time(23, 50)
DECISION_T = dt.time(18, 35)
EXIT_T = dt.time(10, 0)

# Открытие торгового дня по данным брокера (см. докстринг, схема A).
_OPEN_SCHEDULE = ((dt.date(2025, 2, 1), dt.time(6, 50)),
                  (dt.date(2024, 10, 1), dt.time(7, 0)),
                  (dt.date(1900, 1, 1), dt.time(9, 50)))


def open_cutoff(day: dt.date) -> dt.time:
    for since, t in _OPEN_SCHEDULE:
        if day >= since:
            return t
    return _OPEN_SCHEDULE[-1][1]


# ── Привязка поста к тикеру ──────────────────────────────────────────────────

UNIVERSE = ("SBER", "VTBR", "GAZP", "ROSN", "LKOH", "NVTK", "TATN", "SNGS", "SNGSP",
            "GMKN", "PLZL", "MAGN", "CHMF", "ALRS", "IRAO", "FEES", "HYDR", "UPRO",
            "MSNG", "TGKA", "OGKB", "PHOR", "AKRN", "MOEX", "MTSS", "RTKM", "AFLT",
            "FLOT", "NMTP", "MGNT", "X5", "LENT", "FIXR", "YDEX", "VKCO", "ASTR",
            "POSI", "PIKK", "SMLT", "ETLN", "RUAL", "ENPG", "SELG", "BSPB", "CBOM",
            "MVID")

# Старые тикеры тех же компаний.
HASHTAG_ALIASES = {"YNDX": ("YDEX",), "FIVE": ("X5",), "MAIL": ("VKCO",),
                   "SNGS": ("SNGS", "SNGSP"), "SNGSP": ("SNGS", "SNGSP")}

# Названия в тексте. Слова-омонимы («самолёт», «эталон», «лента», «пик»,
# «позитив», «магнит» как предмет) берутся только в кавычках, с «ГК» или с
# учётом регистра — иначе авиа- и прочие новости дали бы ложные совпадения.
# MOEX — только по хештегу: «Мосбиржа» в канале чаще площадка, чем эмитент.
_NAMES = {
    "SBER": r"\bсбер(?:банк\w*|а|у|ом|е)?\b",
    "VTBR": r"\bвтб\b",
    "GAZP": r"\bгазпром(?:а|у|ом|е)?\b(?![\s-]*(?:нефт|банк|межрегион))",
    "ROSN": r"\bроснефт",
    "LKOH": r"\bлукойл",
    "NVTK": r"\bноватэк",
    "TATN": r"\bтатнефт",
    "SNGS": r"\bсургутнефтегаз",
    "GMKN": r"\bнорникел|\bнорильск\w*\s+никел|\bгмк\b",
    "PLZL": r"\bполюс(?:а|у|ом|е)?\b",
    "MAGN": r"\bммк\b|\bмагнитогорск\w*\s+металлург",
    "CHMF": r"\bсеверстал",
    "ALRS": r"\bалрос",
    "IRAO": r"\bинтер\s?рао",
    "FEES": r"\bфск(?:\s+еэс)?\b",
    "HYDR": r"\bрусгидро",
    "UPRO": r"\bюнипро",
    "MSNG": r"\bмосэнерго",
    "TGKA": r"\bтгк-?1\b",
    "OGKB": r"\bогк-?2\b",
    "PHOR": r"\bфосагро",
    "AKRN": r"\bакрон(?:а|у|ом|е)?\b",
    "MTSS": r"\bмтс\b(?![\s-]*банк)",
    "RTKM": r"\bростелеком",
    "AFLT": r"\bаэрофлот",
    "FLOT": r"\bсовкомфлот",
    "NMTP": r"\bнмтп\b",
    "MGNT": r"[«\"]магнит[»\"]|\bритейлер\w*\s+магнит",
    "X5": r"\bx5\b|\bикс\s?5\b|\bпят[её]рочк",
    "LENT": r"[«\"]лент[аыеу][»\"]",
    "FIXR": r"\bfix\s?price\b|\bфикс\s?прайс",
    "YDEX": r"\bяндекс",
    "VKCO": r"\bвконтакте\b|\bvk\b|\bвк\b",
    "ASTR": r"\bгрупп\w*\s+[«\"]?астра|[«\"]астра[»\"]",
    "POSI": r"positive\s+technologies|позитив\s+текнолоджиз|[«\"]позитив[»\"]|группа\s+позитив",
    "SMLT": r"\bгк\s*[«\"]?самол[её]т|[«\"]самол[её]т[»\"]",
    "ETLN": r"\bгк\s*[«\"]?эталон|[«\"]эталон[»\"]|\betalon\b",
    "RUAL": r"\bрусал",
    "ENPG": r"\ben\+|\bэн\+",
    "SELG": r"\bселигдар",
    "BSPB": r"\bбанк\w*\s+санкт-петербург|\bбсп\b",
    "CBOM": r"\bмкб\b|\bмосковск\w*\s+кредитн\w*\s+банк",
    "MVID": r"\bм\.?\s?видео\b|\bmvideo\b",
}
_NAME_RE = {tk: re.compile(p, re.IGNORECASE) for tk, p in _NAMES.items()}
_NAME_RE["PIKK"] = re.compile(r"\bПИК(?:а|у|ом|е)?\b|\bГК\s+ПИК\b")   # с учётом регистра
_HASHTAG = re.compile(r"#([A-Z][A-Z0-9]{1,5})(?![A-Za-z0-9_])")
# «#SMLT = -10%», «SBER +3%» в первой строке — отчёт о движении цены.
_PRICE_REPORT = re.compile(r"^[^\n]{0,40}(?:=\s*[+-−–]?\s*\d|[+-−–]\s*\d+(?:[.,]\d+)?\s*%)")


def ticker_sources(text: str | None) -> dict[str, set[str]]:
    """Тикеры вселенной, о которых пост, и откуда привязка: «hashtag» / «name»."""
    if not text:
        return {}
    found: dict[str, set[str]] = {}

    def add(tk: str, how: str) -> None:
        found.setdefault(tk, set()).add(how)

    for tag in _HASHTAG.findall(text):
        if tag in HASHTAG_ALIASES:
            for tk in HASHTAG_ALIASES[tag]:
                add(tk, "hashtag")
        elif tag in UNIVERSE:
            add(tag, "hashtag")
    for tk, rx in _NAME_RE.items():
        if rx.search(text):
            add(tk, "name")
            if tk == "SNGS":
                add("SNGSP", "name")
    return found


def tickers_in(text: str | None) -> set[str]:
    """Тикеры вселенной, о которых пост: хештеги и названия компаний."""
    return set(ticker_sources(text))


def is_price_report(text: str | None) -> bool:
    return bool(text) and bool(_PRICE_REPORT.search(text))


# ── Тональность: фиксированный словарь, задан до просмотра результатов ───────

_POS = [r"байб[эе]к|обратн\w*\s+выкуп",
        r"рекоменд\w*.{0,40}дивиденд|дивиденд\w*.{0,30}(?:рекоменд|утверд|одобр)",
        r"рекордн\w*\s+(?:прибыл|выручк|добыч|дивиденд)",
        r"(?:прибыль|выручка|ebitda)\w*.{0,40}(?:вырос|увелич)",
        r"повысил\w*.{0,25}(?:прогноз|рейтинг|целев)",
        r"\bupgrade"]
_NEG = [r"санкци", r"убыт",
        r"(?:прибыль|выручка|ebitda)\w*.{0,40}(?:снизил|сократил|упал|сниз)",
        r"понизил\w*.{0,25}(?:прогноз|рейтинг|целев)",
        r"(?:\bне\b|отказ\w*|без|отмен\w*).{0,30}дивиденд|дивиденд\w*.{0,30}(?:не\s+буд|не\s+выплат|отмен|отказ)",
        r"делистинг", r"\bspo\b|допэмисс|вторичн\w*\s+(?:размещ|публичн)",
        r"штраф", r"\bdowngrade"]
_POS_RE = [re.compile(p, re.IGNORECASE) for p in _POS]
_NEG_RE = [re.compile(p, re.IGNORECASE) for p in _NEG]


def tone(text: str | None) -> int:
    """+1 / 0 / −1 по числу совпадений словаря."""
    if not text:
        return 0
    p = sum(1 for rx in _POS_RE if rx.search(text))
    n = sum(1 for rx in _NEG_RE if rx.search(text))
    return (p > n) - (n > p)


# ── Окна ─────────────────────────────────────────────────────────────────────

def _next_day(days: list[dt.date], d: dt.date) -> dt.date | None:
    i = bisect.bisect_right(days, d)
    return days[i] if i < len(days) else None


def _prev_day(days: list[dt.date], d: dt.date) -> dt.date | None:
    i = bisect.bisect_left(days, d)
    return days[i - 1] if i > 0 else None


def daily_window(t_avail: dt.datetime, days: list[dt.date]) -> tuple[str, dt.date] | None:
    """Схема A: («gap», D) — новость в окне гэпа дня D; («day», D) — внутри дня D."""
    d, t = t_avail.date(), t_avail.time()
    daystart = open_cutoff(d)
    idx = bisect.bisect_left(days, d)
    is_trading = idx < len(days) and days[idx] == d
    if is_trading and daystart < t <= CLOSE_T:
        return "day", d
    if is_trading and t <= daystart:
        return "gap", d
    nxt = _next_day(days, d)
    return ("gap", nxt) if nxt else None


def exec_windows(t_avail: dt.datetime, days: list[dt.date]) -> list[tuple[str, dt.date]]:
    """Схема B, окно исполнения 18:35 D → 10:00 D+1.

    («known», D) — новость известна к решению в 18:35 торгового дня D;
    («hold», D) — пришла, пока открыта позиция, купленная в 18:35 дня D.
    Ночная новость попадает в оба окна: это и риск вчерашней позиции, и
    информация к сегодняшнему решению. Выходные: позиция пятницы держится до
    понедельника 10:00.
    """
    d, t = t_avail.date(), t_avail.time()
    idx = bisect.bisect_left(days, d)
    is_trading = idx < len(days) and days[idx] == d
    out: list[tuple[str, dt.date]] = []
    decision = d if (is_trading and t <= DECISION_T) else _next_day(days, d)
    if decision:
        out.append(("known", decision))
    if is_trading and t > DECISION_T:
        out.append(("hold", d))
    elif not is_trading or t < EXIT_T:
        base = _prev_day(days, d)
        if base:
            out.append(("hold", base))
    return out


# ── Статистика ───────────────────────────────────────────────────────────────

def abnormal(frame):
    """Минус равновзвешенное среднее ОСТАЛЬНЫХ бумаг на ту же дату (leave-one-out).
    frame: DataFrame дата × тикер."""
    s = frame.sum(axis=1, min_count=1)
    n = frame.notna().sum(axis=1)
    loo = frame.rsub(s, axis=0).div((n - 1).where(n > 1), axis=0)
    return frame - loo


def by_date(values, dates) -> dict:
    """Среднее и t-статистика по датам (сначала среднее внутри даты)."""
    import pandas as pd
    s = pd.Series(list(values), index=list(dates)).dropna()
    if s.empty:
        return {"n_obs": 0, "n_dates": 0, "mean": None, "t": None, "median": None}
    g = s.groupby(level=0).mean()
    n = len(g)
    sd = g.std(ddof=1) if n > 1 else float("nan")
    t = g.mean() / (sd / math.sqrt(n)) if n > 1 and sd and sd == sd else None
    return {"n_obs": int(len(s)), "n_dates": n, "mean": float(g.mean()),
            "t": float(t) if t is not None else None, "median": float(s.median())}


def paired_by_date(df, flag_col: str, val_col: str) -> dict:
    """Разность средних «с новостью − без» внутри даты, t по датам."""
    import pandas as pd
    d = df.dropna(subset=[val_col])
    m = d.groupby(["date", flag_col])[val_col].mean().unstack()
    if True not in m.columns or False not in m.columns:
        return {"n_dates": 0, "diff": None, "t": None}
    diff = (m[True] - m[False]).dropna()
    n = len(diff)
    if n < 2:
        return {"n_dates": n, "diff": None, "t": None}
    t = diff.mean() / (diff.std(ddof=1) / math.sqrt(n)) if diff.std(ddof=1) else None
    return {"n_dates": n, "diff": float(diff.mean()), "t": float(t) if t is not None else None}


# ── Загрузка ─────────────────────────────────────────────────────────────────

def load_posts(conn, channel: str, since: dt.date):
    import pandas as pd
    df = pd.read_sql("SELECT message_id, posted_at, text FROM news.tg_posts "
                     "WHERE channel = %s AND posted_at >= %s ORDER BY posted_at",
                     conn, params=(channel, since))
    df["msk"] = (pd.to_datetime(df["posted_at"], utc=True)
                 .dt.tz_convert("Europe/Moscow").dt.tz_localize(None))
    return df


def load_daily(conn):
    import pandas as pd
    df = pd.read_sql("SELECT ticker, date, open, close FROM market_data "
                     "WHERE ticker = ANY(%s) AND open > 0 AND close > 0 "
                     "AND EXTRACT(ISODOW FROM date) < 6 ORDER BY date",
                     conn, params=(list(UNIVERSE),))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("open", "close"):
        df[c] = df[c].astype(float)
    return df


def load_exec_bars(conn):
    """Бары 18:30 (цена входа) и 10:00 (цена выхода) из 5-минуток."""
    import pandas as pd
    df = pd.read_sql("""
        SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS msk, open, close
        FROM market_data_5m
        WHERE ticker = ANY(%s) AND close > 0
          AND (ts AT TIME ZONE 'Europe/Moscow')::time IN (TIME '18:30', TIME '10:00')
          AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6""",
                     conn, params=(list(UNIVERSE),))
    df["msk"] = pd.to_datetime(df["msk"])
    df["date"] = df["msk"].dt.date
    df["hhmm"] = df["msk"].dt.strftime("%H:%M")
    for c in ("open", "close"):
        df[c] = df[c].astype(float)
    return df


# ── События ──────────────────────────────────────────────────────────────────

def build_events(posts) -> list[dict]:
    """(пост, тикер) с тональностью; отчёты о движении цены помечены."""
    out = []
    for r in posts.itertuples(index=False):
        tks = tickers_in(r.text)
        if not tks:
            continue
        rep, tn = is_price_report(r.text), tone(r.text)
        avail = r.msk.to_pydatetime() + LAG
        for tk in tks:
            out.append({"message_id": r.message_id, "ticker": tk, "avail": avail,
                        "tone": tn, "price_report": rep})
    return out


def scheme_a(daily, events) -> dict:
    import pandas as pd
    o = daily.pivot(index="date", columns="ticker", values="open").sort_index()
    c = daily.pivot(index="date", columns="ticker", values="close").sort_index()
    days = list(o.index)
    gap = o / c.shift(1) - 1.0
    intra = c / o - 1.0
    ar_gap, ar_day = abnormal(gap) * 100, abnormal(intra) * 100

    # known_news — всё, что известно к решению в 18:35 дня D: окно гэпа D и
    # дневные новости до 18:35. Именно на этом наборе решение принималось бы.
    gap_news, known_news, known_tone = set(), set(), {}
    for e in events:
        if e["price_report"]:
            continue
        w = daily_window(e["avail"], days)
        if not w:
            continue
        kind, d = w
        if kind == "gap":
            gap_news.add((e["ticker"], d))
        if kind == "gap" or e["avail"].time() <= DECISION_T:
            known_news.add((e["ticker"], d))
            known_tone[(e["ticker"], d)] = known_tone.get((e["ticker"], d), 0) + e["tone"]

    rows = []
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}
    for d in days:
        for tk in o.columns:
            g = ar_gap.at[d, tk]
            if g != g:
                continue
            nd = nxt.get(d)
            rows.append({"date": d, "ticker": tk, "ar_gap": g,
                         "gap_news": (tk, d) in gap_news,
                         "known_news": (tk, d) in known_news,
                         "tone": int(math.copysign(1, known_tone[(tk, d)])) if known_tone.get((tk, d)) else 0,
                         "ar_next_gap": ar_gap.at[nd, tk] if nd else None,
                         "ar_next_day": ar_day.at[nd, tk] if nd else None})
    return {"df": pd.DataFrame(rows), "days": days}


def scheme_b(bars, events) -> dict:
    import pandas as pd
    entry = bars[bars["hhmm"] == "18:30"].pivot(index="date", columns="ticker", values="close")
    exitp = bars[bars["hhmm"] == "10:00"].pivot(index="date", columns="ticker", values="open")
    days = sorted(set(entry.index) & set(exitp.index))
    entry, exitp = entry.reindex(days), exitp.reindex(days)
    ret = exitp.shift(-1) / entry - 1.0           # вход 18:35 D → выход 10:00 D+1
    ar = abnormal(ret) * 100

    known, hold, ktone = set(), set(), {}
    for e in events:
        if e["price_report"]:
            continue
        for kind, d in exec_windows(e["avail"], days):
            (known if kind == "known" else hold).add((e["ticker"], d))
            if kind == "known":
                ktone[(e["ticker"], d)] = ktone.get((e["ticker"], d), 0) + e["tone"]
    rows = []
    for d in days[:-1]:
        for tk in ar.columns:
            v = ar.at[d, tk]
            if v != v:
                continue
            rows.append({"date": d, "ticker": tk, "ar_exec": v,
                         "known": (tk, d) in known, "hold": (tk, d) in hold,
                         "tone": int(math.copysign(1, ktone[(tk, d)])) if ktone.get((tk, d)) else 0})
    return {"df": pd.DataFrame(rows), "days": days}


# ── Отчёт ────────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    return "—" if x is None or x != x else f"{x:+.{nd}f}".replace(".", ",")


def _row(name, st, cost_rule=False):
    net = None if st.get("mean") is None else st["mean"] - COST_RT_PCT
    return (f"| {name} | {st['n_obs']} | {st['n_dates']} | {_f(st['mean'])} | "
            f"{_f(st['t'], 2)} | {_f(st['median'])} |"
            + (f" {_f(net)} |" if cost_rule else ""))


def _halves(df, col, mask):
    dates = sorted(df["date"].unique())
    if len(dates) < 20:
        return "—", "—"
    cut = dates[len(dates) // 2]
    a = df[mask & (df["date"] < cut)]
    b = df[mask & (df["date"] >= cut)]
    sa, sb = by_date(a[col], a["date"]), by_date(b[col], b["date"])
    return f"{_f(sa['mean'])} (t {_f(sa['t'], 1)})", f"{_f(sb['mean'])} (t {_f(sb['t'], 1)})"


def report(a: dict, b: dict, meta: dict) -> str:
    L = [f"# Новости markettwits → гэпы и избыточная доходность (без ML)",
         "", f"Сформировано {meta['created']}. Режим: только исследование, торговый контур r3 не затронут.",
         "", "## Покрытие", "",
         f"- постов в периоде: {meta['posts']}; с тикером вселенной: {meta['posts_with_ticker']} "
         f"({meta['share_ticker']:.1f}%); отчётов о движении цены исключено: {meta['price_reports']}",
         f"- пар «пост × тикер»: {meta['events']}; топ-10: {meta['top']}",
         f"- тональность словаря: позитив {meta['pos']}, негатив {meta['neg']}, нейтрально {meta['neu']}",
         "", "Все величины — аномальная доходность в % (минус среднее остальных бумаг), "
         "среднее по датам; t — по датам. Издержки round-trip 0,128%.", ""]

    da = a["df"]
    if not da.empty:
        L += [f"## A. Дневные бары ({a['days'][0]} … {a['days'][-1]})", "",
              "| выборка | наблюд. | дат | среднее, % | t | медиана, % |", "|---|---|---|---|---|---|"]
        for name, mask, col in (
                ("A1 гэп D: новость в окне гэпа", da["gap_news"], "ar_gap"),
                ("A1 гэп D: без новости", ~da["gap_news"], "ar_gap"),
                ("A2 гэп D+1: новость известна к 18:35 D", da["known_news"], "ar_next_gap"),
                ("A2 гэп D+1: без новости", ~da["known_news"], "ar_next_gap"),
                ("A2 доходность D+1: новость к 18:35 D", da["known_news"], "ar_next_day")):
            s = da[mask]
            L.append(_row(name, by_date(s[col], s["date"])))
        L += ["", "Модуль движения (не направление): |аномальный гэп| с новостью и без.", ""]
        for name, mask in (("|гэп D|, новость в окне гэпа", da["gap_news"]),
                           ("|гэп D|, без новости", ~da["gap_news"])):
            s = da[mask]
            L.append(f"- {name}: {_f(by_date(s['ar_gap'].abs(), s['date'])['mean'])}%")
        pa = paired_by_date(da.assign(absgap=da["ar_gap"].abs()), "gap_news", "absgap")
        L.append(f"- разность внутри даты: {_f(pa['diff'])} п.п., t {_f(pa['t'], 2)}, дат {pa['n_dates']}")
        L += ["", "### Правила с учётом издержек (словарь, новость известна к 18:35 D)", "",
              "| правило | наблюд. | дат | среднее, % | t | медиана, % | нетто, % |",
              "|---|---|---|---|---|---|---|"]
        pos = da[da["known_news"] & (da["tone"] > 0)]
        neg = da[da["known_news"] & (da["tone"] < 0)]
        L.append(_row("лонг на ночь при позитиве (гэп D+1)", by_date(pos["ar_next_gap"], pos["date"]), True))
        L.append(_row("шорт на ночь при негативе (−гэп D+1)", by_date(-neg["ar_next_gap"], neg["date"]), True))
        h1, h2 = _halves(da, "ar_next_gap", da["known_news"] & (da["tone"] > 0))
        L.append(f"\nЛонг при позитиве по половинам периода: {h1} → {h2}")

    db = b["df"]
    if not db.empty:
        L += ["", f"## B. Реальное окно исполнения 18:35 → 10:00 ({b['days'][0]} … {b['days'][-1]})", "",
              "| выборка | наблюд. | дат | среднее, % | t | медиана, % |", "|---|---|---|---|---|---|"]
        for name, mask in (("новость известна к 18:35", db["known"]),
                           ("без новости к 18:35", ~db["known"]),
                           ("новость пришла во время удержания", db["hold"])):
            s = db[mask]
            L.append(_row(name, by_date(s["ar_exec"], s["date"])))
        L += ["", "| правило | наблюд. | дат | среднее, % | t | медиана, % | нетто, % |",
              "|---|---|---|---|---|---|---|"]
        pos = db[db["known"] & (db["tone"] > 0)]
        neg = db[db["known"] & (db["tone"] < 0)]
        L.append(_row("лонг при позитиве", by_date(pos["ar_exec"], pos["date"]), True))
        L.append(_row("исключить при негативе (эффект = −среднее)", by_date(-neg["ar_exec"], neg["date"]), True))
        ph = paired_by_date(db.assign(absr=db["ar_exec"].abs()), "hold", "absr")
        L.append(f"\nРиск удержания: |аномальная доходность| при новости ночью минус без неё — "
                 f"{_f(ph['diff'])} п.п., t {_f(ph['t'], 2)}, дат {ph['n_dates']}")

    L += ["", "## Как читать", "",
          "- «Сигнал есть» — только если |t| ≥ 2 в обеих половинах периода и нетто после 0,128% > 0.",
          "- Проверено несколько выборок: отдельный t ≈ 2 при таком числе сравнений ещё не доказательство.",
          "- Словарь тональности грубый; LLM имеет смысл подключать, если уже здесь видна направленность.",
          ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Новости → гэпы и избыточная доходность (без ML)")
    ap.add_argument("--channel", default="markettwits")
    ap.add_argument("--since", default="2024-05-01")
    ap.add_argument("--out", default=None, help="каталог отчёта (по умолчанию audit/news_research/<метка>)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import database
    since = dt.date.fromisoformat(a.since)
    conn = database.get_connection()
    try:
        posts = load_posts(conn, a.channel, since)
        daily = load_daily(conn)
        bars = load_exec_bars(conn)
    finally:
        conn.close()

    events = build_events(posts)
    with_tk = {e["message_id"] for e in events}
    reports = {e["message_id"] for e in events if e["price_report"]}
    from collections import Counter
    top = Counter(e["ticker"] for e in events if not e["price_report"]).most_common(10)
    tones = Counter(e["tone"] for e in events if not e["price_report"])
    a_res, b_res = scheme_a(daily, events), scheme_b(bars, events)
    now = dt.datetime.now()
    meta = {"created": now.strftime("%d.%m.%Y %H:%M"), "posts": len(posts),
            "posts_with_ticker": len(with_tk),
            "share_ticker": 100.0 * len(with_tk) / max(1, len(posts)),
            "price_reports": len(reports), "events": len(events),
            "top": ", ".join(f"{t} {n}" for t, n in top),
            "pos": tones.get(1, 0), "neg": tones.get(-1, 0), "neu": tones.get(0, 0)}
    text = report(a_res, b_res, meta)

    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "audit", "news_research", now.strftime("%Y%m%d-%H%M"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    a_res["df"].to_csv(os.path.join(out, "scheme_a.csv"), index=False)
    b_res["df"].to_csv(os.path.join(out, "scheme_b.csv"), index=False)
    print(text)
    log.info("отчёт: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
