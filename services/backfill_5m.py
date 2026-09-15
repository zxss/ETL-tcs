"""
Разовая догрузка 5-минуток за прошлые годы (ТЗ EDGE-R4 §3, блок A1).

Источник для акций — годовые архивы минуток T-Invest (history-data): один
запрос = год 1-минутных свечей одной бумаги (zip, по CSV на день). 46 бумаг ×
2 года ≈ 100 запросов вместо ≈ 26 тыс. запросов GetCandles по дню. Минутки
сворачиваются в 5-минутки по началу интервала: open первой минуты, close
последней, high/low — экстремумы, объём — сумма. Совпадение с барами
GetCandles проверяется на перекрытии (--verify).

Индекс архивом не отдаётся (IMOEX: 404) — для него GetCandles по дню
(loaders.moex_loader.fetch_5m_candles).

Запись — в market_data_5m с ON CONFLICT DO NOTHING: бары, которые уже загрузил
ETL, не перезаписываются, повторный запуск безопасен. Прогресс по (тикер, год)
хранится в audit/backfill_5m/state.json — оборванная загрузка продолжается с
места обрыва; индекс продолжает с последнего загруженного бара.

Прод эти годы не читает: intraday берёт последние INTRADAY_LOOKBACK_DAYS (60),
quotes — только сегодняшний бар, backfill_daily — дни новее последней дневки.

Запуск на сервере от etl, вне фаз r3 (токен и лимиты API общие с ботом):
    python -m services.backfill_5m --from 2024-05-21 --to 2025-11-30
    python -m services.backfill_5m --index IMOEX --from 2024-05-21 --to 2026-09-14
    python -m services.backfill_5m --verify SBER --year 2025     # без записи
    python -m services.backfill_5m --report --from 2024-05-21 --to 2026-09-11
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import io
import json
import logging
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config                                            # noqa: E402

log = logging.getLogger("backfill_5m")

MSK = dt.timezone(dt.timedelta(hours=3))
HISTORY_URL = os.getenv("INVEST_HISTORY_URL", "https://invest-public-api.tbank.ru/history-data")
ARCHIVE_PAUSE_SEC = 2.1        # лимит history-data — 30 запросов в минуту
ARCHIVE_RETRIES = 5
STATE_PATH = os.path.join(ROOT, "audit", "backfill_5m", "state.json")

INSERT_5M_SQL = """
INSERT INTO {table} (ticker, ts, open, high, low, close, volume)
VALUES %s
ON CONFLICT (ticker, ts) DO NOTHING
RETURNING 1
"""

# Куда писать. research_bars_5m — отложенная история для исследований (Спринт 1,
# 2022 → 20.05.2024): отдельно от market_data_5m, чтобы её никто не «видел»
# до проверки и чтобы прод её не читал.
TABLES = ("market_data_5m", "research_bars_5m")
CREATE_RESEARCH_SQL = """
CREATE TABLE IF NOT EXISTS research_bars_5m (
    id      BIGSERIAL PRIMARY KEY,
    ticker  VARCHAR(10)    NOT NULL,
    ts      TIMESTAMPTZ    NOT NULL,
    open    NUMERIC(18, 6) NOT NULL,
    high    NUMERIC(18, 6) NOT NULL,
    low     NUMERIC(18, 6) NOT NULL,
    close   NUMERIC(18, 6) NOT NULL,
    volume  BIGINT         NOT NULL DEFAULT 0,
    CONSTRAINT uq_research_bars_5m UNIQUE (ticker, ts)
);
"""


def _table(name: str) -> str:
    if name not in TABLES:
        raise ValueError(f"таблица {name!r} не из списка {TABLES}")
    return name


def state_path_for(table: str) -> str:
    return STATE_PATH if table == "market_data_5m" else \
        os.path.join(os.path.dirname(STATE_PATH), f"state_{table}.json")


def ensure_table(conn, table: str) -> None:
    if _table(table) == "research_bars_5m":
        with conn.cursor() as cur:
            cur.execute(CREATE_RESEARCH_SQL)
        conn.commit()


# ── Разбор архива ────────────────────────────────────────────────────────────

def parse_minute_csv(text: str) -> list[tuple]:
    """Строки архива «uid;time;open;close;high;low;volume;» → (ts, o, h, l, c, v).

    Порядок полей в архиве — open, CLOSE, high, low (не OHLC).
    """
    out = []
    for line in text.splitlines():
        p = line.strip().split(";")
        if len(p) < 7 or not p[1]:
            continue
        try:
            ts = dt.datetime.fromisoformat(p[1].replace("Z", "+00:00"))
            o, c, h, lo = (float(x) for x in p[2:6])
            v = int(float(p[6] or 0))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        if o > 0:
            out.append((ts, o, h, lo, c, v))
    return out


def to_5m(minutes: list[tuple]) -> list[tuple]:
    """Минутки → 5-минутки по началу интервала (UTC и МСК кратны 5 минутам)."""
    buckets: dict[dt.datetime, list] = {}
    for ts, o, h, lo, c, v in sorted(minutes, key=lambda r: r[0]):
        key = ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)
        b = buckets.get(key)
        if b is None:
            buckets[key] = [o, h, lo, c, v]
        else:
            b[1] = max(b[1], h)
            b[2] = min(b[2], lo)
            b[3] = c
            b[4] += v
    return [(k, *vals) for k, vals in sorted(buckets.items())]


def _file_day(name: str) -> dt.date | None:
    """«<uid>_20241219.csv» → 2024-12-19 (день по UTC)."""
    stem = os.path.basename(name).rsplit(".", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    try:
        return dt.datetime.strptime(tail, "%Y%m%d").date()
    except ValueError:
        return None


def archive_5m(blob: bytes, date_from: dt.date, date_to: dt.date) -> list[tuple]:
    """5-минутки из zip-архива за [date_from, date_to] по московской дате."""
    one = dt.timedelta(days=1)
    minutes: list[tuple] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(z.namelist()):
            day = _file_day(name)
            # файлы разбиты по UTC-дням: московская дата бывает на день позже
            if day is not None and not (date_from - one <= day <= date_to + one):
                continue
            minutes.extend(parse_minute_csv(z.read(name).decode("utf-8")))
    return [b for b in to_5m(minutes)
            if date_from <= b[0].astimezone(MSK).date() <= date_to]


def _retry_wait(headers, attempt: int) -> float:
    for k in ("x-ratelimit-reset", "Retry-After"):
        v = headers.get(k) if headers else None
        try:
            if v is not None:
                return float(v) + 1.0
        except ValueError:
            pass
    return 5.0 * attempt


# ── Состояние ────────────────────────────────────────────────────────────────

def load_state(path: str, date_from: dt.date, date_to: dt.date) -> dict:
    """Прогресс по (тикер, год). Другой диапазон дат — прогресс с нуля."""
    rng = f"{date_from}:{date_to}"
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    if st.get("range") != rng:
        st = {"range": rng, "done": {}}
    st.setdefault("done", {})
    return st


def save_state(path: str, st: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ── Запись ───────────────────────────────────────────────────────────────────

def insert_bars(conn, ticker: str, bars: list[tuple], table: str = "market_data_5m") -> int:
    """Вставляет (ts, o, h, l, c, v); возвращает число новых строк.

    market_data_5m хранит 4 знака (как ETL); research_bars_5m — 6, чтобы копеечные
    бумаги (TGKA ≈ 0,006 ₽) не превращались в шум округления."""
    if not bars:
        return 0
    from psycopg2.extras import execute_values
    nd = 4 if _table(table) == "market_data_5m" else 6
    rows = [(ticker, ts, round(o, nd), round(h, nd), round(lo, nd), round(c, nd), int(v))
            for ts, o, h, lo, c, v in bars]
    with conn.cursor() as cur:
        got = execute_values(cur, INSERT_5M_SQL.format(table=table), rows,
                             page_size=5000, fetch=True)
    conn.commit()
    return len(got)


# ── Сеть ─────────────────────────────────────────────────────────────────────

def _session():
    import aiohttp
    from loaders import moex_loader
    return aiohttp.ClientSession(headers=moex_loader.make_headers(),
                                 connector=moex_loader.make_connector())


async def find_uid(session, ticker: str, index: bool = False) -> str:
    from loaders import moex_loader
    kind = "INSTRUMENT_TYPE_INDEX" if index else "INSTRUMENT_TYPE_SHARE"
    data = await moex_loader._api_post(session, "InstrumentsService/FindInstrument",
                                       {"query": ticker, "instrumentKind": kind})
    for it in data.get("instruments", []):
        if it.get("ticker") == ticker and (index or it.get("classCode") == "TQBR"):
            return it.get("uid") or it.get("figi")
    raise RuntimeError(f"{ticker}: инструмент не найден ({kind})")


async def fetch_archive(session, uid: str, year: int) -> bytes | None:
    """Годовой архив минуток; None — архива нет (404)."""
    import aiohttp
    from loaders import moex_loader
    url = f"{HISTORY_URL}?instrumentId={uid}&year={year}"
    for attempt in range(1, ARCHIVE_RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as r:
                if r.status == 200:
                    return await r.read()
                if r.status == 404:
                    return None
                if r.status in moex_loader.RETRYABLE and attempt < ARCHIVE_RETRIES:
                    wait = _retry_wait(r.headers, attempt)
                    log.warning("history-data HTTP %s, повтор через %.0f с", r.status, wait)
                    await asyncio.sleep(wait)
                    continue
                body = (await r.text())[:200]
                raise RuntimeError(f"history-data {uid} {year}: HTTP {r.status} {body}")
        except aiohttp.ClientError as e:
            if attempt == ARCHIVE_RETRIES:
                raise RuntimeError(f"history-data {uid} {year}: {e}") from e
            await asyncio.sleep(5.0 * attempt)
    raise RuntimeError(f"history-data {uid} {year}: попытки исчерпаны")


# ── Режимы ───────────────────────────────────────────────────────────────────

def state_key(ticker: str, source: str, year: int) -> str:
    """Ключ прогресса: при загрузке по старому коду — «ИСТОЧНИК->ТИКЕР:год»."""
    return f"{ticker}:{year}" if source == ticker else f"{source}->{ticker}:{year}"


async def backfill_shares(conn, tickers: list[str], date_from: dt.date,
                          date_to: dt.date, state_path: str | None = None,
                          table: str = "market_data_5m",
                          aliases: dict[str, str] | None = None) -> list[dict]:
    """aliases: {тикер в БД: код-источник} — история по старому коду (YNDX для
    YDEX, FIVE для X5) записывается под нынешним тикером."""
    state_path = state_path or state_path_for(table)
    st = load_state(state_path, date_from, date_to)
    stats = []
    async with _session() as s:
        for tk in tickers:
            src = (aliases or {}).get(tk, tk)
            uid = None
            for year in range(date_from.year, date_to.year + 1):
                key = state_key(tk, src, year)
                if key in st["done"]:
                    continue
                uid = uid or await find_uid(s, src)
                blob = await fetch_archive(s, uid, year)
                await asyncio.sleep(ARCHIVE_PAUSE_SEC)
                lo = max(date_from, dt.date(year, 1, 1))
                hi = min(date_to, dt.date(year, 12, 31))
                bars = archive_5m(blob, lo, hi) if blob else []
                new = await asyncio.to_thread(insert_bars, conn, tk, bars, table)
                days = len({b[0].astimezone(MSK).date() for b in bars})
                rec = {"ticker": tk, "year": year, "archive": blob is not None,
                       "bars": len(bars), "inserted": new, "days": days}
                st["done"][key] = rec
                save_state(state_path, st)
                stats.append(rec)
                log.info("%-6s %d: баров %d, новых %d, дней %d%s", tk, year, len(bars),
                         new, days, "" if blob else " (архива нет)")
    return stats


async def backfill_index(conn, ticker: str, date_from: dt.date, date_to: dt.date,
                         table: str = "market_data_5m", source: str | None = None,
                         index: bool = True) -> int:
    """GetCandles по дню; продолжает с последнего загруженного бара.

    Индекс (архивом не отдаётся) или акция без годовых архивов (T до 2024,
    ETLN): source — код-источник, index=False — искать акцию на TQBR."""
    from loaders import moex_loader
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(ts) FROM {_table(table)} WHERE ticker = %s "
                    "AND ts >= %s AND ts < %s",
                    (ticker, dt.datetime.combine(date_from, dt.time(), MSK),
                     dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time(), MSK)))
        last = cur.fetchone()[0]
    start = last.astimezone(MSK).date() if last else date_from
    end = dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time(), MSK)
    end = min(end, dt.datetime.now(MSK))
    total = 0
    async with _session() as s:
        uid = await find_uid(s, source or ticker, index=index)
        cur_dt = dt.datetime.combine(start, dt.time(), MSK)
        while cur_dt < end:
            nxt = min(cur_dt + dt.timedelta(days=30), end)
            rows = await moex_loader.fetch_5m_candles(s, uid, cur_dt, nxt)
            bars = [(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")), o, h, lo, c, v)
                    for ts, o, h, lo, c, v in rows]
            new = await asyncio.to_thread(insert_bars, conn, ticker, bars, table)
            total += new
            log.info("%s %s … %s: баров %d, новых %d", ticker, cur_dt.date(), nxt.date(),
                     len(bars), new)
            cur_dt = nxt
    return total


def compare_bars(arch: list[tuple], db: list[tuple], tol: float = 1e-6) -> dict:
    """Сверка архива со строками БД по общим меткам времени.

    Цены архива округляются до 4 знаков — точности market_data_5m
    (NUMERIC(18,4)); иначе у копеечных бумаг (TGKA ≈ 0,006 ₽) расходится всё.
    """
    a = {r[0]: tuple(round(float(x), 4) for x in r[1:5]) + (r[5],) for r in arch}
    b = {r[0]: tuple(float(x) for x in r[1:]) for r in db}
    common = sorted(set(a) & set(b))
    ohlc = vol = 0
    for k in common:
        x, y = a[k], b[k]
        if all(abs(x[i] - y[i]) <= tol * max(1.0, abs(y[i])) for i in range(4)):
            ohlc += 1
        if int(x[4]) == int(y[4]):
            vol += 1
    n = len(common)
    return {"archive": len(a), "db": len(b), "common": n,
            "only_archive": len(set(a) - set(b)), "only_db": len(set(b) - set(a)),
            "ohlc_equal_share": ohlc / n if n else None,
            "volume_equal_share": vol / n if n else None}


async def verify(conn, ticker: str, year: int) -> dict:
    async with _session() as s:
        blob = await fetch_archive(s, await find_uid(s, ticker), year)
    if not blob:
        return {"ticker": ticker, "year": year, "archive": False}
    arch = archive_5m(blob, dt.date(year, 1, 1), dt.date(year, 12, 31))
    with conn.cursor() as cur:
        cur.execute("SELECT ts, open, high, low, close, volume FROM market_data_5m "
                    "WHERE ticker = %s AND ts >= %s AND ts < %s",
                    (ticker, dt.datetime(year, 1, 1, tzinfo=MSK),
                     dt.datetime(year + 1, 1, 1, tzinfo=MSK)))
        db = cur.fetchall()
    return {"ticker": ticker, "year": year, **compare_bars(arch, db)}


# ── Отчёт полноты (приёмка A) ────────────────────────────────────────────────

COVERAGE_SQL = """
SELECT ticker, d,
       count(*) FILTER (WHERE t >= mo AND t < mc)  AS main_bars,
       count(*)                                    AS all_bars,
       (array_agg(close ORDER BY ts DESC))[1]      AS last_close
FROM (
    SELECT ticker, ts, close,
           (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
           (ts AT TIME ZONE 'Europe/Moscow')::time AS t,
           CASE WHEN (ts AT TIME ZONE 'Europe/Moscow')::date >= %(new)s
                THEN %(mo_new)s::time ELSE %(mo_old)s::time END AS mo,
           CASE WHEN (ts AT TIME ZONE 'Europe/Moscow')::date >= %(new)s
                THEN %(mc_new)s::time ELSE %(mc_old)s::time END AS mc
    FROM market_data_5m
    WHERE ts >= %(f)s AND ts < %(t)s
) s
GROUP BY ticker, d
"""


def coverage_frame(conn, date_from: dt.date, date_to: dt.date):
    import pandas as pd
    from research import session_calendar as sc
    old, new = sc.session(sc.NEW_SCHEDULE_FROM - dt.timedelta(days=1)), sc.session(sc.NEW_SCHEDULE_FROM)
    cov = pd.read_sql(COVERAGE_SQL, conn, params={
        "new": sc.NEW_SCHEDULE_FROM, "mo_new": new.main_open, "mc_new": new.main_close,
        "mo_old": old.main_open, "mc_old": old.main_close,
        "f": dt.datetime.combine(date_from, dt.time(), MSK),
        "t": dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time(), MSK)})
    daily = pd.read_sql("SELECT ticker, date AS d, close AS daily_close FROM market_data "
                        "WHERE date BETWEEN %s AND %s", conn, params=(date_from, date_to))
    cov["d"] = pd.to_datetime(cov["d"]).dt.date
    daily["d"] = pd.to_datetime(daily["d"]).dt.date
    cov = cov.merge(daily, on=["ticker", "d"], how="left")
    cov["expected"] = [sc.session(d).expected_main_bars for d in cov["d"]]
    for c in ("last_close", "daily_close"):
        cov[c] = pd.to_numeric(cov[c], errors="coerce")
    return cov


def gaps(days_by_ticker: dict, min_days: int = 5) -> list[dict]:
    """Разрывы между соседними днями с основной сессией длиннее min_days (A3)."""
    out = []
    for tk, days in days_by_ticker.items():
        ds = sorted(days)
        for a, b in zip(ds, ds[1:]):
            if (b - a).days > min_days:
                out.append({"ticker": tk, "from": a, "to": b, "days": (b - a).days})
    return out


def coverage_report(cov, universe: list[str], date_from, date_to) -> str:
    wk = cov[[d.weekday() < 5 for d in cov["d"]]]
    stocks = wk[wk["ticker"].isin(universe)]
    main = stocks[stocks["main_bars"] > 0]
    per_date = main.groupby("d")["ticker"].nunique()
    per_tk = main.groupby("ticker")["d"].nunique().reindex(universe).fillna(0).astype(int)
    comp = (main["main_bars"] / main["expected"]).clip(upper=1.0)
    rec = stocks.dropna(subset=["last_close", "daily_close"])
    rec = rec[rec["daily_close"] > 0]
    dev = (rec["last_close"] / rec["daily_close"] - 1.0).abs()
    ok = float((dev <= 0.001).mean()) if len(dev) else float("nan")
    gp = gaps({tk: set(g["d"]) for tk, g in main.groupby("ticker")})
    L = [f"# Полнота 5-минуток {date_from} … {date_to}", "",
         f"- будних дат с основной сессией у ≥ 40 из {len(universe)} бумаг: "
         f"**{int((per_date >= 40).sum())}** (приёмка ≥ 560)",
         f"- бумаг с ≥ 560 будними датами: {int((per_tk >= 560).sum())} из {len(universe)}",
         f"- полнота бумаго-дня (баров основной сессии / ожидаемых): медиана "
         f"{comp.median():.3f}, доля ≥ 95 %: {(comp >= 0.95).mean():.3f}",
         f"- close последнего 5-минутного бара = дневной close (±0,1 %): **{ok:.3%}** "
         f"из {len(dev)} бумаго-дней (приёмка ≥ 98 %)",
         f"- разрывов > 5 дней: {len(gp)}", "",
         "| бумага | будних дат | медиана полноты | сверка с дневкой ±0,1 % |",
         "|---|---|---|---|"]
    for tk in universe:
        m = main[main["ticker"] == tk]
        r = rec[rec["ticker"] == tk]
        dv = (r["last_close"] / r["daily_close"] - 1.0).abs()
        c = (m["main_bars"] / m["expected"]).clip(upper=1.0)
        L.append(f"| {tk} | {per_tk.get(tk, 0)} | "
                 f"{c.median():.3f} | {(dv <= 0.001).mean():.3f} |" if len(m) else
                 f"| {tk} | 0 | — | — |")
    if gp:
        L += ["", "## Разрывы > 5 дней (исход через разрыв = NaN)", "",
              "| бумага | с | по | дней |", "|---|---|---|---|"]
        L += [f"| {g['ticker']} | {g['from']} | {g['to']} | {g['days']} |" for g in gp]
    return "\n".join(L) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Догрузка 5-минуток из годовых архивов T-Invest")
    ap.add_argument("--from", dest="date_from", default="2024-05-21")
    ap.add_argument("--to", dest="date_to", default="2025-11-30")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--index", default=None, help="индекс через GetCandles (IMOEX)")
    ap.add_argument("--verify", default=None, help="тикер: сверить архив с БД, без записи")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="отчёт полноты (приёмка A)")
    ap.add_argument("--table", default="market_data_5m", choices=TABLES,
                    help="куда писать; research_bars_5m — отложенная история исследований")
    ap.add_argument("--alias", nargs="*", default=None, metavar="ТИКЕР=ИСТОЧНИК",
                    help="история по старому коду под нынешним тикером: YDEX=YNDX X5=FIVE")
    ap.add_argument("--candles", action="store_true",
                    help="с --alias: GetCandles по дню вместо годовых архивов (T, ETLN)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import database
    d_from, d_to = dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to)
    conn = database.get_connection()
    try:
        if a.verify:
            res = asyncio.run(verify(conn, a.verify.upper(), a.year or d_to.year))
            print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        elif a.report:
            text = coverage_report(coverage_frame(conn, d_from, d_to),
                                   list(config.TICKERS), d_from, d_to)
            out = os.path.join(ROOT, "audit", "backfill_5m",
                               f"coverage-{dt.datetime.now():%Y%m%d-%H%M}.md")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(text)
            log.info("отчёт: %s", out)
        elif a.alias:
            ensure_table(conn, a.table)
            pairs = {k.upper(): v.upper() for k, v in (p.split("=", 1) for p in a.alias)}
            if a.candles:
                for new, old in pairs.items():
                    n = asyncio.run(backfill_index(conn, new, d_from, d_to, table=a.table,
                                                   source=old, index=False))
                    log.info("%s ← %s (GetCandles): новых баров %d", new, old, n)
            else:
                stats = asyncio.run(backfill_shares(conn, list(pairs), d_from, d_to,
                                                    table=a.table, aliases=pairs))
                log.info("готово: запросов %d, новых баров %d", len(stats),
                         sum(s["inserted"] for s in stats))
        elif a.index:
            ensure_table(conn, a.table)
            n = asyncio.run(backfill_index(conn, a.index.upper(), d_from, d_to, table=a.table))
            log.info("%s: новых баров %d", a.index.upper(), n)
        else:
            ensure_table(conn, a.table)
            tickers = [t.upper() for t in (a.tickers or config.TICKERS)]
            stats = asyncio.run(backfill_shares(conn, tickers, d_from, d_to, table=a.table))
            log.info("готово: запросов %d, новых баров %d", len(stats),
                     sum(s["inserted"] for s in stats))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
