#!/usr/bin/env python3
"""
Очистка «мёртвых» тикеров из market_data / market_data_5m.

Зачем: делистнутая бумага оставляет в БД замороженную историю (например, FIXP —
GDR Fix Price, бары кончились 20.06.2025 после редомициляции в FIXR). Такие
ряды продолжают участвовать в кросс-секционных расчётах — перцентилях
ликвидности, ширине рынка, отборе кандидатов — и тянут картину рынка к
состоянию годовой давности.

По умолчанию бары НЕ удаляются, а переносятся в архивные таблицы
market_data_archive / market_data_5m_archive (та же структура + archived_at и
reason). Полное удаление — только явным --purge.

Кандидаты определяются автоматически: тикер есть в БД, но
  * его нет в config.TICKERS / config.INDEX_TICKERS (сопровождение прекращено), И
  * последний бар старше --stale-days (по умолчанию 30) от максимальной даты в БД.
Список можно задать вручную: --ticker FIXP --ticker XXXX.

Запуск:
    python3 -m scripts.cleanup_stale_tickers --dry-run     # только показать
    python3 -m scripts.cleanup_stale_tickers               # архивировать
    python3 -m scripts.cleanup_stale_tickers --ticker FIXP # конкретный тикер
    python3 -m scripts.cleanup_stale_tickers --purge       # удалить без архива
    python3 -m scripts.cleanup_stale_tickers --restore FIXP  # вернуть из архива
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# запуск и как `python3 scripts/cleanup_stale_tickers.py`, и как `-m scripts...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config      # noqa: E402
import database    # noqa: E402

log = logging.getLogger("cleanup_stale_tickers")

CREATE_ARCHIVE_SQL = """
CREATE TABLE IF NOT EXISTS market_data_archive (
    LIKE market_data INCLUDING DEFAULTS,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_data_archive_ticker
    ON market_data_archive (ticker, date DESC);

CREATE TABLE IF NOT EXISTS market_data_5m_archive (
    LIKE market_data_5m INCLUDING DEFAULTS,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_data_5m_archive_ticker
    ON market_data_5m_archive (ticker, ts DESC);
"""

_STATS_SQL = """
SELECT ticker, count(*) AS n, min(date) AS first, max(date) AS last
FROM market_data
GROUP BY ticker
ORDER BY last ASC, ticker;
"""

_MAX_DATE_SQL = "SELECT max(date) FROM market_data;"


def _tracked_tickers() -> set[str]:
    """Тикеры, которые проект сопровождает сейчас."""
    tracked = set(t.upper() for t in getattr(config, "TICKERS", []))
    tracked |= set(t.upper() for t in getattr(config, "INDEX_TICKERS", []))
    tracked |= set(t.upper() for t in getattr(config, "TFT_TICKERS", []))
    tracked |= set(t.upper() for t in getattr(config, "VALIDATION_TICKERS", []))
    return tracked


def find_stale(conn, stale_days: int) -> list[dict]:
    """Тикеры-кандидаты на архивацию: вне сопровождения и отставшие по датам."""
    tracked = _tracked_tickers()
    with conn.cursor() as cur:
        cur.execute(_MAX_DATE_SQL)
        row = cur.fetchone()
        max_date = row[0] if row else None
        if max_date is None:
            return []
        cur.execute(_STATS_SQL)
        stats = cur.fetchall()

    out = []
    for ticker, n, first, last in stats:
        lag = (max_date - last).days
        if ticker.upper() in tracked:
            continue                      # бумага в сопровождении — не трогаем
        if lag < stale_days:
            continue                      # отстала недостаточно, чтобы считать мёртвой
        out.append({"ticker": ticker, "rows": n, "first": first,
                    "last": last, "lag_days": lag})
    return out


def describe(conn, tickers: list[str]) -> list[dict]:
    """Статистика по явно заданным тикерам (даже если они в сопровождении)."""
    if not tickers:
        return []
    with conn.cursor() as cur:
        cur.execute(_MAX_DATE_SQL)
        row = cur.fetchone()
        max_date = row[0] if row else None
        cur.execute("""SELECT ticker, count(*), min(date), max(date)
                       FROM market_data WHERE ticker = ANY(%s) GROUP BY ticker
                       ORDER BY ticker;""", ([t.upper() for t in tickers],))
        rows = cur.fetchall()
    return [{"ticker": t, "rows": n, "first": f, "last": l,
             "lag_days": (max_date - l).days if max_date else None}
            for t, n, f, l in rows]


def _count_5m(conn, ticker: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM market_data_5m WHERE ticker = %s;", (ticker,))
        return int(cur.fetchone()[0])


def archive_ticker(conn, ticker: str, reason: str) -> tuple[int, int]:
    """Переносит дневные и 5-минутные бары тикера в архивные таблицы."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO market_data_archive
            SELECT m.*, NOW(), %s FROM market_data m WHERE m.ticker = %s;
        """, (reason, ticker))
        n_daily = cur.rowcount
        cur.execute("DELETE FROM market_data WHERE ticker = %s;", (ticker,))

        cur.execute("""
            INSERT INTO market_data_5m_archive
            SELECT m.*, NOW(), %s FROM market_data_5m m WHERE m.ticker = %s;
        """, (reason, ticker))
        n_5m = cur.rowcount
        cur.execute("DELETE FROM market_data_5m WHERE ticker = %s;", (ticker,))
    return n_daily, n_5m


def purge_ticker(conn, ticker: str) -> tuple[int, int]:
    """Удаляет бары тикера безвозвратно (без архива)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM market_data WHERE ticker = %s;", (ticker,))
        n_daily = cur.rowcount
        cur.execute("DELETE FROM market_data_5m WHERE ticker = %s;", (ticker,))
        n_5m = cur.rowcount
    return n_daily, n_5m


def restore_ticker(conn, ticker: str) -> tuple[int, int]:
    """Возвращает тикер из архива обратно в рабочие таблицы.

    Колонки перечислены явно: id не переносим (его выдаст sequence), archived_at
    и reason — служебные поля архива. Каст строки целиком (a.*)::market_data
    здесь не работает — типы таблиц разные."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO market_data
                (ticker, date, open, high, low, close, volume, source, created_at)
            SELECT ticker, date, open, high, low, close, volume, source, created_at
            FROM market_data_archive WHERE ticker = %s
            ON CONFLICT (ticker, date) DO NOTHING;
        """, (ticker,))
        n_daily = cur.rowcount
        cur.execute("DELETE FROM market_data_archive WHERE ticker = %s;", (ticker,))
        cur.execute("""
            INSERT INTO market_data_5m
                (ticker, ts, open, high, low, close, volume, created_at)
            SELECT ticker, ts, open, high, low, close, volume, created_at
            FROM market_data_5m_archive WHERE ticker = %s
            ON CONFLICT (ticker, ts) DO NOTHING;
        """, (ticker,))
        n_5m = cur.rowcount
        cur.execute("DELETE FROM market_data_5m_archive WHERE ticker = %s;", (ticker,))
    return n_daily, n_5m


def _print_table(rows: list[dict], conn) -> None:
    print(f"\n  {'Ticker':<8}{'1D баров':>10}{'5M баров':>10}  {'Первый':<12}{'Последний':<12}{'Отставание':>11}")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {r['ticker']:<8}{r['rows']:>10}{_count_5m(conn, r['ticker']):>10}  "
              f"{str(r['first']):<12}{str(r['last']):<12}"
              f"{(str(r['lag_days']) + ' дн.') if r['lag_days'] is not None else '—':>11}")
    print("  " + "-" * 66)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s %(message)s")
    p = argparse.ArgumentParser(
        description="Архивация/удаление мёртвых тикеров из market_data.")
    p.add_argument("--ticker", action="append", default=[],
                   help="конкретный тикер (можно повторять). По умолчанию — "
                        "автопоиск тикеров вне сопровождения с устаревшими барами.")
    p.add_argument("--stale-days", type=int, default=30,
                   help="порог отставания последнего бара, дней (по умолч. 30).")
    p.add_argument("--dry-run", action="store_true",
                   help="только показать, что было бы сделано.")
    p.add_argument("--purge", action="store_true",
                   help="удалить безвозвратно вместо переноса в архив.")
    p.add_argument("--restore", action="append", default=[],
                   help="вернуть тикер из архива в рабочие таблицы.")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждение.")
    args = p.parse_args(argv)

    with database.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_ARCHIVE_SQL)

        # ── восстановление из архива ──
        if args.restore:
            for tk in args.restore:
                n_d, n_5 = restore_ticker(conn, tk.upper())
                print(f"  {tk.upper()}: возвращено из архива 1D={n_d}, 5M={n_5}")
            return 0

        rows = (describe(conn, args.ticker) if args.ticker
                else find_stale(conn, args.stale_days))

        if not rows:
            print("Мёртвых тикеров не найдено — чистить нечего.")
            return 0

        action = "УДАЛЕНЫ БЕЗВОЗВРАТНО" if args.purge else "перенесены в архив"
        print(f"\nКандидаты на очистку (будут {action}):")
        _print_table(rows, conn)

        if args.dry_run:
            print("\n[DRY-RUN] ничего не изменено.")
            return 0

        if not args.yes:
            ans = input(f"\n{action.capitalize()}: {', '.join(r['ticker'] for r in rows)}. Продолжить? (y/n): ")
            if ans.strip().lower() not in ("y", "yes", "д", "да"):
                print("[INFO] Отменено пользователем.")
                return 0

        reason = f"stale ticker: последний бар отстаёт >= {args.stale_days} дн."
        total_d = total_5 = 0
        for r in rows:
            tk = r["ticker"]
            if args.purge:
                n_d, n_5 = purge_ticker(conn, tk)
                log.info("%s: удалено 1D=%d, 5M=%d", tk, n_d, n_5)
            else:
                n_d, n_5 = archive_ticker(conn, tk, reason)
                log.info("%s: в архив 1D=%d, 5M=%d", tk, n_d, n_5)
            total_d += n_d
            total_5 += n_5

        print(f"\nГотово: обработано тикеров {len(rows)}, строк 1D={total_d}, 5M={total_5}.")
        if not args.purge:
            print("Вернуть: python3 -m scripts.cleanup_stale_tickers --restore <TICKER>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
