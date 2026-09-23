#!/usr/bin/env python3
"""
Одноразовая миграция: разметка исторического слоя market_data.source.

Колонка source появилась 2026-09-08. Все строки, существовавшие до этого,
получили значение по DEFAULT — 'api', то есть выглядят как официальные бары
биржи. На самом деле среди них могут быть бары, реконструированные из
5-минуток (services/backfill_daily.py) ещё до того, как появился трекинг:
задним числом их уже не различить.

Скрипт переводит такие строки в 'api_legacy' — честную метку «происхождение
неизвестно». Новые вставки из брокерского API продолжают писаться как 'api'.

Идемпотентен: повторный запуск не найдёт строк (у них уже 'api_legacy').

Запуск:
    python3 -m scripts.mark_legacy_source --dry-run
    python3 -m scripts.mark_legacy_source
    python3 -m scripts.mark_legacy_source --cutoff 2026-09-08 --yes
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database  # noqa: E402
from models.market_data import (  # noqa: E402
    MIGRATE_MARKET_DATA_SOURCE_LEGACY_SQL,
    SOURCE_API, SOURCE_API_LEGACY,
)

log = logging.getLogger("mark_legacy_source")

# Момент развёртывания трекинга источника. Важно: значение по умолчанию —
# ВРЕМЯ добавления колонки, а не полночь. Строки, вставленные в этот день, но
# ДО ALTER TABLE, тоже получили 'api' по DEFAULT и так же не отслеживались —
# отсечка по полуночи оставила бы их ошибочно помеченными как официальные.
# Полночь ('2026-09-08') можно задать явно через --cutoff.
DEFAULT_CUTOFF = "2026-09-08T22:00"

_PREVIEW_SQL = """
SELECT count(*), min(date), max(date), min(created_at), max(created_at)
FROM market_data
WHERE source = %(src)s AND created_at < %(cutoff)s;
"""

_DIST_SQL = "SELECT source, count(*) FROM market_data GROUP BY source ORDER BY source;"


MSK = dt.timezone(dt.timedelta(hours=3))


def _parse_cutoff(text: str):
    """Принимает дату или дату+время; возвращает aware datetime в MSK."""
    for parse in (dt.datetime.fromisoformat,):
        try:
            v = parse(text)
        except ValueError:
            continue
        return v if v.tzinfo else v.replace(tzinfo=MSK)
    try:
        d = dt.date.fromisoformat(text)
    except ValueError:
        return None
    return dt.datetime(d.year, d.month, d.day, tzinfo=MSK)


def _distribution(conn) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(_DIST_SQL)
        return [(r[0], int(r[1])) for r in cur.fetchall()]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Разметка исторических баров как api_legacy.")
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                   help=f"строки, созданные РАНЬШЕ этого момента (по умолч. "
                        f"{DEFAULT_CUTOFF}). Принимает YYYY-MM-DD или "
                        f"YYYY-MM-DDTHH:MM.")
    p.add_argument("--dry-run", action="store_true", help="только показать, ничего не менять.")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждение.")
    args = p.parse_args(argv)

    cutoff = _parse_cutoff(args.cutoff)
    if cutoff is None:
        print(f"Некорректный --cutoff: {args.cutoff!r} "
              "(нужен YYYY-MM-DD или YYYY-MM-DDTHH:MM)", file=sys.stderr)
        return 2

    params = {"src": SOURCE_API, "cutoff": cutoff}
    with database.get_db_connection() as conn:
        print("\nsource ДО миграции:")
        for src, n in _distribution(conn):
            print(f"  {src or '(NULL)':<14}{n:>8}")

        with conn.cursor() as cur:
            cur.execute(_PREVIEW_SQL, params)
            n, d_min, d_max, c_min, c_max = cur.fetchone()

        if not n:
            print(f"\nСтрок с source='{SOURCE_API}' и created_at < {cutoff} нет — "
                  "миграция уже выполнена.")
            return 0

        print(f"\nБудет размечено как '{SOURCE_API_LEGACY}': {n} строк")
        print(f"  даты баров : {d_min} … {d_max}")
        print(f"  created_at : {c_min} … {c_max}")

        if args.dry_run:
            print("\n[DRY-RUN] ничего не изменено.")
            return 0

        if not args.yes:
            ans = input(f"\nПеревести {n} строк в '{SOURCE_API_LEGACY}'? (y/n): ")
            if ans.strip().lower() not in ("y", "yes", "д", "да"):
                print("[INFO] Отменено пользователем.")
                return 0

        with conn.cursor() as cur:
            cur.execute(MIGRATE_MARKET_DATA_SOURCE_LEGACY_SQL, params)
            updated = cur.rowcount
        log.info("Размечено строк: %d", updated)

        print("\nsource ПОСЛЕ миграции:")
        for src, n2 in _distribution(conn):
            print(f"  {src or '(NULL)':<14}{n2:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
