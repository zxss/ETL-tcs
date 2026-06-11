"""
Шаг расчёта: после загрузки данных выгружает дневные свечи из PostgreSQL
в CSV и прогоняет статистический контур валидации стратегий
(validation/verdict.py из скила strategy-edge-validator).

Контур отвечает не на вопрос «сколько заработала стратегия», а на вопрос
«отличается ли результат от случайности» — White RC, SPA, PBO, BH-FDR,
Ljung-Box, Hurst, энтропия, Probability of Ruin → вердикт REJECTED /
WEAK / CANDIDATE EDGE.

Вызывается из main.py после run(conn). Управляется конфигом:
RUN_VALIDATION, VALIDATOR_DIR, VALIDATION_DATA_DIR, VALIDATION_TICKERS,
VALIDATION_STRATS, VALIDATION_BOOT.
"""

from __future__ import annotations

import csv
import logging
import os
import sys

import config

log = logging.getLogger("run_validation")

# Свечи дневные: validation/advanced_stats.load_daily ждёт колонки
# date, open, high, low, close, volume.
_EXPORT_SQL = """
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = %s
    ORDER BY date ASC;
"""

_HEADER = ["date", "open", "high", "low", "close", "volume"]


def export_ticker_csv(conn, ticker: str, out_dir: str) -> tuple[str, int]:
    """Выгружает дневные свечи тикера в {ticker}_candles.csv. Возвращает (path, n)."""
    with conn.cursor() as cur:
        cur.execute(_EXPORT_SQL, (ticker,))
        rows = cur.fetchall()

    path = os.path.join(out_dir, f"{ticker.lower()}_candles.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HEADER)
        for date, o, h, l, c, v in rows:
            w.writerow([str(date)[:10], float(o), float(h), float(l), float(c), int(v or 0)])
    return path, len(rows)


def run(conn, quiet: bool = False) -> list | None:
    """
    Выгружает выбранные тикеры из БД и запускает verdict.run().
    Возвращает список строк-результатов (по одной на ticker×strategy) либо
    None, если контур пропущен/недоступен.

    quiet=True — не печатать собственный отчёт verdict.print_report (строки
    всё равно возвращаются для сводной итоговой таблицы).
    """
    if not config.RUN_VALIDATION:
        log.info("RUN_VALIDATION=0 — шаг расчёта пропущен.")
        return None

    scripts_dir = config.VALIDATOR_DIR
    verdict_path = os.path.join(scripts_dir, "validation", "verdict.py")
    if not os.path.isdir(scripts_dir) or not os.path.exists(verdict_path):
        log.warning(
            "Валидатор не найден по пути VALIDATOR_DIR=%s — шаг расчёта пропущен. "
            "Укажите корректный путь через переменную окружения VALIDATOR_DIR.",
            scripts_dir,
        )
        return None

    out_dir = config.VALIDATION_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 1. Выгрузка свечей из БД в CSV
    exported = []
    for tk in config.VALIDATION_TICKERS:
        try:
            path, n = export_ticker_csv(conn, tk, out_dir)
        except Exception as e:  # noqa: BLE001 — не валим ETL из-за одного тикера
            log.warning("  [%s] ошибка выгрузки: %s", tk, e)
            continue
        if n == 0:
            log.warning("  [%s] нет данных в БД — пропуск.", tk)
            continue
        exported.append(tk)
        log.info("  [%s] выгружено %d дневных свечей → %s", tk, n, os.path.basename(path))

    if not exported:
        log.warning("Нет данных для валидации — шаг расчёта пропущен.")
        return None

    # 2. Запуск статистического контура из скила
    sys.path.insert(0, scripts_dir)
    try:
        from validation import verdict  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось импортировать validation.verdict: %s", e)
        return None

    log.info(
        "Запуск контура валидации: %d тикеров × %d стратегий (B=%d)...",
        len(exported), len(config.VALIDATION_STRATS), config.VALIDATION_BOOT,
    )
    rows = verdict.run(
        [t.lower() for t in exported],
        config.VALIDATION_STRATS,
        out_dir,
        B=config.VALIDATION_BOOT,
    )
    if not quiet:
        verdict.print_report(rows)
    return rows
