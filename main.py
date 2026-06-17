#!/usr/bin/env python3
"""
Точка входа ETL-приложения.

Запуск:
    python3 main.py

Переменные окружения (или .env):
    INVEST_TOKEN  — токен T-Инвестиций
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD — параметры PostgreSQL

Логика:
  - Проверяет наличие данных в БД для каждого тикера.
  - Скачивает только новые данные (инкрементальный режим).
  - Все тикеры загружаются параллельно (asyncio + aiohttp).
"""

import logging
import sys

import database
from services.load_history import run
from services import run_validation
from services import backfill_daily
from services import load_index
import config
import tft_forecast


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("main")

    log.info("Подключение к PostgreSQL...")
    try:
        conn = database.get_connection()
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)

    try:
        database.init_db(conn)
        run(conn)

        # Индексы (IMOEX) — отдельный шаг для слоя рыночного контекста.
        log.info("Загрузка биржевых индексов (IMOEX)...")
        try:
            load_index.run(conn)
        except Exception as e:  # noqa: BLE001 — индекс не должен валить ETL
            log.warning("Шаг загрузки индексов завершился с ошибкой: %s", e)

        # Достраиваем дневные свечи из 5M, если официальный дневной бар ещё
        # не опубликован брокером (вечер/ночь). Самокорректируется позже.
        try:
            backfill_daily.backfill(conn, config.TICKERS)
        except Exception as e:  # noqa: BLE001 — не валим конвейер
            log.warning("Backfill дневных из 5M завершился с ошибкой: %s", e)

        combined = getattr(config, "COMBINED_TABLE", True)

        log.info("Данные загружены — запуск статистической валидации стратегий...")
        val_rows = None
        try:
            val_rows = run_validation.run(conn, quiet=combined)
        except Exception as e:  # noqa: BLE001 — расчёт не должен валить ETL
            log.warning("Шаг валидации завершился с ошибкой: %s", e)

        log.info("Запуск Multi-Asset TFT — прогноз диапазона на следующий день...")
        forecasts = None
        try:
            forecasts = tft_forecast.run(conn, quiet=combined)
        except Exception as e:  # noqa: BLE001 — прогноз не должен валить ETL
            log.warning("Шаг TFT-прогноза завершился с ошибкой: %s", e)

        if combined:
            try:
                # Вселенная дашборда — все бумаги с прогнозом (а не только
                # VALIDATION_TICKERS), чтобы вывести топ-N по всему рынку.
                fc = forecasts or {}
                universe = [k for k in fc if k != "__meta__"] or config.VALIDATION_TICKERS
                tft_forecast.print_combined(
                    val_rows, forecasts,
                    universe, config.VALIDATION_STRATS,
                    show_all=getattr(config, "SHOW_ALL_INTRADAY", False),
                    top_n=getattr(config, "DASHBOARD_TOP_N", 30),
                )
            except Exception as e:  # noqa: BLE001 — печать не должна валить ETL
                log.warning("Шаг сводной таблицы завершился с ошибкой: %s", e)
    except Exception as e:
        log.exception("Критическая ошибка: %s", e)
        sys.exit(1)
    finally:
        conn.close()

    log.info("ETL + валидация завершены.")


if __name__ == "__main__":
    main()
