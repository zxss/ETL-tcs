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
    except Exception as e:
        log.exception("Критическая ошибка: %s", e)
        sys.exit(1)
    finally:
        conn.close()

    log.info("ETL завершён.")


if __name__ == "__main__":
    main()
