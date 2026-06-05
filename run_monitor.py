#!/usr/bin/env python3
"""
Точка входа мониторинга новостей @markettwits.

Использование:
    # Разовый запуск (загрузить всё новое):
    python3 run_monitor.py

    # Демон (каждые 15 мин):
    python3 run_monitor.py --loop --interval 15

    # Первичная авторизация в Telegram (один раз):
    python3 run_monitor.py --auth

Переменные окружения (в .env):
    TG_API_ID       — с https://my.telegram.org
    TG_API_HASH     — с https://my.telegram.org
    TG_PHONE        — номер телефона в формате +79001234567
    ANTHROPIC_API_KEY — ключ Claude API (опционально; без него только regex)
"""

import argparse
import asyncio
import logging
import sys

import database
from services.monitor_news import run_once, run_loop


def parse_args():
    p = argparse.ArgumentParser(description="Мониторинг новостей @markettwits → PostgreSQL")
    p.add_argument("--loop",     action="store_true", help="Запустить как демон")
    p.add_argument("--interval", type=int, default=15, help="Интервал в минутах (для --loop)")
    p.add_argument("--auth",     action="store_true", help="Авторизация в Telegram (первый раз)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("run_monitor")
    args = parse_args()

    # Интерактивная авторизация
    if args.auth:
        from monitors.tg_reader import auth
        asyncio.run(auth())
        log.info("Авторизация завершена. Теперь запустите без --auth.")
        return

    # Подключение к БД
    log.info("Подключение к PostgreSQL...")
    try:
        conn = database.get_connection()
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)

    try:
        database.init_db(conn)

        if args.loop:
            run_loop(conn, interval_sec=args.interval * 60)
        else:
            saved = run_once(conn)
            log.info("Готово. Новых записей: %d", saved)
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    except Exception as e:
        log.exception("Критическая ошибка: %s", e)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
