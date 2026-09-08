#!/usr/bin/env python3
"""
Точка входа мониторинга новостей @markettwits.

Использование:
    # Разовый запуск (загрузить всё новое):
    python3 -m contrib.experimental_news.run_monitor

    # Демон (каждые 15 мин):
    python3 -m contrib.experimental_news.run_monitor --loop --interval 15

    # Первичная авторизация в Telegram (один раз):
    python3 -m contrib.experimental_news.run_monitor --auth

Переменные окружения (в .env):
    TG_API_ID       — с https://my.telegram.org
    TG_API_HASH     — с https://my.telegram.org
    TG_PHONE        — номер телефона в формате +79001234567
    ANTHROPIC_API_KEY — ключ Claude API (опционально; без него только regex)

ВРЕМЕННО ОТКЛЮЧЕНО: запросы к Telegram закомментированы до настройки авторизации.
Раскомментировать блоки с пометкой # TG_ENABLED для включения.
"""

import argparse
# import asyncio          # TG_ENABLED
import logging
import os
import sys

# запуск из корня проекта: python3 -m contrib.experimental_news.run_monitor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contrib.experimental_news import news_db
# from contrib.experimental_news.monitor_news import run_once, run_loop  # TG_ENABLED


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

    # ── Авторизация Telegram ──────────────────────────────────────────────────
    # TG_ENABLED: раскомментировать после настройки TG_API_ID / TG_HASH / TG_PHONE
    #
    # if args.auth:
    #     from contrib.experimental_news.monitors.tg_reader import auth
    #     asyncio.run(auth())
    #     log.info("Авторизация завершена. Теперь запустите без --auth.")
    #     return

    if args.auth:
        log.warning(
            "Telegram-мониторинг отключён. "
            "Заполните TG_API_ID, TG_API_HASH, TG_PHONE в .env и "
            "раскомментируйте TG_ENABLED-блоки в contrib/experimental_news/run_monitor.py"
        )
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

        # ── Запуск мониторинга ────────────────────────────────────────────────
        # TG_ENABLED: раскомментировать оба блока ниже
        #
        # if args.loop:
        #     run_loop(conn, interval_sec=args.interval * 60)
        # else:
        #     saved = run_once(conn)
        #     log.info("Готово. Новых записей: %d", saved)

        log.warning(
            "Telegram-мониторинг отключён. "
            "Раскомментируйте TG_ENABLED-блоки в contrib/experimental_news/run_monitor.py."
        )

    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    except Exception as e:
        log.exception("Критическая ошибка: %s", e)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
