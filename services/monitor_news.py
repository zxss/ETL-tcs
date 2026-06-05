from __future__ import annotations
"""
Сервис мониторинга новостей @markettwits.

Логика:
  1. Читаем MAX(message_id) из news_sentiment для данного источника.
  2. Загружаем новые сообщения из Telegram (min_id = last+1).
  3. Парсим батчами через regex + Claude haiku.
  4. Сохраняем в news_sentiment.

Запуск разовый:
    python3 run_monitor.py

Запуск в режиме демона (каждые N минут):
    python3 run_monitor.py --loop --interval 15

ВРЕМЕННО ОТКЛЮЧЕНО: блоки Telegram закомментированы до настройки авторизации.
Пометка # TG_ENABLED — что нужно раскомментировать для включения.
"""

import asyncio
import logging
from datetime import datetime, timezone

# import anthropic                              # TG_ENABLED

import config
import database
# from monitors.tg_reader import fetch_messages  # TG_ENABLED
# from monitors.news_parser import parse_messages  # TG_ENABLED

log = logging.getLogger("monitor_news")


async def run_once_async(conn) -> int:
    """
    Один цикл мониторинга.
    Возвращает количество новых записей в БД.

    TG_ENABLED: раскомментировать весь блок внутри функции.
    """
    log.warning("Telegram-мониторинг отключён. Раскомментируйте TG_ENABLED-блоки.")
    return 0

    # ── TG_ENABLED: раскомментировать ниже ───────────────────────────────────
    #
    # # 1. Последний обработанный message_id
    # last_id = await asyncio.to_thread(
    #     database.get_last_message_id, conn, config.TG_CHANNEL
    # )
    # log.info("Последний обработанный message_id: %s", last_id or "нет (первый запуск)")
    #
    # # 2. Загрузка новых сообщений из Telegram
    # messages = await fetch_messages(
    #     min_id=last_id or 0,
    #     limit=config.TG_FETCH_LIMIT,
    # )
    # if not messages:
    #     log.info("Новых сообщений нет")
    #     return 0
    #
    # log.info("Получено %d новых сообщений", len(messages))
    #
    # # 3. Парсинг
    # anthropic_client = None
    # if config.ANTHROPIC_API_KEY:
    #     anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    #
    # records = parse_messages(messages, anthropic_client)
    # if not records:
    #     log.info("Тикеров из вотчлиста не найдено")
    #     return 0
    #
    # # 4. Сохранение в БД
    # rows = [
    #     (
    #         r["message_id"],
    #         r["ts"],
    #         r["ticker"],
    #         r["sentiment"],
    #         r["headline"],
    #         r["raw_text"],
    #         config.TG_CHANNEL,
    #     )
    #     for r in records
    # ]
    # saved = await asyncio.to_thread(database.upsert_news, conn, rows)
    # log.info("Сохранено записей: %d", saved)
    # return saved


def run_once(conn) -> int:
    """Синхронная обёртка."""
    return asyncio.run(run_once_async(conn))


async def run_loop_async(conn, interval_sec: int = 900) -> None:
    """Бесконечный цикл мониторинга с паузой interval_sec."""
    log.info("Мониторинг запущен (интервал %d сек)", interval_sec)
    while True:
        try:
            saved = await run_once_async(conn)
            log.info("Цикл завершён, новых записей: %d", saved)
        except Exception as e:
            log.exception("Ошибка в цикле мониторинга: %s", e)
        await asyncio.sleep(interval_sec)


def run_loop(conn, interval_sec: int = 900) -> None:
    asyncio.run(run_loop_async(conn, interval_sec))
