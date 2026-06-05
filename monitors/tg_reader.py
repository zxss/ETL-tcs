"""
Асинхронный читатель Telegram-канала @markettwits через Telethon.

Возвращает список сообщений начиная с min_message_id+1.
Использует файловую сессию (.session) для хранения авторизации.

Первый запуск:
    python3 -c "from monitors.tg_reader import auth; import asyncio; asyncio.run(auth())"
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Message

import config

log = logging.getLogger("tg_reader")

SESSION_FILE = "tg_session"   # .session файл в рабочей директории


def _make_client() -> TelegramClient:
    return TelegramClient(
        SESSION_FILE,
        config.TG_API_ID,
        config.TG_API_HASH,
    )


async def auth() -> None:
    """
    Интерактивная авторизация (нужна один раз).
    Запустить вручную: python3 -c "from monitors.tg_reader import auth; import asyncio; asyncio.run(auth())"
    """
    async with _make_client() as client:
        await client.start(phone=config.TG_PHONE)
        me = await client.get_me()
        log.info("Авторизован как: %s", me.username or me.first_name)


async def fetch_messages(
    min_id: int = 0,
    batch_size: int = 100,
    limit: int = 500,
) -> list[dict]:
    """
    Загружает сообщения из TG_CHANNEL после min_id.

    Возвращает список словарей:
        {message_id, ts (UTC datetime), text}
    Отсортированы по возрастанию message_id.
    """
    messages: list[dict] = []

    async with _make_client() as client:
        try:
            channel = await client.get_entity(config.TG_CHANNEL)
        except Exception as e:
            raise RuntimeError(f"Не удалось получить канал {config.TG_CHANNEL}: {e}") from e

        fetched = 0
        async for msg in client.iter_messages(
            channel,
            min_id=min_id,
            limit=limit,
            reverse=True,   # от старых к новым
        ):
            if not isinstance(msg, Message) or not msg.text:
                continue

            ts = msg.date
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)

            messages.append({
                "message_id": msg.id,
                "ts":         ts,
                "text":       msg.text,
            })

            fetched += 1
            if fetched >= limit:
                break

    log.info("Загружено %d сообщений из %s (после id=%d)",
             len(messages), config.TG_CHANNEL, min_id)
    return messages
