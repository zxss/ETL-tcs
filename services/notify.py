"""
Уведомления в Telegram: транспорт и ничего больше.

Формированием текста занимается вызывающий код (services/stage2_demo.py) —
здесь только отправка, экранирование и правило «уведомление не имеет права
уронить торговлю».

Это правило — главное в модуле. Сбой сети, отозванный токен, заблокированный
ботом чат, таймаут Telegram: ни один из этих случаев не должен превратиться в
исключение, поднимающееся в фазу. Поэтому send() возвращает bool и никогда не
бросает, а любая ошибка уходит в лог как warning.

Секреты: TELEGRAM_BOT_TOKEN живёт только в .env (chmod 600, вне git). Токен
никогда не пишется в лог — при ошибке из текста вырезается подстрокой.

Настройка:
    TELEGRAM_ENABLED=1
    TELEGRAM_BOT_TOKEN=<id>:<секрет>
    TELEGRAM_CHAT_ID=<id получателя>

Проверка:
    python3 -m services.notify "проверка связи"
"""
from __future__ import annotations

import html
import json
import logging
import ssl
import sys
import time
import urllib.error
import urllib.request

import config

log = logging.getLogger("notify")

_API = "https://api.telegram.org"
_TIMEOUT = 10
_RETRIES = 2          # первая попытка + одна повторная
_MAX_LEN = 4096       # жёсткий лимит Telegram на sendMessage


def _token() -> str:
    return str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "")


def _chat_id() -> str:
    return str(getattr(config, "TELEGRAM_CHAT_ID", "") or "")


def enabled() -> bool:
    """Уведомления включены и настроены полностью."""
    return bool(getattr(config, "TELEGRAM_ENABLED", False)) and bool(_token()) and bool(_chat_id())


def esc(value) -> str:
    """Экранирует значение для parse_mode=HTML.

    Telegram упадёт с 400 на неэкранированном '<' в тексте, а тикеры и тексты
    ошибок приходят из внешних источников — экранируем всё, что подставляем.
    """
    return html.escape("—" if value is None else str(value), quote=False)


def _scrub(text: str) -> str:
    """Убирает токен из текста перед записью в лог."""
    tok = _token()
    return text.replace(tok, "<токен скрыт>") if tok else text


def _truncate(text: str) -> str:
    if len(text) <= _MAX_LEN:
        return text
    tail = "\n… сообщение обрезано"
    return text[: _MAX_LEN - len(tail)] + tail


def send(text: str, *, parse_mode: str | None = "HTML", silent: bool = False) -> bool:
    """Отправляет сообщение. Возвращает True при успехе, False при любой ошибке.

    Никогда не бросает исключений: см. докстринг модуля.
    """
    if not enabled():
        log.debug("notify: уведомления выключены или не настроены")
        return False

    payload = {
        "chat_id": _chat_id(),
        "text": _truncate(text),
        "disable_web_page_preview": True,
        "disable_notification": bool(silent),
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    # ensure_ascii=False: кириллица уходит как UTF-8, а не \uXXXX —
    # тело запроса вшестеро короче, Telegram принимает UTF-8 штатно.
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # Публичный УЦ: api.telegram.org не имеет отношения к брокерскому корню
    # из INVEST_CA_BUNDLE, поэтому системного хранилища достаточно.
    ctx = ssl.create_default_context()
    url = f"{_API}/bot{_token()}/sendMessage"

    for attempt in range(1, _RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as r:
                if json.loads(r.read().decode("utf-8")).get("ok"):
                    return True
                log.warning("notify: Telegram вернул ok=false")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("description", "")
            except Exception:  # noqa: BLE001
                pass
            log.warning("notify: HTTP %s %s", e.code, _scrub(detail))
            if e.code in (400, 401, 403):
                return False      # неверный токен/чат — повтор не поможет
        except Exception as e:  # noqa: BLE001 — уведомление не роняет торговлю
            log.warning("notify: попытка %d не удалась: %s", attempt, _scrub(str(e)))
        if attempt < _RETRIES:
            time.sleep(2)
    return False


def _main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not enabled():
        print("Уведомления не настроены: нужны TELEGRAM_ENABLED=1, "
              "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в .env")
        return 1
    text = " ".join(argv) or "проверка связи"
    ok = send(f"<b>ETL-tcs</b>\n{esc(text)}")
    print("отправлено" if ok else "не отправлено — см. лог выше")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
