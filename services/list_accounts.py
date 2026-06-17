"""
services/list_accounts.py — список брокерских счетов в T-Invest API.

Эндпоинт: UsersService/GetAccounts (REST, тот же, что и для свечей).
Авторизация: bearer-токен config.INVEST_TOKEN (env INVEST_TOKEN).
Без внешних зависимостей — только stdlib (urllib).

Запуск:
    python3 -m services.list_accounts          # человеко-читаемая таблица
    python3 -m services.list_accounts --json   # сырой JSON (для скриптов)
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

import config

# Расшифровки enum-значений T-Invest для читаемого вывода.
_TYPE = {
    "ACCOUNT_TYPE_TINKOFF":     "Брокерский",
    "ACCOUNT_TYPE_TINKOFF_IIS": "ИИС",
    "ACCOUNT_TYPE_INVEST_BOX":  "Инвесткопилка",
    "ACCOUNT_TYPE_UNSPECIFIED": "—",
}
_STATUS = {
    "ACCOUNT_STATUS_NEW":         "Новый",
    "ACCOUNT_STATUS_OPEN":        "Открыт",
    "ACCOUNT_STATUS_CLOSED":      "Закрыт",
    "ACCOUNT_STATUS_UNSPECIFIED": "—",
}
_ACCESS = {
    "ACCOUNT_ACCESS_LEVEL_FULL_ACCESS": "FULL",
    "ACCOUNT_ACCESS_LEVEL_READ_ONLY":   "READ-ONLY",
    "ACCOUNT_ACCESS_LEVEL_NO_ACCESS":   "NO-ACCESS",
    "ACCOUNT_ACCESS_LEVEL_UNSPECIFIED": "—",
}


def _ssl_context(verify: bool) -> ssl.SSLContext:
    """SSL-контекст. По умолчанию проверка ВЫКЛЮЧЕНА — основной ETL проекта
    уже так работает (см. aiohttp.TCPConnector(ssl=False) в loaders),
    значит на машине MITM-перехват TLS (корп. прокси / антивирус). Можно
    вернуть проверку через INVEST_TLS_VERIFY=1 или флаг --verify."""
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def get_accounts(verify_tls: bool = False) -> dict:
    """POST {} к UsersService/GetAccounts. Возвращает распарсенный JSON."""
    url = f"{config.API_BASE_URL}/{config.API_SERVICE}.UsersService/GetAccounts"
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {config.INVEST_TOKEN}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15,
                                    context=_ssl_context(verify_tls)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}\n{body}") from e


def print_table(accounts: list[dict]) -> None:
    if not accounts:
        print("Счетов нет (или у токена нет прав).")
        return

    print("\n  СЧЕТА T-INVEST")
    print("  " + "─" * 96)
    print(f"  {'#':>2}  {'ID':<22}  {'Тип':<14}  {'Статус':<8}"
          f"  {'Доступ':<10}  {'Открыт':<10}  Имя")
    print("  " + "─" * 96)
    for i, a in enumerate(accounts, 1):
        opened = (a.get("openedDate") or "")[:10] or "—"
        print(f"  {i:>2}  {a.get('id', '—'):<22}  "
              f"{_TYPE.get(a.get('type'), a.get('type', '—')):<14}  "
              f"{_STATUS.get(a.get('status'), a.get('status', '—')):<8}  "
              f"{_ACCESS.get(a.get('accessLevel'), '—'):<10}  "
              f"{opened:<10}  {a.get('name') or '—'}")
    print("  " + "─" * 96)
    print(f"  Всего счетов: {len(accounts)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Список брокерских счетов T-Invest")
    p.add_argument("--json", action="store_true", help="вывести сырой JSON-ответ")
    p.add_argument("--verify", action="store_true",
                   help="включить проверку TLS-сертификата (по умолчанию выкл,"
                        " как в основном ETL — из-за MITM-перехвата TLS)")
    args = p.parse_args()

    # Глобальный фолбэк включения проверки: INVEST_TLS_VERIFY=1
    verify = args.verify or os.getenv("INVEST_TLS_VERIFY", "0") not in ("0", "false", "False")

    try:
        data = get_accounts(verify_tls=verify)
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    print_table(data.get("accounts", []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
