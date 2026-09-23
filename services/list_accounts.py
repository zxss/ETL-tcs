"""
services/list_accounts.py — список брокерских счетов в T-Invest API.

Эндпоинт: UsersService/GetAccounts (REST, тот же, что и для свечей).
Авторизация: bearer-токен из env INVEST_TOKEN (config.require_invest_token()).
Без внешних зависимостей — только stdlib (urllib).

Запуск:
    python3 -m services.list_accounts          # человеко-читаемая таблица
    python3 -m services.list_accounts --json   # сырой JSON (для скриптов)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

import config
import tls

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


def get_accounts(verify_tls: bool | None = None) -> dict:
    """POST {} к UsersService/GetAccounts. Возвращает распарсенный JSON."""
    url = f"{config.API_BASE_URL}/{config.API_SERVICE}.UsersService/GetAccounts"
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {config.require_invest_token()}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15,
                                    context=tls.ssl_context(verify_tls)) as resp:
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
    p.add_argument("--no-verify", action="store_true",
                   help="отключить проверку TLS-сертификата (эквивалент "
                        "INVEST_TLS_VERIFY=0; по умолчанию проверка ВКЛЮЧЕНА)")
    args = p.parse_args()

    # None → политика из config.INVEST_TLS_VERIFY (по умолчанию проверка вкл).
    verify = False if args.no_verify else None

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
