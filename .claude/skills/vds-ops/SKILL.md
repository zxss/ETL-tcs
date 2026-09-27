---
name: vds-ops
description: Работа с сервером ETL-tcs (VDS; адрес — в git remote vds) — запуск команд от etl с нужным профилем окружения, проверка состояния, выкладка, расписания, бэкапы перед правками. Использовать при любом действии на сервере, при диагностике контура, выкладке и проверке счетов.
---

# Сервер ETL-tcs

> Адрес сервера не хранится в репозитории (он публичный): `VDS=$(git remote get-url vds | sed 's/.*@//; s/:.*//')`.

## Доступ
- `ssh root@$VDS`. «Permission denied (publickey)» после перезагрузки Mac →
  `ssh-add --apple-load-keychain` (ключ с паролем в связке ключей).
- Приложение работает от `etl`. PostgreSQL — peer-аутентификация: БД читать только
  `sudo -u etl …` или `sudo -u postgres psql -d market_data`.
- Профили: песочница r4 — `.env`; боевой «Робот» — `set -a && . .env.tournament && set +a`
  поверх `.env`. Шаблон:
  `ssh root@$VDS "sudo -u etl bash -c 'cd /opt/etl-tcs && [профиль] && .venv/bin/python -m <модуль>'"`
- Сложный Python на сервер — файлом через `scp` в `/tmp`, не строкой в кавычках
  (экранирование ломается); после запуска файл удалить.

## Раскладка
| что | где |
|---|---|
| живой код | `/opt/etl-tcs` (work-tree), bare `/opt/etl-tcs.git`, venv `/opt/etl-tcs/.venv` (Python 3.14) |
| ветка выкладки | `/opt/etl-tcs.git/DEPLOY_BRANCH` |
| исследования | `/opt/etl-r4` (ветка r4, `git archive`), `/opt/research-vrp` + `/opt/research-data` |
| расписания | `sudo -u etl crontab -l`, `/etc/cron.d/etl-*`, systemd `etl-pipeline.timer` (main.py 19:30) |
| бэкапы правок | `/root/*.backup-YYYYMMDD-<суть>` |

## Правила
1. Сначала только чтение: статус, логи, холостой прогон. Изменение — после.
2. Перед правкой `.env`, `.env.tournament`, crontab, хука — бэкап в `/root/…backup-<дата>-<суть>`.
3. Боевые заявки не отправлять. `services.stage2_demo protect` холостого режима НЕ имеет.
   Состояние боевого счёта — read-only скриптом (позиции, `get_active_stop_orders`, операции).
4. Длинные задачи — `setsid nohup … > log 2>&1 < /dev/null &`; обрыв SSH их не убивает.
   Проверять через лог и число строк результата: `pgrep -f` находит и сам SSH-запрос.
5. Выкладка — поток `flows/deploy/КАК_ЗАПУСТИТЬ.md`.

## Известные ловушки API
- `GetOrderState` боевого API по вчерашней заявке → 404: факт брать из операций.
- В песочнице `get_money_rub` — только СВОБОДНЫЕ рубли; капитал = money + blocked.
- Клиент песочницы в `get_positions` раньше не видел фьючерсы; общий парсер —
  `tinkoff_base.parse_positions`.
- Выходные: MOEX не торгует, статус `SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING` —
  биржевые рыночные заявки не проходят (`scripts.flatten_account --dealer`).
- Дневная свеча брокера — весь день (open 06:55, close 23:45); модель — на
  `market_data_session` (основная сессия).
