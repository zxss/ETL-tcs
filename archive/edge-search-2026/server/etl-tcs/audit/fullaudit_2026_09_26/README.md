# Комплексный аудит 26.09.2026 — скрипты воспроизведения

Отчёт: `~/Desktop/трейдинг исследование/AUDIT-FULL-2026-09-26.md`.

Все скрипты только читают боевую БД и файлы фаз; ничего не пишут. Запуск на VDS
от пользователя `etl` (peer-аутентификация PostgreSQL):

    scp audit/fullaudit_2026_09_26/*.py root@<VDS>:/tmp/
    ssh root@<VDS> 'cd /opt/etl-tcs && sudo -u etl .venv/bin/python /tmp/audit_mech.py'

| скрипт | что проверяет | раздел отчёта |
|---|---|---|
| `audit_mech.py` | ночной рейтинг против уже случившегося утреннего гэпа; где стояли реально купленные бумаги | F1 |
| `audit_window.py` | цель модели (23:45→06:55) против торгуемого окна (18:45→09:10) | F2 |
| `audit_fwd.py` | ex-ante Rank IC, калибровка ExpPnL/ProbProfit, покрытие коридора по журналу `forecasts` | F3 |
| `audit_data_exec.py` | качество дневных свечей; журнал исполнения, стопы, выходы | F4, F7, данные |

Разложение баланса (F6) — существующим `audit/stage2_decompose.py` с ветки r4:

    ssh root@<VDS> 'cd /opt/etl-r4 && sudo -u etl /opt/etl-tcs/.venv/bin/python -m audit.stage2_decompose --dir /opt/etl-tcs/audit/stage2-demo'

Результаты зависят от даты запуска: журнал `forecasts` и фазы пополняются
каждый день. Цифры отчёта — на 26.09.2026.
