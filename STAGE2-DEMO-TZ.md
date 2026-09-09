# ТЗ: Этап 2 — функциональный тест на демо-счёте, 15 торговых дней

**Проект:** etl-tcs · **Ветка:** `fix/audit-remediation` · **Дата:** 09.09.2026
**Идентификатор теста:** `stage2-demo-15d`

---

## 0. Что изменено относительно исходной постановки и почему

Пять изменений. Каждое вызвано устройством системы, а не предпочтением.

| # | Было в постановке | Стало | Причина |
|---|---|---|---|
| 1 | PREP 09:00 | **PREP 09:45** | Рынок открывается в 10:00. В 09:00 живой цены нет, `quotes.py` вернёт вчерашнее закрытие, и план будет построен на вчерашнем якоре. 09:45 — после утреннего аукциона, до открытия. |
| 2 | ORDER 10:00 | **ORDER 10:05** | 10:00 — первая минута непрерывных торгов, самый широкий спред дня. Моделирование исполнения показало медиану заливки на 6-м пятиминутном баре (≈30 мин после открытия), так что пять минут задержки ничего не стоят. |
| 3 | ORDER пересчитывает сигналы | **ORDER исполняет ПЛАН, сохранённый в PREP** | `compute_orders` заново обучает TFT и якорит прогноз на текущей цене (`TFT_USE_LIVE_PRICE=1`). В 09:45 и 10:05 сигналы будут разными. Без снапшота требование «соответствие заявки сигналу» невыполнимо. |
| 4 | CLEANUP «вечером», OVERNIGHT после | **CLEANUP 18:20, OVERNIGHT 18:35** | Основная сессия закрывается в 18:40, аукцион 18:40–18:50. При cleanup в 18:35 овернайт попадал бы в последние 3 минуты — худшая ликвидность дня. |
| 5 | Овернайт-заявки вечером | **Так же, но это НОВОЕ поведение** | Сейчас `long_overnight` выставляется утром вместе с интрадеем. Перенос на вечер соответствует определению стратегии (`next_open / today_close − 1` — вход на закрытии), но формально меняет момент входа. |

**Изменение №5 требует явного согласия:** оно меняет момент входа боевой стратегии. Логика отбора, скоринг и размер позиции не трогаются.

---

## 1. Цель и границы

**Цель:** проверить работоспособность исполнительного контура, сохранность данных, воспроизводимость, корректность перехода intraday → cleanup → overnight и отсутствие обращений к боевому API.

**Критерий успеха — функциональный.** Прибыльность, Sharpe, CAGR и Profit Factor **не являются** критериями этого теста. Причина: переаудит показал Deflated Sharpe 0,251 и концентрацию 119% прибыли в 5 днях из 151 — на пятнадцати днях любая оценка доходности статистически бессмысленна. Баланс отслеживается как **диагностика исполнения**, не как оценка стратегии.

**Вне объёма:** изменение торговой стратегии, алгоритма отбора, весов скоринга, размера позиции.

---

## 2. Расписание торгового дня

| Фаза | Время (МСК) | Команда | Обращается к брокеру |
|---|---|---|---|
| `PREP` | 09:45 | `stage2_demo prep` | нет (только чтение котировок) |
| `ORDER` | 10:05 | `stage2_demo order` | да — выставление лимиток |
| `CLEANUP` | 18:20 | `stage2_demo cleanup` | да — закрытие интрадея |
| `OVERNIGHT` | 18:35 | `stage2_demo overnight` | да — выставление овернайт-заявок |

Времена берутся из конфигурации (§9), а не из кода планировщика.

Основание для 18:20 / 18:35: `SESSION_END_MIN = 18:40` в [`tft_forecast/intraday.py:34`](tft_forecast/intraday.py), аукцион закрытия 18:40–18:50 (комментарий в [`services/backfill_daily.py:50`](services/backfill_daily.py)).

---

## 3. Механизм снапшота плана — ключевое требование

Между PREP и ORDER проходит 20 минут. За это время меняются якорная цена, реализованная часть дневного хода и результат переобучения модели. Поэтому:

**PREP** вычисляет план **один раз** и замораживает его на диске.
**ORDER** читает замороженный план и **ничего не пересчитывает**.

### Формат плана

`audit/stage2-demo/runs/<RUN_ID>/plan.json`:

```json
{
  "test_id": "stage2-demo-15d",
  "trading_day": "2026-09-10",
  "run_id": "20260910-094500-PREP",
  "created_at": "2026-09-10T09:45:12+03:00",
  "dataset": {
    "market_data_max_date": "2026-09-09",
    "market_data_rows": 32803,
    "market_data_5m_max_ts": "2026-09-09T18:45:00+03:00",
    "dataset_hash": "sha256:..."
  },
  "config_hash": "sha256:...",
  "model": {"backend": "torch", "epochs": 30, "tickers": 46},
  "orders": [
    {
      "signal_id": "20260910-SBER-intraday_short",
      "ticker": "SBER", "strategy_type": "intraday_short",
      "direction": "SHORT", "order_type": "LIMIT",
      "anchor_price": 312.45, "entry_price": 313.20,
      "stop_price": 318.10, "tp_price": 309.80,
      "quantity_lots": 32, "lot_size": 1, "total_rub": 10022.40,
      "exp_pnl_pct": 0.41, "final_score": 0.7312,
      "expected_cost_pct": 0.104, "verdict": "REJECTED"
    }
  ]
}
```

`dataset_hash` — SHA-256 от `(ticker, date, open, high, low, close, volume)` всех строк `market_data`, участвовавших в расчёте. `config_hash` — SHA-256 от отсортированного списка всех значимых параметров `config`.

### Правила исполнения плана

ORDER **обязан отказаться** от исполнения, если:
- `plan.json` отсутствует или относится к другому торговому дню;
- `config_hash` в плане не совпадает с текущим (конфигурацию поменяли между фазами);
- `dataset_hash` изменился, а `STAGE2_ALLOW_DATASET_DRIFT=0`.

Отказ фиксируется как `FAIL` в дневном аудите, заявки не выставляются.

---

## 4. Фаза PREP (09:45)

1. Создать `RUN_ID = <YYYYMMDD-HHMMSS>-PREP`, каталог запуска.
2. Проверить, что сегодня торговый день (`services/calendar.is_trading_day`). Если нет — выйти с кодом 0 и записать `skipped: not_a_trading_day`.
3. Догрузить данные и проверить свежесть (`ensure_fresh_data`). При `StaleDataError` — `FAIL`, план не создаётся.
4. Зафиксировать `dataset_hash`, сохранить `input_snapshot.json`.
5. Сохранить `config_snapshot.env` — все значения `config`, включая дефолты, не только `.env`.
6. Выполнить расчёт: `run_validation` → `tft_forecast.run` → `select_top_rows` → `build_orders`.
7. Сохранить `signals.json` (все строки дашборда) и `plan.json` (только исполнимые заявки).
8. Снять снимок счёта → `positions_before.json`, `balance.json`.
9. Прогнать предполётные проверки (§8) и записать результат в `preflight.json`.
10. **Ни одной заявки не выставлять.** Технически: путь PREP не вызывает `place_limits`/`sync_portfolio`.

---

## 5. Фаза ORDER (10:05)

1. `RUN_ID = <...>-ORDER`. Загрузить `plan.json` от PREP того же торгового дня.
2. Проверить `config_hash` и `dataset_hash` (§3).
3. Для **каждой** заявки перед отправкой проверить:
   - контур `SANDBOX` и `TRADING_MODE != prod`;
   - `strategy_type` входит в `TRADING_STRATEGIES`;
   - тикер не в `non_shortable_tickers()`, если направление SHORT;
   - `quantity_lots > 0` и `total_rub <= BEST_TRADES_POSITION_RUB`;
   - по инструменту нет открытой позиции и активной заявки (защита от задвоения);
   - `order_id` ещё не использован в этом торговом дне.
4. Выставить лимитки. **Только `intraday_*`** — овернайт откладывается до вечера (§7).
5. По каждой заявке записать в `execution_audit` через `record_intent()`.
6. Дождаться `ORDER_FILL_WAIT_SEC` (текущее значение 60 с; **не менять автоматически**), затем привязать SL/TP к залившимся.
7. Сохранить `orders.json`, `fills.json`, `positions_after.json`, `balance.json`.

---

## 6. Фаза CLEANUP (18:20)

1. `RUN_ID = <...>-CLEANUP`. Снять `positions_before.json` и `balance.json`.
2. Вызвать существующий `square_off_intraday()` — закрывает только `intraday_long`/`intraday_short`, `long_overnight` не трогает, снимает SL/TP до закрытия.
3. Дождаться подтверждения: повторный `GetPositions` должен вернуть ноль внутридневных позиций.
4. Снять оставшиеся дневные лимитки, не залившиеся за день.
5. Сохранить `positions_after.json`, `balance.json`, записать события в `execution_audit`.
6. **Acceptance-проверка:** после cleanup внутридневных позиций быть не должно. Невыполнение — критический `FAIL` дня.

---

## 7. Фаза OVERNIGHT (18:35)

1. `RUN_ID = <...>-OVERNIGHT`. Проверить, что CLEANUP этого дня завершился успешно; иначе — отказ.
2. Пересчитать сигналы **только для `long_overnight`** (актуальная цена закрытия — правильный якорь для входа через ночь).
3. Проверить ограничения: инструмент, размер, лимит, отсутствие дубля.
4. Выставить овернайт-заявки, пометив `strategy_type = "long_overnight"`.
5. Сохранить отдельным файлом `overnight_orders.json` — **не смешивать с `orders.json`** интрадея.
6. Записать в `execution_audit`.
7. Снять финальный снимок дня → `positions_after.json`, `balance.json`.
8. Сформировать дневной аудит (§10) и отчёт по балансу (§9).

Овернайт-позиции переходят на следующий торговый день и **не закрываются** фазой CLEANUP.

---

## 8. Предполётные и контрольные проверки

Выполняются каждый день, результат — в `daily_audit`.

**Критические** (невыполнение → `FAIL` дня, торговля останавливается):

- [ ] контур `SANDBOX`, `TRADING_MODE != prod`, `--prod` не передан
- [ ] обращений к боевому `API_BASE_URL` за день: 0
- [ ] `dataset_hash` сохранён и совпадает между PREP и ORDER
- [ ] `config_snapshot.env` сохранён, `config_hash` совпадает
- [ ] после CLEANUP внутридневных позиций нет
- [ ] предыдущие запуски не перезаписаны (каталог `RUN_ID` создавался как новый)

**Некритические** (→ `WARNING`):

- [ ] все четыре фазы отработали
- [ ] `signals.json`, `orders.json`, `fills.json`, `positions_*.json`, `execution_audit.json` сохранены
- [ ] овернайт-заявки сохранены отдельно
- [ ] нет дублирующихся `order_id`
- [ ] расхождение фактического проскальзывания с расчётным ≤ 20%

---

## 9. Отчёт по тестовому балансу

Баланс снимается **четыре раза в день** — на каждой фазе, что позволяет отделить внутридневной ход от овернайт-переноса.

### Источник данных

`services/account_status.snapshot()` → `OperationsService/GetPortfolio`. Поля (проверено на живом sandbox):

| Поле API | Смысл |
|---|---|
| `totalAmountPortfolio` | полная стоимость счёта |
| `totalAmountCurrencies` | свободные деньги |
| `totalAmountShares` | стоимость позиций в акциях |
| `expectedYield` | нереализованный P&L |

### Файл `balance.json` (в каждом каталоге запуска)

```json
{
  "run_id": "20260910-183500-OVERNIGHT",
  "phase": "OVERNIGHT",
  "captured_at": "2026-09-10T18:35:04+03:00",
  "account_id": "1639899c-...",
  "total_portfolio_rub": 99999.17,
  "free_cash_rub": 99999.17,
  "shares_value_rub": 0.0,
  "unrealised_pnl_rub": -0.0008,
  "open_positions": 0,
  "active_orders": 0,
  "active_stop_orders": 0
}
```

### Ежедневная сводка `audit/stage2-demo/balance/YYYY-MM-DD.json`

Баланс на всех четырёх фазах плюс дневные дельты:

```json
{
  "trading_day": "2026-09-10",
  "opening_balance_rub": 100000.00,
  "closing_balance_rub": 99871.40,
  "day_change_rub": -128.60,
  "day_change_pct": -0.129,
  "by_phase": {
    "PREP": 100000.00, "ORDER": 99994.10,
    "CLEANUP": 99880.20, "OVERNIGHT": 99871.40
  },
  "intraday_pnl_rub": -119.80,
  "overnight_carry_rub": -8.80,
  "commission_paid_rub": 12.40,
  "positions_overnight": 2,
  "positions_overnight_value_rub": 19842.00
}
```

### Отчёт `audit/stage2-demo/balance_report.md`

Пересобирается после каждой фазы OVERNIGHT. Формат:

```
ОТЧЁТ ПО ТЕСТОВОМУ БАЛАНСУ — stage2-demo-15d
Счёт: SANDBOX 1639899c-…   Стартовый баланс: 100 000,00 ₽
Торговых дней пройдено: 7 из 15

День         Открытие    PREP     ORDER   CLEANUP  OVERNIGHT   Δ день   Δ нараст.  Поз.
2026-09-10  100 000,00  100 000  99 994   99 880    99 871    −128,60    −0,13%     2
2026-09-11   99 871,40   99 871  99 902   99 940    99 933     +61,60    −0,07%     1
…
ИТОГО                                                          −67,00    −0,07%

РАЗЛОЖЕНИЕ
  Внутридневной P&L        −119,80 ₽
  Перенос через ночь         −8,80 ₽
  Комиссия                  −12,40 ₽
  Нереализованный P&L       +73,20 ₽

ИСПОЛНЕНИЕ
  Заявок выставлено                        43
  Исполнено                                26   (60,5%)
  Среднее проскальзывание, факт          0,031%
  Среднее проскальзывание, расчёт        0,028%
  Расхождение                             +11%   в пределах порога 20%
```

Отчёт **не содержит** Sharpe, CAGR и Profit Factor — см. §1.

---

## 10. Сохранение данных

### Структура

```
audit/stage2-demo/
├── state.json                       # счётчик дней, статус теста
├── runs/<RUN_ID>/                   # по каталогу на каждую фазу
│   ├── run_meta.json
│   ├── config_snapshot.env
│   ├── input_snapshot.json
│   ├── plan.json                    # только PREP
│   ├── signals.json
│   ├── orders.json
│   ├── overnight_orders.json        # только OVERNIGHT
│   ├── fills.json
│   ├── positions_before.json
│   ├── positions_after.json
│   ├── balance.json
│   ├── execution_audit.json
│   ├── preflight.json
│   ├── stdout.log
│   ├── stderr.log
│   └── exit_code
├── daily_audit/YYYY-MM-DD.json
├── balance/YYYY-MM-DD.json
├── balance_report.md
├── final_audit.md
└── final_audit.json
```

### Правила

- Каталог `RUN_ID` создаётся с `os.makedirs(exist_ok=False)`. Коллизия — ошибка, не перезапись.
- Идентификаторы: `RUN_ID = <YYYYMMDD-HHMMSS>-<PHASE>`, где `PHASE ∈ {PREP, ORDER, CLEANUP, OVERNIGHT}`.
- Ничего из `audit/stage2-demo/` не удаляется, в том числе после завершения теста.
- Дублирование в БД: `execution_audit` (таблица) — источник истины, JSON в каталоге запуска — снимок для автономного чтения.

---

## 11. Дневной аудит

`audit/stage2-demo/daily_audit/YYYY-MM-DD.json`:

```json
{
  "trading_day": "2026-09-10",
  "prep_run_id": "20260910-094500-PREP",
  "order_run_id": "20260910-100500-ORDER",
  "cleanup_run_id": "20260910-182000-CLEANUP",
  "overnight_run_id": "20260910-183500-OVERNIGHT",
  "signals_count": 46, "orders_count": 5, "fills_count": 3,
  "intraday_positions_opened": 3, "intraday_positions_closed": 3,
  "overnight_orders": 2,
  "execution_audit_count": 13,
  "dataset_hash": "sha256:…", "config_hash": "sha256:…",
  "checks": {"critical_passed": 6, "critical_failed": 0, "warnings": 1},
  "errors": [], "warnings": ["fill rate 60% ниже ожидаемых 57%±10"],
  "verdict": "PASS"
}
```

`verdict ∈ {PASS, PASS_WITH_WARNINGS, FAIL}`. При `FAIL` автоматические запуски следующего дня **не выполняются** до ручного снятия блокировки.

---

## 12. Счётчик 15 торговых дней

`audit/stage2-demo/state.json`:

```json
{
  "test_id": "stage2-demo-15d",
  "started_at": "2026-09-10",
  "completed_trading_days": 7,
  "target_trading_days": 15,
  "status": "running",
  "days": ["2026-09-10", "2026-09-11", "…"],
  "halted_reason": null
}
```

- День засчитывается только при завершившейся фазе OVERNIGHT.
- Выходные и праздники не считаются — используется `services/calendar.is_trading_day`.
- При `completed_trading_days == 15` статус → `finished`, все фазы выходят с кодом 0 и сообщением, ничего не делая. Результаты не удаляются.
- `status = "halted"` при критическом `FAIL`; снимается вручную через `stage2_demo resume`.

---

## 13. Конфигурация

### Обязательно зафиксировать перед стартом

Переаудит обнаружил расхождение между `.env.example` и дефолтами кода. **Без явного `.env` система возьмёт заниженные издержки.**

```ini
VALIDATION_COST_RT=0.128     # в коде сейчас 0.08 — занижение в 1,6 раза
TFT_COST_RT=0.128            # в коде сейчас 0.08
VALIDATION_FULL_UNIVERSE=1   # в коде сейчас False
```

Требование: привести **дефолты кода** к этим значениям, чтобы расхождение не воспроизводилось.

### Новые параметры

```ini
STAGE2_ENABLED=1
STAGE2_TEST_ID=stage2-demo-15d
STAGE2_TARGET_DAYS=15
STAGE2_PREP_TIME=09:45
STAGE2_ORDER_TIME=10:05
STAGE2_CLEANUP_TIME=18:20
STAGE2_OVERNIGHT_TIME=18:35
STAGE2_ALLOW_DATASET_DRIFT=0
STAGE2_HALT_ON_FAIL=1
STAGE2_START_BALANCE_RUB=100000
```

### Изменяется

```ini
INTRADAY_SQUARE_OFF_TIME=18:20   # было 18:35 — освобождает окно для овернайта
```

### Не изменяется автоматически

```ini
ORDER_FILL_WAIT_SEC=60           # зафиксировать фактическое значение
BEST_TRADES_TOP_N=5
BEST_TRADES_POSITION_RUB=10000
SCORE_MODE=heuristic
APPLY_RISK_PENALTIES=0
```

---

## 14. Cron

Четыре отдельных задания, не один общий запуск.

```cron
# stage2-demo-15d — демо-тест, только будни
45  9 * * 1-5  cd /Users/llmifistoll/my-project/ETL-tcs && python3 -m services.stage2_demo prep      >> audit/stage2-demo/cron.log 2>&1
 5 10 * * 1-5  cd /Users/llmifistoll/my-project/ETL-tcs && python3 -m services.stage2_demo order     >> audit/stage2-demo/cron.log 2>&1
20 18 * * 1-5  cd /Users/llmifistoll/my-project/ETL-tcs && python3 -m services.stage2_demo cleanup   >> audit/stage2-demo/cron.log 2>&1
35 18 * * 1-5  cd /Users/llmifistoll/my-project/ETL-tcs && python3 -m services.stage2_demo overnight >> audit/stage2-demo/cron.log 2>&1
```

Cron не знает о праздниках — проверку торгового дня делает сам оркестратор через `services/calendar`. В праздник фаза выходит с кодом 0 и отметкой `skipped`.

---

## 15. Acceptance-тесты

Обязательны до первого запуска:

1. **Square-off закрывает всё внутридневное.** После CLEANUP `GetPositions` не содержит позиций со `strategy_type ∈ {intraday_long, intraday_short}`.
2. **Овернайт переживает cleanup.** Позиция `long_overnight`, открытая накануне, после CLEANUP остаётся.
3. **PREP не выставляет заявок.** Мок брокера: ни одного вызова `post_limit_order` в фазе PREP.
4. **ORDER исполняет план, а не пересчитывает.** Мок `compute_orders`: в фазе ORDER не вызывается.
5. **Отказ при смене конфигурации.** `config_hash` в плане ≠ текущему → ORDER возвращает ненулевой код, заявок нет.
6. **Каталоги не перезаписываются.** Повторный запуск с тем же `RUN_ID` — ошибка.
7. **Боевой контур недостижим.** `TRADING_MODE=sandbox` + `--prod` → отказ (уже покрыто `tests/test_final_config.py`).
8. **Счётчик считает торговые дни.** Суббота не увеличивает `completed_trading_days`.
9. **Автостоп на 15-м дне.** При `completed_trading_days = 15` все фазы — no-op.
10. **Баланс собирается на всех четырёх фазах** и агрегируется в дневную сводку.

---

## 16. Финальный отчёт

После 15-го дня: `audit/stage2-demo/final_audit.md` и `final_audit.json`.

```
ЭТАП 2 — ФУНКЦИОНАЛЬНЫЙ ТЕСТ НА ДЕМО-СЧЁТЕ
Период: 2026-09-10 … 2026-10-01     Торговых дней: 15 из 15

ЗАПУСКИ            запланировано  успешно  сбоев  пропущено  дублей
  PREP                    15         15       0        0        0
  ORDER                   15         15       0        0        0
  CLEANUP                 15         15       0        0        0
  OVERNIGHT               15         15       0        0        0

ИСПОЛНЕНИЕ
  Сигналов сформировано                690
  Заявок выставлено                     71
  Исполнено                             41   (57,7%)
  Записей в execution_audit            183
  Внутридневных позиций открыто         38
  Внутридневных позиций закрыто         38   ✔ совпадает
  Square-off выполнен                   15   из 15 дней
  Овернайт-заявок                       23

ЦЕЛОСТНОСТЬ
  Изменений датасета                     0
  Изменений конфигурации                 0
  Перезаписей предыдущих запусков        0
  Обращений к боевому API                0

БАЛАНС (диагностика, не оценка стратегии)
  Старт                          100 000,00 ₽
  Финал                           99 640,20 ₽
  Изменение                         −359,80 ₽  (−0,36%)

ФУНКЦИОНАЛЬНЫЙ ВЕРДИКТ: PASS
```

Вердикт: `PASS` — все критические проверки пройдены во все 15 дней; `PASS WITH WARNINGS` — есть некритические замечания; `FAIL` — хотя бы одна критическая проверка не прошла.

---

## 17. Что нужно построить

| Файл | Статус | Работа |
|---|---|---|
| `services/stage2_demo.py` | новый | Оркестратор четырёх фаз, снапшот плана, счётчик, аудит |
| `services/stage2_balance.py` | новый | Снятие баланса, дневная сводка, `balance_report.md` |
| `audit/execution_audit.py` | есть, **не подключён** | Подключить: `record_intent`, `record_fill`, `record_exit` |
| `database.py` | правка | Создание таблицы `execution_audit` в `init_db` |
| `services/place_orders.py` | правка | `run_id`/`phase` в журнале, фильтр по `strategy_type` при выставлении, исполнение готового плана |
| `config.py` | правка | `STAGE2_*`, издержки 0.08 → 0.128, `VALIDATION_FULL_UNIVERSE=1`, `INTRADAY_SQUARE_OFF_TIME=18:20` |
| `scripts/stage2_crontab` | новый | Четыре задания |
| `tests/test_stage2_demo.py` | новый | 10 acceptance-тестов из §15 |

Оценка: около 900 строк кода и тестов. Торговая стратегия и алгоритм отбора не затрагиваются.

---

## 18. Открытые вопросы

Требуют подтверждения до начала работ:

1. **Перенос `long_overnight` на вечер** (§0, изменение 5) — меняет момент входа боевой стратегии.
2. **`INTRADAY_SQUARE_OFF_TIME` 18:35 → 18:20** — освобождает окно для овернайта.
3. **Стартовый баланс демо-счёта.** Сейчас на sandbox-счёте `1639899c-…` лежит 99 999,17 ₽. Обнулять и пополнять заново до ровных 100 000 ₽ или стартовать с текущего остатка?
