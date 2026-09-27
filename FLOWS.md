# Потоки проекта

Поток — повторяемая работа: у неё есть папка, одна команда запуска, известные
входы и выходы. Разовое — в scratchpad, не сюда. Правила — в CLAUDE.md.
Проверка, что каждая точка входа описана здесь или в архиве:
`python -m tools.code_audit.audit`.

| поток | инструкция | команда | выход |
|---|---|---|---|
| ETL: свечи, справочник, дивиденды, ставка ЦБ, бары сессии, прогноз, валидация | [flows/etl](flows/etl/КАК_ЗАПУСТИТЬ.md) | `python main.py` | таблицы БД `market_data*`, `forecasts`, `dividends`, `macro_rates`, `market_data_session` |
| Этап 2: торговый контур r4 / «Робот» (сейчас ОСТАНОВЛЕН) | [flows/stage2](flows/stage2/КАК_ЗАПУСТИТЬ.md) | `python -m services.stage2_demo <фаза>` | `audit/stage2-*/runs/…`, `data/order_log/` |
| Выкладка на VDS | [flows/deploy](flows/deploy/КАК_ЗАПУСТИТЬ.md) | `git push vds <DEPLOY_BRANCH>` | `/opt/etl-tcs` на коммите ветки |
| Контур валидации стратегий | [flows/validation](flows/validation/КАК_ЗАПУСТИТЬ.md) | `python -m validation.verdict …` | вердикты REJECTED / WEAK / CANDIDATE EDGE |
| Аудит системы на живых данных | [audit/fullaudit_2026_09_26](audit/fullaudit_2026_09_26/КАК_ЗАПУСТИТЬ.md) | `python /tmp/audit_mech.py` (на VDS) | итоги в stdout, `audit/out/walkforward_session_wNN.json` |
| Аудит кода | [tools/code_audit](tools/code_audit/КАК_ЗАПУСТИТЬ.md) | `python -m tools.code_audit.audit` | `audit/out/code_audit_wNN.{json,md}` |
| Инструменты потоков, синхронизация `.env.example` | [tools](tools/КАК_ЗАПУСТИТЬ.md) | `python -m tools.env_example_sync` | авто-блок в `.env.example` |
| Обслуживание данных | [flows/maintenance](flows/maintenance/КАК_ЗАПУСТИТЬ.md) | `python -m scripts.cleanup_stale_tickers --dry-run` | правки таблиц БД |
| Портфель ВДО: бэктест по предрегистрации и живая система в песочнице (ветка `feature/bonds`) | [bonds](bonds/КАК_ЗАПУСТИТЬ.md) | `python -m bonds.trader rebalance --if-due` · `monitor` · `status`; `python -m bonds.backtest --cache …`, `python -m bonds.distress --cache …` | `$BONDS_DIR/target_YYYY-MM_wNN.json`, `ledger.jsonl`, `cache/results/bonds_backtest_wNN.json` |

Исследовательские потоки живут в ветках `research/*`, у каждого свой
`КАК_ЗАПУСТИТЬ.md` рядом с кодом (например, `research/vrp`, `research/levmom` в
ветке `research/vrp`). Отчёты — `~/Desktop/трейдинг исследование/`.

Модули, которые принадлежат потокам выше (упоминание нужно аудиту кода):
`main`, `services.load_instruments`, `services.news_tg`, `services.news_tg_import`,
`services.stage2_demo`, `services.place_orders`, `services.cancel_orders`,
`services.account_status`, `services.list_accounts`, `services.adopt_position`,
`services.notify`, `scripts.flatten_account`, `scripts.cleanup_stale_tickers`,
`scripts.mark_legacy_source`, `strategy_validation.validation.verdict`,
`strategy_validation.advanced_stats`, `audit.walkforward_session`,
`audit.fullaudit_2026_09_26.audit_mech`, `audit.fullaudit_2026_09_26.audit_window`,
`audit.fullaudit_2026_09_26.audit_fwd`, `audit.fullaudit_2026_09_26.audit_data_exec`,
`tools.code_audit.audit`, `tools.code_audit.fix_getattr`, `tools.env_example_sync`,
`bonds.trader`, `bonds.backtest`, `bonds.distress`,
`strategy_validation.validation.bootstrap`, `strategy_validation.validation.entropy`,
`strategy_validation.validation.monte_carlo`, `strategy_validation.validation.probability_of_ruin`,
`strategy_validation.validation.risk_metrics`, `strategy_validation.validation.serial_dependence`,
`strategy_validation.validation.significance`, `strategy_validation.validation.walk_forward`,
`strategy_validation.validation.white_rc` (самотесты модулей контура валидации).

## Архив

Разовые прогоны и отключённый код. Не потоки: повторно не запускаются, выходы
перезаписывают — так и задумано. Результаты — в отчётах AUDIT-*.md и на рабочем столе.

- `audit/reaudit` — повторный аудит 09.09 (AUDIT-REAUDIT-REPORT.md)
- `audit/reaudit_r3` — аудит r3 14.09 (AUDIT-R3-REPORT.md)
- `audit/replay`, `audit/run_alternatives`, `audit/run_analysis`,
  `audit/run_validation_full`, `audit/run_walkforward` — аудит доходности 08.09
  (AUDIT-PROFITABILITY-REPORT.md)
- `research/news_event_study`, `research/ta/lowvol` — закрытые исследования
- `contrib/experimental_news` — отключённый контур новостей (сбор идёт через `services.news_tg`)
- `tft_forecast/feature_engineering` — мёртвый код: никем не импортируется, продакшн
  учит TFT на признаках `tft_forecast/features.py` (аудит 26.09)
