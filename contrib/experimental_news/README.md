# Контур новостей @markettwits (отключён)

Экспериментальный контур: Telegram-канал → парсинг сентимента (regex + Claude API)
→ таблица `news_sentiment` в PostgreSQL → view `news_with_candles` (джойн новости
с дневными свечами).

**Статус: не работает и не входит в штатный конвейер.** Запросы к Telegram
закомментированы (метки `# TG_ENABLED`) до настройки авторизации. Код вынесен
сюда из корня проекта, чтобы не мешать основному пайплайну.

## Что изменилось при выносе

* `run_monitor.py`, `services/monitor_news.py`, `monitors/` переехали в этот каталог;
* DDL новостей (`news_sentiment`, view `news_with_candles`) — в `news_schema.py`;
* функции доступа к БД — в `news_db.py`;
* **`database.init_db()` больше не создаёт `news_sentiment` и `news_with_candles`.**
  Уже созданные таблицы в существующих базах не удаляются — они просто перестали
  обновляться при запуске штатного ETL.

## Как включить обратно

```bash
pip install -r contrib/experimental_news/requirements.txt
```

1. Заполнить в `.env`: `TG_API_ID`, `TG_API_HASH`, `TG_PHONE`, `ANTHROPIC_API_KEY`
   (переменные остались в общем `config.py`).
2. Раскомментировать блоки `# TG_ENABLED` в `run_monitor.py`, `monitor_news.py`,
   `monitors/tg_reader.py`, вернуть `import anthropic` в `monitors/news_parser.py`.
3. Поднять схему — она больше не создаётся автоматически:

```python
from contrib.experimental_news.news_db import init_news_schema
init_news_schema()
```

4. Разовая авторизация в Telegram и запуск:

```bash
python3 -m contrib.experimental_news.run_monitor --auth
python3 -m contrib.experimental_news.run_monitor
python3 -m contrib.experimental_news.run_monitor --loop --interval 15
```
