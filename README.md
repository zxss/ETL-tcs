# ETL — исторические биржевые данные (T-Invest API → PostgreSQL)

Загружает дневные свечи по тикерам **SBER, GAZP, ROSN, LKOH, GMKN, NVTK, PLZL**
за последние 24 месяца и сохраняет их в PostgreSQL.

## Структура проекта

```
ETL-tcs/
├── main.py                  # точка входа
├── config.py                # параметры из .env
├── database.py              # подключение к PG, upsert
├── loaders/
│   └── moex_loader.py       # HTTP-клиент T-Invest API
├── models/
│   └── market_data.py       # DDL таблицы и SQL-запросы
├── services/
│   └── load_history.py      # оркестрация ETL по тикерам
├── .env                     # переменные окружения (не коммитить!)
├── requirements.txt
└── README.md
```

## Быстрый старт

### 1. Зависимости

```bash
pip install -r requirements.txt
```

### 2. Настройка `.env`

```env
INVEST_TOKEN=t.ВАШ_ТОКЕН

DB_HOST=localhost
DB_PORT=5432
DB_NAME=market_data
DB_USER=postgres
DB_PASSWORD=postgres
```

### 3. Создание базы данных

```sql
CREATE DATABASE market_data;
```

### 4. Запуск

```bash
python3 main.py
```

Таблица `market_data` создаётся автоматически при первом запуске.

## Схема таблицы

| Поле       | Тип            | Описание                        |
|------------|----------------|---------------------------------|
| id         | BIGSERIAL PK   | Суррогатный ключ                |
| ticker     | VARCHAR(10)     | Тикер (SBER, LKOH, ...)         |
| date       | DATE           | Дата торговой сессии            |
| open       | NUMERIC(18,4)  | Цена открытия                   |
| high       | NUMERIC(18,4)  | Максимум дня                    |
| low        | NUMERIC(18,4)  | Минимум дня                     |
| close      | NUMERIC(18,4)  | Цена закрытия                   |
| volume     | BIGINT         | Объём в лотах                   |
| created_at | TIMESTAMPTZ    | Время загрузки                  |

Уникальный ключ: `(ticker, date)` — повторные запуски безопасны (upsert).

## Добавление нового тикера

В `config.py` добавьте тикер в список `TICKERS`:

```python
TICKERS = ["SBER", "GAZP", "ROSN", "LKOH", "GMKN", "NVTK", "PLZL", "НОВЫЙ_ТИКЕР"]
```
