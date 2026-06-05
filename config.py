"""
Конфигурация приложения — читается из переменных окружения / .env файла.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- T-Invest API -----------------------------------------------------------
INVEST_TOKEN: str = os.environ["INVEST_TOKEN"]
API_BASE_URL: str = "https://invest-public-api.tbank.ru/rest"
API_SERVICE:  str = "tinkoff.public.invest.api.contract.v1"

# --- PostgreSQL -------------------------------------------------------------
DB_HOST:     str = os.getenv("DB_HOST", "localhost")
DB_PORT:     int = int(os.getenv("DB_PORT", 5432))
DB_NAME:     str = os.getenv("DB_NAME", "market_data")
DB_USER:     str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# --- ETL параметры ----------------------------------------------------------
TICKERS: list[str] = ["SBER", "GAZP", "ROSN", "LKOH", "GMKN", "NVTK", "PLZL"]

# Глубина первоначальной загрузки (если данных ещё нет в БД)
INITIAL_MONTHS_DAILY: int = 24   # дневные свечи — 24 мес
INITIAL_MONTHS_5M:    int = 6    # 5-мин свечи   —  6 мес

# Окно одного запроса к API (чанкинг)
CHUNK_DAYS_DAILY: int = 365   # дневные: до года за запрос
CHUNK_DAYS_5M:    int = 1     # 5-мин: не более 1 дня за запрос (лимит API)

# Параллельность
MAX_CONCURRENT_TICKERS: int = 3   # сколько тикеров грузить одновременно

# HTTP / ретраи
MAX_RETRIES:   int   = 4
BASE_SLEEP:    float = 0.5    # базовая пауза для exponential backoff
REQUEST_SLEEP: float = 0.2    # пауза между чанк-запросами одного тикера
