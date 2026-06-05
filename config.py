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
TICKERS: list[str] = [
    # Банки
    "SBER",
    "VTBR",

    # Нефть и газ
    "GAZP",
    "ROSN",
    "LKOH",
    "NVTK",
    "TATN",
    "SNGS",
    "SNGSP",

    # Металлы и добыча
    "GMKN",
    "PLZL",
    "MAGN",
    "CHMF",
    "ALRS",

    # Энергетика
    "IRAO",
    "FEES",
    "HYDR",    # РусГидро
    "UPRO",    # Юнипро
    "MSNG",    # Мосэнерго
    "TGKA",    # ТГК-1
    "OGKB",    # ОГК-2

    # Химия и удобрения
    "PHOR",
    "AKRN",    # Акрон

    # Финансы
    "MOEX",

    # Телеком
    "MTSS",
    "RTKM",    # Ростелеком

    # Транспорт
    "AFLT",
    "FLOT",    # Совкомфлот
    "NMTP",    # НМТП

    # Ритейл
    "MGNT",    # Магнит
    "X5",      # X5 Group
    "LENT",    # Лента
    "FIXP",    # Fix Price

    # IT и технологии
    "YDEX",    # Яндекс
    "VKCO",    # VK
    "ASTR",    # Астра
    "POSI",    # Positive Technologies

    # Строительство и недвижимость
    "PIKK",    # ПИК
    "SMLT",    # Самолет
    "ETLN",    # Эталон

    # Прочее
    "RUAL",    # Русал
    "ENPG",    # Эн+
    "SELG",    # Селигдар
    "BSPB",    # Банк Санкт-Петербург
    "CBOM",    # МКБ
    "MVID",    # М.Видео
]

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

# --- Telegram мониторинг @markettwits ----------------------------------------
# Получить: https://my.telegram.org → API development tools
TG_API_ID:   int = int(os.getenv("TG_API_ID") or "0")
TG_API_HASH: str = os.getenv("TG_API_HASH", "")
TG_PHONE:    str = os.getenv("TG_PHONE", "")     # +79001234567
TG_CHANNEL:  str = os.getenv("TG_CHANNEL", "@markettwits")
TG_FETCH_LIMIT: int = int(os.getenv("TG_FETCH_LIMIT", "500"))  # макс. постов за цикл

# --- Claude API (для LLM-парсинга сентимента) --------------------------------
# Получить: https://console.anthropic.com/keys
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
