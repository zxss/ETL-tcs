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

# Биржевые индексы (грузятся отдельным шагом services/load_index.py).
# Нужны слою рыночного контекста (режим рынка по IMOEX, относительная сила).
LOAD_INDEX: bool = os.getenv("LOAD_INDEX", "1") not in ("0", "false", "False")
INDEX_TICKERS: list[str] = (
    os.getenv("INDEX_TICKERS").split() if os.getenv("INDEX_TICKERS") else ["IMOEX"]
)

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

# --- Strategy Edge Validator -------------------------------------------------
# После загрузки данных main.py может прогнать статистический контур валидации
# стратегий (validation/verdict.py из скила strategy-edge-validator).
# RUN_VALIDATION=0 — отключить шаг расчёта.
RUN_VALIDATION: bool = os.getenv("RUN_VALIDATION", "1") not in ("0", "false", "False")

# Каталог с движком валидации: advanced_stats.py, pnl_engine.py и пакет
# validation/. По умолчанию используется локальная копия внутри проекта
# (strategy_validation/), скопированная из скила strategy-edge-validator —
# внешней зависимости от пути к скилу больше нет. Можно переопределить
# через переменную окружения VALIDATOR_DIR.
VALIDATOR_DIR: str = os.getenv(
    "VALIDATOR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_validation"),
)

# Куда выгружать CSV-свечи из БД для валидатора.
VALIDATION_DATA_DIR: str = os.getenv(
    "VALIDATION_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_data"),
)

# Тикеры и стратегии для прогона валидации.
# TGKA исключён намеренно (битый масштаб цены) — см. validate_candles.
VALIDATION_TICKERS: list[str] = (
    os.getenv("VALIDATION_TICKERS", "ETLN SELG SMLT MGNT ALRS UPRO").split()
)
VALIDATION_STRATS: list[str] = (
    os.getenv("VALIDATION_STRATS", "long_overnight intraday_short intraday_long").split()
)
VALIDATION_BOOT: int = int(os.getenv("VALIDATION_BOOT", "2000"))

# --- Реалистичные издержки (PR-1) -------------------------------------------
# Round-trip издержки в % по умолчанию (комиссия + half-spread×2 + проскальзывание).
# 0.08% = 2×0.04% — оптимистичная оценка для ликвидных голубых фишек. Для мид-капов
# (ETLN/SELG/SMLT) реальный round-trip ближе к 0.15–0.25% из-за спреда. Переопределяется
# по тикеру через VALIDATION_COST_RT_MAP="ETLN:0.20 SELG:0.18 SMLT:0.20".
VALIDATION_COST_RT: float = float(os.getenv("VALIDATION_COST_RT", "0.08"))
VALIDATION_COST_RT_MAP: dict[str, float] = {
    kv.split(":")[0].upper(): float(kv.split(":")[1])
    for kv in os.getenv("VALIDATION_COST_RT_MAP", "").split() if ":" in kv
}

# --- Честная мультитестовая вселенная (PR-2) --------------------------------
# При True контур считает HAC-t p-values по ВСЕМ config.TICKERS (а не только по
# отобранным VALIDATION_TICKERS) и применяет BH-FDR по полной сетке — это
# корректирует data snooping при выборе тикеров. Отчёт по-прежнему печатается
# только по VALIDATION_TICKERS. Тяжелее (выгрузка всех тикеров в CSV).
VALIDATION_FULL_UNIVERSE: bool = os.getenv("VALIDATION_FULL_UNIVERSE", "0") not in ("0", "false", "False")

# --- Винзоризация ex-div / новостных гэпов (PR-4) ---------------------------
# T-Invest свечи НЕ скорректированы на дивиденды: в ex-div дату overnight/total
# показывают ложный гэп вниз. При True экстремальные overnight-гэпы винзоризуются
# (прокси истинной total-return корректировки; для точного учёта нужен дивидендный
# фид). По умолчанию выключено, чтобы не менять результаты молча.
VALIDATION_WINSORIZE_GAPS: bool = os.getenv("VALIDATION_WINSORIZE_GAPS", "0") not in ("0", "false", "False")

# --- Walk-forward OOS (PR-5) ------------------------------------------------
# Доля хвоста ряда, отводимая под честный хронологический out-of-sample тест.
VALIDATION_OOS_FRACTION: float = float(os.getenv("VALIDATION_OOS_FRACTION", "0.30"))

# --- Multi-Asset TFT Next-Day Range Forecast --------------------------------
# Единый Temporal Fusion Transformer на ВСЕХ тикерах прогнозирует диапазон
# цены следующего торгового дня (ForecastLow/High, RangePct, CoverageProb).
# Запускается из main.py после контура валидации. TFT_FORECAST=0 — отключить.
TFT_FORECAST: bool = os.getenv("TFT_FORECAST", "1") not in ("0", "false", "False")
# Тикеры для обучения общей модели (по умолчанию — весь список TICKERS).
TFT_TICKERS: list[str] = (
    os.getenv("TFT_TICKERS").split() if os.getenv("TFT_TICKERS") else TICKERS
)
TFT_EPOCHS: int = int(os.getenv("TFT_EPOCHS", "30"))
TFT_HIDDEN: int = int(os.getenv("TFT_HIDDEN", "32"))
# Издержки round-trip (%) для направленного прогноза PnL стратегий.
# По умолчанию 0.08% = 2 × 0.04% (как cost-side в контуре валидации).
TFT_COST_RT: float = float(os.getenv("TFT_COST_RT", "0.08"))

# --- Сводная итоговая таблица ------------------------------------------------
# COMBINED_TABLE=1 — вместо четырёх отдельных таблиц (валидация, диапазон по
# тикерам, диапазон по стратегиям, направленный PnL) вывести ОДНУ сводную
# таблицу «ticker + strategy» со всеми метриками + расшифровкой столбцов и
# топ-5 кандидатов. COMBINED_TABLE=0 — старое поведение (отдельные таблицы).
COMBINED_TABLE: bool = os.getenv("COMBINED_TABLE", "1") not in ("0", "false", "False")

# SHOW_ALL_INTRADAY=1 — подробный режим: показывать обе дневные стратегии
# (intraday_short и intraday_long) с флагом Selected ✓/-. По умолчанию (0)
# в таблице остаётся только лучшая дневная стратегия по каждому тикеру.
SHOW_ALL_INTRADAY: bool = os.getenv("SHOW_ALL_INTRADAY", "0") not in ("0", "false", "False")

# DASHBOARD_TOP_N — сколько строк (бумаг × стратегий) выводить в сводном
# дашборде, отсортированных по итоговому рейтингу FinalScore. По умолчанию 50.
# 0 — без ограничения (показать все).
DASHBOARD_TOP_N: int = int(os.getenv("DASHBOARD_TOP_N", "50"))

# BEST_TRADES_TOP_N — сколько сигналов показывать в блоке «ЛУЧШИЕ СДЕЛКИ».
BEST_TRADES_TOP_N: int = int(os.getenv("BEST_TRADES_TOP_N", "10"))

# BEST_TRADES_POSITION_RUB — целевой размер позиции на одну бумагу (₽)
# для расчёта объёма лотов в торговых инструкциях.
BEST_TRADES_POSITION_RUB: float = float(os.getenv("BEST_TRADES_POSITION_RUB", "10000"))

# LIMIT_ENTRY_FRACTION — насколько лимитка прижата к экстремуму прогнозного
# коридора (доля 0..1): 0 = спот, 1 = ровно на F.High/F.Low.
# SHORT тянется к ВЕРХНЕЙ границе, LONG — к НИЖНЕЙ. Default 0.8 — агрессивно,
# но не на самом краю, чтобы заявка имела шанс залиться.
LIMIT_ENTRY_FRACTION: float = float(os.getenv("LIMIT_ENTRY_FRACTION", "0.8"))

# --- Актуальные котировки на момент запуска ----------------------------------
# TFT_USE_LIVE_PRICE=1 — перед расчётом получить последнюю цену (Last Price) по
# каждому тикеру и якорить прогноз диапазона/PnL на ней, а не на вчерашнем
# закрытии. =0 — считать на последней цене закрытия (Previous Close).
TFT_USE_LIVE_PRICE: bool = os.getenv("TFT_USE_LIVE_PRICE", "1") not in ("0", "false", "False")
# Возраст котировки (сек), после которого выводится предупреждение об устаревании.
TFT_STALE_SECONDS: int = int(os.getenv("TFT_STALE_SECONDS", "900"))

# --- Max Pos ₽ (максимальный размер позиции по ликвидности) -------------------
# Допустимое ценовое воздействие (доля): сколько можно «сдвинуть» цену входом.
TFT_IMPACT_TOL: float = float(os.getenv("TFT_IMPACT_TOL", "0.005"))   # 0.5%
# Максимальная доля участия в среднедневном рублёвом обороте.
TFT_PARTICIPATION: float = float(os.getenv("TFT_PARTICIPATION", "0.01"))  # 1%
# Глубина окна (дней) для оценки ликвидности.
TFT_LIQUIDITY_DAYS: int = int(os.getenv("TFT_LIQUIDITY_DAYS", "60"))

# --- Dashboard 2.0: рыночный контекст и риск-фильтры -------------------------
# Жёсткий рыночный фильтр: запрещать LONG при BEAR и SHORT при BULL.
STRICT_MARKET_FILTER: bool = os.getenv("STRICT_MARKET_FILTER", "0") not in ("0", "false", "False")
