"""
Конфигурация приложения — читается из переменных окружения / .env файла.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_bool_env(name: str, default: bool | str | int = False) -> bool:
    """Канонический парсер булевых переменных окружения.

    Единственная точка, где решается, что считать «выключено». Раньше по файлу
    было рассыпано 17 копий идиомы `os.getenv(...) not in ("0","false","False")`,
    и любая правка списка ложных значений в одном месте не доезжала до
    остальных.

    Ложью считаются (без учёта регистра и пробелов): 0, false, no, off, n, f
    и пустая строка. Всё остальное — истина. Вариант через int(os.getenv(...))
    сознательно не используется: он падает с ValueError на значении вида
    «false», а боевой конфиг не должен ронять процесс из-за опечатки в .env.
    """
    raw = os.getenv(name)
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "n", "f", "")

# --- T-Invest API -----------------------------------------------------------
# Токен НЕ обязателен на этапе импорта: аналитические скрипты и работа с БД
# должны импортировать config без него. Наличие токена проверяют сетевые
# клиенты брокера при инициализации — см. require_invest_token().
INVEST_TOKEN: str | None = os.getenv("INVEST_TOKEN")
API_BASE_URL: str = "https://invest-public-api.tbank.ru/rest"
API_SERVICE:  str = "tinkoff.public.invest.api.contract.v1"


def require_invest_token() -> str:
    """Токен для обращения к API брокера. Вызывается в инициализации сетевых
    клиентов (services/broker/, loaders/), а не при импорте config."""
    if not INVEST_TOKEN:
        raise ValueError("INVEST_TOKEN не задан в .env")
    return INVEST_TOKEN


# Проверка TLS-сертификатов брокера. По умолчанию ВКЛЮЧЕНА. Отключать только
# явно (INVEST_TLS_VERIFY=0) и осознанно — например при MITM-перехвате TLS
# корпоративным прокси; при отключении в лог пишется предупреждение (tls.py).
INVEST_TLS_VERIFY: int = 0 if os.getenv(
    "INVEST_TLS_VERIFY", "1").strip().lower() in ("0", "false", "no", "") else 1

# Сертификат T-Invest выпущен «Russian Trusted Root CA» (Минцифры), которого нет
# в дефолтном доверенном хранилище Python. Путь к PEM-бандлу с этим корнем —
# INVEST_CA_BUNDLE; без него проверка TLS упадёт CERTIFICATE_VERIFY_FAILED.
INVEST_CA_BUNDLE: str = os.getenv("INVEST_CA_BUNDLE", "")

# --- T-Invest автозаявки (services/place_orders.py) -------------------------
# КОНТУР ПО УМОЛЧАНИЮ — SANDBOX (виртуальные деньги). Боевой контур требует
# явного флага --prod. Боевой счёт берётся ТОЛЬКО из окружения: захардкоженного
# номера счёта здесь быть не должно.
PROD_ACCOUNT_ID: str = os.getenv("PROD_ACCOUNT_ID", "")

# Предел позиции на БОЕВОМ контуре — предохранитель от опечатки в
# BEST_TRADES_POSITION_RUB (лишний ноль превращает 20 000 ₽ в 200 000 ₽).
# 0 отключает проверку.
PROD_MAX_POSITION_RUB: float = float(os.getenv("PROD_MAX_POSITION_RUB", "25000"))


def require_prod_account_id() -> str:
    """Номер боевого счёта. Вызывается только при реальном обращении к PROD."""
    if not PROD_ACCOUNT_ID:
        raise ValueError("PROD_ACCOUNT_ID не задан в .env")
    return PROD_ACCOUNT_ID

# Песочница: тестовый контур (виртуальные счета/деньги, исполнение по last price).
SANDBOX_API_BASE_URL: str = os.getenv(
    "SANDBOX_API_BASE_URL",
    "https://sandbox-invest-public-api.tbank.ru/rest",
)
# Если задан — переиспользуем этот sandbox-счёт; иначе OpenSandboxAccount + PayIn.
# Счета песочницы живут 3 месяца от последнего обращения, потом удаляются.
SANDBOX_ACCOUNT_ID: str = os.getenv("SANDBOX_ACCOUNT_ID", "")
SANDBOX_PAYIN_RUB: float = float(os.getenv("SANDBOX_PAYIN_RUB", "100000"))
SANDBOX_PAYIN_CURRENCY: str = os.getenv("SANDBOX_PAYIN_CURRENCY", "rub")

# Поведение по риску раннего стопа (см. ТЗ §7.5.3): в двухфазной модели стоп
# ставится отдельной фазой --attach-stops по факту наличия позиции, поэтому
# флаг влияет только на режим --immediate-stop.
ORDER_STOP_MODE_WAIT_FILL: bool = (
    get_bool_env("ORDER_STOP_MODE_WAIT_FILL", 1)
)

# --- PostgreSQL -------------------------------------------------------------
DB_HOST:     str = os.getenv("DB_HOST", "localhost")
DB_PORT:     int = int(os.getenv("DB_PORT", 5432))
DB_NAME:     str = os.getenv("DB_NAME", "market_data")
DB_USER:     str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# Пул соединений (database.init_pool). Загрузка идёт в MAX_CONCURRENT_TICKERS
# потоков, каждому нужен свой коннект — иначе commit одного потока фиксирует
# незавершённую транзакцию другого.
DB_POOL_MIN: int = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX: int = int(os.getenv("DB_POOL_MAX", "5"))

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
    "FIXR",    # Fix Price (новая рос. акция после редомициляции; старый FIXP/US-GDR не торгуется через API)

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
LOAD_INDEX: bool = get_bool_env("LOAD_INDEX", 1)
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
RUN_VALIDATION: bool = get_bool_env("RUN_VALIDATION", 1)

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
# 0.128% — round-trip по переаудиту: комиссия 0.04%×2 + half-spread + проскальзывание.
# Дефолт 0.08 занижал издержки в 1,6 раза, и расхождение с .env.example приводило
# к тому, что без явного .env система считала альфу по заниженной планке.
VALIDATION_COST_RT: float = float(os.getenv("VALIDATION_COST_RT", "0.128"))
VALIDATION_COST_RT_MAP: dict[str, float] = {
    kv.split(":")[0].upper(): float(kv.split(":")[1])
    for kv in os.getenv("VALIDATION_COST_RT_MAP", "").split() if ":" in kv
}

# --- Честная мультитестовая вселенная (PR-2) --------------------------------
# При True контур считает HAC-t p-values по ВСЕМ config.TICKERS (а не только по
# отобранным VALIDATION_TICKERS) и применяет BH-FDR по полной сетке — это
# корректирует data snooping при выборе тикеров. Отчёт по-прежнему печатается
# только по VALIDATION_TICKERS. Тяжелее (выгрузка всех тикеров в CSV).
VALIDATION_FULL_UNIVERSE: bool = get_bool_env("VALIDATION_FULL_UNIVERSE", 1)

# --- Винзоризация ex-div / новостных гэпов (PR-4) ---------------------------
# T-Invest свечи НЕ скорректированы на дивиденды: в ex-div дату overnight/total
# показывают ложный гэп вниз. При True экстремальные overnight-гэпы винзоризуются
# (прокси истинной total-return корректировки; для точного учёта нужен дивидендный
# фид). По умолчанию выключено, чтобы не менять результаты молча.
VALIDATION_WINSORIZE_GAPS: bool = get_bool_env("VALIDATION_WINSORIZE_GAPS", 0)

# --- Walk-forward OOS (PR-5) ------------------------------------------------
# Доля хвоста ряда, отводимая под честный хронологический out-of-sample тест.
VALIDATION_OOS_FRACTION: float = float(os.getenv("VALIDATION_OOS_FRACTION", "0.30"))

# --- Multi-Asset TFT Next-Day Range Forecast --------------------------------
# Единый Temporal Fusion Transformer на ВСЕХ тикерах прогнозирует диапазон
# цены следующего торгового дня (ForecastLow/High, RangePct, CoverageProb).
# Запускается из main.py после контура валидации. TFT_FORECAST=0 — отключить.
TFT_FORECAST: bool = get_bool_env("TFT_FORECAST", 1)
# Тикеры для обучения общей модели (по умолчанию — весь список TICKERS).
TFT_TICKERS: list[str] = (
    os.getenv("TFT_TICKERS").split() if os.getenv("TFT_TICKERS") else TICKERS
)
TFT_EPOCHS: int = int(os.getenv("TFT_EPOCHS", "30"))
TFT_HIDDEN: int = int(os.getenv("TFT_HIDDEN", "32"))
# Издержки round-trip (%) для направленного прогноза PnL стратегий.
# По умолчанию 0.08% = 2 × 0.04% (как cost-side в контуре валидации).
# Та же планка, что и в контуре валидации: источник значения должен быть один.
TFT_COST_RT: float = float(os.getenv("TFT_COST_RT", "0.128"))

# --- Разложение издержек на статьи (execution_audit) --------------------------
# TFT_COST_RT — одно число round-trip. Чтобы сравнить ФАКТИЧЕСКОЕ проскальзывание
# с расчётным (критерий ТЗ Этапа 2: расхождение ≤ 20%), нужна отдельно взятая
# доля проскальзывания на одну сторону сделки — иначе сравнивать не с чем.
#
# Разложение то же, из которого получена 0.128: комиссия 0.04% × 2 стороны +
# half-spread + проскальзывание. Отсюда на одну сторону:
#     (0.128 − 2 × 0.04) / 2 = 0.024%
# Значение НЕ новое и не выдуманное: это та же константа, которой уже доверяет
# модель, разложенная на слагаемые. Переопределяется явно, если брокерский
# тариф отличается.
BROKER_COMMISSION_PCT: float = float(os.getenv("BROKER_COMMISSION_PCT", "0.04"))
EXPECTED_SLIPPAGE_PCT: float = float(os.getenv(
    "EXPECTED_SLIPPAGE_PCT",
    str(round(max(0.0, TFT_COST_RT - 2 * float(os.getenv("BROKER_COMMISSION_PCT", "0.04"))) / 2, 4))))

# --- Сводная итоговая таблица ------------------------------------------------
# COMBINED_TABLE=1 — вместо четырёх отдельных таблиц (валидация, диапазон по
# тикерам, диапазон по стратегиям, направленный PnL) вывести ОДНУ сводную
# таблицу «ticker + strategy» со всеми метриками + расшифровкой столбцов и
# топ-5 кандидатов. COMBINED_TABLE=0 — старое поведение (отдельные таблицы).
COMBINED_TABLE: bool = get_bool_env("COMBINED_TABLE", 1)

# SHOW_ALL_INTRADAY=1 — подробный режим: показывать обе дневные стратегии
# (intraday_short и intraday_long) с флагом Selected ✓/-. По умолчанию (0)
# в таблице остаётся только лучшая дневная стратегия по каждому тикеру.
SHOW_ALL_INTRADAY: bool = get_bool_env("SHOW_ALL_INTRADAY", 0)

# DASHBOARD_TOP_N — сколько строк (бумаг × стратегий) выводить в сводном
# дашборде, отсортированных по итоговому рейтингу FinalScore. По умолчанию 50.
# 0 — без ограничения (показать все).
DASHBOARD_TOP_N: int = int(os.getenv("DASHBOARD_TOP_N", "50"))

# SAVE_FORECASTS=1 (по умолчанию) — сохранять строки дневного и недельного
# дашбордов в таблицу forecasts (upsert по asof_date+ticker+strategy), чтобы
# потом сверять прогноз с фактом. =0 — только печать, без записи в БД.
SAVE_FORECASTS: bool = get_bool_env("SAVE_FORECASTS", 1)

# BEST_TRADES_TOP_N — сколько сигналов показывать в блоке «ЛУЧШИЕ СДЕЛКИ».
# Концентрация в топ-5: при обоих предохранителях top_n=5 даёт CAGR 10.4%
# против 9.3% при top_n=10 и лучшую просадку (-19.7% против -22.1%).
BEST_TRADES_TOP_N: int = int(os.getenv("BEST_TRADES_TOP_N", "5"))

# BEST_TRADES_POSITION_RUB — целевой размер позиции на одну бумагу (₽)
# для расчёта объёма лотов в торговых инструкциях. С прогона r4 (депозит 5 млн ₽,
# решение пользователя 15.09.2026) — 100 000 ₽: 5 позиций × 100 000 = 10 % депозита
# в активном риске. До r4 было 10 000 ₽ при депозите 2 млн.
BEST_TRADES_POSITION_RUB: float = float(os.getenv("BEST_TRADES_POSITION_RUB", "100000"))

# LIMIT_ENTRY_FRACTION — насколько лимитка прижата к экстремуму прогнозного
# коридора (доля 0..1): 0 = спот, 1 = ровно на F.High/F.Low.
# SHORT тянется к ВЕРХНЕЙ границе, LONG — к НИЖНЕЙ. Default 0.2 — заявка стоит
# близко к рынку (~0.5–1% от спота), чтобы реально заливаться; 0.8 ставило вход
# на 2–5% от рынка, и заявки висели днями не исполняясь.
LIMIT_ENTRY_FRACTION: float = float(os.getenv("LIMIT_ENTRY_FRACTION", "0.2"))

# OVERNIGHT_ENTRY_MODE — как фаза OVERNIGHT 18:35 ставит ночные лонги.
#   "marketable" — перед постановкой берётся ТЕКУЩАЯ цена у брокера и лимит
#                  ставится на OVERNIGHT_MARKETABLE_SLIP_PCT выше неё: заявка
#                  исполняется сразу (аукцион закрытия / вечерняя сессия).
#                  Стоп и тейк пересчитываются от фактического входа с теми же
#                  процентами, лоты — под ту же сумму позиции.
#   "forecast"   — прежний вход внутри прогнозного коридора (LIMIT_ENTRY_FRACTION).
# Решение пользователя 22.09.2026: в песочнице задача — проверять покупки и
# продажи каждый день. За 18–21.09 при "forecast" не исполнилась ни одна из 9
# ночных заявок: лимит стоял на 0,7–1,3 % ниже якоря, а якорь вечерней фазы —
# вчерашний бар (дефект 4), цена до лимита не доходила.
OVERNIGHT_ENTRY_MODE: str = os.getenv("OVERNIGHT_ENTRY_MODE", "marketable").strip().lower()
OVERNIGHT_MARKETABLE_SLIP_PCT: float = float(os.getenv("OVERNIGHT_MARKETABLE_SLIP_PCT", "0.1"))

# LIMIT_TP_FRACTION — цель take-profit как доля пути от ВХОДА к ПРОТИВОПОЛОЖНОЙ
# границе прогнозного коридора (анализ диапазона):
#   SHORT → к НИЖНЕЙ границе F.Low (прибыль на возврате вниз);
#   LONG  → к ВЕРХНЕЙ границе F.High (прибыль на возврате вверх).
# 1.0 — ровно дальняя граница прогноза (q0.9 для LONG, q0.1 для SHORT);
# 0.0 — цель совпадает со входом.
#
# Default 0.5 вместо прежнего 1.0. Основание — моделирование исполнения на
# 5-минутном пути цены (2 577 сигналов, дек-2025 … сен-2026): при 1.0 цель
# достигалась лишь в 1,9% сделок, то есть заявка TAKE_PROFIT была почти
# декоративной, и 79% позиций доживали до закрытия сессии. При 0.5 цель
# срабатывает в 11,2% случаев — в шесть раз чаще. Полный свип:
#
#   tp_frac   доля достижения цели   win rate   средний нетто-PnL
#     1.0            1,9%             38,1%         −0,280%
#     0.5           11,2%             38,6%         −0,294%
#     0.2           36,2%             45,8%         −0,320%
#
# ВАЖНО: по среднему PnL более близкая цель СЛЕГКА ХУЖЕ — она срезает
# прибыльные хвосты, оставляя стоп на месте. Значение 0.5 выбрано ради
# работоспособности механизма TP (при 1.0 он фактически не участвует в
# сделке), а не ради прироста доходности; на текущем сигнале все варианты
# убыточны. Пересмотреть после того, как появится положительная альфа.
# См. AUDIT-PROFITABILITY-REPORT.md, раздел 6, и audit/out/tp_sweep.csv.
LIMIT_TP_FRACTION: float = float(os.getenv("LIMIT_TP_FRACTION", "0.5"))

# WEEK_HORIZON_DAYS — горизонт недельного прогноза (торговых дней). Модель
# обучается на НАСТОЯЩИХ недельных целях: min(low)/max(high)/close за следующие
# H дней (features.TARGET_COLS: week_low_pct/week_high_pct/week_total), квантили
# предсказываются той же TFT-головой, покрытие недельного коридора калибруется
# отдельно на held-out хвосте. Смена значения требует переобучения (следующий
# запуск main.py / place_orders обучит с новым горизонтом автоматически).
WEEK_HORIZON_DAYS: int = int(os.getenv("WEEK_HORIZON_DAYS", "5"))

# WEEK_TARGET_COVERAGE — целевое покрытие недельного коридора. Сырые квантили
# q0.1/q0.9 на недельном горизонте недокрывают (экстремумы за 5 дней шире, чем
# модель им выучила) — конформная поправка расширяет коридор на held-out хвосте
# до этой цели (см. forecast._conformal_week_margin).
WEEK_TARGET_COVERAGE: float = float(os.getenv("WEEK_TARGET_COVERAGE", "0.80"))

# INTRADAY_ADJUST — обучаемая внутридневная поправка (Шаг 1): при открытом рынке
# остаток дневного хода (сейчас→close) предсказывается моделью от времени запуска
# τ и реализованного движения (из market_data_5m), а не арифметикой remaining =
# predicted − realized. =0 → прежняя арифметика. Валидация покрытия по времени
# суток пишется в лог (см. intraday.log_report).
INTRADAY_ADJUST: bool = get_bool_env("INTRADAY_ADJUST", 1)
INTRADAY_LOOKBACK_DAYS: int = int(os.getenv("INTRADAY_LOOKBACK_DAYS", "60"))
INTRADAY_BUCKETS: int = int(os.getenv("INTRADAY_BUCKETS", "6"))

# ORDER_FILL_WAIT_SEC — сколько секунд ждать исполнения только что выставленных
# лимиток ПЕРЕД привязкой стопов (в едином прогоне --top-n). При entry_frac≈0.2
# заявки стоят у рынка и заливаются за секунды; 0 — не ждать (стопы привяжет
# следующий прогон / --attach-stops). ORDER_FILL_POLL_SEC — интервал опроса.
# 60 секунд вместо 30: медиана заливки — 6-й пятиминутный бар (≈30 минут после
# открытия), так что единый прогон всё равно почти всегда уходит, не дождавшись;
# 60 с ловит хотя бы те заявки, что заливаются сразу.
ORDER_FILL_WAIT_SEC: float = float(os.getenv("ORDER_FILL_WAIT_SEC", "60"))
ORDER_FILL_POLL_SEC: float = float(os.getenv("ORDER_FILL_POLL_SEC", "5"))

# --- Актуальные котировки на момент запуска ----------------------------------
# TFT_USE_LIVE_PRICE=1 — перед расчётом получить последнюю цену (Last Price) по
# каждому тикеру и якорить прогноз диапазона/PnL на ней, а не на вчерашнем
# закрытии. =0 — считать на последней цене закрытия (Previous Close).
TFT_USE_LIVE_PRICE: bool = get_bool_env("TFT_USE_LIVE_PRICE", 1)
# Возраст котировки (сек), после которого выводится предупреждение об устаревании.
TFT_STALE_SECONDS: int = int(os.getenv("TFT_STALE_SECONDS", "900"))

# --- Max Pos ₽ (максимальный размер позиции по ликвидности) -------------------
# Допустимое ценовое воздействие (доля): сколько можно «сдвинуть» цену входом.
TFT_IMPACT_TOL: float = float(os.getenv("TFT_IMPACT_TOL", "0.005"))   # 0.5%
# Максимальная доля участия в среднедневном рублёвом обороте.
TFT_PARTICIPATION: float = float(os.getenv("TFT_PARTICIPATION", "0.01"))  # 1%
# Глубина окна (дней) для оценки ликвидности.
TFT_LIQUIDITY_DAYS: int = int(os.getenv("TFT_LIQUIDITY_DAYS", "60"))

# --- Торговый календарь (weekend trading) ------------------------------------
# В market_data есть бары за субботы/воскресенья (сессии выходного дня): объём
# в них примерно в 8 раз ниже будничного, и они искажают ATR, EWMA и z-оценки
# объёма, а шаг модели (H торговых дней) расходится с окном Пн–Пт в шапке
# дашборда.
#   0 (по умолчанию) — строгий биржевой календарь: бары выходных отбрасываются
#       при формировании датасета валидации и TFT, дашборд считает окно по Пн–Пт.
#   1 — торговля 7 дней в неделю: бары выходных остаются, а торговые дни
#       определяются по фактическому наличию торгов в БД (trading_calendar).
# Смена режима меняет датасет — модель переобучается на следующем прогоне.
INCLUDE_WEEKEND_TRADING: int = int(os.getenv("INCLUDE_WEEKEND_TRADING", "0"))

# --- Dashboard 2.0: рыночный контекст и риск-фильтры -------------------------
# Жёсткий рыночный фильтр: запрещать LONG при BEAR и SHORT при BULL.
STRICT_MARKET_FILTER: bool = get_bool_env("STRICT_MARKET_FILTER", 0)

# --- Спринт 2: детоксикация рейтинга и гейт валидации -------------------------
# Основание — квант-аудит (AUDIT-PROFITABILITY-REPORT.md, разделы 3 и 4).
# Замер Rank IC по walk-forward реплею (456 дат, ~182 бумаги в кросс-секции):
#
#   сигнал          вес      Rank IC     t      вывод
#   exp_score       0.30     +0.0544   +2.99    единственный работающий
#   prob_score      0.15     +0.0404   +2.32    значим, но corr с exp = 0.881
#   liq_score       0.15     +0.0020   +0.84    шум
#   rs_score        0.10     -0.0020   -0.15    шум, а на long_overnight ВРЕДИТ
#                                               (IC -0.0371, t -4.30)
#   regime_score    0.10     +0.0088   +0.26    неотличим от нуля: принимает
#                                               2 значения в день (LONG/SHORT)
#   vol_score       0.05     -0.0025   -1.06    шум
#   FinalScore       —       +0.0262   +1.30    ШУМ по порогу |t| >= 2
#
# Обёртка разбавляет единственный сигнал пятью шумовыми: 0.054 → 0.026.

# SCORE_MODE — чем ранжировать кандидатов. Основной переключатель Спринта 2.
#   "heuristic"  (ДЕФОЛТ) exp 0.70 / prob 0.20 / liq 0.10, всё в [0,1];
#   "raw_alpha"  Score = exp_pnl (сырая квантильная альфа модели);
#   "trade_score" Score = exp_pnl / ATR% — нормировка прогноза на волатильность.
#
# Почему дефолт heuristic, а не raw_alpha при БОЛЬШЕМ Rank IC у последней:
# высокий кросс-секционный IC не означает прибыльный Top-K портфель. IC мерит
# правильность рангов по всей сетке из 46 бумаг, а торгуем мы только топ-10 —
# важно поведение хвоста, а не середины. Сырой exp_pnl — абсолютная доходность
# в процентах, и её МОДУЛЬ связан с волатильностью (корреляция Спирмена с
# шириной прогнозного коридора +0.285), поэтому топ-10 по сырой альфе
# систематически набирается из бумаг с широким размахом. Компоненты prob и liq
# в эвристике работают неявным регуляризатором: медианный балл ликвидности в
# топ-10 у эвристики 69.6 против 54.3 у сырой альфы.
#
# Замер на walk-forward реплее (455 дней, нетто реальных издержек, без
# риск-штрафов), audit/out/penalty_ablation.csv:
#
#   режим                 Rank IC    t     Sharpe   CAGR     альфа
#   эвристика 70/20/10    +0.051   +2.81   -0.035   -2.8%    +2.1%   <- лучший
#   сырая альфа           +0.054   +2.98   -0.222   -6.6%    -2.5%
#   TradeScore / ATR%     +0.051   +2.86   -0.293   -7.3%    -5.2%
#
# Обратите внимание: IC у всех трёх практически одинаков (0.051-0.054), а CAGR
# расходится втрое. На этой выборке IC почти не информативен для выбора режима.
#
# Устойчивость: при расколе выборки пополам эвристика первая в ОБЕИХ половинах,
# порядок двух других меняется местами. Это единственный вывод, который
# переживает раскол, — абсолютные величины сильно зависят от периода
# (все режимы теряют в 1-й половине и зарабатывают во 2-й).
SCORE_MODE: str = os.getenv("SCORE_MODE", "heuristic").strip().lower()

# Обратная совместимость: USE_RAW_ALPHA_SCORE=1 эквивалентен SCORE_MODE=raw_alpha.
# Явно заданный SCORE_MODE имеет приоритет.
USE_RAW_ALPHA_SCORE: bool = get_bool_env("USE_RAW_ALPHA_SCORE", 0)
if USE_RAW_ALPHA_SCORE and not os.getenv("SCORE_MODE"):
    SCORE_MODE = "raw_alpha"

# APPLY_RISK_PENALTIES — применять ли мультипликативные риск-штрафы
# (High Risk Short, климакс объёма, риск гэпа) и штраф за контртренд.
#
# ДЕФОЛТ 0 — штрафы ВЫКЛЮЧЕНЫ. На walk-forward реплее (455 дней) они ухудшают
# результат во всех трёх режимах ранжирования:
#
#   режим         штрафы вкл   штрафы выкл   разница по альфе
#   heuristic       -1.2%        +2.1%          +3.3 п.п.
#   raw_alpha       -2.2%        -6.4%          -4.2 п.п.
#   trade_score     -3.8%        -2.9%          +0.9 п.п.
#
# Жёсткое отсечение бумаг по тем же условиям (RAW_ALPHA_HARD_EXCLUDE) тоже
# проверено и тоже вредит: альфа эвристики падает с +2.1% до -1.2%, причём
# в ОБЕИХ половинах выборки (-16.7 → -21.4 и +20.3 → +18.6).
#
# СЛЕДСТВИЕ, которое надо понимать: при штрафах=0 и режиме heuristic условия
# «климакс объёма» и «риск гэпа» не влияют на отбор ВООБЩЕ — ни штрафом, ни
# отсечением. Флаги в таблице остаются, но они теперь чисто информационные.
#
# ОГОВОРКА: вывод получен сравнением девяти вариантов на одной выборке в
# 455 дней и сам подвержен переобучению на отбор. Устойчиво лишь то, что ни
# один риск-фильтр не показал пользы ни в одной половине выборки.
APPLY_RISK_PENALTIES: bool = get_bool_env("APPLY_RISK_PENALTIES", 0)

# INTRADAY_SQUARE_OFF_ENABLED — принудительно закрывать внутридневные позиции
# (intraday_long / intraday_short) перед закрытием основной сессии.
#
# ВНИМАНИЕ: ФЛАГ ПОКА НИ К ЧЕМУ НЕ ПОДКЛЮЧЁН. В services/place_orders.py нет
# логики закрытия по концу сессии: позиция выходит только по STOP_LOSS или
# TAKE_PROFIT. Значение 1 здесь фиксирует ЦЕЛЕВОЕ поведение, а не текущее —
# сам по себе флаг сейчас ничего не делает.
#
# Зачем он нужен (AUDIT-PROFITABILITY-REPORT.md, §6.1): на симуляции 5-минутного
# пути цены 79,1% внутридневных позиций не достигают ни стопа, ни цели и
# переносятся через ночь. Последствия два:
#   * торгуется не та стратегия, которую валидировали: intraday_long определена
#     как close/open - 1, а с переносом реализуется close_{t+1}/open_t - 1;
#   * шорт через ночь стоит 0,0575% (ключевая ставка + маржа брокера), и этой
#     статьи нет ни в VALIDATION_COST_RT, ни в TFT_COST_RT.
# Цена дефекта на реплее — 14,6 п.п. итоговой доходности (-17,8% против -32,4%)
# и удвоение отрицательного Sharpe.
INTRADAY_SQUARE_OFF_ENABLED: bool = get_bool_env("INTRADAY_SQUARE_OFF_ENABLED", 1)

# Время закрытия внутридневных позиций (МСК). Аукцион закрытия основной сессии
# идёт 18:40–18:50, поэтому выходить нужно до него. 18:20, а не 18:35: фаза
# OVERNIGHT Этапа 2 выставляет ночные заявки в 18:35, и при cleanup в 18:35
# овернайт попадал бы в последние три минуты сессии — худшая ликвидность дня.
INTRADAY_SQUARE_OFF_TIME: str = os.getenv("INTRADAY_SQUARE_OFF_TIME", "18:20")

# В режиме сырой альфы жёстко отсекать бумаги с климаксом объёма (VolSpike > 4)
# и высоким риском гэпа вниз (>0.5 для long_overnight), а не просто штрафовать.
RAW_ALPHA_HARD_EXCLUDE: bool = get_bool_env("RAW_ALPHA_HARD_EXCLUDE", 1)

# STRICT_VALIDATION_GATE=1 — не торговать стратегии с вердиктом REJECTED.
# ВНИМАНИЕ: на момент аудита REJECTED имеют 138 комбинаций из 138, поэтому
# включение гейта останавливает торговлю полностью. Это осознанное состояние:
# дефолт 0 сохраняет прежнее поведение, но теперь при торговле отвергнутыми
# стратегиями печатается явное предупреждение (см. combined.warn_unvalidated).
# Включать после перекалибровки валидатора на реальных издержках.
STRICT_VALIDATION_GATE: bool = get_bool_env("STRICT_VALIDATION_GATE", 0)

# --- Путь А: специализация TFT ------------------------------------------------
# Идея: перестать требовать от модели универсальности и оставить только то,
# где есть измеренное преимущество. Основание — walk-forward реплей, 455 дней.
#
# Фактическая доходность стратегий (нетто издержек, по всей выборке):
#
#   стратегия        сделок    средняя      t       вывод
#   intraday_long     20727    -0.2393   -17.26   систематически убыточна
#   intraday_short    20727    -0.0217    -1.57   около нуля
#   long_overnight    20727    -0.0751   -16.60   убыточна БЕЗ отбора,
#                                                 но Rank IC 0.058 (t 7.42) —
#                                                 ранжирует хорошо, уровень плохой

# TRADING_STRATEGIES — что РАЗРЕШЕНО торговать. Отдельно от VALIDATION_STRATS:
# контур валидации продолжает считать все три (иначе мы перестанем видеть, что
# происходит с исключённой стратегией), а в стакан уходят только эти.
#
# intraday_long исключена: её удаление поднимает CAGR портфеля с -2.8% до +0.5%
# и альфу с +2.2% до +4.9%, причём Sharpe улучшается в ОБЕИХ половинах выборки
# (H1 -0.690 -> -0.411, H2 +0.522 -> +0.609). Это самый устойчивый результат
# Пути А.
TRADING_STRATEGIES: list[str] = (
    os.getenv("TRADING_STRATEGIES").split() if os.getenv("TRADING_STRATEGIES")
    else ["long_overnight", "intraday_short"]
)

# INTRADAY_SHORT_REQUIRE_MOMENTUM — шортить интрадей только в дни
# подтверждённого импульса продавцов: вчерашний день закрылся падением
# И волатильность рынка выше медианы.
#
# ВНИМАНИЕ, ЭТО НЕ АЛЬФА. Измерение: в дни срабатывания фильтра равновзвешенный
# рынок на следующий день даёт -0.2100% против -0.0017% в остальные дни. То
# есть шорт в эти дни зарабатывает +52.9% годовых ОДНОЙ ЛИШЬ БЕТОЙ, что больше
# всей доходности стратегии (+15.3%). Фильтр — это тайминг рыночного падения,
# а не отбор бумаг. На выборке из одного медвежьего рынка он выглядит отлично
# (Sharpe 1.51, положителен в обеих половинах), но его знак зависит от того,
# продолжится ли режим. Включён по постановке задачи; выключать при смене
# режима рынка.
SELLER_MOMENTUM_SHORT_ENABLED: bool = get_bool_env(
    "SELLER_MOMENTUM_SHORT_ENABLED",
    get_bool_env("INTRADAY_SHORT_REQUIRE_MOMENTUM", 1))   # старое имя — обратная совместимость
INTRADAY_SHORT_REQUIRE_MOMENTUM: bool = SELLER_MOMENTUM_SHORT_ENABLED  # алиас

# SHORT_IMOEX_MAX_TREND — предохранитель от шорт-сквиза: не шортить интрадей,
# когда индекс выше своей EMA50. Пустое значение отключает предохранитель.
#
# ЭТО СТРАХОВКА, А НЕ ИСТОЧНИК ДОХОДНОСТИ. На выборке из одного медвежьего
# рынка фильтр СТОИТ около 3 п.п. CAGR (12.4% -> 9.3%, Sharpe 0.661 -> 0.530,
# альфа 26.5% -> 23.0%) и не улучшает просадку (-22.0% в обоих случаях) —
# он убирает только прибыльные шорт-дни, потому что в этой выборке рост
# индекса выше EMA50 (34.3% дней) почти всегда оказывался коротким отскоком.
# Смысл фильтра — защита в режиме, которого в данных НЕТ: устойчивый бычий
# тренд, где шорты выносит. Премия за страховку видна, выплата — нет.
SHORT_IMOEX_MAX_TREND: str = os.getenv("SHORT_IMOEX_MAX_TREND", "EMA50").strip()

# OVERNIGHT_MAX_MARKET_ATR_PCTL — не покупать овернайт при панической
# волатильности рынка (медиана ATR-перцентиля по вселенной выше порога).
# Влияние на выборке минимально (CAGR 12.363% -> 12.347%): порог задевает
# 13.5% строк, а овернайт — небольшая часть книги. Оставлен как дешёвый
# предохранитель против покупки через ночь в момент максимального разброса гэпов.
OVERNIGHT_MAX_MARKET_ATR_PCTL: float = float(
    os.getenv("OVERNIGHT_MAX_MARKET_ATR_PCTL", "70"))

# OVERNIGHT_MIN_EDGE_X_COST — торговать long_overnight только когда ожидаемая
# доходность превышает издержки round-trip в k раз (ExpPnL уже НЕТТО издержек,
# так что это порог сверх безубыточности).
#
# ДОКАЗАТЕЛЬНОЙ БАЗЫ НЕТ. По порогу на cost_rt t-статистика средней доходности
# нигде не превышает 1.0, а Sharpe второй половины выборки отрицателен при
# ЛЮБОМ пороге (k=0.25: -1.00; k=0.5: -0.31; k=0.75: -1.61). Отбор срезает
# выборку до 0.7-1.6% сигналов (144-326 сделок за два года). Порог введён по
# постановке задачи; считать его работающим фильтром пока нельзя.
OVERNIGHT_MIN_EDGE_X_COST: float = float(os.getenv("OVERNIGHT_MIN_EDGE_X_COST", "0.5"))

# --- Контур по умолчанию и сайзинг --------------------------------------------
# TRADING_MODE — контур, в котором работает система, если не передан явный флаг.
# "sandbox" (дефолт) — виртуальные деньги; "prod" — боевой счёт.
# CLI-флаг --prod имеет приоритет над этой переменной; обратного флага нет,
# то есть переменная может только ОГРАНИЧИТЬ, но не расширить права запуска.
TRADING_MODE: str = os.getenv("TRADING_MODE", "sandbox").strip().lower()

# FIXED_POSITION_OVERFLOW_MODE — что делать, когда один лот дороже лимита позиции.
#   "skip"             — бумага пропускается (дефолт и единственное поведение);
#   "force_min_1_lot"  — НЕ ПОДДЕРЖИВАЕТСЯ, оставлено только чтобы явно
#                        зафиксировать отказ от него.
# Прежний max(1, ...) покупал БОЛЬШЕ, чем посчитала модель (при лоте GMKN
# 1296 ₽ и лимите 500 ₽ — в 2.6 раза), обходя риск-сайзинг и создавая риск
# овердрафта. Значение, отличное от "skip", вызывает ошибку на старте.
FIXED_POSITION_OVERFLOW_MODE: str = os.getenv(
    "FIXED_POSITION_OVERFLOW_MODE", "skip").strip().lower()
if FIXED_POSITION_OVERFLOW_MODE != "skip":
    raise ValueError(
        f"FIXED_POSITION_OVERFLOW_MODE={FIXED_POSITION_OVERFLOW_MODE!r} не поддержан. "
        "Допустимо только 'skip': округление вверх до одного лота обходит "
        "риск-сайзинг и может привести к овердрафту.")


# --- Этап 2: функциональный тест на демо-счёте (STAGE2-DEMO-TZ.md) -----------
# Оркестратор четырёх фаз: PREP (расчёт и заморозка плана) → ORDER (исполнение
# замороженного плана) → CLEANUP (закрытие интрадея) → OVERNIGHT (ночные заявки).
# Времена берутся ОТСЮДА, а не из планировщика: cron только вызывает фазу.
STAGE2_ENABLED: bool = get_bool_env("STAGE2_ENABLED", 1)
STAGE2_TEST_ID: str = os.getenv("STAGE2_TEST_ID", "stage2-demo-15d")
STAGE2_TARGET_DAYS: int = int(os.getenv("STAGE2_TARGET_DAYS", "15"))

# STAGE2_TRADE_TARGET — счётчик сделок для зачётных условий (турнир и т.п.).
# 0 (по умолчанию) — счётчик выключен, в карточке Telegram не показывается.
# Считаются ЗАЛИТЫЕ (filled) строки execution_audit с asof_date >= дня старта
# теста, отдельно по каждому контуру (SANDBOX/PROD — колонка account_env),
# чтобы параллельные прогоны (например, песочница r4 и боевой турнирный
# профиль) не путали друг другу счётчик. Решение пользователя 23.09.2026.
STAGE2_TRADE_TARGET: int = int(os.getenv("STAGE2_TRADE_TARGET", "0"))

# STAGE2_NOTIFY_LABEL — метка в шапке карточки Telegram (например «ТУРНИР»).
# "" (по умолчанию) — ничего не добавляется, карточка выглядит как сейчас.
# Нужна, когда несколько параллельных контуров (песочница r4 и боевой
# турнирный профиль) шлют уведомления в ОДИН И ТОТ ЖЕ чат одним и тем же
# ботом — решение пользователя 23.09.2026 не заводить отдельный канал.
STAGE2_NOTIFY_LABEL: str = os.getenv("STAGE2_NOTIFY_LABEL", "").strip()

# С 14.09.2026 Мосбиржа открывается аукционом в 09:00, непрерывные торги с 09:10
# (T-Invest TradingSchedules). Утро сдвинуто вслед за открытием с тем же шагом,
# что при открытии в 10:00: PREP — до аукциона (план от вчерашнего закрытия, как
# задумано), CLOSE — первая минута торгов, ORDER — через 5 минут после CLOSE.
STAGE2_PREP_TIME: str = os.getenv("STAGE2_PREP_TIME", "08:45")
STAGE2_ORDER_TIME: str = os.getenv("STAGE2_ORDER_TIME", "09:15")
STAGE2_CLEANUP_TIME: str = os.getenv("STAGE2_CLEANUP_TIME", "18:20")
STAGE2_OVERNIGHT_TIME: str = os.getenv("STAGE2_OVERNIGHT_TIME", "18:35")

# Между PREP и ORDER проходит 20 минут, за которые догружаются 5-минутки.
# 0 — ORDER отказывается исполнять план, если датасет изменился (по умолчанию:
# план должен исполняться ровно на тех данных, на которых считался).
STAGE2_ALLOW_DATASET_DRIFT: bool = get_bool_env("STAGE2_ALLOW_DATASET_DRIFT", 0)

# Критический FAIL останавливает тест: следующие фазы не выполняются до
# ручного снятия блокировки командой `stage2_demo resume`.
STAGE2_HALT_ON_FAIL: bool = get_bool_env("STAGE2_HALT_ON_FAIL", 1)

STAGE2_START_BALANCE_RUB: float = float(os.getenv("STAGE2_START_BALANCE_RUB", "100000"))

# Каталог результатов теста. Ничего отсюда не удаляется, в том числе по завершении.
STAGE2_DIR: str = os.getenv(
    "STAGE2_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "audit", "stage2-demo"))


# ── Уведомления в Telegram (services/notify.py) ───────────────────────────────
# Транспорт отчётности Этапа 2: статус каждой фазы и дневная сводка приходят
# в чат сразу после отработки крона.
#
# TELEGRAM_BOT_TOKEN — СЕКРЕТ. Живёт только в .env (chmod 600, вне git), сюда
# не подставляется дефолтом сознательно: пустой токен просто выключает
# уведомления, а не роняет запуск.
#
# TELEGRAM_CHAT_ID — кому слать. Токен говорит, ОТ ЧЬЕГО имени, но не КОМУ:
# без chat_id отправка невозможна. Узнать: написать боту любое сообщение и
# вызвать getUpdates.
#
# TELEGRAM_SILENT_PHASES=1 — рутинные PASS приходят без звука; FAIL и дневная
# сводка звучат всегда.
TELEGRAM_ENABLED: bool = get_bool_env("TELEGRAM_ENABLED", 0)
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_SILENT_PHASES: bool = get_bool_env("TELEGRAM_SILENT_PHASES", 1)


# ── Сбор новостей из Telegram-каналов (services/news_tg.py) ───────────────────
# Публичное веб-превью t.me/s/<канал>: авторизация не нужна вообще, в отличие
# от Bot API (требует прав админа в канале) и MTProto (api_id + вход по номеру).
#
# ЭТО КОНТУР СБОРА. Данные складываются в схему news и НЕ участвуют в скоринге,
# отборе бумаг и постановке заявок — решение вынести новости из торгового
# пайплайна (contrib/experimental_news) остаётся в силе.
#
# NEWS_TG_MAX_PAGES — потолок страниц за один опрос. При опросе раз в 10 минут
# хватает одной (20 постов ≈ 2 часа канала); запас нужен, чтобы подобрать дыру
# после простоя сервера, но не выкачать весь архив при пустой таблице.
NEWS_TG_ENABLED: bool = get_bool_env("NEWS_TG_ENABLED", 0)
NEWS_TG_CHANNELS: str = os.getenv("NEWS_TG_CHANNELS", "markettwits")
NEWS_TG_MAX_PAGES: int = int(os.getenv("NEWS_TG_MAX_PAGES", "3"))
NEWS_TG_TIMEOUT: int = int(os.getenv("NEWS_TG_TIMEOUT", "20"))
NEWS_TG_USER_AGENT: str = os.getenv(
    "NEWS_TG_USER_AGENT", "Mozilla/5.0 (compatible; etl-tcs/1.0)")


# ── Хотфиксы 11.09: выход из овернайта, стопы, лимиты брокера ─────────────────
# MIN_STOP_PCT — пол расстояния стопа от входа, %. Когда q0.10 прогноза выше
# издержек, Downside положительный, и прежняя формула ставила стоп лонга ВЫШЕ
# входа (ENPG 10.09: вход 310,81 → стоп 312,47). Теперь стоп всегда на
# убыточной стороне и не ближе этого пола, как бы ни был оптимистичен квантиль.
MIN_STOP_PCT: float = float(os.getenv("MIN_STOP_PCT", "1.0"))

# STAGE2_CLOSE_TIME — утреннее закрытие позиций long_overnight (фаза CLOSE).
# Первая минута непрерывных торгов после аукциона открытия 09:00–09:10:
# рыночная заявка в аукционе не нужна. До 14.09.2026 было 10:00.
STAGE2_CLOSE_TIME: str = os.getenv("STAGE2_CLOSE_TIME", "09:10")

# STAGE2_PARK_TIME — казначейство после открытия СПБ (фаза PARK). TMON@ торгуется
# через API только на СПБ бирже, ETF-сессия с 10:00 МСК; до этого листинг закрыт
# для API, и CLOSE в 09:10 паркует виртуально. 10:05, а не 10:00: в первую
# минуту бывают выбросы цены (15.09: 161,00–165,20 при цене 164,3).
STAGE2_PARK_TIME: str = os.getenv("STAGE2_PARK_TIME", "10:05")

# STAGE2_START_DATE — день, с которого считается тест (ISO). До него фазы не
# выполняются и даже каталог пропуска не создают. Пусто — без ограничения.
STAGE2_START_DATE: str = os.getenv("STAGE2_START_DATE", "")

# Лимиты запросов брокера. 10.09 четыре PostSandboxOrder ушли в одну секунду,
# и брокер отбил два по HTTP 429. Между заявками — минимальный интервал, на
# 429 — повтор с паузой 1 → 2 → 4 с.
BROKER_ORDER_MIN_INTERVAL_SEC: float = float(os.getenv("BROKER_ORDER_MIN_INTERVAL_SEC", "0.25"))
BROKER_429_RETRIES: int = int(os.getenv("BROKER_429_RETRIES", "3"))

# Аудит r4 (17.09). CLEANUP ждёт, пока позиция у брокера обнулится после
# рыночного закрытия; реестр стопов блокируется на время фазы (flock), protect
# при занятом реестре пропускает прогон, фаза ждёт до REGISTRY_LOCK_TIMEOUT_SEC.
SQUARE_OFF_FILL_TIMEOUT_SEC: float = float(os.getenv("SQUARE_OFF_FILL_TIMEOUT_SEC", "30"))
REGISTRY_LOCK_TIMEOUT_SEC: float = float(os.getenv("REGISTRY_LOCK_TIMEOUT_SEC", "600"))


# =====================================================================
# КАЗНАЧЕЙСТВО И ДЕНЕЖНЫЙ РЫНОК (TMON) — services/treasury.py
# =====================================================================
# Свободный кэш сверх буфера утром паркуется в фонд денежного рынка, вечером
# паи продаются ровно под ночные лонги. Интрадей-шорты паи не трогают.
TREASURY_ENABLED = get_bool_env("TREASURY_ENABLED", default=True)
TREASURY_TICKER = os.getenv("TREASURY_TICKER", "TMON")
# Класс листинга. Пусто — первый листинг фонда, доступный через API: у TMON на
# Мосбирже (TQBR, TQTF) apiTradeAvailableFlag=false, через API торгуется TMON@
# на СПБ бирже (SPBRU). Недоступный класс в песочнице → виртуальный реестр.
# «# …» отрезается: python-dotenv 1.2 при пустом значении с комментарием в той
# же строке отдаёт сам комментарий как значение (14.09 листинг так и не нашёлся).
TREASURY_CLASS_CODE = os.getenv("TREASURY_CLASS_CODE", "").split("#", 1)[0].strip()
# Неснижаемый остаток свободного кэша на комиссии, округление лотов и вариационку.
# С r4 — 30 000 ₽ при депозите 5 млн: песочница при постановке заявки резервирует
# 0,501 % сверх её суммы (замерено 16.09.2026 двоичным поиском: при кэше 5 000 000 ₽
# максимальная принимаемая заявка 4 975 050 ₽). При буфере 5 000 ₽ парковка всего
# кэша отклонялась с «Not enough balance». Минимум для 5 млн — ~23 100 ₽, взято с запасом.
TREASURY_CASH_BUFFER_RUB = float(os.getenv("TREASURY_CASH_BUFFER_RUB", "30000.0"))
# Минимальная сумма для покупки TMON (не гонять ордера ради 100 рублей; с r4 — 10 000 ₽)
TREASURY_MIN_SWEEP_RUB = float(os.getenv("TREASURY_MIN_SWEEP_RUB", "10000.0"))
# Фаза PARK покупает паи только лимитной заявкой: не дороже последней сделки +
# этот процент (округление вниз к шагу цены). Рыночная заявка на открытии СПБ
# собрала бы выбросы первой минуты.
TREASURY_LIMIT_MAX_PREMIUM_PCT = float(os.getenv("TREASURY_LIMIT_MAX_PREMIUM_PCT", "0.05"))
