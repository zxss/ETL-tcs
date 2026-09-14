"""
DDL и SQL для таблиц:
  market_data      — дневные свечи
  market_data_5m   — 5-минутные свечи
  forecasts        — рассчитанные прогнозы и сигналы дашбордов
  execution_audit  — намерение → факт по каждой заявке (проскальзывание, PnL)

Схема отключённого контура новостей (news_sentiment + view news_with_candles)
живёт отдельно: contrib/experimental_news/news_schema.py — штатный init_db()
её не создаёт.
"""

# --- Дневные свечи -----------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_data (
    id         BIGSERIAL PRIMARY KEY,
    ticker     VARCHAR(10)    NOT NULL,
    date       DATE           NOT NULL,
    open       NUMERIC(18, 4) NOT NULL,
    high       NUMERIC(18, 4) NOT NULL,
    low        NUMERIC(18, 4) NOT NULL,
    close      NUMERIC(18, 4) NOT NULL,
    volume     BIGINT         NOT NULL DEFAULT 0,
    -- Происхождение бара:
    --   'api'         — официальный дневной бар биржи (значение по умолчанию
    --                   для всех новых вставок из брокерского API);
    --   'backfill_5m' — локальная интрадей-реконструкция из 5-минуток
    --                   (см. services/backfill_daily.py);
    --   'api_legacy'  — исторические данные, загруженные ДО внедрения трекинга
    --                   источника: происхождение достоверно неизвестно, среди
    --                   них могут быть реконструированные бары.
    source     VARCHAR(20)    DEFAULT 'api',
    created_at TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_data_ticker_date UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date
    ON market_data (ticker, date DESC);
"""

# Миграция для уже существующих баз (init_db выполняет её при каждом старте).
MIGRATE_MARKET_DATA_SOURCE_SQL = """
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'api';
"""

# Разметка исторического слоя: строки, созданные до внедрения трекинга, получили
# source='api' по DEFAULT, хотя часть из них могла быть реконструкцией из
# 5-минуток. Помечаем их отдельным значением, чтобы не выдавать за официальные
# бары биржи. Запускается ОДНОРАЗОВО скриптом scripts/mark_legacy_source.py
# (в init_db не входит: это разовая правка данных, а не эволюция схемы).
# Идемпотентна — повторный запуск не находит строк.
MIGRATE_MARKET_DATA_SOURCE_LEGACY_SQL = """
UPDATE market_data
SET source = 'api_legacy'
WHERE source = 'api' AND created_at < %(cutoff)s;
"""

# Допустимые значения market_data.source.
SOURCE_API = "api"                  # официальный дневной бар биржи
SOURCE_BACKFILL_5M = "backfill_5m"  # локальная реконструкция из 5-минуток
SOURCE_API_LEGACY = "api_legacy"    # история до внедрения трекинга источника

UPSERT_SQL = """
INSERT INTO market_data (ticker, date, open, high, low, close, volume, source)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date) DO UPDATE SET
    open   = EXCLUDED.open,
    high   = EXCLUDED.high,
    low    = EXCLUDED.low,
    close  = EXCLUDED.close,
    volume = EXCLUDED.volume,
    source = EXCLUDED.source;
"""

LAST_DATE_SQL = """
SELECT MAX(date) FROM market_data WHERE ticker = %s;
"""

# --- 5-минутные свечи --------------------------------------------------------
CREATE_TABLE_5M_SQL = """
CREATE TABLE IF NOT EXISTS market_data_5m (
    id         BIGSERIAL PRIMARY KEY,
    ticker     VARCHAR(10)    NOT NULL,
    ts         TIMESTAMPTZ    NOT NULL,
    open       NUMERIC(18, 4) NOT NULL,
    high       NUMERIC(18, 4) NOT NULL,
    low        NUMERIC(18, 4) NOT NULL,
    close      NUMERIC(18, 4) NOT NULL,
    volume     BIGINT         NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_data_5m_ticker_ts UNIQUE (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_market_data_5m_ticker_ts
    ON market_data_5m (ticker, ts DESC);
"""

UPSERT_5M_SQL = """
INSERT INTO market_data_5m (ticker, ts, open, high, low, close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, ts) DO UPDATE SET
    open   = EXCLUDED.open,
    high   = EXCLUDED.high,
    low    = EXCLUDED.low,
    close  = EXCLUDED.close,
    volume = EXCLUDED.volume;
"""

LAST_TS_5M_SQL = """
SELECT MAX(ts) FROM market_data_5m WHERE ticker = %s;
"""

# --- Прогнозы и сигналы дашбордов --------------------------------------------
# Одна строка = (дата расчёта, тикер, стратегия). Дневные строки дашборда пишутся
# со стратегией из VALIDATION_STRATS (long_overnight / intraday_long / ...),
# недельный дашборд — со стратегией 'weekly'.
#
# q10/q50/q90 — ЦЕНОВЫЕ квантили прогнозного коридора (₽), приведённые к
# anchor_price: для дневных строк это ForecastLow / медиана дневного total /
# ForecastHigh, для недельных — WeekLow / медиана week_total / WeekHigh.
# exp_pnl и prob_profit — нетто издержек (доли %/вероятность), final_score —
# итоговый рейтинг дашборда. Всё, что не влезло в колонки (рыночный контекст,
# метрики валидации, ликвидность), лежит в raw_payload.
CREATE_FORECASTS_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
    id BIGSERIAL PRIMARY KEY,
    asof_date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    anchor_price NUMERIC(18, 4),
    q10 NUMERIC(18, 4),
    q50 NUMERIC(18, 4),
    q90 NUMERIC(18, 4),
    exp_pnl NUMERIC(10, 4),
    prob_profit NUMERIC(6, 4),
    final_score NUMERIC(6, 4),
    verdict VARCHAR(20),
    raw_payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asof_date, ticker, strategy)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_date_ticker ON forecasts(asof_date, ticker);
"""

# Порядок/состав полей, которые принимает database.save_forecasts.
FORECAST_COLUMNS = (
    "asof_date", "ticker", "strategy", "anchor_price",
    "q10", "q50", "q90", "exp_pnl", "prob_profit", "final_score",
    "verdict", "raw_payload",
)

UPSERT_FORECAST_SQL = """
INSERT INTO forecasts (asof_date, ticker, strategy, anchor_price,
                       q10, q50, q90, exp_pnl, prob_profit, final_score,
                       verdict, raw_payload)
VALUES (%(asof_date)s, %(ticker)s, %(strategy)s, %(anchor_price)s,
        %(q10)s, %(q50)s, %(q90)s, %(exp_pnl)s, %(prob_profit)s, %(final_score)s,
        %(verdict)s, %(raw_payload)s)
ON CONFLICT (asof_date, ticker, strategy) DO UPDATE SET
    anchor_price = EXCLUDED.anchor_price,
    q10          = EXCLUDED.q10,
    q50          = EXCLUDED.q50,
    q90          = EXCLUDED.q90,
    exp_pnl      = EXCLUDED.exp_pnl,
    prob_profit  = EXCLUDED.prob_profit,
    final_score  = EXCLUDED.final_score,
    verdict      = EXCLUDED.verdict,
    raw_payload  = EXCLUDED.raw_payload,
    created_at   = NOW();
"""


# --- Аудит исполнения заявок --------------------------------------------------
# Одна строка на заявку: что модель хотела (цена, стоп, цель, ожидаемые издержки)
# и что получилось (цена заливки, фактическое проскальзывание, выход, PnL).
# Заполняется через audit/execution_audit.py: record_intent → record_fill →
# record_exit. Источник истины по исполнению; JSON в каталоге запуска Этапа 2 —
# лишь автономный снимок этой таблицы.
CREATE_EXECUTION_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS execution_audit (
    id             BIGSERIAL PRIMARY KEY,
    order_id       VARCHAR(64) NOT NULL,
    account_env    VARCHAR(10) NOT NULL,        -- SANDBOX | PROD
    asof_date      DATE        NOT NULL,        -- дата прогноза, породившего заявку
    ticker         VARCHAR(10) NOT NULL,
    strategy       VARCHAR(30) NOT NULL,
    side           VARCHAR(5)  NOT NULL,        -- BUY | SELL

    -- Намерение: что модель хотела
    final_score    NUMERIC(6, 4),
    exp_pnl_pct    NUMERIC(10, 4),              -- ожидаемый нетто-PnL, %
    anchor_price   NUMERIC(18, 4),              -- цена, на которой строился прогноз
    requested_price NUMERIC(18, 4) NOT NULL,    -- цена лимитной заявки
    expected_slippage_pct NUMERIC(10, 4),       -- расчётное проскальзывание модели
    expected_cost_pct     NUMERIC(10, 4),       -- расчётные издержки RT, %
    stop_price     NUMERIC(18, 4),
    target_price   NUMERIC(18, 4),
    qty_lots       INTEGER,
    lot_size       INTEGER,

    -- Факт: что получилось
    filled         BOOLEAN     NOT NULL DEFAULT FALSE,
    filled_price   NUMERIC(18, 4),
    filled_at      TIMESTAMPTZ,
    slippage_rub   NUMERIC(18, 4),
    slippage_pct   NUMERIC(10, 4),
    fee_rub        NUMERIC(18, 4),
    exit_price     NUMERIC(18, 4),
    exit_at        TIMESTAMPTZ,
    exit_reason    VARCHAR(20),                 -- target | stop | manual | eod
    hold_time_sec  INTEGER,
    pnl_gross_rub  NUMERIC(18, 4),
    pnl_net_rub    NUMERIC(18, 4),

    -- Этап 2: привязка к фазе оркестратора (STAGE2-DEMO-TZ §5-§7)
    run_id         VARCHAR(64),                 -- <YYYYMMDD-HHMMSS>-<PHASE>
    phase          VARCHAR(16),                 -- PREP | ORDER | CLEANUP | OVERNIGHT

    raw_payload    JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (order_id)
);
CREATE INDEX IF NOT EXISTS idx_exec_audit_date   ON execution_audit (asof_date DESC);
CREATE INDEX IF NOT EXISTS idx_exec_audit_ticker ON execution_audit (ticker, asof_date DESC);
CREATE INDEX IF NOT EXISTS idx_exec_audit_run    ON execution_audit (run_id);
"""


# --- Казначейство: сделки с фондом денежного рынка -----------------------------
# Одна строка на покупку или продажу паёв (services/treasury.py). mode='broker' —
# реальная заявка, след для аудита; mode='virtual' — виртуальное владение в
# песочнице, когда брокер отказал в заявке по фонду (ТЗ Treasury, задача 4).
# Виртуальная позиция = Σ BUY − Σ SELL по строкам mode='virtual'.
CREATE_TREASURY_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS treasury_ledger (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_env VARCHAR(10) NOT NULL,          -- SANDBOX | PROD
    account_id  VARCHAR(64) NOT NULL,
    ticker      VARCHAR(16) NOT NULL,          -- TREASURY_TICKER, не тикер листинга
    mode        VARCHAR(10) NOT NULL,          -- broker | virtual
    side        VARCHAR(5)  NOT NULL,          -- BUY | SELL
    lots        INTEGER     NOT NULL CHECK (lots > 0),
    price       NUMERIC(18, 6) NOT NULL,       -- цена лота, ₽
    amount_rub  NUMERIC(18, 4) NOT NULL,
    reason      VARCHAR(20),                   -- sweep | overnight | cover
    run_id      VARCHAR(64),
    order_id    VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS idx_treasury_ledger_account
    ON treasury_ledger (account_id, ticker, ts);
"""
