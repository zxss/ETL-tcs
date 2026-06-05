"""
DDL и SQL для таблиц market_data (дневные) и market_data_5m (5-мин).
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
    created_at TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_data_ticker_date UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date
    ON market_data (ticker, date DESC);
"""

UPSERT_SQL = """
INSERT INTO market_data (ticker, date, open, high, low, close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date) DO UPDATE SET
    open   = EXCLUDED.open,
    high   = EXCLUDED.high,
    low    = EXCLUDED.low,
    close  = EXCLUDED.close,
    volume = EXCLUDED.volume;
"""

# Запрос последней записи для инкрементального режима
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
