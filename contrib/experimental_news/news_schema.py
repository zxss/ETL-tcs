"""
DDL и SQL отключённого контура новостей (@markettwits).

Вынесено из models/market_data.py: штатный database.init_db() эти объекты
БОЛЬШЕ НЕ СОЗДАЁТ. Чтобы поднять схему при включении контура, вызовите
contrib.experimental_news.news_db.init_news_schema().

VIEW news_with_candles джойнит новости с дневными свечами market_data.
"""

# --- Новости и сентимент (TG @markettwits) -----------------------------------
CREATE_NEWS_SQL = """
CREATE TABLE IF NOT EXISTS news_sentiment (
    id          BIGSERIAL PRIMARY KEY,
    message_id  BIGINT         NOT NULL,          -- ID сообщения в Telegram
    ts          TIMESTAMPTZ    NOT NULL,           -- время публикации (UTC)
    ticker      VARCHAR(10)    NOT NULL,           -- тикер из вотчлиста
    sentiment   VARCHAR(10)    NOT NULL,           -- pos | neg | neutral
    headline    TEXT           NOT NULL DEFAULT '',-- краткая суть (≤120 симв)
    raw_text    TEXT           NOT NULL DEFAULT '',-- полный текст поста
    source      VARCHAR(64)    NOT NULL DEFAULT '@markettwits',
    created_at  TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_news_message_ticker UNIQUE (message_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_ts
    ON news_sentiment (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_news_ts
    ON news_sentiment (ts DESC);
"""

UPSERT_NEWS_SQL = """
INSERT INTO news_sentiment (message_id, ts, ticker, sentiment, headline, raw_text, source)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (message_id, ticker) DO UPDATE SET
    sentiment  = EXCLUDED.sentiment,
    headline   = EXCLUDED.headline,
    raw_text   = EXCLUDED.raw_text;
"""

LAST_MESSAGE_ID_SQL = """
SELECT MAX(message_id) FROM news_sentiment WHERE source = %s;
"""

# --- Полезный VIEW: связка новости ↔ свечи -----------------------------------
CREATE_NEWS_CANDLES_VIEW_SQL = """
CREATE OR REPLACE VIEW news_with_candles AS
SELECT
    n.ts                                   AS news_ts,
    n.ticker,
    n.sentiment,
    n.headline,
    n.message_id,
    d.date                                 AS candle_date,
    d.open, d.high, d.low, d.close, d.volume,
    -- overnight-гэп следующего дня после новости
    LEAD(d.open) OVER (PARTITION BY d.ticker ORDER BY d.date)
        / d.close - 1                      AS next_overnight_pct,
    -- интрадей в день новости
    d.close / d.open - 1                   AS intraday_pct
FROM news_sentiment n
JOIN market_data d
    ON d.ticker = n.ticker
   AND d.date = n.ts::date;
"""
