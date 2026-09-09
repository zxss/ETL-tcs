"""
DDL и SQL для таблицы instruments — кэш справочника T-Invest.

Зачем нужна: размер лота, шаг цены и доступность шорта нельзя держать в
хардкоде. Лотность меняется при сплитах и редомициляциях (GMKN 1:100 в апреле
2024, VTBR, FIXP→FIXR), и захардкоженная таблица неизбежно расходится с
реальностью. Сверка ручного списка _LOT_SIZES с API показала расхождение по
11 из 46 тикеров, вплоть до 10 000× (VTBR), и полное отсутствие TGKA.

Наполняется services/load_instruments.py из InstrumentsService/ShareBy.
Потребители: tft_forecast/liquidity.py (рублёвый оборот = close × volume × lot),
tft_forecast/combined.py (размер лота для инструкций по заявкам).

Источник истины в момент выставления заявки остаётся живой ответ API
(services/place_orders.py сверяет Instrument.lot); кэш нужен расчётному контуру,
который в API не ходит.
"""

CREATE_INSTRUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS instruments (
    ticker              VARCHAR(10)  PRIMARY KEY,
    figi                VARCHAR(32),
    uid                 VARCHAR(64),
    name                TEXT,
    class_code          VARCHAR(16),
    lot                 INTEGER      NOT NULL DEFAULT 1,
    min_price_increment NUMERIC(18, 9),
    short_enabled       BOOLEAN,
    buy_available       BOOLEAN,
    sell_available      BOOLEAN,
    api_trade_available BOOLEAN,
    for_qual_investor   BOOLEAN,
    dlong_client        NUMERIC(10, 6),
    dshort_client       NUMERIC(10, 6),
    trading_status      VARCHAR(48),
    sector              VARCHAR(64),
    fetched_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""

UPSERT_INSTRUMENT_SQL = """
INSERT INTO instruments (
    ticker, figi, uid, name, class_code, lot, min_price_increment,
    short_enabled, buy_available, sell_available, api_trade_available,
    for_qual_investor, dlong_client, dshort_client, trading_status, sector,
    fetched_at)
VALUES (%(ticker)s, %(figi)s, %(uid)s, %(name)s, %(class_code)s, %(lot)s,
        %(min_price_increment)s, %(short_enabled)s, %(buy_available)s,
        %(sell_available)s, %(api_trade_available)s, %(for_qual_investor)s,
        %(dlong_client)s, %(dshort_client)s, %(trading_status)s, %(sector)s,
        NOW())
ON CONFLICT (ticker) DO UPDATE SET
    figi                = EXCLUDED.figi,
    uid                 = EXCLUDED.uid,
    name                = EXCLUDED.name,
    class_code          = EXCLUDED.class_code,
    lot                 = EXCLUDED.lot,
    min_price_increment = EXCLUDED.min_price_increment,
    short_enabled       = EXCLUDED.short_enabled,
    buy_available       = EXCLUDED.buy_available,
    sell_available      = EXCLUDED.sell_available,
    api_trade_available = EXCLUDED.api_trade_available,
    for_qual_investor   = EXCLUDED.for_qual_investor,
    dlong_client        = EXCLUDED.dlong_client,
    dshort_client       = EXCLUDED.dshort_client,
    trading_status      = EXCLUDED.trading_status,
    sector              = EXCLUDED.sector,
    fetched_at          = NOW();
"""

SELECT_LOTS_SQL = """
SELECT ticker, lot, fetched_at FROM instruments;
"""

SELECT_INSTRUMENTS_SQL = """
SELECT ticker, lot, min_price_increment, short_enabled, api_trade_available,
       fetched_at
FROM instruments;
"""
