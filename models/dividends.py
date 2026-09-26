"""
DDL и SQL для таблицы dividends — календарь дивидендных отсечек.

Зачем нужна. Свечи T-Invest НЕ скорректированы на дивиденды: в день отсечки
open проваливается на величину выплаты, и этот провал попадает и в торговый
контур, и в обучающую выборку.

  * ИСПОЛНЕНИЕ. Стратегия long_overnight входит на закрытии и выходит на
    открытии, то есть вход вечером `last_buy_date` попадает ровно в гэп. По
    деньгам это примерно ноль (замер на 97 отсечках: нетто после НДФЛ и
    издержек −0,08 % в медиане, гэп закрывается не полностью). Ломается
    исполнение: ценовой гэп в медиане −4,61 %, стоп MIN_STOP_PCT = 1 % пробит в
    85 из 97 ночей, в 44 — глубже 5 %. Позиция уходит в аукционе открытия по
    неконтролируемой цене, а дивиденд приходит отдельным платежом через недели.
  * ОБУЧЕНИЕ. Признак `overnight` и цель `next_overnight` считаются как
    open/prev_close. В дни отсечек это не доходность, а дивиденд с обратным
    знаком: 0,13 % наблюдений с амплитудой в десять раз больше обычной ночной
    доходности. Модель училась на том, что высокодивидендные бумаги проваливаются
    за ночь без причины, и закладывала это в квантиль q10 — то есть в стопы.

Ключ (ticker, last_buy_date) — последний день покупки с правом на дивиденд.
Гэп приходится на открытие СЛЕДУЮЩЕГО торгового дня после last_buy_date.

Наполняется services/load_dividends.py из InstrumentsService/GetDividends.
Метод отдаёт и ОБЪЯВЛЕННЫЕ будущие выплаты, поэтому календарь работает вперёд,
а не только по истории.
"""

CREATE_DIVIDENDS_SQL = """
CREATE TABLE IF NOT EXISTS dividends (
    ticker        VARCHAR(10)    NOT NULL,
    last_buy_date DATE           NOT NULL,
    record_date   DATE,
    payment_date  DATE,
    dividend_net  NUMERIC(18, 6) NOT NULL,
    close_price   NUMERIC(18, 4),
    currency      VARCHAR(8),
    fetched_at    TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_dividends_ticker_lastbuy UNIQUE (ticker, last_buy_date)
);
CREATE INDEX IF NOT EXISTS idx_dividends_last_buy
    ON dividends (last_buy_date DESC);
"""

UPSERT_DIVIDEND_SQL = """
INSERT INTO dividends (ticker, last_buy_date, record_date, payment_date,
                       dividend_net, close_price, currency, fetched_at)
VALUES (%(ticker)s, %(last_buy_date)s, %(record_date)s, %(payment_date)s,
        %(dividend_net)s, %(close_price)s, %(currency)s, NOW())
ON CONFLICT (ticker, last_buy_date) DO UPDATE SET
    record_date  = EXCLUDED.record_date,
    payment_date = EXCLUDED.payment_date,
    dividend_net = EXCLUDED.dividend_net,
    close_price  = EXCLUDED.close_price,
    currency     = EXCLUDED.currency,
    fetched_at   = NOW();
"""

# Все выплаты по бумаге — для дивидендной поправки обучающей выборки.
SELECT_DIVIDENDS_SQL = """
SELECT ticker, last_buy_date, dividend_net, close_price
FROM dividends
WHERE ticker = %s AND dividend_net > 0
ORDER BY last_buy_date;
"""

# Отсечки на конкретный день — для торгового контура.
SELECT_DIVIDENDS_ON_SQL = """
SELECT ticker, last_buy_date, record_date, dividend_net, close_price
FROM dividends
WHERE last_buy_date = %s AND dividend_net > 0;
"""
