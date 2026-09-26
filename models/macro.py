"""
DDL и SQL для таблицы macro_rates — безрисковая ставка.

Зачем нужна. Робот паркует свободные деньги в фонд денежного рынка (TMON@), и
доходность фонда идёт примерно за ключевой ставкой. Значит любая торговая
стратегия контура соревнуется не с нулём, а со ставкой: при ключевой 14 %
позиция, простоявшая ночь впустую, стоила 0,038 % альтернативного дохода.

Без этой строки в отчёте «плюс за день» читается как успех, хотя деньги в фонде
дали бы столько же без риска. Поэтому ставка — не признак модели (макро как
предиктор доходностей закрыто литературой и нашими же тестами), а ЗНАМЕНАТЕЛЬ:
она задаёт планку, выше которой торговля вообще имеет смысл.

Наполняется services/load_macro.py из SOAP-сервиса ЦБ РФ (DailyInfo.asmx,
метод KeyRate). Ставка меняется решениями совета директоров, то есть рядом
ступенек: между заседаниями значение одно и то же, и последнее известное
значение действует до следующего заседания.
"""

CREATE_MACRO_RATES_SQL = """
CREATE TABLE IF NOT EXISTS macro_rates (
    date         DATE           PRIMARY KEY,
    key_rate_pct NUMERIC(8, 4)  NOT NULL,
    source       VARCHAR(16)    NOT NULL DEFAULT 'cbr',
    fetched_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
"""

UPSERT_MACRO_RATE_SQL = """
INSERT INTO macro_rates (date, key_rate_pct, source, fetched_at)
VALUES (%(date)s, %(key_rate_pct)s, %(source)s, NOW())
ON CONFLICT (date) DO UPDATE SET
    key_rate_pct = EXCLUDED.key_rate_pct,
    source       = EXCLUDED.source,
    fetched_at   = NOW();
"""

# Ставка, действующая на дату: последнее известное значение НЕ ПОЗЖЕ этой даты.
# Именно так ставка и работает — ступенькой до следующего заседания.
SELECT_KEY_RATE_ASOF_SQL = """
SELECT date, key_rate_pct FROM macro_rates
WHERE date <= %s ORDER BY date DESC LIMIT 1;
"""

# Средняя ставка за период — для пересчёта безрисковой доходности отрезка.
SELECT_KEY_RATE_AVG_SQL = """
SELECT avg(key_rate_pct) FROM macro_rates WHERE date BETWEEN %s AND %s;
"""
