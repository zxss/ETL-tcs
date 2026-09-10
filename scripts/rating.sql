-- Рейтинг бумаг со скорингом за последний посчитанный день.
--
-- Источник — таблица forecasts, которую наполняет фаза PREP при SAVE_FORECASTS=1.
-- Это снимок того, что модель посчитала, а НЕ список того, что будет куплено:
-- колонка «торгуемость» показывает, какие ворота строка не проходит.
--
-- Запуск:
--   psql -d $DB_NAME -f scripts/rating.sql
--
-- Важное ограничение: предохранитель шорт-сквиза (SHORT_IMOEX_MAX_TREND) здесь
-- не воспроизводится — признак index_above_ema50 в forecasts не сохраняется.
-- Поэтому строка intraday_short с пометкой «проходит ворота таблицы» всё равно
-- может быть отсеяна специализацией. Единственный источник истины о том, что
-- реально пойдёт в стакан, — plan.json соответствующего запуска PREP.

\pset border 2

WITH params AS (
    -- Держать в соответствии с .env: при их изменении колонка «торгуемость» врёт.
    SELECT 0.128::numeric AS cost_rt,      -- TFT_COST_RT
           0.5::numeric   AS overnight_k   -- OVERNIGHT_MIN_EDGE_X_COST
),
last AS (SELECT max(asof_date) AS d FROM forecasts)
SELECT
    row_number() OVER (ORDER BY f.final_score DESC NULLS LAST)  AS "#",
    f.asof_date                                                 AS "дата",
    f.ticker                                                    AS "тикер",
    f.strategy                                                  AS "стратегия",
    f.raw_payload->>'direction'                                 AS "напр",
    round(f.final_score, 3)                                     AS "скоринг",
    round(f.exp_pnl, 3)                                         AS "ExpPnL%",
    round(f.prob_profit, 3)                                     AS "P(проф)",
    coalesce(f.verdict, '—')                                    AS "вердикт",
    round(f.anchor_price, 2)                                    AS "цена",
    round((f.raw_payload->>'rs')::numeric, 1)                   AS "RS",
    round((f.raw_payload->>'vol_spike')::numeric, 2)            AS "vol",
    round((f.raw_payload->>'atr_pctl')::numeric, 0)             AS "ATR%",
    coalesce(nullif(array_to_string(
        ARRAY(SELECT jsonb_array_elements_text(f.raw_payload->'flags')), ', '), ''),
        '—')                                                    AS "штрафы",
    CASE
        WHEN f.strategy = 'weekly'
            THEN 'недельный горизонт, не торгуется'
        WHEN f.strategy NOT IN ('long_overnight', 'intraday_short')
            THEN 'нет в TRADING_STRATEGIES'
        WHEN f.raw_payload->>'direction' = 'SHORT'
             AND f.ticker IN ('AKRN', 'CBOM', 'MVID')
            THEN 'шорт недоступен у брокера'
        WHEN f.strategy = 'long_overnight'
             AND f.exp_pnl <= p.overnight_k * p.cost_rt
            THEN 'ниже порога овернайта'
        ELSE 'проходит ворота таблицы'
    END                                                         AS "торгуемость"
FROM forecasts f, params p, last
WHERE f.asof_date = last.d
ORDER BY f.final_score DESC NULLS LAST;
