"""
Сборка баров основной сессии из 5-минуток (market_data_5m → market_data_session).

Смысл и мотивация — в models/session_bars.py. Здесь: часы основной сессии по
эпохам, SQL агрегации и точки вызова.

Часы. Метки 5-минуток — начало бара по МСК.
  до 14.09.2026: основная сессия с 10:00 (аукцион открытия 09:50–10:00 — бар 09:55
                 в бар сессии НЕ входит: робот тогда выходил в 10:00);
  с 14.09.2026:  с 09:10 (аукцион 09:00, непрерывные торги с 09:10, CLOSE в 09:10).
  закрытие:      бары, начавшиеся до 18:40, — ночной вход идёт в 18:38–18:50.
Выходные и вечер/утро в бар не входят: их движение попадает в overnight, и это
правильно — именно его робот несёт через ночь.

Вызовы:
  * main.py — после загрузки свечей (первый запуск на пустой таблице собирает
    всю историю);
  * data_freshness.ensure_fresh_data(refresh=True) — перед каждым расчётом
    заявок, чтобы в PREP был вчерашний бар, а в OVERNIGHT — сегодняшний частичный.
"""
from __future__ import annotations

import datetime as dt
import logging

from models.session_bars import CREATE_SESSION_SQL

log = logging.getLogger("session_bars")

SWITCH_DATE = dt.date(2026, 9, 14)
OPEN_BEFORE = dt.time(10, 0)
OPEN_AFTER = dt.time(9, 10)
CLOSE_CUTOFF = dt.time(18, 40)
COMPLETE_FROM = dt.time(18, 35)        # последняя 5-минутка перед закрытием
MIN_BARS = 12                          # час торгов — меньше не бар, а обрывок
RECENT_DAYS = 10                       # окно пересборки при обычном запуске


def main_open(d: dt.date) -> dt.time:
    """Начало основной сессии на дату (эпоха расписания MOEX)."""
    return OPEN_BEFORE if d < SWITCH_DATE else OPEN_AFTER


_BUILD_SQL = """
INSERT INTO market_data_session
    (ticker, date, open, high, low, close, volume, n_bars, complete, updated_at)
SELECT ticker, d,
       (array_agg(open  ORDER BY ts))[1],
       max(high), min(low),
       (array_agg(close ORDER BY ts DESC))[1],
       sum(volume), count(*),
       max(t) >= %(complete_from)s,
       NOW()
FROM (
    SELECT ticker, ts, open, high, low, close, volume,
           (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
           (ts AT TIME ZONE 'Europe/Moscow')::time AS t
    FROM market_data_5m
    WHERE ts >= %(since)s
) b
WHERE extract(isodow FROM d) < 6
  AND t >= CASE WHEN d < %(switch)s THEN %(open_before)s ELSE %(open_after)s END
  AND t < %(close_cutoff)s
GROUP BY ticker, d
HAVING count(*) >= %(min_bars)s
ON CONFLICT (ticker, date) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume, n_bars = EXCLUDED.n_bars,
    complete = EXCLUDED.complete, updated_at = NOW();
"""


# Выравнивание масштаба по сплитам. Дневные свечи брокера история приходит уже
# скорректированной на сплиты, 5-минутки — нет. 26.09.2026 это дало «ночь» VTBR
# +502 663 % (обратный сплит 5000:1 15.07.2024) и FIXR −99 % (20.08.2025), а
# pinball обучения вырос с 0,55 до 22. Блок дней, где дневное закрытие и закрытие
# сессии расходятся больше чем в полтора раза, умножается на МЕДИАННОЕ отношение
# по блоку: одно число на блок не вносит в бар вечернее движение конкретного дня.
SCALE_BAND = (1 / 1.5, 1.5)

_BREAKS_SQL = """
SELECT s.ticker, s.date, m.close / s.close AS ratio
FROM market_data_session s
JOIN market_data m USING (ticker, date)
WHERE s.close > 0 AND m.close > 0
  AND (m.close / s.close < %(lo)s OR m.close / s.close > %(hi)s)
ORDER BY s.ticker, s.date;
"""

_RESCALE_SQL = """
UPDATE market_data_session
SET open = open * %(f)s, high = high * %(f)s, low = low * %(f)s,
    close = close * %(f)s,
    volume = GREATEST(1, ROUND(volume / %(f)s))::bigint,
    updated_at = NOW()
WHERE ticker = %(tk)s AND date BETWEEN %(d0)s AND %(d1)s;
"""


def adjust_splits(conn) -> list[tuple]:
    """Выровнять блоки с другим масштабом цены. Идемпотентно: после поправки
    отношение ≈ 1, и повторный вызов этих дней не трогает."""
    import statistics
    lo, hi = SCALE_BAND
    with conn.cursor() as cur:
        cur.execute(_BREAKS_SQL, {"lo": lo, "hi": hi})
        rows = cur.fetchall()
    blocks: list[list] = []
    for tk, d, r in rows:
        r = float(r)
        last = blocks[-1] if blocks else None
        # один блок — подряд идущие дни одного тикера с одним порядком множителя
        if (last and last[0] == tk and (d - last[2]).days <= 10
                and 0.8 < r / last[3][-1] < 1.25):
            last[2] = d
            last[3].append(r)
        else:
            blocks.append([tk, d, d, [r]])
    done = []
    with conn.cursor() as cur:
        for tk, d0, d1, ratios in blocks:
            f = statistics.median(ratios)
            cur.execute(_RESCALE_SQL, {"f": f, "tk": tk, "d0": d0, "d1": d1})
            done.append((tk, d0, d1, round(f, 6), len(ratios)))
    conn.commit()
    for tk, d0, d1, f, n in done:
        log.info("Масштаб %s %s…%s (%d дн.) выровнен по дневным свечам: ×%s",
                 tk, d0, d1, n, f)
    return done


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_SESSION_SQL)
    conn.commit()


def build(conn, since: dt.date | None = None) -> int:
    """Пересобрать бары основной сессии начиная с since (None — вся история).

    Пустая таблица → вся история, иначе последние RECENT_DAYS дней. Возвращает
    число записанных баров. Идемпотентно: частичный бар дня перезаписывается.
    """
    ensure_table(conn)
    if since is None:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM market_data_session;")
            last = cur.fetchone()[0]
        since = (last - dt.timedelta(days=RECENT_DAYS)) if last else dt.date(2000, 1, 1)
    since_ts = dt.datetime.combine(since, dt.time(0, 0), tzinfo=dt.timezone.utc)
    with conn.cursor() as cur:
        cur.execute(_BUILD_SQL, {
            "since": since_ts, "switch": SWITCH_DATE,
            "open_before": OPEN_BEFORE, "open_after": OPEN_AFTER,
            "close_cutoff": CLOSE_CUTOFF, "complete_from": COMPLETE_FROM,
            "min_bars": MIN_BARS})
        n = cur.rowcount
    conn.commit()
    log.info("Бары основной сессии пересобраны с %s: %d", since, n)
    adjust_splits(conn)
    return n


def run(conn) -> int:
    """Шаг ETL: не бросает наружу — без свежих баров сработает гард свежести."""
    try:
        return build(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("Сборка баров основной сессии не удалась: %s", e)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return 0
