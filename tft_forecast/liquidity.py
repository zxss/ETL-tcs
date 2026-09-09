"""
Оценка максимально допустимого размера позиции (Max Pos ₽) — сколько рублей
можно ввести в инструмент, не сдвинув цену существенно.

Учитывается РЕАЛЬНАЯ ликвидность (ценовое воздействие на рубль оборота), а не
только объём торгов. Используется модель Amihud illiquidity по дневным OHLCV:

    ILLIQ = median_t( |ret_t| / оборот_в_рублях_t )

ILLIQ — это во сколько (в долях) сдвигается цена на каждый вложенный рубль.
Чтобы удержать воздействие в пределах допуска λ (TFT_IMPACT_TOL, напр. 0.5%):

    MaxPos_impact = λ / ILLIQ

Дополнительно ограничиваем долей участия в среднем дневном рублёвом обороте
(чтобы для сверхликвидных бумаг оценка оставалась консервативной):

    MaxPos_adv = participation × median(оборот_в_рублях)

Итог: Max Pos ₽ = min(MaxPos_impact, MaxPos_adv) — наиболее консервативная.

РАЗМЕР ЛОТА (важно)
-------------------
market_data.volume приходит из T-Invest В ЛОТАХ, а не в штуках. Рублёвый оборот
равен close × volume × lot. Раньше здесь считалось close × volume, из-за чего
оборот занижался ровно в размер лота: для TGKA (лот 100 000) — в сто тысяч раз,
для FEES (10 000) — в десять тысяч. Занижение оборота завышает ILLIQ и режет
Max Pos ₽, а также искажает балл ликвидности Liq, который весит 15% в
FinalScore: IRAO с оборотом 306 млн ₽/день попадал в один разряд с бумагами
на 3 млн ₽/день.

Лотность берётся из кэша справочника (таблица instruments, наполняется
services/load_instruments.py из InstrumentsService/ShareBy). Хардкод-таблица
для этого не годится: сверка ручного списка с API дала расхождение по 11
тикерам из 46, вплоть до 10 000× (VTBR).

Если лот тикера в кэше отсутствует, оценка по этому тикеру НЕ выдаётся
(возвращается None) и пишется предупреждение. Молча подставить lot=1 нельзя:
для TGKA это ошибка в 100 000 раз, а Max Pos ₽ ограничивает размер реальной
позиции. Отсутствие числа честнее заведомо неверного числа.

Полностью graceful: при нехватке данных по тикеру возвращается None.
"""

from __future__ import annotations

import logging
from statistics import median

import config
import database
import trading_calendar

log = logging.getLogger("tft.liquidity")

_SQL_TMPL = """
    SELECT close, volume
    FROM market_data
    WHERE ticker = %(tk)s{session_filter}
    ORDER BY date DESC
    LIMIT %(n)s;
"""


def _max_pos_for(rows, lot: int) -> dict | None:
    """rows: список (close, volume) от свежих к старым, volume В ЛОТАХ.

    lot — размер лота (число акций в лоте); рублёвый оборот = close × volume × lot.
    """
    if not lot or lot <= 0:
        return None

    closes, vols = [], []
    for c, v in rows:
        try:
            c = float(c)
            v = float(v or 0)
        except (TypeError, ValueError):
            continue
        if c > 0:
            closes.append(c)
            vols.append(v)
    if len(closes) < 20:
        return None

    # восстанавливаем хронологический порядок (был DESC)
    closes = closes[::-1]
    vols = vols[::-1]

    # Рублёвый оборот: volume в лотах → умножаем на число акций в лоте.
    dollar_vol = [c * (v * lot) for c, v in zip(closes, vols)]
    adv_rub = median([d for d in dollar_vol if d > 0] or [0.0])
    if adv_rub <= 0:
        return None

    # дневные доходности (по модулю) и Amihud по дням с ненулевым оборотом
    illiq_daily = []
    for i in range(1, len(closes)):
        dv = dollar_vol[i]
        if dv <= 0:
            continue
        ret = abs(closes[i] / closes[i - 1] - 1.0)
        illiq_daily.append(ret / dv)
    illiq = median(illiq_daily) if illiq_daily else None

    impact_tol = float(getattr(config, "TFT_IMPACT_TOL", 0.005))
    participation = float(getattr(config, "TFT_PARTICIPATION", 0.01))

    candidates = []
    if illiq and illiq > 0:
        candidates.append(impact_tol / illiq)
    candidates.append(participation * adv_rub)
    max_pos = min(candidates) if candidates else None

    return {"max_pos": max_pos, "adv_rub": adv_rub, "illiq": illiq, "lot": lot}


def compute(conn, tickers: list[str], lots: dict[str, int] | None = None) -> dict:
    """
    Возвращает {ticker: {"max_pos","adv_rub","illiq","lot"}} по тикерам, для
    которых хватило данных И известен размер лота. Никогда не бросает наружу.

    lots — {TICKER: размер_лота}. Если не передан, читается из кэша справочника
    (таблица instruments). Тикеры с неизвестным лотом пропускаются: см. модульную
    документацию о том, почему подстановка lot=1 недопустима.
    """
    n = int(getattr(config, "TFT_LIQUIDITY_DAYS", 60))
    out: dict[str, dict] = {}

    if lots is None:
        try:
            lots = database.get_instrument_lots(conn)
        except Exception as e:  # noqa: BLE001 — кэша может не быть
            log.warning("Кэш инструментов недоступен (%s) — Max Pos ₽ не считается. "
                        "Запустите: python3 -m services.load_instruments", e)
            return {}
    lots = {str(k).upper(): int(v) for k, v in (lots or {}).items() if v}

    if not lots:
        log.warning("Кэш лотности пуст — Max Pos ₽ и балл ликвидности не считаются. "
                    "Запустите: python3 -m services.load_instruments")
        return {}

    # Сессии выходного дня дают объём примерно в 8 раз ниже будничного —
    # в строгом календаре они не должны занижать ADV и Max Pos.
    sql = _SQL_TMPL.format(session_filter=trading_calendar.sql_session_filter("date"))
    missing: list[str] = []
    try:
        with conn.cursor() as cur:
            for tk in tickers:
                TK = tk.upper()
                lot = lots.get(TK)
                if not lot:
                    missing.append(TK)
                    continue
                try:
                    cur.execute(sql, {"tk": TK, "n": n})
                    rows = cur.fetchall()
                except Exception:  # noqa: BLE001 — один тикер не должен валить всё
                    continue
                m = _max_pos_for(rows, lot)
                if m:
                    out[TK] = m
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось оценить ликвидность (%s) — столбец Max Pos будет пуст.", e)

    if missing:
        log.warning("Лот неизвестен для %d бумаг — Max Pos ₽ по ним не считается: %s. "
                    "Обновите кэш: python3 -m services.load_instruments",
                    len(missing), " ".join(sorted(missing)))
    return out
