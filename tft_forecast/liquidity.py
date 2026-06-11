"""
Оценка максимально допустимого размера позиции (Max Pos ₽) — сколько рублей
можно ввести в инструмент, не сдвинув цену существенно.

Учитывается РЕАЛЬНАЯ ликвидность (ценовое воздействие на рубль оборота), а не
только объём торгов. Используется модель Amihud illiquidity по дневным OHLCV:

    ILLIQ = median_t( |ret_t| / (close_t * volume_t) )

ILLIQ — это во сколько (в долях) сдвигается цена на каждый вложенный рубль.
Чтобы удержать воздействие в пределах допуска λ (TFT_IMPACT_TOL, напр. 0.5%):

    MaxPos_impact = λ / ILLIQ

Дополнительно ограничиваем долей участия в среднем дневном рублёвом обороте
(чтобы для сверхликвидных бумаг оценка оставалась консервативной):

    MaxPos_adv = participation × median(close_t * volume_t)

Итог: Max Pos ₽ = min(MaxPos_impact, MaxPos_adv) — наиболее консервативная.

Полностью graceful: при нехватке данных по тикеру возвращается None.
"""

from __future__ import annotations

import logging
from statistics import median

import config

log = logging.getLogger("tft.liquidity")

_SQL = """
    SELECT close, volume
    FROM market_data
    WHERE ticker = %(tk)s
    ORDER BY date DESC
    LIMIT %(n)s;
"""


def _max_pos_for(rows) -> dict | None:
    """rows: список (close, volume) от свежих к старым. Возвращает метрики."""
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

    dollar_vol = [c * v for c, v in zip(closes, vols)]
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

    return {"max_pos": max_pos, "adv_rub": adv_rub, "illiq": illiq}


def compute(conn, tickers: list[str]) -> dict:
    """
    Возвращает {ticker: {"max_pos","adv_rub","illiq"}} для тикеров, по которым
    хватило данных. Никогда не бросает наружу.
    """
    n = int(getattr(config, "TFT_LIQUIDITY_DAYS", 60))
    out: dict[str, dict] = {}
    try:
        with conn.cursor() as cur:
            for tk in tickers:
                TK = tk.upper()
                try:
                    cur.execute(_SQL, {"tk": TK, "n": n})
                    rows = cur.fetchall()
                except Exception:  # noqa: BLE001 — один тикер не должен валить всё
                    continue
                m = _max_pos_for(rows)
                if m:
                    out[TK] = m
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось оценить ликвидность (%s) — столбец Max Pos будет пуст.", e)
    return out
