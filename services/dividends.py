"""
Дивидендные отсечки для торгового контура.

Стратегия long_overnight входит на закрытии дня D и выходит на открытии D+1.
Гэп отсечки приходится ровно на это открытие, если D == last_buy_date.

Арифметика входа в такой вечер (не гипотеза, а тождество):
  * право на дивиденд получено — держатель на дату фиксации;
  * утренний гэп ≈ полная величина выплаты, съедается сразу;
  * выплата придёт через недели и за вычетом НДФЛ (брокер — налоговый агент);
  * итог ≈ −13 % от дивиденда. При медианном гэпе 5,2 % это −0,68 % на позицию
    против издержек круга 0,08 %, то есть в восемь раз дороже самой торговли.
Плюс исполнение: стоп в MIN_STOP_PCT = 1 % пробит уже в аукционе открытия, и
позиция закрывается по неконтролируемой цене (SNGSP 17.07.2024: гэп 20,4 %).

Поэтому вечер last_buy_date для ночного входа блокируется (OVERNIGHT_SKIP_EX_DIV).
Внутридневные стратегии не трогаем: они работают close/open, обе цены уже после
гэпа, дивиденд их не задевает.
"""

from __future__ import annotations

import datetime as dt
import logging

import config

log = logging.getLogger("dividends")

SKIP_REASON = "отсечка по дивидендам: вход в ночь перед гэпом"


def ex_div_block(day, conn=None) -> dict[str, dict]:
    """{TICKER: инфо} — бумаги, вечер которых закрыт для ночного входа.

    Пустой словарь означает «блокировать нечего»: либо выключено флагом, либо
    календарь пуст. Контур обязан работать без календаря — деградация до
    прежнего поведения, но громкая (см. warn_if_empty).
    """
    if not bool(getattr(config, "OVERNIGHT_SKIP_EX_DIV", True)):
        return {}
    try:
        import database
        return database.get_dividends_on(day, conn)
    except Exception as e:  # noqa: BLE001 — календарь не критичен для заявок
        log.warning("[DIV] календарь отсечек недоступен (%s) — блокировка не применена", e)
        return {}


def gap_today(day, conn=None) -> dict[str, dict]:
    """{TICKER: инфо} — бумаги, у которых гэп отсечки приходится на ЭТО утро.

    Гэп ложится на открытие первого торгового дня строго после last_buy_date,
    поэтому ищем отсечки последних дней и сверяем по календарю биржи. Нужно
    защите: стоп в 1 % от входа при гэпе 5 % срабатывает в аукционе открытия по
    неконтролируемой цене, и это не движение рынка, а выплата.
    """
    try:
        import database
        import trading_calendar as cal
        with (database.get_db_connection() if conn is None else _noop(conn)) as c:
            with c.cursor() as cur:
                cur.execute("SELECT ticker, last_buy_date, dividend_net, close_price "
                            "FROM dividends WHERE dividend_net > 0 "
                            "AND last_buy_date BETWEEN %s AND %s",
                            (day - dt.timedelta(days=10), day))
                rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        log.warning("[DIV] проверка гэпа на сегодня не удалась: %s", e)
        return {}
    out = {}
    for tk, last_buy, amount, close in rows:
        try:
            if cal.next_trading_day(last_buy) != day:
                continue
        except Exception:  # noqa: BLE001 — календарь не должен ломать защиту
            continue
        px = float(close) if close else 0.0
        out[str(tk).upper()] = {
            "last_buy_date": last_buy, "dividend_net": float(amount),
            "close_price": px,
            "gap_pct": (float(amount) / px * 100.0) if px > 0 else None,
        }
    return out


def describe(ticker: str, info: dict) -> str:
    """Строка для журнала и отчёта фазы."""
    gap = info.get("gap_pct")
    tail = f", гэп ≈ {gap:.2f} %" if gap else ""
    return (f"{ticker}: последний день покупки {info.get('last_buy_date')}, "
            f"дивиденд {info.get('dividend_net'):.4f} ₽{tail}")


def calendar_stale(day, conn=None, horizon_days: int = 45) -> bool:
    """True, если впереди на horizon_days нет ни одной объявленной отсечки.

    Не ошибка сама по себе (бывают затишья, особенно в феврале–апреле), но в
    мае–июле пустой горизонт означает, что календарь не обновляется: 63 %
    годовых отсечек приходятся на май–июль.
    """
    try:
        import database
        with database.get_db_connection() if conn is None else _noop(conn) as c:
            with c.cursor() as cur:
                cur.execute("SELECT count(*) FROM dividends "
                            "WHERE last_buy_date BETWEEN %s AND %s",
                            (day, day + dt.timedelta(days=horizon_days)))
                (n,) = cur.fetchone()
        return int(n) == 0
    except Exception as e:  # noqa: BLE001
        log.warning("[DIV] проверка свежести календаря не удалась: %s", e)
        return False


class _noop:
    """Контекст, отдающий уже открытый коннект без закрытия."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False
