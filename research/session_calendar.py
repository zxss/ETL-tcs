"""
Календарь сессий Мосбиржи для исследований (ТЗ EDGE-R4 §3, блок A2).

Окна исследований задаются СОБЫТИЯМИ сессии, а не часами: «первая цена
основной сессии», «начало основной + 5 минут». Часы этих событий менялись, и
окно, записанное часами, молча переносит результат из одного режима рынка в
другой (аудит r3: сдвиг конца окна меняет знак IC long_overnight).

| Точка                                   | до 14.09.2026   | с 14.09.2026    |
|-----------------------------------------|-----------------|-----------------|
| аукцион открытия                        | 09:50           | 09:00           |
| первая цена основной сессии             | open бара 10:00 | open бара 09:10 |
| вход утреннего шорта (основная + 5 мин) | open бара 10:05 | open бара 09:15 |
| конец непрерывной основной сессии       | 18:40           | 18:54           |
| выход дневного шорта (фаза 18:20)       | close бара 18:15| close бара 18:15|
| вход вечером (фаза 18:35)               | close бара 18:30| close бара 18:30|

Утренняя сессия акций — по данным (research/session_timing, первая сделка по
корзине из 10 бумаг): с 14.08.2024 около 07:00, с февраля 2025 — 06:50. С 14.09.2026
TradingSchedules показывает только основную (аукцион 09:00, торги с 09:10), но
сделки с 06:50 идут и 14–15.09 (5–24 % объёма дня до 09:00): утренняя сессия
осталась, сдвинулось начало основной. Выходная сессия — факт наличия
баров в субботу или воскресенье; регламент по датам здесь не зашит, потому что
её наличие проверяется по данным.

Модуль только для исследований: торговый контур его не импортирует.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

NEW_SCHEDULE_FROM = dt.date(2026, 9, 14)
_MORNING_FROM = dt.date(2024, 8, 14)

SHORT_EXIT_BAR = dt.time(18, 15)       # close бара 18:15 — цена фазы 18:20
EVENING_ENTRY_BAR = dt.time(18, 30)    # close бара 18:30 — цена фазы 18:35
DECISION_TIME = dt.time(18, 35)        # вечернее решение


@dataclass(frozen=True)
class Session:
    day: dt.date
    opening_auction: dt.time
    main_open: dt.time        # начало основной сессии = open первого бара
    main_close: dt.time       # конец непрерывной основной сессии (начало аукциона закрытия)
    morning: bool
    weekend: bool

    @property
    def short_entry_bar(self) -> dt.time:
        """Бар входа утреннего шорта: начало основной сессии + 5 минут."""
        return _plus_minutes(self.main_open, 5)

    @property
    def expected_main_bars(self) -> int:
        """Сколько 5-минутных баров в непрерывной основной сессии."""
        a = self.main_open.hour * 60 + self.main_open.minute
        b = self.main_close.hour * 60 + self.main_close.minute
        return (b - a) // 5


def _plus_minutes(t: dt.time, m: int) -> dt.time:
    x = dt.datetime.combine(dt.date(2000, 1, 1), t) + dt.timedelta(minutes=m)
    return x.time()


def session(day: dt.date) -> Session:
    """Параметры сессии на дату (часы по Москве)."""
    weekend = day.weekday() >= 5
    if day >= NEW_SCHEDULE_FROM:
        return Session(day, dt.time(9, 0), dt.time(9, 10), dt.time(18, 54),
                       morning=True, weekend=weekend)
    return Session(day, dt.time(9, 50), dt.time(10, 0), dt.time(18, 40),
                   morning=day >= _MORNING_FROM, weekend=weekend)


def main_open(day: dt.date) -> dt.time:
    """Время бара с первой ценой основной сессии (выход ночной позиции)."""
    return session(day).main_open


def short_entry(day: dt.date) -> dt.time:
    """Время бара входа утреннего шорта."""
    return session(day).short_entry_bar
