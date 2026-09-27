"""
Разбор кода опционной серии FORTS (ISS не отдаёт метаданные в дневной истории).

Формат: <актив><страйк><тип расчёта><буква месяца><цифра года>[буква недели]
  RI100000BC7  — RTS, страйк 100 000, расчёт B, март (C), 2027, месячная;
  Si82500BJ6D  — Si, страйк 82 500, октябрь-колл (J), 2026, недельная серия D.
Буква месяца: A–L — колл январь–декабрь, M–X — пут январь–декабрь.
Год — последняя цифра; десятилетие восстанавливается по дате торгов. Серии старше
примерно десяти лет ISS переименовывает с суффиксом года: RI100000BA5_2015 — тогда
год берётся из суффикса (без этого до 27.09.2026 все циклы до 2019 года терялись).

Экспирация недельных серий из кода не выводится однозначно — берётся как
последний день, когда серия встречается в истории (см. download.py).
"""
from __future__ import annotations

import re

_RX = re.compile(r"^(?P<asset>RI|Si)(?P<strike>\d+(?:\.\d+)?)(?P<settle>[A-Z])"
                 r"(?P<month>[A-X])(?P<year>\d)(?P<week>[A-E])?(?:_(?P<full>\d{4}))?$")


def parse(secid: str, trade_year: int | None = None) -> dict | None:
    """{asset, strike, type 'C'|'P', month 1..12, year, weekly} или None."""
    m = _RX.match(secid)
    if not m:
        return None
    letter = m["month"]
    idx = ord(letter) - ord("A")
    kind, month = ("C", idx + 1) if idx < 12 else ("P", idx - 11)
    digit = int(m["year"])
    year = None
    if m["full"]:
        year = int(m["full"])
    elif trade_year is not None:
        # ближайший год с этой последней цифрой, не раньше года торгов минус 1
        base = trade_year - trade_year % 10 + digit
        year = base if base >= trade_year - 1 else base + 10
    return {"asset": m["asset"], "strike": float(m["strike"]), "type": kind,
            "month": month, "year": year, "weekly": m["week"] is not None,
            "week": m["week"]}
