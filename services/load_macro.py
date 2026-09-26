"""
Загрузка ключевой ставки ЦБ РФ (SOAP DailyInfo.asmx, метод KeyRate).

Зачем: ставка — планка доходности контура. Робот держит свободные деньги в фонде
денежного рынка, фонд идёт за ставкой, значит торговля обязана обыгрывать её, а
не ноль. Подробно — models/macro.py.

Почему SOAP, а не JSON: у ЦБ нет открытого JSON-эндпоинта по ключевой ставке.
GET на /KeyRate сервис отклоняет («URL неожиданно закончился»), работает только
POST с конвертом SOAP 1.1 и заголовком SOAPAction. Ответ — DataSet в
diffgram-обёртке, где каждая запись это <KR><DT>дата</DT><Rate>ставка</Rate></KR>.

Ставка публикуется ступеньками: между заседаниями значение не меняется. Грузим
диапазоном и пишем каждый календарный день, который отдал сервис.
"""

from __future__ import annotations

import datetime as dt
import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import config
import database
import tls

log = logging.getLogger("load_macro")

CBR_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
SOAP_ACTION = "http://web.cbr.ru/KeyRate"
_ENVELOPE = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
    '<soap:Body><KeyRate xmlns="http://web.cbr.ru/">'
    "<fromDate>{frm}</fromDate><ToDate>{to}</ToDate>"
    "</KeyRate></soap:Body></soap:Envelope>"
)


def parse_key_rate(xml_text: str) -> list[dict]:
    """XML-ответ KeyRate → [{date, key_rate_pct, source}, ...].

    Элементы KR лежат в diffgram без пространства имён, поэтому ищем их по
    локальному имени в любом месте дерева: структура обёртки менялась и может
    поменяться снова, а сами записи стабильны с 2013 года.
    """
    rows: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"KeyRate: ответ не разобран как XML ({e})") from e
    for node in root.iter():
        if not node.tag.endswith("KR"):
            continue
        day = rate = None
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "DT" and child.text:
                day = child.text.strip()[:10]
            elif tag == "Rate" and child.text:
                try:
                    rate = float(child.text.strip().replace(",", "."))
                except ValueError:
                    rate = None
        if day and rate is not None and rate > 0:
            rows.append({"date": day, "key_rate_pct": rate, "source": "cbr"})
    return rows


def fetch(frm: dt.date, to: dt.date, timeout: float = 25.0) -> list[dict]:
    body = _ENVELOPE.format(frm=frm.isoformat(), to=to.isoformat()).encode("utf-8")
    req = urllib.request.Request(
        CBR_URL, data=body, method="POST",
        headers={"Content-Type": "text/xml; charset=utf-8",
                 "SOAPAction": f'"{SOAP_ACTION}"'})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=tls.ssl_context()) as resp:
            return parse_key_rate(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"KeyRate: HTTP {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"KeyRate: сетевая ошибка: {e}") from e


def run(conn=None) -> int:
    """Обновляет macro_rates. Не бросает наружу — шаг ETL не критичен.

    Окно берётся с запасом назад: ставка ступенчатая, повторная запись тех же
    значений идемпотентна, а пропуск заседания испортил бы бенчмарк.
    """
    if not getattr(config, "LOAD_MACRO", True):
        log.info("LOAD_MACRO=0 — ключевая ставка не обновляется.")
        return 0
    days = int(getattr(config, "MACRO_HISTORY_DAYS", 400))
    today = dt.date.today()
    try:
        rows = fetch(today - dt.timedelta(days=days), today)
        n = database.save_key_rates(rows, conn)
        if rows:
            log.info("Ключевая ставка обновлена: %d дней, последняя %.2f %% на %s",
                     n, rows[0]["key_rate_pct"], rows[0]["date"])
        else:
            log.warning("Сервис ЦБ не вернул ни одной записи по ключевой ставке.")
        return n
    except Exception as e:  # noqa: BLE001 — шаг ETL не должен валить конвейер
        log.warning("Загрузка ключевой ставки завершилась с ошибкой: %s", e)
        return 0
