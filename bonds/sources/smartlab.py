"""
smart-lab.ru — рейтинг и эмитент облигаций по ISIN (вторичный источник).

Таблица /q/bonds/ отсортирована по обороту, 100 строк на странице, до 16 страниц —
самые ликвидные ~1 600 выпусков. Из строки берём ISIN (ссылка /q/bonds/<ISIN>/),
эмитента (ссылка /bonds/<эмитент>), название и колонку «Рейтинг».

Вежливость: robots.txt разрешает /q/bonds/ для обычных агентов; запрос раз в
сутки с паузой 2 с между страницами, честный User-Agent, кэш на день.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.request

BASE = "https://smart-lab.ru"
UA = "ETL-tcs bonds module (personal research; 1 request / 2 s, once a day)"
PAGES = 16


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse(html: str) -> dict:
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.S)
    if not tables:
        return {}
    t = max(tables, key=len)
    heads = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<th[^>]*>(.*?)</th>", t, re.S)]
    try:
        ri = heads.index("Рейтинг")
    except ValueError:
        return {}
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S)[1:]:
        m_isin = re.search(r'href="/q/bonds/([A-Z]{2}[A-Z0-9]{10})/"', row)
        if not m_isin:
            continue
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        m_iss = re.search(r'href="/bonds/([^"/]+)"', row)
        out[m_isin.group(1)] = {"rating": cells[ri] if ri < len(cells) else None,
                                "issuer": m_iss.group(1) if m_iss else None,
                                "name": cells[1] if len(cells) > 1 else None}
    return out


def ratings(cache: str | None, pages: int = PAGES) -> dict:
    """{ISIN: {rating, issuer, name}} с кэшем на день."""
    day = dt.date.today().isoformat()
    path = os.path.join(cache, "smartlab", f"bonds_{day}.json") if cache else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out = parse(_fetch(f"{BASE}/q/bonds/"))
    for p in range(2, pages + 1):
        time.sleep(2)
        try:
            page = parse(_fetch(f"{BASE}/q/bonds/order_by_val_to_day/desc/page{p}/"))
        except OSError:
            break
        if not page:
            break
        out.update(page)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:  # flow-state: кэш дня
            json.dump(out, f, ensure_ascii=False)
    return out
