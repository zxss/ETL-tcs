"""
Парсер новостей @markettwits с двухуровневым извлечением тикеров:

  Уровень 1 (быстрый, бесплатный) — regex:
    • #SBER, $SBER, #GAZP и т.п.
    • маппинг старых тикеров: YNDX→YDEX, FIVE→X5, FIVEP→X5
    • русские названия по словарю

  Уровень 2 (LLM, дешёвый haiku) — только если уровень 1 нашёл ≥1 тикер
    или текст содержит название компании из словаря:
    • уточняет сентимент и заголовок
    • снимает неоднозначности (Магнит vs ММК)

Батчевая обработка: до BATCH_SIZE постов за один API-вызов → экономия токенов.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

import config

log = logging.getLogger("news_parser")

# ── Вотчлист и маппинги ──────────────────────────────────────────────────────

WATCHLIST: set[str] = set(config.TICKERS)

# Старые тикеры → текущие
OLD_TICKER_MAP: dict[str, str] = {
    "YNDX":  "YDEX",
    "FIVE":  "X5",
    "FIVEP": "X5",
    "LNTA":  "LENT",
    "TCSG":  "SBER",   # ТКС уже не самостоятельный — убрать если нежелательно
}

# Русские названия → тикер (только из вотчлиста)
RU_NAME_MAP: dict[str, str] = {
    "сбербанк": "SBER", "сбер": "SBER",
    "втб": "VTBR",
    "газпром": "GAZP",
    "роснефть": "ROSN",
    "лукойл": "LKOH",
    "новатэк": "NVTK",
    "татнефть": "TATN",
    "сургутнефтегаз": "SNGS", "сургут": "SNGS",
    "сургут-ап": "SNGSP", "сургутнефтегаз-п": "SNGSP",
    "норникель": "GMKN", "норильский никель": "GMKN",
    "полюс": "PLZL",
    "ммк": "MAGN", "магнитка": "MAGN",
    "северсталь": "CHMF",
    "алроса": "ALRS",
    "интер рао": "IRAO", "интерэнерго": "IRAO",
    "фск": "FEES", "россети": "FEES",
    "русгидро": "HYDR",
    "юнипро": "UPRO",
    "мосэнерго": "MSNG",
    "тгк-1": "TGKA",
    "огк-2": "OGKB",
    "фосагро": "PHOR",
    "акрон": "AKRN",
    "московская биржа": "MOEX", "мосбиржа": "MOEX",
    "мтс": "MTSS",
    "ростелеком": "RTKM",
    "аэрофлот": "AFLT",
    "совкомфлот": "FLOT",
    "нмтп": "NMTP",
    "магнит": "MGNT",
    "x5": "X5", "пятёрочка": "X5", "пятерочка": "X5", "перекрёсток": "X5",
    "лента": "LENT",
    "fix price": "FIXR", "фикс прайс": "FIXR",
    "яндекс": "YDEX",
    "вк": "VKCO",
    "астра": "ASTR", "группа астра": "ASTR",
    "позитив": "POSI", "positive technologies": "POSI",
    "пик": "PIKK",
    "самолёт": "SMLT", "самолет": "SMLT",
    "эталон": "ETLN",
    "русал": "RUAL",
    "эн+": "ENPG",
    "селигдар": "SELG",
    "банк санкт-петербург": "BSPB", "банк спб": "BSPB",
    "мкб": "CBOM",
    "м.видео": "MVID", "мвидео": "MVID",
}

# regex: #TICKER, $TICKER (1–6 букв/цифр)
_CASHTAG_RE = re.compile(r'[#$]([A-Za-zА-Яа-я0-9]{1,6})', re.UNICODE)

BATCH_SIZE = 20   # постов за один LLM-вызов


# ── Уровень 1: быстрый regex-парсер ─────────────────────────────────────────

def _extract_tickers_regex(text: str) -> list[str]:
    """Извлекает тикеры из текста без LLM."""
    found: set[str] = set()
    text_lower = text.lower()

    # 1. Кэштеги
    for m in _CASHTAG_RE.finditer(text):
        raw = m.group(1).upper()
        ticker = OLD_TICKER_MAP.get(raw, raw)
        if ticker in WATCHLIST:
            found.add(ticker)

    # 2. Русские названия (по словарю)
    for ru_name, ticker in RU_NAME_MAP.items():
        if ru_name in text_lower:
            found.add(ticker)

    return sorted(found)


def _needs_llm(text: str, regex_tickers: list[str]) -> bool:
    """Нужна ли LLM для уточнения?"""
    # LLM нужна если найден тикер — для сентимента и заголовка
    return len(regex_tickers) > 0


# ── Уровень 2: LLM (Claude haiku) ───────────────────────────────────────────

_SYSTEM_PROMPT = f"""Ты парсер финансовых новостей из Telegram-канала @markettwits (русский).

ВОТЧЛИСТ: {', '.join(sorted(WATCHLIST))}

ЗАДАЧА: для каждого поста верни JSON-объект:
{{
  "tickers": ["..."],       // только из вотчлиста; [] если нет
  "sentiment": "pos|neg|neutral",  // тон новости ДЛЯ ТИКЕРА (не для рынка в целом)
  "headline": "<=120 симв"  // суть на русском, кратко
}}

ПРАВИЛА:
- Кэштеги #SBER, $SBER — основной маркер
- Старые тикеры: YNDX→YDEX, FIVE→X5, FIVEP→X5
- Русские названия → тикер: Сбербанк→SBER, ВТБ→VTBR, Газпром→GAZP и т.д.
- «Магнит»→MGNT (ритейл), «ММК»/«Магнитка»→MAGN (металл) — не путать
- Если нет тикеров из вотчлиста → {{"tickers":[],"sentiment":"neutral","headline":""}}
- sentiment: pos=рост/прибыль/позитив, neg=падение/убыток/негатив, neutral=нейтрально/факт
- ФОРМАТ: только JSON без пояснений"""

_BATCH_PROMPT_TEMPLATE = """Обработай следующие посты. Верни JSON-массив объектов в том же порядке.

{posts}"""


def _make_batch_prompt(messages: list[dict]) -> str:
    posts_text = "\n\n".join(
        f"[{i+1}] ID={m['message_id']} ({m['ts'].strftime('%Y-%m-%d %H:%M')} UTC)\n{m['text']}"
        for i, m in enumerate(messages)
    )
    return _BATCH_PROMPT_TEMPLATE.format(posts=posts_text)


def parse_batch_llm(messages: list[dict], client: anthropic.Anthropic) -> list[dict]:
    """
    Вызывает Claude для батча сообщений.
    Возвращает список результатов в том же порядке.
    """
    if not messages:
        return []

    prompt = _make_batch_prompt(messages)
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # Убираем markdown-обёртку если есть
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed

    except Exception as e:
        log.error("LLM ошибка батча: %s", e)
        # Фолбэк: вернуть regex-результаты
        return [
            {
                "tickers": _extract_tickers_regex(m["text"]),
                "sentiment": "neutral",
                "headline": m["text"][:120],
            }
            for m in messages
        ]


# ── Главная функция парсинга ─────────────────────────────────────────────────

def parse_messages(
    messages: list[dict],
    anthropic_client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """
    Парсит список сообщений и возвращает список записей для БД:
      [{message_id, ts, ticker, sentiment, headline, raw_text}, ...]

    Один пост → несколько записей (если несколько тикеров).
    """
    if not messages:
        return []

    # Шаг 1: быстрый regex-фильтр
    pre: list[dict] = []   # сообщения с тикерами → в LLM
    results: list[dict] = []

    for msg in messages:
        tickers = _extract_tickers_regex(msg["text"])
        if tickers:
            pre.append({**msg, "_regex_tickers": tickers})
        # Сообщения без тикеров пропускаем

    if not pre:
        return []

    # Шаг 2: LLM-обработка батчами
    llm_results: list[dict] = []
    if anthropic_client and pre:
        for i in range(0, len(pre), BATCH_SIZE):
            batch = pre[i:i + BATCH_SIZE]
            batch_results = parse_batch_llm(batch, anthropic_client)
            llm_results.extend(batch_results)
    else:
        # Без LLM: используем только regex
        llm_results = [
            {
                "tickers":   m["_regex_tickers"],
                "sentiment": "neutral",
                "headline":  m["text"][:120],
            }
            for m in pre
        ]

    # Шаг 3: объединяем результаты LLM с оригинальными сообщениями
    for msg, llm_res in zip(pre, llm_results):
        tickers = llm_res.get("tickers") or msg["_regex_tickers"]
        tickers = [t for t in tickers if t in WATCHLIST]
        if not tickers:
            continue

        sentiment = llm_res.get("sentiment", "neutral")
        if sentiment not in ("pos", "neg", "neutral"):
            sentiment = "neutral"

        headline = (llm_res.get("headline") or msg["text"][:120]).strip()[:120]

        for ticker in tickers:
            results.append({
                "message_id": msg["message_id"],
                "ts":         msg["ts"],
                "ticker":     ticker,
                "sentiment":  sentiment,
                "headline":   headline,
                "raw_text":   msg["text"][:4000],
            })

    log.info("Сообщений обработано: %d → %d записей", len(messages), len(results))
    return results
