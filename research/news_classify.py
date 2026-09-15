"""
Классификатор постов markettwits: категория, тональность и связь бумаги с постом
(Спринт 1, ТЗ 50D). Только исследование, торговым контуром не импортируется.

Детерминированный: регулярные выражения из research/dicts/news_categories.json,
заморожены коммитом. Внешних LLM нет ни в рантайме, ни в исследовании (решение
пользователя: Claude — только офлайн, при разметке учебной выборки). Кодбук —
research/dicts/news_codebook.md.

Как устроено (для каждой пары пост × бумага):
  1. Контекст. В дайджестах («календарь на сегодня», «попсовые отчетности»)
     берутся только строки с бумагой: остальное — про другие компании.
  2. Отчёт о цене («#GAZP = мин за 5 мес», «#GAZP > 138», «#VKCO в моменте»)
     — категория OTHER, тональность 0, связь «упоминание»: пост описывает
     случившееся движение, а не его причину.
  3. Категория — та, чей шаблон встретился в тексте раньше (заголовок решает),
     при равенстве — порядок category_order; вторая — следующая найденная.
     Хештеги канала (#дивиденд, #отчетность) шумные и работают только как
     запасной вариант, если в тексте ничего не нашлось.
  4. Тональность — словарь категории: сильные слова ±1, обычные ±0,5, сумма
     обрезается до [−1, 1]. Для отчётности добавляется сравнение «X против Y».
  5. Связь: омоним → отчёт о цене → строка дайджеста → пост о чужой компании
     (чужой тикер в хештеге, бумага только названием) → источник мнения
     («по оценке ВТБ», «— Газпром», «Сбер: …») → много бумаг названием →
     объект.

Оценка на размеченной выборке:
    python -m research.news_classify --eval texts.csv labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import news_event_study as ns             # noqa: E402

DICT_PATH = os.path.join(ROOT, "research", "dicts", "news_categories.json")
CATEGORIES = ("DIVIDEND", "FINANCIAL", "CORPORATE", "SANCTIONS_MACRO", "OTHER")
TZ_CODES = {c: f"CAT_{c}" for c in CATEGORIES}
RELATIONS = ("объект", "источник", "упоминание", "омоним")
_UNITS = {"трлн": 1e12, "млрд": 1e9, "млн": 1e6, "тыс": 1e3}
_NUM = r"([+\-]?\d[\d ]*(?:[.,]\d+)?)\s*(трлн|млрд|млн|тыс)?"
_VS = re.compile(_NUM + r"\.?\s*(?:руб\w*\.?)?[^\n]{0,35}?против\s+(прибыли\s+|убытка\s+|выручки\s+)?\+?"
                 + _NUM, re.IGNORECASE)
_HASHTAG_ANY = re.compile(r"#([A-Za-zА-Яа-яЁё0-9_]+)")
_FOREIGN_TICKER = re.compile(r"#([A-Z][A-Z0-9]{2,5})(?![A-Za-z0-9_])")


def normalize(text: str | None) -> str:
    return (text or "").lower().replace("ё", "е")


def _name_pattern(ticker: str) -> str:
    if ticker == "PIKK":
        return r"\bпик(?:а|у|ом|е)?\b"
    p = ns._NAMES.get(ticker)
    return f"(?:{p})" if p else r"(?!)"


class Classifier:
    def __init__(self, path: str = DICT_PATH):
        with open(path, encoding="utf-8") as f:
            self.d = json.load(f)
        self.version = self.d["version"]
        self.cat_re = {c: [re.compile(p) for p in ps] for c, ps in self.d["categories"].items()}
        self.order = self.d["category_order"]
        self.sent = {c: {k: [re.compile(p) for p in v] for k, v in lex.items()}
                     for c, lex in self.d["sentiment"].items()}
        self.omonyms = {t: [re.compile(p) for p in ps] for t, ps in self.d["omonyms"].items()}

    # ── контекст и отчёты о цене ──

    def is_digest(self, text: str) -> bool:
        t = normalize(text)
        return any(m in t for m in self.d["digest_markers"])

    def context(self, text: str, ticker: str) -> tuple[str, bool]:
        """Строки дайджеста с этой бумагой; не дайджест — весь пост."""
        if not self.is_digest(text):
            return text, False
        lines = [ln for ln in text.splitlines() if ticker in ns.ticker_sources(ln)]
        return ("\n".join(lines) if lines else text), True

    def price_report(self, ctx: str, ticker: str) -> bool:
        if ns.is_price_report(ctx):
            return True
        tags = [ticker] + [a for a, t in ns.HASHTAG_ALIASES.items() if ticker in t and a != ticker]
        t = normalize(ctx)
        for tag in tags:
            for p in self.d["price_report"]:
                if re.search(p.replace("{T}", re.escape(tag.lower())), t):
                    return True
        return False

    # ── категория ──

    def category(self, ctx: str) -> tuple[str, str | None]:
        body = normalize(_HASHTAG_ANY.sub(" ", ctx))
        first: dict[str, int] = {}
        for c, pats in self.cat_re.items():
            pos = [m.start() for p in pats for m in [p.search(body)] if m]
            if pos:
                first[c] = min(pos)
        if not first:
            tags = {normalize(h) for h in _HASHTAG_ANY.findall(ctx)}
            for tag, c in self.d["fallback_hashtags"].items():
                if tag in tags:
                    return c, None
            return "OTHER", None
        ranked = sorted(first, key=lambda c: (first[c], self.order.index(c)))
        return ranked[0], (ranked[1] if len(ranked) > 1 else None)

    # ── тональность ──

    def _lex(self, body: str, cat: str) -> float:
        lex = self.sent.get(cat, {})
        s = 0.0
        for key, w in (("strong_pos", 1.0), ("pos", 0.5), ("strong_neg", -1.0), ("neg", -0.5)):
            s += w * sum(1 for p in lex.get(key, []) if p.search(body))
        return s

    @staticmethod
    def compare_numbers(text: str) -> float:
        """«X против Y» в отчётности: ±1 при изменении более чем в 1,5 раза,
        ±0,5 — более 5 %, иначе 0. Прошлый убыток → новая прибыль = +1."""
        m = _VS.search(normalize(text))
        if not m:
            return 0.0

        def val(num, unit):
            x = float(num.replace(" ", "").replace(",", "."))
            return x * _UNITS.get(unit or "", 1.0)
        try:
            a, b = val(m.group(1), m.group(2)), val(m.group(4), m.group(5))
        except ValueError:
            return 0.0
        if m.group(3) and "убытк" in m.group(3):
            return 1.0
        if a <= 0 or b <= 0:
            return 0.0
        r = a / b
        return 1.0 if r > 1.5 else 0.5 if r > 1.05 else -1.0 if r < 1 / 1.5 else -0.5 if r < 0.95 else 0.0

    def sentiment(self, ctx: str, cat: str, cat2: str | None) -> float:
        body = normalize(_HASHTAG_ANY.sub(" ", ctx))
        s = self._lex(body, cat)
        if cat2 and cat2 != cat:
            s += 0.5 * self._lex(body, cat2)
        if "FINANCIAL" in (cat, cat2):
            s += self.compare_numbers(ctx)
        return max(-1.0, min(1.0, s))

    # ── связь бумаги с постом ──

    def relation(self, text: str, ticker: str, sources: dict, digest: bool, ctx: str,
                 is_price: bool, cat: str) -> str:
        how = sources.get(ticker, set())
        t = normalize(text)
        if "hashtag" not in how:
            if any(p.search(t) for p in self.omonyms.get(ticker, [])):
                return "омоним"
        if is_price:
            return "упоминание"
        if digest:
            return "объект" if cat != "OTHER" else "упоминание"
        name = _name_pattern(ticker)
        if "hashtag" not in how:
            foreign = [h for h in _FOREIGN_TICKER.findall(text)
                       if h not in ns.UNIVERSE and h not in ns.HASHTAG_ALIASES]
            if foreign:
                return "упоминание"
            for p in self.d["source_patterns"]:
                if re.search(p.replace("{N}", name), t):
                    return "источник"
            if len(sources) >= 3:
                return "упоминание"
        else:
            head = text.splitlines()[0] if text else ""
            in_head = ticker in ns.ticker_sources(head)
            if not in_head and len(sources) >= 2:
                return "упоминание"
        return "объект"

    # ── пост целиком ──

    def classify(self, text: str, ticker: str, sources: dict | None = None) -> dict:
        sources = sources if sources is not None else ns.ticker_sources(text)
        ctx, digest = self.context(text, ticker)
        is_price = self.price_report(ctx, ticker)
        if is_price:
            cat, cat2, sent = "OTHER", None, 0.0
        else:
            cat, cat2 = self.category(ctx)
            sent = self.sentiment(ctx, cat, cat2)
        rel = self.relation(text, ticker, sources, digest, ctx, is_price, cat)
        return {"ticker": ticker, "category": cat, "category2": cat2, "sentiment": sent,
                "relation": rel, "price_report": is_price, "digest": digest}

    def classify_post(self, text: str) -> list[dict]:
        src = ns.ticker_sources(text)
        return [self.classify(text, tk, src) for tk in sorted(src)]


# ── Оценка на разметке ────────────────────────────────────────────────────────

def snap(x: float) -> float:
    """Ближайшая точка шкалы кодбука: −1, −0,5, 0, +0,5, +1."""
    return min((-1.0, -0.5, 0.0, 0.5, 1.0), key=lambda v: abs(v - x))


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def evaluate(texts: dict, labels: list[dict], clf: Classifier | None = None) -> dict:
    clf = clf or Classifier()
    rows = []
    for lb in labels:
        text = texts[int(lb["message_id"])]
        p = clf.classify(text, lb["ticker"])
        rows.append((lb, p))
    n = len(rows)
    rel_ok = sum(1 for lb, p in rows if p["relation"] == lb["relation"])
    obj_pred = [(lb, p) for lb, p in rows if p["relation"] == "объект"]
    obj_true = [(lb, p) for lb, p in rows if lb["relation"] == "объект"]
    tp = sum(1 for lb, p in obj_pred if lb["relation"] == "объект")
    ev = obj_true
    cat_ok = sum(1 for lb, p in ev if p["category"] == lb["category"])
    cat_any = sum(1 for lb, p in ev if p["category"] in (lb["category"], lb.get("category2") or None))
    s_true = [float(lb["sentiment"]) for lb, p in ev]
    s_pred = [p["sentiment"] for lb, p in ev]
    sign_ok = sum(1 for a, b in zip(s_true, s_pred) if sign(a) == sign(b))
    per_cat = {}
    for c in CATEGORIES:
        tp_c = sum(1 for lb, p in ev if p["category"] == c and lb["category"] == c)
        pp = sum(1 for lb, p in ev if p["category"] == c)
        tt = sum(1 for lb, p in ev if lb["category"] == c)
        per_cat[c] = {"precision": tp_c / pp if pp else None, "recall": tp_c / tt if tt else None,
                      "support": tt}
    conf = Counter((lb["relation"], p["relation"]) for lb, p in rows)
    misses = [(lb["message_id"], lb["ticker"], lb["category"], p["category"], lb["sentiment"],
               round(p["sentiment"], 2), lb["relation"], p["relation"]) for lb, p in rows
              if p["relation"] != lb["relation"] or (lb["relation"] == "объект"
                                                     and p["category"] != lb["category"])]
    return {"n": n, "version": clf.version,
            "relation_accuracy": rel_ok / n if n else None,
            "object_precision": tp / len(obj_pred) if obj_pred else None,
            "object_recall": tp / len(obj_true) if obj_true else None,
            "category_accuracy_on_objects": cat_ok / len(ev) if ev else None,
            "category_hit_primary_or_second": cat_any / len(ev) if ev else None,
            "sentiment_sign_accuracy": sign_ok / len(ev) if ev else None,
            "sentiment_mae": (sum(abs(snap(b) - a) for a, b in zip(s_true, s_pred)) / len(ev)) if ev else None,
            "per_category": per_cat,
            "relation_confusion": {f"{a}→{b}": v for (a, b), v in sorted(conf.items())},
            "misses": misses}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % интервал Уилсона для доли k/n."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def _load_texts(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return {int(r["message_id"]): r["text"] for r in csv.DictReader(f)}


def _load_labels(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("ticker")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Классификатор постов markettwits")
    ap.add_argument("--eval", nargs=2, metavar=("TEXTS_CSV", "LABELS_CSV"), required=True)
    ap.add_argument("--misses", action="store_true", help="показать расхождения")
    a = ap.parse_args(argv)
    res = evaluate(_load_texts(a.eval[0]), _load_labels(a.eval[1]))
    misses = res.pop("misses")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.misses:
        for m in misses:
            print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
