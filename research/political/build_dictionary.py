"""
Сборка словаря «переговоры и встречи» из истории markettwits — только по
текстам, без цен (словарь не подгоняется под реакцию рынка).

Метод: политические посты (есть политический якорь) делятся на «ядро
переговоров» (затравочные маркеры) и остальные. Для каждой основы слова и пары
основ считается взвешенное лог-отношение шансов с информативным априорным
распределением Дирихле (Monroe, Colaresi, Quinn 2008) — z-оценка того,
насколько терм характерен именно для переговорной повестки. Основа — первые
7 букв слова (морфологических библиотек на сервере нет).

Выход — только термы и частоты; тексты постов не выгружаются.

Запуск (на сервере): python -m research.political.build_dictionary
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.political.political_news import load_rules      # noqa: E402

OUT = os.path.join(ROOT, "audit", "r4_research", "political", "dictionary_candidates.json")
SEEDS = r"переговор|перемири|прекращени[ея] огня|мирн\w* (план|договор|соглашени|урегулировани|процесс)|саммит|встреч\w* (путина|трампа|путин|трамп)"
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж
вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего
раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы
нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им
более всегда конечно всю между это также заявил заявила сообщил сообщает сказал может будут""".split())
WORD = re.compile(r"[а-яёa-z]{3,}")


def stem(w: str) -> str:
    return w[:7]


def tokens(text: str) -> list[str]:
    return [stem(w) for w in WORD.findall(text.lower().replace("ё", "е")) if w not in STOP]


def terms(text: str) -> set[str]:
    t = tokens(text)
    return set(t) | {f"{a} {b}" for a, b in zip(t, t[1:])}


def log_odds(fa: collections.Counter, fb: collections.Counter, prior: collections.Counter,
             min_count: int = 15) -> list[tuple]:
    na, nb, n0 = sum(fa.values()), sum(fb.values()), sum(prior.values())
    out = []
    for w, a0 in prior.items():
        ya, yb = fa.get(w, 0), fb.get(w, 0)
        if ya < min_count:
            continue
        a = a0 / n0 * 500.0                       # масштаб априорного распределения
        la = math.log((ya + a) / (na + 500.0 - ya - a))
        lb = math.log((yb + a) / (nb + 500.0 - yb - a))
        var = 1.0 / (ya + a) + 1.0 / (yb + a)
        out.append((w, (la - lb) / math.sqrt(var), ya, yb))
    return sorted(out, key=lambda x: -x[1])


def main() -> int:
    import database
    rules = load_rules()
    anchor = rules["classify"]["anchor"]
    conn = database.get_connection()
    cur = conn.cursor(name="pol_posts")                  # серверный курсор — без выгрузки всей таблицы
    cur.itersize = 5000
    cur.execute("SELECT text ~* %s AS seed, text FROM news.tg_posts WHERE text ~* %s", (SEEDS, anchor))
    fa, fb, prior = collections.Counter(), collections.Counter(), collections.Counter()
    na = nb = 0
    for seed, text in cur:
        ts = terms(text or "")
        prior.update(ts)
        if seed:
            fa.update(ts); na += 1
        else:
            fb.update(ts); nb += 1
    conn.close()
    ranked = log_odds(fa, fb, prior)
    top = [{"term": w, "z": round(z, 2), "in_negotiation_posts": a, "in_other_political": b}
           for w, z, a, b in ranked[:300]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seed_posts": na, "other_political_posts": nb, "seeds": SEEDS, "top": top},
                  f, ensure_ascii=False, indent=1)
    print(f"ядро переговоров: {na} постов, прочие политические: {nb}")
    for r in top[:300]:
        print(f"{r['z']:7.2f}  {r['in_negotiation_posts']:6d}  {r['in_other_political']:6d}  {r['term']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
