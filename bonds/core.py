"""
Ядро модуля ВДО: правила, рейтинговая шкала, отбор выпусков и риск-ограничения.

Чистые функции без сети и брокера — одни и те же для бэктеста (bonds/backtest.py)
и живой торговли (bonds/trader.py). Правила — последняя версия bonds/rules_wNN.json.

Выпуск на входе — словарь:
  secid, isin, issuer, sector, yield_pct (к ближайшему событию), duration_y,
  price_pct (% номинала), days_to_event, value_rub (оборот), bid_ask_pct | None,
  issue_size_rub | None, rating_num | None, negative_watch (bool),
  floating, subordinated, perpetual, qual_only, required_tests (bool флаги),
  tradable (bool).
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

# Национальная шкала: AAA = 1 … D = 22. Разные агентства пишут по-разному:
# ruAA- (Эксперт РА), AA-(RU) (АКРА), A.ru (НКР), A|ru| (НРА).
SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
         "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
_NUM = {r: i + 1 for i, r in enumerate(SCALE)}
_RX = re.compile(r"(AAA|AA|A|BBB|BB|B|CCC|CC|C|D)([+-]?)")


def rating_num(text: str | None) -> int | None:
    """'ruBB+' / 'BB+(RU)' / 'BB+.ru' / 'BB+|ru|' → 11. Нераспознанное → None."""
    if not text:
        return None
    s = str(text).upper().replace("RU", "").replace("(", " ").replace("|", " ")
    m = _RX.search(s.replace(".", " "))
    return _NUM.get(m.group(1) + m.group(2)) if m else None


def rating_name(n: int | None) -> str:
    return SCALE[n - 1] if n and 1 <= n <= len(SCALE) else "—"


def worst_rating(texts) -> int | None:
    """Консервативно: худший из текущих рейтингов агентств."""
    nums = [n for n in (rating_num(t) for t in texts) if n]
    return max(nums) if nums else None


def load_rules(path: str | None = None) -> dict:
    """Последняя версия правил rules_wNN.json (или явный путь)."""
    if path is None:
        files = sorted(glob.glob(os.path.join(HERE, "rules_w*.json")))
        if not files:
            raise FileNotFoundError("нет bonds/rules_wNN.json")
        path = files[-1]
    with open(path, encoding="utf-8") as f:
        rules = json.load(f)
    rules["_path"] = os.path.basename(path)
    return rules


# ── отбор ────────────────────────────────────────────────────────────────────

def exclusion_reason(b: dict, rules: dict, key_rate: float, use_credit: bool = True) -> str | None:
    """Почему выпуск не проходит фильтры (None — проходит)."""
    u, y, p, d, lq, cr = (rules["universe"], rules["yield"], rules["price"],
                          rules["duration_years"], rules["liquidity"], rules["credit"])
    if u.get("exclude_qual_only") and b.get("qual_only"):
        return "только для квалов"
    if u.get("exclude_required_tests") and b.get("required_tests"):
        return "нужен тест"
    if u.get("exclude_floating_coupon") and b.get("floating"):
        return "плавающий купон"
    if u.get("exclude_subordinated") and b.get("subordinated"):
        return "субординированный"
    if u.get("exclude_perpetual") and b.get("perpetual"):
        return "бессрочный"
    if not b.get("tradable", True):
        return "не торгуется через API"
    if b.get("issue_size_rub") is not None and b["issue_size_rub"] < u["min_issue_size_rub"]:
        return "мал объём выпуска"
    if b.get("days_to_event") is None or b["days_to_event"] < u["min_days_to_event"]:
        return "близко событие"
    if b.get("yield_pct") is None:
        return "нет доходности"
    spread = b["yield_pct"] - key_rate
    if spread < y["min_spread_to_key_pp"]:
        return "мала доходность"
    if spread > y["max_spread_to_key_pp"]:
        return "доходность дистресса"
    if b.get("price_pct") is None or not p["min_pct_of_face"] <= b["price_pct"] <= p["max_pct_of_face"]:
        return "цена вне коридора"
    if b.get("duration_y") is None or not d["min"] <= b["duration_y"] <= d["max"]:
        return "дюрация вне коридора"
    if (b.get("value_rub") or 0) < lq["min_day_value_rub"]:
        return "мала ликвидность"
    if b.get("bid_ask_pct") is not None and b["bid_ask_pct"] > lq["max_bid_ask_pct"]:
        return "широкий спред"
    if use_credit:
        floor, cap = rating_num(cr["rating_floor"]), rating_num(cr["rating_cap"])
        r = b.get("rating_num")
        if r is None:
            if cr.get("require_rating"):
                return "нет рейтинга"
        elif r > floor:
            return "рейтинг ниже порога"
        elif r < cap:
            return "рейтинг выше коридора ВДО"
        if cr.get("exclude_negative_watch") and b.get("negative_watch"):
            return "негативный прогноз"
    return None


def bucket(b: dict, use_credit: bool) -> str:
    """Корзина сравнения: рейтинг (живая система) или дюрация (бэктест)."""
    if use_credit and b.get("rating_num"):
        return f"r{b['rating_num']}"
    d = b.get("duration_y") or 0
    return "d0-1" if d < 1 else ("d1-2" if d < 2 else "d2+")


def rank(eligible: list[dict], use_credit: bool) -> list[dict]:
    """Дешевизна: доходность минус медиана своей корзины; по убыванию."""
    by = {}
    for b in eligible:
        by.setdefault(bucket(b, use_credit), []).append(b["yield_pct"])
    med = {k: statistics.median(v) for k, v in by.items()}
    out = [{**b, "score": b["yield_pct"] - med[bucket(b, use_credit)]} for b in eligible]
    return sorted(out, key=lambda x: (-x["score"], x["secid"]))


def sector_key(b: dict) -> str:
    """Неизвестный сектор — отдельная корзина на выпуск, а не общая «?»: иначе лимит
    на сектор режет портфель до 5 бумаг, когда сектора нет (бэктест, 27.09.2026)."""
    return b.get("sector") or f"?{b['secid']}"


def select(ranked: list[dict], rules: dict, holdings: set[str] | None = None) -> list[dict]:
    """Целевой портфель с ограничениями. Удерживаемые выпуски в пределах
    hold_buffer_rank идут первыми (меньше оборота), остальное — по рангу."""
    pf = rules["portfolio"]
    n, w = pf["n_bonds"], min(pf["max_weight_bond"], (1 - pf["cash_reserve"]) / pf["n_bonds"])
    holdings = holdings or set()
    buffer_ids = {b["secid"] for b in ranked[: pf["hold_buffer_rank"]]}
    order = ([b for b in ranked if b["secid"] in holdings and b["secid"] in buffer_ids]
             + [b for b in ranked if not (b["secid"] in holdings and b["secid"] in buffer_ids)])
    picked, issuers, sectors = [], set(), {}
    for b in order:
        if len(picked) >= n:
            break
        iss = b.get("issuer") or b["secid"]
        if iss in issuers:
            continue
        sec = sector_key(b)
        if (sectors.get(sec, 0) + 1) * w > pf["max_weight_sector"] + 1e-9:
            continue
        cand = picked + [b]
        dur = sum(x["duration_y"] for x in cand) * w / (len(cand) * w)
        if dur > pf["max_portfolio_duration"] + 1e-9:
            continue
        picked.append(b)
        issuers.add(iss)
        sectors[sec] = sectors.get(sec, 0) + 1
    return [{**b, "weight": w} for b in picked]


def check_portfolio(port: list[dict], rules: dict) -> list[str]:
    """Нарушения ограничений (пусто — всё соблюдено). Независимый пересчёт."""
    pf, bad = rules["portfolio"], []
    tot = sum(b["weight"] for b in port)
    if tot > 1 - pf["cash_reserve"] + 1e-9:
        bad.append(f"вложено {tot:.3f} > {1 - pf['cash_reserve']:.3f}")
    by_iss, by_sec = {}, {}
    for b in port:
        if b["weight"] > pf["max_weight_bond"] + 1e-9:
            bad.append(f"{b['secid']}: вес {b['weight']:.3f}")
        by_iss[b.get("issuer") or b["secid"]] = by_iss.get(b.get("issuer") or b["secid"], 0) + b["weight"]
        by_sec[sector_key(b)] = by_sec.get(sector_key(b), 0) + b["weight"]
    bad += [f"эмитент {k}: {v:.3f}" for k, v in by_iss.items() if v > pf["max_weight_issuer"] + 1e-9]
    bad += [f"сектор {k}: {v:.3f}" for k, v in by_sec.items() if v > pf["max_weight_sector"] + 1e-9]
    if tot > 0:
        dur = sum(b["weight"] * b["duration_y"] for b in port) / tot
        if dur > pf["max_portfolio_duration"] + 1e-9:
            bad.append(f"дюрация портфеля {dur:.2f}")
    return bad


def build_portfolio(bonds: list[dict], rules: dict, key_rate: float, *,
                    use_credit: bool = True, holdings: set[str] | None = None) -> dict:
    """Вся цепочка: фильтры → ранжирование → отбор → проверка ограничений."""
    reasons, eligible = {}, []
    for b in bonds:
        r = exclusion_reason(b, rules, key_rate, use_credit)
        if r:
            reasons[r] = reasons.get(r, 0) + 1
        else:
            eligible.append(b)
    ranked = rank(eligible, use_credit)
    port = select(ranked, rules, holdings)
    return {"universe": len(bonds), "eligible": len(eligible), "excluded": reasons,
            "portfolio": port, "violations": check_portfolio(port, rules),
            "ranked_ids": [b["secid"] for b in ranked]}


def exit_signals(held: dict, rules: dict, key_rate: float, use_credit: bool = True) -> str | None:
    """Внеплановый выход по удерживаемому выпуску (ежедневный контроль)."""
    ex = rules["exit"]
    if held.get("halted") and ex.get("on_trading_halt"):
        return "остановка торгов"
    if held.get("price_pct") is not None and held["price_pct"] < ex["price_stop_pct_of_face"]:
        return f"цена {held['price_pct']:.1f} < стоп {ex['price_stop_pct_of_face']}"
    if held.get("days_to_event") is not None and held["days_to_event"] < ex["days_to_event_roll"]:
        return "близко оферта/погашение"
    if use_credit and ex.get("on_downgrade_below_floor") and held.get("rating_num") \
            and held["rating_num"] > rating_num(rules["credit"]["rating_floor"]):
        return f"рейтинг {rating_name(held['rating_num'])} ниже порога"
    if use_credit and rules["credit"].get("exclude_negative_watch") and held.get("negative_watch"):
        return "негативный прогноз рейтинга"
    return None
