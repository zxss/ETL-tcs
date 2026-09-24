"""
Лимит ночного риска по VaR (доработка 24.09.2026).

Откуда число. Единственная проверенная в программе модель хвостового риска —
историческое моделирование на 250 днях: она прошла Kupiec и Christoffersen на
уровнях 95 % и 99 % на обеих выборках (research/vol/tail_risk.py, волна 4).
GARCH-EVT-копула, которую рекомендует литература, тесты НЕ прошла — слишком
консервативна, поэтому за основу взята простая модель.

Замер: ночной VaR 99 % прокси-корзины (топ-5 по обороту, равный вес,
close → следующее открытие) составил **2,3 % от суммарной стоимости корзины**
(2,31 % историческое моделирование, 2,28 % EVT — сходятся). То есть раз в сто
ночей корзина теряет больше 2,3 % своей стоимости.

Что это значит в рублях: 5 позиций по 100 000 ₽ = 500 000 ₽ экспозиции →
ожидаемый убыток худшей ночи из ста около 11 500 ₽.

ОГРАНИЧЕНИЯ, которые надо знать, прежде чем включать:
  * оценка откалибрована на корзине из ПЯТИ бумаг. Чем меньше бумаг, тем хуже
    диверсификация и тем выше настоящий VaR — для N < 5 оценка расширяется
    множителем √(5/N), это консервативная поправка, а не замер;
  * корзина прокси, а не сам BEST_TRADES: правило контура зависит от прогнозов,
    которых на всей истории нет;
  * VaR не является максимумом убытка. Один раз из ста бывает хуже, и насколько
    хуже — говорит ES, а не VaR.

ПО УМОЛЧАНИЮ ЛИМИТ ВЫКЛЮЧЕН (OVERNIGHT_VAR_BUDGET_RUB = 0). Включение меняет
поведение торгового контура и является решением пользователя.
"""
from __future__ import annotations

import math

import config

# Ночной VaR 99 % в % от стоимости корзины (замер на holdout 2024-05…2026-09).
DEFAULT_VAR_PCT = 2.3
CALIBRATED_NAMES = 5


def var_pct() -> float:
    return float(getattr(config, "OVERNIGHT_VAR_PCT", DEFAULT_VAR_PCT))


def budget_rub() -> float:
    """0 — лимит выключен (поведение контура прежнее)."""
    return float(getattr(config, "OVERNIGHT_VAR_BUDGET_RUB", 0.0) or 0.0)


def enabled() -> bool:
    return budget_rub() > 0.0


def basket_var_rub(notional_rub: float, n_names: int) -> float:
    """Ночной VaR 99 % корзины в рублях.

    Для N < 5 множитель √(5/N): диверсификации меньше, риск выше. Для N ≥ 5
    оценка не занижается — множитель не опускается ниже 1.
    """
    if notional_rub <= 0 or n_names <= 0:
        return 0.0
    widen = math.sqrt(CALIBRATED_NAMES / n_names) if n_names < CALIBRATED_NAMES else 1.0
    return notional_rub * var_pct() / 100.0 * widen


def trim_to_budget(orders: list[dict], *, budget: float | None = None,
                   amount_key: str = "total_rub",
                   rank_key: str = "final_score") -> tuple[list[dict], list[dict], dict]:
    """Урезает ночную корзину так, чтобы её VaR укладывался в бюджет.

    Отбрасываются заявки с наименьшим rank_key (худшие по оценке стратегии).
    Возвращает (оставленные, отброшенные, справка).
    """
    b = budget_rub() if budget is None else float(budget)
    total = sum(float(o.get(amount_key) or 0.0) for o in orders)
    info = {"enabled": b > 0.0, "budget_rub": b, "var_pct": var_pct(),
            "notional_before_rub": round(total, 2),
            "var_before_rub": round(basket_var_rub(total, len(orders)), 2),
            "dropped": 0}
    if b <= 0.0 or not orders:
        info["notional_after_rub"] = round(total, 2)
        info["var_after_rub"] = info["var_before_rub"]
        return list(orders), [], info

    kept = sorted(orders, key=lambda o: (o.get(rank_key) is None, -(o.get(rank_key) or 0.0)))
    dropped: list[dict] = []
    while kept:
        amt = sum(float(o.get(amount_key) or 0.0) for o in kept)
        if basket_var_rub(amt, len(kept)) <= b:
            break
        dropped.append(kept.pop())
    amt = sum(float(o.get(amount_key) or 0.0) for o in kept)
    info["notional_after_rub"] = round(amt, 2)
    info["var_after_rub"] = round(basket_var_rub(amt, len(kept)), 2)
    info["dropped"] = len(dropped)
    return kept, dropped, info
