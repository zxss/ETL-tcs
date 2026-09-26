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

# --- Поправка на режим волатильности (26.09.2026) ----------------------------
# 2,3 % — замер на конкретном окне, и вола в нём не была постоянной: 21-дневная
# реализованная вола IMOEX в окне калибровки шла от 0,64 до 2,24 %/день (5-й и
# 95-й процентили) при медиане 1,3946 %/день. Фиксированный процент означает, что
# в спокойном рынке лимит завышен, а в стрессе — занижен ровно тогда, когда он
# нужен.
#
# Единственное, что программа из 281 испытания подтвердила, — прогнозируемость
# волатильности (HAR-RV, волна 1). Здесь она и применяется: оценка VaR
# расширяется пропорционально отношению текущей волы к медиане окна калибровки.
# Множитель НИКОГДА не опускается ниже 1: занижать откалиброванный уровень в
# спокойном рынке значит торговать риском, которого не замеряли.
VOL_REF_PCT = 1.3946      # медиана 21д реализованной волы IMOEX, %/день
VOL_WINDOW = 21           # окно реализованной волы, торговых дней
VOL_WIDEN_MAX = 2.5       # предохранитель от битой свечи, а не суждение о риске


def var_pct() -> float:
    return float(getattr(config, "OVERNIGHT_VAR_PCT", DEFAULT_VAR_PCT))


def budget_rub() -> float:
    """0 — лимит выключен (поведение контура прежнее)."""
    return float(getattr(config, "OVERNIGHT_VAR_BUDGET_RUB", 0.0) or 0.0)


def enabled() -> bool:
    return budget_rub() > 0.0


def vol_widen(conn=None, index_ticker: str = "IMOEX") -> tuple[float, dict]:
    """Множитель расширения VaR по текущему режиму волатильности.

    Возвращает (множитель ≥ 1, справка). Мера режима — 21-дневная реализованная
    вола индекса относительно медианы окна, на котором замерён VaR. Ниже 1
    множитель не опускается: спокойный рынок не повод торговать риском, который
    не замеряли.

    Любая проблема с данными → (1.0, причина): лимит остаётся на
    откалиброванном уровне, а не отключается и не раздувается.
    """
    info = {"scaling": bool(getattr(config, "OVERNIGHT_VAR_VOL_SCALING", True)),
            "vol_ref_pct": VOL_REF_PCT, "vol_now_pct": None, "widen": 1.0}
    if not info["scaling"]:
        info["reason"] = "выключено OVERNIGHT_VAR_VOL_SCALING"
        return 1.0, info
    try:
        import database
        with (database.get_db_connection() if conn is None else _keep(conn)) as c:
            with c.cursor() as cur:
                cur.execute("SELECT close FROM market_data WHERE ticker = %s "
                            "ORDER BY date DESC LIMIT %s",
                            (index_ticker, VOL_WINDOW + 1))
                rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001 — лимит важнее замера режима
        info["reason"] = f"индекс недоступен: {e}"
        return 1.0, info
    closes = [float(r[0]) for r in reversed(rows) if r[0] and float(r[0]) > 0]
    if len(closes) < VOL_WINDOW + 1:
        info["reason"] = f"истории индекса {len(closes)} < {VOL_WINDOW + 1} дней"
        return 1.0, info
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    now_pct = math.sqrt(var) * 100.0
    widen = min(VOL_WIDEN_MAX, max(1.0, now_pct / VOL_REF_PCT)) if VOL_REF_PCT > 0 else 1.0
    info["vol_now_pct"] = round(now_pct, 4)
    info["widen"] = round(widen, 4)
    return widen, info


def basket_var_rub(notional_rub: float, n_names: int,
                   vol_factor: float = 1.0) -> float:
    """Ночной VaR 99 % корзины в рублях.

    Для N < 5 множитель √(5/N): диверсификации меньше, риск выше. Для N ≥ 5
    оценка не занижается — множитель не опускается ниже 1.
    vol_factor — поправка на режим волатильности (см. vol_widen); 1.0 означает
    вола как в окне калибровки.
    """
    if notional_rub <= 0 or n_names <= 0:
        return 0.0
    widen = math.sqrt(CALIBRATED_NAMES / n_names) if n_names < CALIBRATED_NAMES else 1.0
    return notional_rub * var_pct() / 100.0 * widen * max(1.0, float(vol_factor))


class _keep:
    """Контекст, отдающий уже открытый коннект и не закрывающий его."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


def trim_to_budget(orders: list[dict], *, budget: float | None = None,
                   amount_key: str = "total_rub",
                   rank_key: str = "final_score",
                   vol_factor: float = 1.0) -> tuple[list[dict], list[dict], dict]:
    """Урезает ночную корзину так, чтобы её VaR укладывался в бюджет.

    Отбрасываются заявки с наименьшим rank_key (худшие по оценке стратегии).
    vol_factor замеряется вызывающим через vol_widen(): здесь нет обращений к
    БД, чтобы расчёт лимита оставался чистой функцией от входа.
    Возвращает (оставленные, отброшенные, справка).
    """
    b = budget_rub() if budget is None else float(budget)
    vf = max(1.0, float(vol_factor))
    total = sum(float(o.get(amount_key) or 0.0) for o in orders)
    info = {"enabled": b > 0.0, "budget_rub": b, "var_pct": var_pct(),
            "vol_factor": round(vf, 4),
            "notional_before_rub": round(total, 2),
            "var_before_rub": round(basket_var_rub(total, len(orders), vf), 2),
            "dropped": 0}
    if b <= 0.0 or not orders:
        info["notional_after_rub"] = round(total, 2)
        info["var_after_rub"] = info["var_before_rub"]
        return list(orders), [], info

    kept = sorted(orders, key=lambda o: (o.get(rank_key) is None, -(o.get(rank_key) or 0.0)))
    dropped: list[dict] = []
    while kept:
        amt = sum(float(o.get(amount_key) or 0.0) for o in kept)
        if basket_var_rub(amt, len(kept), vf) <= b:
            break
        dropped.append(kept.pop())
    amt = sum(float(o.get(amount_key) or 0.0) for o in kept)
    info["notional_after_rub"] = round(amt, 2)
    info["var_after_rub"] = round(basket_var_rub(amt, len(kept), vf), 2)
    info["dropped"] = len(dropped)
    return kept, dropped, info
