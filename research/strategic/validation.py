"""
Валидация стратегического исследования (ТЗ 18.09.2026, шаги 2–4).

CPCV — комбинаторная очищенная кросс-валидация (López de Prado, гл. 12):
календарь режется на N блоков, тестом по очереди становится каждая пара
блоков (C(5,2) = 10 разбиений), из обучения удаляются наблюдения, чьи метки
пересекаются с тестом (purging), плюс карантин в E дней после теста (embargo).
Так out-of-sample оценка не подглядывает в тест через перекрывающиеся окна.

PBO — вероятность переподгонки (Bailey, Borwein, López de Prado, Zhu 2017),
считается методом CSCV по матрице «блок × конфигурация»: если конфигурация,
лучшая на обучении, в тесте систематически оказывается ниже медианы, стратегия
подогнана. PBO < 40 % — гейт разработки.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── CPCV ─────────────────────────────────────────────────────────────────────

def date_blocks(dates: list[dt.date], n_blocks: int) -> list[list[dt.date]]:
    """Непрерывные блоки календаря примерно равной длины."""
    uniq = sorted(set(dates))
    if n_blocks < 2 or len(uniq) < n_blocks:
        raise ValueError(f"мало дат ({len(uniq)}) для {n_blocks} блоков")
    return [list(a) for a in np.array_split(np.array(uniq, dtype=object), n_blocks)]


def cpcv_splits(dates: list[dt.date], n_blocks: int = 5, k_test: int = 2,
                embargo_days: int = 5, label_days: int = 5) -> list[dict]:
    """Все разбиения CPCV с очисткой и карантином.

    Возвращает список {'test': set дат, 'train': set дат, 'blocks': (i, j)}.
    Число независимых путей бэктеста = C(N,k)·k/N.
    """
    blocks = date_blocks(dates, n_blocks)
    uniq = sorted(set(dates))
    out = []
    for combo in itertools.combinations(range(n_blocks), k_test):
        test = sorted(d for i in combo for d in blocks[i])
        test_set = set(test)
        train = []
        for d in uniq:
            if d in test_set:
                continue
            # purging: метка обучающего наблюдения не должна заходить в тест
            if any(0 <= (t - d).days <= label_days for t in test):
                continue
            # embargo: карантин после каждого тестового блока
            if any(0 < (d - t).days <= embargo_days for t in test):
                continue
            train.append(d)
        out.append({"blocks": combo, "test": test_set, "train": set(train)})
    return out


def n_paths(n_blocks: int = 5, k_test: int = 2) -> int:
    return math.comb(n_blocks, k_test) * k_test // n_blocks


# ── PBO ──────────────────────────────────────────────────────────────────────

def cscv_pbo(perf: pd.DataFrame, s: int = 8) -> dict:
    """PBO по матрице доходностей (строки — отрезки времени, столбцы — конфигурации).

    perf должна содержать ≥ s строк и ≥ 2 столбцов. Возвращает PBO и медиану
    логита ранга: отрицательная медиана означает, что выбор лучшей конфигурации
    в среднем не переносится на новые данные.
    """
    perf = perf.dropna(how="all").dropna(axis=1, how="any")
    n_cfg = perf.shape[1]
    if n_cfg < 2 or len(perf) < s:
        return {"pbo": None, "configs": int(n_cfg), "rows": int(len(perf)),
                "note": "недостаточно конфигураций или отрезков"}
    s = s - s % 2
    parts = [p for p in np.array_split(np.arange(len(perf)), s)]
    lam = []
    for train_ids in itertools.combinations(range(s), s // 2):
        tr = np.concatenate([parts[i] for i in train_ids])
        te = np.concatenate([parts[i] for i in range(s) if i not in train_ids])
        is_perf = perf.iloc[tr].mean()
        oos_perf = perf.iloc[te].mean()
        best = is_perf.idxmax()
        rank = float(oos_perf.rank(ascending=True)[best])          # 1 — худшая
        w = rank / (n_cfg + 1.0)
        lam.append(math.log(w / (1.0 - w)))
    lam = np.asarray(lam, float)
    return {"pbo": float((lam <= 0).mean()), "median_logit": float(np.median(lam)),
            "configs": int(n_cfg), "combinations": int(len(lam))}


# ── Метрики и гейты ──────────────────────────────────────────────────────────

def t_by_date(values, dates) -> float | None:
    s = pd.Series(np.asarray(values, float), index=np.asarray(dates)).dropna()
    if s.empty:
        return None
    g = s.groupby(level=0).mean()
    if len(g) < 3 or not g.std(ddof=1):
        return None
    return float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g))))


def max_drawdown(daily_rub: pd.Series) -> float:
    eq = daily_rub.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def deflated_sharpe(daily: pd.Series, n_trials: int) -> float | None:
    from scipy import stats
    r = pd.Series(daily).dropna()
    if len(r) < 30 or n_trials < 2 or not r.std(ddof=1):
        return None
    sr = float(r.mean() / r.std(ddof=1))
    var = (1 + 0.5 * sr ** 2) / len(r)
    g = 0.5772156649015329
    sr0 = math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                            + g * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / den))


def summarize(trades: pd.DataFrame, d_from: dt.date, d_to: dt.date, capital_rub: float,
              n_trials: int | None = None) -> dict:
    """Сводка по таблице сделок с колонками entry_day/exit_day/notional/
    net_excess_pct/pnl_excess_rub."""
    if trades is None or trades.empty:
        return {"trades": 0}
    tr = trades.copy()
    tr["hold_days"] = [max(1, (b - a).days) for a, b in zip(tr["entry_day"], tr["exit_day"])]
    deployed = float((tr["notional"] * tr["hold_days"] / 365.0).sum())      # рубле-годы
    excess = float(tr["pnl_excess_rub"].sum())
    years = max(1e-9, ((d_to - d_from).days + 1) / 365.25)
    days = [d.date() for d in pd.bdate_range(d_from, d_to)]
    daily = (tr.groupby("exit_day")["pnl_excess_rub"].sum().reindex(days).fillna(0.0))
    rel = daily / capital_rub
    sd = rel.std(ddof=1)
    out = {"trades": int(len(tr)), "trades_per_year": len(tr) / years,
           "dates": int(pd.Series(tr["entry_day"]).nunique()),
           "avg_hold_days": float(tr["hold_days"].mean()),
           "net_excess_pct_trade": float(tr["net_excess_pct"].mean()),
           "win_rate": float((tr["net_excess_pct"] > 0).mean()),
           "t": t_by_date(tr["net_excess_pct"], tr["entry_day"]),
           "excess_annual_pct": excess / deployed * 100.0 if deployed > 0 else None,
           "contribution_pct_capital_year": excess / capital_rub / years * 100.0,
           "pnl_excess_rub": excess,
           "ir": float(rel.mean() / sd * math.sqrt(TRADING_DAYS)) if sd else None,
           "mdd_rub": max_drawdown(daily)}
    if n_trials:
        out["dsr"] = deflated_sharpe(rel, max(2, n_trials))
        out["trials"] = int(n_trials)
    return out


def dev_gate(summary: dict, pbo: float | None, capacity_rub: float | None, gates: dict) -> tuple[bool, list[str]]:
    """Гейт шага 3: все условия одновременно."""
    fails = []
    ex, t = summary.get("excess_annual_pct"), summary.get("t")
    if ex is None or ex < gates["excess_annual_min_pct"]:
        fails.append(f"сверх фонда {'—' if ex is None else f'{ex:+.2f}'} % год < {gates['excess_annual_min_pct']}")
    if t is None or t < gates["t_min"]:
        fails.append(f"t {'—' if t is None else f'{t:+.2f}'} < {gates['t_min']} (Harvey–Liu–Zhu)")
    if pbo is None or pbo >= gates["pbo_max"]:
        fails.append(f"PBO {'—' if pbo is None else f'{pbo:.0%}'} ≥ {gates['pbo_max']:.0%}")
    if capacity_rub is None or capacity_rub < gates["capacity_min_rub"]:
        cap = "—" if capacity_rub is None else f"{capacity_rub:,.0f}".replace(",", " ")
        fails.append(f"ёмкость {cap} ₽ < {gates['capacity_min_rub']:,.0f} ₽".replace(",", " "))
    return (not fails), fails


def holdout_gate(summary: dict, gates: dict) -> tuple[bool, list[str]]:
    fails = []
    t, ex, dsr = summary.get("t"), summary.get("excess_annual_pct"), summary.get("dsr")
    if t is None or t < gates["t_min"]:
        fails.append(f"t {'—' if t is None else f'{t:+.2f}'} < {gates['t_min']}")
    if ex is None or ex <= gates["net_excess_min_pct"]:
        fails.append(f"сверх фонда {'—' if ex is None else f'{ex:+.2f}'} % год ≤ {gates['net_excess_min_pct']}")
    if dsr is None or dsr < gates["dsr_min"]:
        fails.append(f"DSR {'—' if dsr is None else f'{dsr:.3f}'} < {gates['dsr_min']}")
    return (not fails), fails
