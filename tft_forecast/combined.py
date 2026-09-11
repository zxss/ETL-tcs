"""
Сводная итоговая таблица: одна строка на связку (ticker + strategy), все
ключевые метрики из контура валидации (White RC / SPA / PBO / FDR / Ruin30 /
LB / Verdict) и из TFT-прогноза (направленный PnL коридора + ценовой диапазон).

Заменяет четыре отдельные таблицы (валидация, диапазон по тикерам, диапазон
по стратегиям, направленный PnL) единым представлением для принятия решения.

Для дневных стратегий (intraday_short / intraday_long) система автоматически
оставляет по каждому тикеру ТОЛЬКО ОДНУ — лучшую — идею (best_intraday),
чтобы не показывать одновременно два противоположных направления. Ночная
стратегия long_overnight остаётся всегда.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger("tft.combined")

# ── Цвет (ANSI) ────────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color_enabled() -> bool:
    return sys.stdout.isatty()


def _wrap(text: str, color: str) -> str:
    if not color or not _color_enabled():
        return text
    return f"{color}{text}{_RESET}"


# Конкурирующие дневные стратегии — из них выбирается одна лучшая.
DAILY_STRATS = ("intraday_short", "intraday_long")

# Направление сделки по названию стратегии.
_DIRECTION = {
    "long_overnight": "LONG",
    "intraday_long":  "LONG",
    "long_intraday":  "LONG",
    "intraday_short": "SHORT",
    "short_hold":     "SHORT",
}


def _direction(strategy: str) -> str:
    return _DIRECTION.get(strategy, "—")


# Канонический вердикт → (короткая метка, приоритет сортировки, цвет)
_VERDICT_INFO = {
    "CANDIDATE EDGE":      ("CAND", 0, _GREEN),
    "WEAK / INCONCLUSIVE": ("WEAK", 1, _YELLOW),
    "REJECTED":            ("REJ",  2, _RED),
}
_UNKNOWN = ("—", 1.5, "")


def _verdict_meta(verdict: str | None):
    if verdict is None:
        return _UNKNOWN
    return _VERDICT_INFO.get(verdict, _UNKNOWN)


def _pnl_color(v: float | None) -> str:
    if v is None:
        return ""
    return _GREEN if v > 0 else _RED


def _prob_color(p: float | None) -> str:
    if p is None:
        return ""
    if p > 0.60:
        return _GREEN
    if p >= 0.50:
        return _YELLOW
    return _RED


# ── Сборка строк ───────────────────────────────────────────────────────────────

def _build_rows(val_rows, forecasts, tickers, strats):
    val_idx = {}
    for r in (val_rows or []):
        val_idx[(r["ticker"].upper(), r["strategy"])] = r

    rows = []
    for tk in tickers:
        TK = tk.upper()
        cor = (forecasts or {}).get(TK) or (forecasts or {}).get(tk) or {}
        dirmap = cor.get("directional", {}) if isinstance(cor, dict) else {}
        for st in strats:
            v = val_idx.get((TK, st))
            d = dirmap.get(st, {})
            rows.append({
                "ticker": TK,
                "strategy": st,
                "direction": _direction(st),
                "selected": None,   # проставляется при отборе лучшей дневной
                "verdict": v["verdict"] if v else None,
                "white_rc": v["white_rc_p"] if v else None,
                "spa": v["spa_p"] if v else None,
                "pbo": v["pbo"] if v else None,
                "fdr": v["fdr_pass"] if v else None,
                "ruin30": v["ruin30"] if v else None,
                "lb": v["lb_struct"] if v else None,
                "exp_pnl": d.get("ExpPnL"),
                "prob_profit": d.get("ProbProfit"),
                "down": d.get("Downside"),
                "up": d.get("Upside"),
                "f_low": cor.get("ForecastLow") if isinstance(cor, dict) else None,
                "f_high": cor.get("ForecastHigh") if isinstance(cor, dict) else None,
                "range_pct": cor.get("RangePct") if isinstance(cor, dict) else None,
                "coverage": cor.get("CoverageProb") if isinstance(cor, dict) else None,
                "anchor_price": cor.get("anchor_price") if isinstance(cor, dict) else None,
                "last_date": cor.get("last_date") if isinstance(cor, dict) else None,
                "price_ts": cor.get("price_ts") if isinstance(cor, dict) else None,
                "max_pos": cor.get("MaxPos") if isinstance(cor, dict) else None,
                "liq_score": cor.get("LiqScore") if isinstance(cor, dict) else None,
                # рыночный контекст (Dashboard 2.0)
                "regime": cor.get("Regime") if isinstance(cor, dict) else None,
                "rs": cor.get("RS") if isinstance(cor, dict) else None,
                "vol_spike": cor.get("VolSpike") if isinstance(cor, dict) else None,
                "atr_pctl": cor.get("ATRpctl") if isinstance(cor, dict) else None,
                "atr_pct": cor.get("ATRpct") if isinstance(cor, dict) else None,
                "ret1": cor.get("Ret1") if isinstance(cor, dict) else None,
                "market_atr_pctl": (cor.get("MarketATRpctl")
                                    if isinstance(cor, dict) else None),
                "index_above_ema50": (cor.get("IndexAboveEMA50")
                                      if isinstance(cor, dict) else None),
                "cost_rt": cor.get("CostRT") if isinstance(cor, dict) else None,
                "gap_down_prob": cor.get("GapDownProb") if isinstance(cor, dict) else None,
            })
    return rows


# ── Отбор лучшей дневной стратегии ─────────────────────────────────────────────

_EXP_TIE = 0.05   # порог «близости» по ExpPnL%, при котором включаются тай-брейки


def _daily_better(a, b) -> bool:
    """True если дневная стратегия a лучше b по правилам ТЗ:
    ExpPnL% → (при разнице <0.05%) ProbProfit → Verdict → Ruin30."""
    ea = a["exp_pnl"] if a["exp_pnl"] is not None else -1e9
    eb = b["exp_pnl"] if b["exp_pnl"] is not None else -1e9
    if abs(ea - eb) >= _EXP_TIE:
        return ea > eb
    # тай-брейк 1: ProbProfit (выше лучше)
    pa = a["prob_profit"] if a["prob_profit"] is not None else -1.0
    pb = b["prob_profit"] if b["prob_profit"] is not None else -1.0
    if pa != pb:
        return pa > pb
    # тай-брейк 2: Verdict (ниже ранг = лучше)
    ra = _verdict_meta(a["verdict"])[1]
    rb = _verdict_meta(b["verdict"])[1]
    if ra != rb:
        return ra < rb
    # тай-брейк 3: Ruin30 (ниже лучше)
    ua = a["ruin30"] if a["ruin30"] is not None else 1e9
    ub = b["ruin30"] if b["ruin30"] is not None else 1e9
    return ua < ub


def _apply_selection(rows, show_all: bool):
    """Помечает selected=✓/- у дневных стратегий и (если не show_all)
    убирает проигравшие дневные. Возвращает отфильтрованный список."""
    by_tk: dict[str, list] = {}
    for r in rows:
        by_tk.setdefault(r["ticker"], []).append(r)

    out = []
    for tk, group in by_tk.items():
        daily = [r for r in group if r["strategy"] in DAILY_STRATS]
        non_daily = [r for r in group if r["strategy"] not in DAILY_STRATS]

        winner = None
        if daily:
            winner = daily[0]
            for cand in daily[1:]:
                if _daily_better(cand, winner):
                    winner = cand
            for r in daily:
                r["selected"] = (r is winner)

        out.extend(non_daily)
        if show_all:
            out.extend(daily)
        elif winner is not None:
            out.append(winner)
    return out


def _sort_key(r):
    _, rank, _ = _verdict_meta(r["verdict"])
    exp = r["exp_pnl"] if r["exp_pnl"] is not None else -1e9
    prob = r["prob_profit"] if r["prob_profit"] is not None else -1.0
    # вердикт по возрастанию приоритета, ExpPnL и ProbProfit — по убыванию
    return (rank, -exp, -prob)


# ── Итоговый рейтинг с рыночным контекстом (Dashboard 2.0) ─────────────────────

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _apply_penalty(score: float, factor: float) -> float:
    """Знак-безопасное применение риск-штрафа.

    Штраф обязан УХУДШАТЬ рейтинг независимо от знака. Простое умножение это
    свойство ломает на отрицательных значениях: −0.5 × 0.70 = −0.35, то есть
    штраф ПОДНИМАЕТ плохой сигнал вверх по рейтингу. В режиме эвристики
    (score ∈ [0,1]) проблемы нет, но в режиме сырой альфы score — это ExpPnL,
    который регулярно отрицателен, поэтому штраф на отрицательной стороне
    применяется делением.
    """
    if factor <= 0:
        return score
    return score * factor if score >= 0 else score / factor


def _risk_penalties(r, long: bool) -> tuple[float, list[str], bool]:
    """Мультипликативные риск-штрафы и флаги — общие для обоих режимов.

    Возвращает (множитель, флаги, severe): severe=True означает, что сработало
    условие, по которому бумагу можно не просто штрафовать, а отсекать
    (климакс объёма или высокий риск гэпа вниз).

    Штраф за режим рынка сюда НЕ входит: см. _score_row.
    """
    rs = r["rs"]
    vs = r["vol_spike"]
    gdp = r["gap_down_prob"]
    factor = 1.0
    flags: list[str] = []
    severe = False

    # High Risk Short: SHORT при сильной бумаге (RS > +5%) → −25%.
    # Здесь rs используется как РИСК-ФИЛЬТР одного края, а не как компонент
    # рейтинга: как компонент он шум (IC −0.002), а на long_overnight вреден.
    if (not long) and rs is not None and rs > 5.0:
        factor *= 0.75
        flags.append("High Risk Short")

    # Климакс объёма.
    if vs is not None and vs > 2.5:
        flags.append("⚠ Volume Climax")
    if vs is not None and vs > 4.0:
        factor *= 0.70
        severe = True

    # Риск гэпа вниз — только для ночной стратегии.
    if r["strategy"] == "long_overnight" and gdp is not None:
        if gdp > 0.40:
            flags.append("⚠ High Overnight Risk")
        if gdp > 0.50:
            factor *= 0.70
            severe = True

    return factor, flags, severe


SCORE_MODES = ("heuristic", "raw_alpha", "trade_score")

# Пол для знаменателя нормировки на волатильность: ниже 0.2% ATR не бывает у
# ликвидных бумаг, а деление на околонулевую величину даёт выбросы в рейтинге.
_MIN_ATR_PCT = 0.2


def _trade_score(r) -> float | None:
    """Прогноз, нормированный на волатильность: ExpPnL / ATR%.

    Смысл: бумага с ожиданием +0,8% при ATR 1% должна стоять выше бумаги с
    ожиданием +1,2% при ATR 3% — вторая просто шумнее, а не лучше.

    ВАЖНО: издержки здесь ВТОРОЙ РАЗ НЕ ВЫЧИТАЮТСЯ. ExpPnL уже приходит нетто
    round-trip (см. directional.strategy_pnl: exp_net = med - cost_rt), поэтому
    формула вида (exp_pnl - CostRT) / ATR% вычла бы издержки дважды.

    Знаменатель — ATR(14)/close*100 (market._atr_pct), а НЕ ATRpctl: перцентиль
    сравнивает бумагу с её собственной историей и между бумагами несопоставим.
    При отсутствии ATR% откатываемся на ширину прогнозного коридора RangePct —
    это тоже волатильность в процентах, только оценённая моделью.
    """
    exp = r.get("exp_pnl")
    if exp is None:
        return None
    vol = r.get("atr_pct")
    if vol is None or not vol or vol != vol:
        rng = r.get("range_pct")
        # Коридор q0.1..q0.9 примерно вчетверо шире дневного ATR — приводим
        # к сопоставимому масштабу, чтобы режим не менял смысл при откате.
        vol = (rng / 4.0) if rng else None
    if vol is None or vol != vol:
        return None
    return exp / max(float(vol), _MIN_ATR_PCT)


def _resolve_mode(raw_alpha, mode):
    """Обратная совместимость: булев raw_alpha старше строкового mode."""
    if raw_alpha is True:
        return "raw_alpha"
    if raw_alpha is False and mode is None:
        return "heuristic"
    if mode:
        m = str(mode).strip().lower()
        return m if m in SCORE_MODES else "heuristic"
    import config as _cfg
    m = str(getattr(_cfg, "SCORE_MODE", "heuristic")).strip().lower()
    return m if m in SCORE_MODES else "heuristic"


def _score_row(r, strict: bool, raw_alpha: bool | None = None,
               hard_exclude: bool | None = None,
               mode: str | None = None,
               apply_penalties: bool | None = None):
    """
    Считает итоговый рейтинг строки, флаги риска и допустимость сделки.
    Возвращает (final_score, allowed, flags).

    ТРИ РЕЖИМА (config.SCORE_MODE):

      heuristic (ДЕФОЛТ) — exp 0.70 / prob 0.20 / liq 0.10, результат в [0,1].
        Веса пересчитаны после аудита: из формулы убраны rs, regime и vol,
        чей измеренный вклад неотличим от нуля, и validation_score (вердикт
        теперь работает отдельным гейтом, а не слагаемым весом 0.05).

      raw_alpha — Score = ExpPnL. Даёт максимальный Rank IC (+0.054 против
        +0.051 у эвристики), но ХУДШИЙ портфель: модуль ExpPnL связан с
        волатильностью (корреляция с шириной коридора +0.285), поэтому топ-10
        набирается из бумаг с широким размахом.

      trade_score — Score = ExpPnL / ATR%. Нормировка прогноза на риск.
        Измеренный результат: волатильность топ-10 действительно снижается,
        но доходность падает (альфа -5.2% против +2.1% у эвристики), и режим
        неустойчив при расколе выборки. Оставлен как доступный режим, но не
        рекомендован — см. таблицу в config.py.

    Общее для всех режимов: риск-штрафы (_risk_penalties), штраф за контртренд
    и жёсткий рыночный фильтр strict. Штрафы отключаются
    config.APPLY_RISK_PENALTIES=0 (на реплее они ухудшают результат).
    """
    import config as _cfg
    mode = _resolve_mode(raw_alpha, mode)
    if hard_exclude is None:
        hard_exclude = bool(getattr(_cfg, "RAW_ALPHA_HARD_EXCLUDE", True))
    if apply_penalties is None:
        apply_penalties = bool(getattr(_cfg, "APPLY_RISK_PENALTIES", True))

    long = r["direction"] == "LONG"
    regime = r["regime"]
    exp = r["exp_pnl"]

    penalty, flags, severe = _risk_penalties(r, long)
    if not apply_penalties:
        penalty = 1.0

    if mode == "raw_alpha":
        final = exp if exp is not None else 0.0
    elif mode == "trade_score":
        ts = _trade_score(r)
        final = ts if ts is not None else 0.0
    else:
        exp_score = _clamp01(0.5 + (exp or 0.0) / 2.0)      # ±1% → 0..1
        prob_score = r["prob_profit"] if r["prob_profit"] is not None else 0.5
        liq_score = (r["liq_score"] or 50) / 100.0
        final = 0.70 * exp_score + 0.20 * prob_score + 0.10 * liq_score

    final = _apply_penalty(final, penalty)

    # Штраф за контртренд остаётся: в отличие от regime_score, он не ранжирует
    # бумаги между собой, а наклоняет весь блок LONG против блока SHORT.
    counter_trend = (long and regime == "BEAR") or ((not long) and regime == "BULL")
    if apply_penalties and counter_trend:
        final = _apply_penalty(final, 0.70)

    # — допустимость сделки —
    allowed = True
    if strict and counter_trend:
        allowed = False
    # В режимах, где рейтинг может быть отрицательным, множительный штраф слабо
    # меняет порядок — тяжёлые риск-условия отсекают бумагу целиком.
    if mode in ("raw_alpha", "trade_score") and hard_exclude and severe:
        allowed = False

    return final, allowed, flags


# ── Форматирование ячеек ───────────────────────────────────────────────────────

def _f(v, fmt, na="—"):
    return na if v is None else format(v, fmt)


def _yn(v):
    if v is None:
        return "—"
    return "Y" if v else "n"


def _sel(v):
    if v is None:
        return "—"
    return "✓" if v else "-"


# ── Печать ─────────────────────────────────────────────────────────────────────

def _money(v):
    """Компактный рублёвый формат: 12.5M, 850K, 1.23B."""
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _hms(ts):
    try:
        return ts.strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001
        return "—"


def _print_header(meta):
    """Шапка отчёта: дата расчёта, время/источник данных, свежесть."""
    if not meta:
        return
    as_of = meta.get("as_of")
    date_txt = as_of.strftime("%Y-%m-%d") if as_of else "—"
    time_txt = as_of.strftime("%H:%M:%S") if as_of else "—"
    source = meta.get("source", "Previous Close")
    print(f"\nДата расчёта: {date_txt}")
    if source == "Real-time":
        print(f"Время данных: {time_txt} MSK")
        print("Источник цены: Real-time")
    else:
        print("Источник цены: Previous Close")
        if not meta.get("market_open", False):
            print("Рынок закрыт")
    if meta.get("stale"):
        age_min = int(meta.get("max_age_sec", 0) // 60)
        print(_wrap(f"WARNING: Market data is stale ({age_min} min old)", _RED))
        print(_wrap("Results may be inaccurate.", _RED))


def _rs_txt(rs):
    return f"{rs:+.1f}%" if rs is not None else "—"


def _vol_txt(vs):
    return f"{vs:.1f}x" if vs is not None else "—"


def _atr_txt(a):
    return f"{a:.0f}%" if a is not None else "—"


def _gap_txt(r):
    # GapRisk показываем только для long_overnight; иначе «-»
    if r["strategy"] != "long_overnight" or r["gap_down_prob"] is None:
        return "-"
    return f"{r['gap_down_prob'] * 100:.0f}%"


def _regime_color(reg):
    return {"BULL": _GREEN, "BEAR": _RED, "NEUTRAL": _YELLOW}.get(reg, "")


def _price_time_txt(r):
    """Дата/время цены, на которой построен прогноз. Real-time котировка →
    'MM-DD HH:MM'; иначе дата дневной свечи (закрытие) → 'YYYY-MM-DD'."""
    ts = r.get("price_ts")
    if ts is not None:
        try:
            return ts.strftime("%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            pass
    d = r.get("last_date")
    if d is not None:
        try:
            return d.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            return str(d)[:16]
    return "—"


# ── Путь А: специализация по стратегиям ───────────────────────────────────────

def trading_strategies() -> set[str]:
    """Что разрешено торговать (config.TRADING_STRATEGIES).

    Отдельно от VALIDATION_STRATS: контур валидации продолжает считать все
    стратегии, иначе мы перестанем видеть, что происходит с исключённой.
    """
    import config as _cfg
    return set(getattr(_cfg, "TRADING_STRATEGIES", None)
               or ["long_overnight", "intraday_short"])


def _seller_momentum(r: dict) -> bool | None:
    """Подтверждён ли импульс продавцов: вчера падение И волатильность рынка
    выше медианы.

    None — данных не хватает (нет ret1 или оценки волатильности рынка). Вызывающий
    решает сам; здесь мы НЕ выдаём False, чтобы не спутать «нет импульса» с
    «не знаем».
    """
    ret1 = r.get("ret1")
    mkt = r.get("market_atr_pctl")
    if ret1 is None or mkt is None:
        return None
    return bool(ret1 < 0 and mkt > 50.0)


def _overnight_edge_ok(r: dict, k: float) -> bool | None:
    """ExpPnL превышает издержки round-trip в k раз.

    ExpPnL приходит УЖЕ нетто издержек (directional.strategy_pnl), поэтому это
    порог сверх безубыточности, а не «покрывает ли сделка комиссию».
    """
    exp = r.get("exp_pnl")
    if exp is None:
        return None
    cost = r.get("cost_rt")
    if cost is None:
        import config as _cfg
        cost = float(getattr(_cfg, "TFT_COST_RT", 0.08))
    return bool(exp > k * float(cost))


def _short_squeeze_risk(r: dict) -> bool | None:
    """Индекс выше своей EMA50 — шортить опасно (риск шорт-сквиза).

    None — признак недоступен.
    """
    v = r.get("index_above_ema50")
    return None if v is None else bool(v)


def _overnight_market_too_hot(r: dict, cap: float) -> bool | None:
    """Волатильность рынка выше потолка — овернайт-лонги запрещены.

    Покупка через ночь на панической волатильности — это ставка на гэп вверх
    в момент, когда распределение гэпов шире всего.
    """
    v = r.get("market_atr_pctl")
    return None if v is None else bool(float(v) > cap)


def apply_strategy_specialisation(rows: list[dict], *,
                                  allowed: set[str] | None = None,
                                  require_momentum: bool | None = None,
                                  overnight_k: float | None = None,
                                  block_short_uptrend: bool | None = None,
                                  overnight_max_market_atr: float | None = None,
                                  verbose: bool = True) -> list[dict]:
    """Фильтр Пути А: оставить только те сигналы, где есть преимущество.

      1. Торгуются только стратегии из TRADING_STRATEGIES (intraday_long убрана:
         её средняя доходность -0.2393% при t -17.26).
      2. intraday_short — только при подтверждённом импульсе продавцов.
      3. intraday_short запрещён, когда индекс выше своей EMA50 (шорт-сквиз).
      4. long_overnight — только когда ExpPnL превышает издержки в k раз.
      5. long_overnight запрещён при рыночной волатильности выше потолка.

    Пункты 3 и 5 — ПРЕДОХРАНИТЕЛИ, а не источники доходности. На выборке из
    одного медвежьего рынка запрет шорта при растущем индексе стоит около
    3 п.п. CAGR (12.4% -> 9.3%), потому что убирает только прибыльные шорт-дни;
    его смысл — защита в режиме, которого в выборке нет. Потолок волатильности
    для овернайта почти ни на что не влияет (12.363% -> 12.347%).

    Строки, по которым не хватает данных для решения, ПРОПУСКАЮТСЯ (остаются),
    а не отбрасываются: отсутствие признака не есть отрицательный сигнал.
    """
    import config as _cfg
    allowed = allowed if allowed is not None else trading_strategies()
    if require_momentum is None:
        require_momentum = bool(getattr(_cfg, "SELLER_MOMENTUM_SHORT_ENABLED", True))
    if overnight_k is None:
        overnight_k = float(getattr(_cfg, "OVERNIGHT_MIN_EDGE_X_COST", 0.5))
    if block_short_uptrend is None:
        block_short_uptrend = bool(getattr(_cfg, "SHORT_IMOEX_MAX_TREND", "EMA50"))
    if overnight_max_market_atr is None:
        overnight_max_market_atr = float(
            getattr(_cfg, "OVERNIGHT_MAX_MARKET_ATR_PCTL", 70.0))

    out = []
    dropped = {"strategy": 0, "momentum": 0, "squeeze": 0, "edge": 0, "hot": 0}
    for r in rows:
        st = r.get("strategy")
        if st not in allowed:
            dropped["strategy"] += 1
            continue
        if st == "intraday_short":
            if require_momentum and _seller_momentum(r) is False:
                dropped["momentum"] += 1
                continue
            if block_short_uptrend and _short_squeeze_risk(r) is True:
                dropped["squeeze"] += 1
                continue
        if st == "long_overnight":
            if overnight_k > 0 and _overnight_edge_ok(r, overnight_k) is False:
                dropped["edge"] += 1
                continue
            if (overnight_max_market_atr
                    and _overnight_market_too_hot(r, overnight_max_market_atr) is True):
                dropped["hot"] += 1
                continue
        out.append(r)

    if verbose and any(dropped.values()):
        log.info("Специализация: отсеяно по стратегии %d, без импульса %d, "
                 "риск шорт-сквиза %d, ниже порога %d, рынок перегрет %d; "
                 "осталось %d.",
                 dropped["strategy"], dropped["momentum"], dropped["squeeze"],
                 dropped["edge"], dropped["hot"], len(out))
    return out


REJECTED_VERDICT = "REJECTED"


def is_rejected(row: dict) -> bool:
    """Вердикт контура валидации — REJECTED («отличие от случайности не доказано»)."""
    return row.get("verdict") == REJECTED_VERDICT


def warn_unvalidated(rows: list[dict], *, env: str = "", force: bool = False) -> int:
    """Печатает предупреждение, если среди кандидатов есть REJECTED-стратегии.

    Возвращает число таких кандидатов. Ничего не блокирует — блокировка живёт
    в select_top_rows под STRICT_VALIDATION_GATE. Смысл предупреждения: до
    аудита система молча отправляла в стакан заявки по стратегиям, которые её
    собственный контур валидации отверг, и об этом нигде не говорилось.
    """
    bad = [r for r in rows if is_rejected(r)]
    if not bad:
        return 0
    head = f"{_BOLD}{_RED}" if _color_enabled() else ""
    tail = _RESET if _color_enabled() else ""
    print(f"\n{head}{'!' * 88}{tail}")
    print(f"{head}!!  ВНИМАНИЕ: {len(bad)} из {len(rows)} сигналов имеют вердикт "
          f"REJECTED{tail}")
    print(f"{head}!!  Контур валидации не смог отличить эти стратегии от "
          f"случайности.{tail}")
    if env == "PROD":
        print(f"{head}!!  КОНТУР БОЕВОЙ — это реальные деньги на непроверенном "
              f"сигнале.{tail}")
    if force:
        print(f"{head}!!  Передан --force-trade-unvalidated — торговля продолжится.{tail}")
    else:
        print(f"{head}!!  STRICT_VALIDATION_GATE=1 остановит торговлю такими "
              f"сигналами.{tail}")
    for r in bad[:10]:
        print(f"{head}!!    {r['ticker']:<6} {r['strategy']:<16} "
              f"ExpPnL={_f(r.get('exp_pnl'), '+.3f')}%{tail}")
    if len(bad) > 10:
        print(f"{head}!!    … ещё {len(bad) - 10}{tail}")
    print(f"{head}{'!' * 88}{tail}\n")
    return len(bad)


def select_top_rows(val_rows, forecasts, tickers, strats, *,
                    show_all: bool = False, top_n: int = 10,
                    strict: bool | None = None,
                    validation_gate: bool | None = None,
                    specialise: bool = True) -> list[dict]:
    """Та же пайплайн-логика, что в print_combined → _print_best_trades,
    но без печати. Возвращает топ-N строк-кандидатов (сортированы по рейтингу).

    Используется services/place_orders.py: для выставления заявок нужна
    ТА ЖЕ выборка, что показана пользователю в блоке «ЛУЧШИЕ СДЕЛКИ».

    ГЕЙТ ВАЛИДАЦИИ (config.STRICT_VALIDATION_GATE, по умолчанию выключен).
    При включении из кандидатов исключаются стратегии с вердиктом REJECTED.
    До аудита вердикт вообще не участвовал в отборе: он влиял только на
    слагаемое validation_score весом 0.15 (то есть менял рейтинг на ~0.05) и
    не мог помешать сделке. На момент аудита REJECTED имели 138 комбинаций из
    138, поэтому включение гейта останавливает торговлю полностью — это
    ожидаемое поведение, а не сбой: см. AUDIT-PROFITABILITY-REPORT.md, §4.1.
    Пока гейт выключен, вызывающий обязан показать warn_unvalidated().
    """
    import config as _cfg
    if strict is None:
        strict = bool(getattr(_cfg, "STRICT_MARKET_FILTER", False))
    if validation_gate is None:
        validation_gate = bool(getattr(_cfg, "STRICT_VALIDATION_GATE", False))

    fc = dict(forecasts or {})
    fc.pop("__meta__", None)
    rows = _build_rows(val_rows, fc, tickers, strats)
    rows = _drop_blocked_shorts(rows)
    if not rows:
        return []
    rows = _apply_selection(rows, show_all)
    # Путь А: специализация — торгуем только там, где измерено преимущество.
    if specialise:
        rows = apply_strategy_specialisation(rows)
        if not rows:
            return []
    scored = []
    for r in rows:
        score, allowed, flags = _score_row(r, strict)
        r["final_score"] = score
        r["flags"] = flags
        if allowed:
            scored.append(r)
    if not scored:
        return []
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    cand = [r for r in scored
            if r["exp_pnl"] is not None and r["selected"] is not False]

    if validation_gate:
        before = len(cand)
        cand = [r for r in cand if not is_rejected(r)]
        if before and not cand:
            log.warning("STRICT_VALIDATION_GATE=1: все %d кандидатов отвергнуты "
                        "контуром валидации (REJECTED) — сделок нет.", before)
        elif before != len(cand):
            log.info("STRICT_VALIDATION_GATE=1: отсеяно %d кандидатов с вердиктом "
                     "REJECTED, осталось %d.", before - len(cand), len(cand))

    if top_n and top_n > 0:
        cand = cand[:top_n]
    return cand


def print_combined(val_rows, forecasts, tickers, strats, show_all: bool = False,
                   top_n: int = 50):
    import config
    strict = bool(getattr(config, "STRICT_MARKET_FILTER", False))

    forecasts = dict(forecasts or {})
    meta = forecasts.pop("__meta__", None)

    rows = _build_rows(val_rows, forecasts, tickers, strats)
    # Тот же фильтр, что и в select_top_rows: блок «ЛУЧШИЕ СДЕЛКИ» и реальные
    # заявки строятся из одной выборки — расхождений быть не должно.
    rows = _drop_blocked_shorts(rows)
    if not rows:
        return
    rows = _apply_selection(rows, show_all)

    # Итоговый рейтинг с рыночным контекстом + жёсткий фильтр.
    scored = []
    for r in rows:
        score, allowed, flags = _score_row(r, strict)
        r["final_score"] = score
        r["flags"] = flags
        if allowed:
            scored.append(r)
    rows = scored
    if not rows:
        _print_header(meta)
        print("\n  Все сигналы отфильтрованы жёстким рыночным фильтром "
              "(STRICT_MARKET_FILTER).")
        _print_market_summary(meta)
        return
    rows.sort(key=lambda x: x["final_score"], reverse=True)

    # Сохранение в БД (forecasts) — ДО отсечения по top_n, чтобы в базе была
    # вся посчитанная выборка, а не только видимый топ.
    from . import persist
    persist.save_daily(rows, meta, forecasts)

    # Топ-N бумаг по итоговому рейтингу (FinalScore). По умолчанию 50.
    total_rows = len(rows)
    if top_n and top_n > 0:
        rows = rows[:top_n]

    _print_header(meta)

    W = 172
    print("\n" + "=" * W)
    shown = len(rows)
    title = f" СВОДНЫЙ ДАШБОРД 2.0 (ИИ-ПРОГНОЗ + РЫНОЧНЫЙ КОНТЕКСТ) — ТОП-{shown}"
    if shown < total_rows:
        title += f" из {total_rows}"
    print(title)
    print("=" * W)

    hdr = (f" {'Ticker':<7}{'Strategy':<15}{'Dir':<6}{'ExpPnL':>8}{'PProf':>7}"
           f"{'FDR':>5}{'PBO':>7}{'Liq':>5}{'MaxPos':>9}"
           f"{'Price':>10}{'PriceTime':>14}{'F.Low':>10}{'F.High':>10}{'Range%':>9}"
           f"{'IMOEX':>9}{'RS':>8}{'VolSpike':>10}{'ATR%':>7}{'GapRisk':>9}")
    print(hdr)
    print("-" * W)

    for r in rows:
        exp = r["exp_pnl"]
        exp_cell = _wrap(f"{_f(exp, '+.2f'):>7}%" if exp is not None else f"{'—':>8}",
                         _pnl_color(exp))
        prob = r["prob_profit"]
        prob_txt = f"{prob * 100:.0f}%" if prob is not None else "—"
        prob_cell = _wrap(f"{prob_txt:>7}", _prob_color(prob))
        reg = r["regime"]
        reg_cell = _wrap(f"{(reg or '—'):>9}", _regime_color(reg))
        rng = r["range_pct"]
        rng_cell = f"{_f(rng, '.2f'):>8}%" if rng is not None else f"{'—':>9}"

        print(
            f" {r['ticker']:<7}{r['strategy']:<15}{r['direction']:<6}"
            f"{exp_cell}{prob_cell}{_yn(r['fdr']):>5}{_f(r['pbo'], '.2f'):>7}"
            f"{_f(r['liq_score'], '.0f'):>5}{_money(r['max_pos']):>9}"
            f"{_f(r['anchor_price'], '.2f'):>10}{_price_time_txt(r):>14}"
            f"{_f(r['f_low'], '.2f'):>10}{_f(r['f_high'], '.2f'):>10}{rng_cell}"
            f"{reg_cell}{_rs_txt(r['rs']):>8}{_vol_txt(r['vol_spike']):>10}"
            f"{_atr_txt(r['atr_pctl']):>7}{_gap_txt(r):>9}"
        )
    print("=" * W)

    best_top_n    = int(getattr(config, "BEST_TRADES_TOP_N",         10))
    best_position = float(getattr(config, "BEST_TRADES_POSITION_RUB", 10_000.0))
    entry_frac    = float(getattr(config, "LIMIT_ENTRY_FRACTION",    0.8))

    _print_legend()
    _print_flags(rows)

    # Блок «ЛУЧШИЕ СДЕЛКИ» и инструкции для автозаявок ДЕЙСТВЕННЫЕ: из них
    # напрямую строятся заявки. Поэтому здесь применяется та же специализация
    # Пути А, что и в select_top_rows, — иначе дашборд предлагал бы сигналы по
    # стратегиям, которые торговать запрещено (intraday_long), и «что показано»
    # расходилось бы с «что отправлено».
    # Таблица ВЫШЕ намеренно остаётся полной: контур валидации продолжает
    # считать все стратегии из VALIDATION_STRATS, иначе мы перестанем видеть,
    # что происходит с исключённой.
    tradable = apply_strategy_specialisation(rows, verbose=False)
    if len(tradable) != len(rows):
        allowed = ", ".join(sorted(trading_strategies()))
        print(f"\n  К торговле допущено {len(tradable)} из {len(rows)} сигналов "
              f"(TRADING_STRATEGIES: {allowed} + предохранители Пути А).")
        print("  Строки выше — полный мониторинг, включая нетоварные стратегии.")
    _print_best_trades(tradable, top_n=best_top_n, position_rub=best_position,
                       entry_frac=entry_frac)
    _print_market_summary(meta)


# ── НЕДЕЛЬНЫЙ ДАШБОРД (формат дневного) ───────────────────────────────────────

_DOW_RU = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def _next_trading_day(d):
    """Следующий торговый день по календарю биржи (services/calendar.py).

    Раньше здесь было `while d.weekday() >= 5` — жёсткий Пн–Пт: окно в шапке
    расходилось с шагом модели при торговле по выходным и не знало о
    праздниках и переносах."""
    from services.calendar import get_calendar
    days = get_calendar().get_next_trading_days(d, 1)
    return days[0] if days else d


def print_weekly_dashboard(forecasts, top_n: int = 50) -> None:
    """Недельный дашборд по аналогии с дневным «СВОДНЫЙ ДАШБОРД 2.0».

    Рекомендация LONG/SHORT — по знаку медианы предсказанной 5-дневной
    доходности (week_total, пересчитанной в ОСТАВШУЮСЯ отн. текущей цены);
    ExpPnL5 — медиана нетто издержек, PProf — P(доходность > издержек) по
    квантильной функции (как в дневном направленном прогнозе).

    Текущий день недели учитывается дважды: (1) модель получает dow/next_dow
    как признаки — прогноз с пятницы (окно через выходные) отличается от
    прогноза со вторника; (2) в шапке печатается фактическое окно прогноза
    (следующие H торговых дней от сегодняшнего дня недели).
    """
    import config as _cfg
    import datetime as dt
    from .directional import QUANTILES as _Q, _prob_above

    fc = dict(forecasts or {})
    meta = fc.pop("__meta__", None) or {}
    cost_rt = float(getattr(_cfg, "TFT_COST_RT", 0.08))

    rows = []
    for tk, r in fc.items():
        if not isinstance(r, dict):
            continue
        q = r.get("WeekTotalQ")
        if not q or r.get("WeekLow") is None:
            continue
        med = float(q[len(q) // 2])
        sign = 1.0 if med >= 0 else -1.0
        sq = sorted(sign * float(v) for v in q)   # знаковая доходность, возр.
        exp_net = abs(med) - cost_rt
        prob = _prob_above(_Q, sq, cost_rt)
        rows.append({
            "ticker": tk,
            "dir": "LONG" if sign > 0 else "SHORT",
            "exp": exp_net,
            "prob": prob,
            "w_low": r.get("WeekLow"), "w_high": r.get("WeekHigh"),
            "w_range": r.get("WeekRangePct"), "w_cov": r.get("WeekCoverage"),
            "med5": med,
            "liq_score": r.get("LiqScore"), "max_pos": r.get("MaxPos"),
            "anchor_price": r.get("anchor_price"),
            "price_ts": r.get("price_ts"), "last_date": r.get("last_date"),
            "regime": r.get("Regime"), "rs": r.get("RS"),
            "vol_spike": r.get("VolSpike"), "atr_pctl": r.get("ATRpctl"),
        })
    if not rows:
        return
    rows.sort(key=lambda x: x["exp"], reverse=True)

    # Сохранение в БД (forecasts, strategy='weekly') — тоже до отсечения top_n.
    from . import persist
    _h = int(fc[rows[0]["ticker"]].get("WeekHorizon", 5))
    persist.save_weekly(rows, meta, _h)

    total = len(rows)
    if top_n and top_n > 0:
        rows = rows[:top_n]

    # Окно прогноза = ровно те H торговых дней, которые покрывает горизонт
    # модели. Календарь один: TradingCalendar знает праздники, переносы и
    # расписание биржи из API, а маску дней недели берёт из режима проекта.
    from services.calendar import get_calendar
    h = int(fc[rows[0]["ticker"]].get("WeekHorizon", 5))
    as_of = meta.get("as_of") or dt.datetime.now(dt.timezone(dt.timedelta(hours=3)))
    today = as_of.date() if hasattr(as_of, "date") else as_of
    _cal = get_calendar()
    horizon_days = _cal.get_next_trading_days(today, h)
    start = horizon_days[0] if horizon_days else today
    end = _cal.get_target_horizon_date(today, h) or start

    W = 146
    print("\n" + "=" * W)
    shown = len(rows)
    title = f" НЕДЕЛЬНЫЙ ДАШБОРД (ИИ-ПРОГНОЗ НА {h} ТОРГ. ДНЕЙ) — ТОП-{shown}"
    if shown < total:
        title += f" из {total}"
    print(title)
    print(f" Сегодня {_DOW_RU[today.weekday()]} {today.strftime('%d.%m')} → окно прогноза: "
          f"{_DOW_RU[start.weekday()]} {start.strftime('%d.%m')} — "
          f"{_DOW_RU[end.weekday()]} {end.strftime('%d.%m')} "
          f"(день недели — признак модели: прогноз с Пт ≠ прогнозу со Вт)")
    print("=" * W)
    hdr = (f" {'Ticker':<7}{'Dir':<6}{'Rec':>4}{'ExpPnL5':>9}{'PProf':>7}{'Cover':>7}"
           f"{'Liq':>5}{'MaxPos':>9}{'Price':>10}{'PriceTime':>14}"
           f"{'W.Low':>10}{'W.High':>10}{'Range%':>9}{'Med5d%':>8}"
           f"{'IMOEX':>9}{'RS':>8}{'VolSpike':>10}{'ATR%':>7}")
    print(hdr)
    print("-" * W)

    for r in rows:
        rec = "✓" if (r["exp"] is not None and r["exp"] > 0
                      and r["prob"] is not None and r["prob"] > 0.5) else ""
        exp_cell = _wrap(f"{_f(r['exp'], '+.2f'):>8}%", _pnl_color(r["exp"]))
        prob_txt = f"{r['prob'] * 100:.0f}%" if r["prob"] is not None else "—"
        prob_cell = _wrap(f"{prob_txt:>7}", _prob_color(r["prob"]))
        cov = r["w_cov"]
        cov_txt = f"{cov * 100:.0f}%" if cov is not None and cov == cov else "—"
        reg_cell = _wrap(f"{(r['regime'] or '—'):>9}", _regime_color(r["regime"]))
        rng = r["w_range"]
        rng_cell = f"{_f(rng, '.2f'):>8}%" if rng is not None else f"{'—':>9}"
        print(
            f" {r['ticker']:<7}{r['dir']:<6}{rec:>4}{exp_cell}{prob_cell}{cov_txt:>7}"
            f"{_f(r['liq_score'], '.0f'):>5}{_money(r['max_pos']):>9}"
            f"{_f(r['anchor_price'], '.2f'):>10}{_price_time_txt(r):>14}"
            f"{_f(r['w_low'], '.2f'):>10}{_f(r['w_high'], '.2f'):>10}{rng_cell}"
            f"{_f(r['med5'], '+.2f'):>7}%{reg_cell}{_rs_txt(r['rs']):>8}"
            f"{_vol_txt(r['vol_spike']):>10}{_atr_txt(r['atr_pctl']):>7}"
        )
    print("=" * W)
    print("  Dir/Rec — рекомендация по знаку медианы 5-дневной доходности; "
          "✓ = ExpPnL5>0 и PProf>50%.")
    print("  ExpPnL5 — медианная ОСТАВШАЯСЯ доходность за окно (нетто издержек, "
          "отн. текущей цены); Med5d% — то же брутто.")
    print("  W.Low/W.High — недельный коридор (конформно калиброван), Cover — его "
          "покрытие на held-out; остальные столбцы — как в дневном дашборде.")
    print("  ВНИМАНИЕ: недельный прогноз не проходил контур валидации стратегий "
          "(White RC/PBO/FDR) — это сырой прогноз модели, не подтверждённый edge.")


def _print_legend():
    print("\n  РАСШИФРОВКА СТОЛБЦОВ")
    print("  " + "-" * 74)
    legend = [
        ("Ticker",   "Тикер инструмента"),
        ("Strategy", "Название стратегии"),
        ("Dir",      "Направление сделки (LONG / SHORT)"),
        ("ExpPnL",   "Ожидаемая доходность следующего дня (нетто)"),
        ("PProf",    "ProbProfit — вероятность положительного результата"),
        ("FDR",      "Прошла ли контроль False Discovery Rate (Y/n)"),
        ("PBO",      "Probability of Backtest Overfitting"),
        ("Liq",      "Балл ликвидности 0..100 (перцентиль оборота по рынку)"),
        ("MaxPos",   "Макс. размер позиции ₽ (Amihud; сжат при высоком ATR%)"),
        ("Price",    "Цена, на которой построен прогноз (real-time котировка или закрытие)"),
        ("PriceTime","Дата/время этой цены: 'MM-DD HH:MM' (real-time) или дата закрытия"),
        ("F.Low",    "Прогноз нижней границы цены на завтра (q0.1, ₽)"),
        ("F.High",   "Прогноз верхней границы цены на завтра (q0.9, ₽)"),
        ("Range%",   "Ширина прогнозного коридора (F.High−F.Low) к якорной цене"),
        ("IMOEX",    "Режим широкого рынка по EMA50/EMA200 (BULL/BEAR/NEUTRAL)"),
        ("RS",       "Rel.Strength — сила бумаги к рынку за 10 дней"),
        ("VolSpike", "Объём вчера / SMA20 (кратность)"),
        ("ATR%",     "Перцентиль текущего ATR(14) за 252 дня"),
        ("GapRisk",  "Вероятность гэпа вниз (>-0.5%); только long_overnight"),
    ]
    for name, desc in legend:
        print(f"  {name:<10}{desc}")
    print("  " + "-" * 74)


def _print_flags(rows):
    flagged = [r for r in rows if r.get("flags")]
    if not flagged:
        return
    print("\n  РИСК-ФЛАГИ")
    print("  " + "-" * 74)
    for r in flagged:
        joined = ", ".join(r["flags"])
        print(f"  {r['ticker']:<6}{r['strategy']:<16}{joined}")
    print("  " + "-" * 74)


def _print_market_summary(meta):
    m = (meta or {}).get("market") if meta else None
    if not m:
        return
    breadth = m.get("breadth")
    atr = m.get("atr_pctl")
    risk = m.get("risk_level", "NORMAL")
    regime = m.get("regime", "NEUTRAL")
    src = m.get("index_source", "proxy")

    print("\n  MARKET SUMMARY")
    print("  " + "-" * 74)
    reg_lbl = regime + (" (proxy-индекс)" if src != "IMOEX" else "")
    print(f"  IMOEX Regime:    {reg_lbl}")
    print(f"  Market Breadth:  {breadth * 100:.0f}%" if breadth is not None else "  Market Breadth:  —")
    print(f"  ATR Percentile:  {atr:.0f}%" if atr is not None else "  ATR Percentile:  —")
    print(f"  Risk Level:      {risk}")
    print("  Recommendation:")
    for line in _recommendation(regime, risk):
        print(f"    {line}")
    print("  " + "-" * 74)


def _recommendation(regime, risk):
    rec = []
    if risk == "HIGH":
        rec.append("Снизить размеры позиций на 50%.")
    elif risk == "ELEVATED":
        rec.append("Снизить размеры позиций на 25%.")
    if regime == "BEAR":
        rec.append("Предпочитать SHORT-сигналы.")
        rec.append("Избегать long_overnight позиций.")
    elif regime == "BULL":
        rec.append("Предпочитать LONG-сигналы.")
        rec.append("Осторожно с SHORT против тренда.")
    else:
        rec.append("Рынок без выраженного тренда — торговать выборочно.")
    if not rec:
        rec.append("Условия нормальные — действовать по рейтингу.")
    return rec


# Размер лота (кол-во акций в 1 лоте) для основных тикеров MOEX.
# Источник: спецификации режимов TQBR/TQTF на MOEX, актуально на середину 2025.
# Для отсутствующих тикеров используется лот = 1 (помечается «(?)»).
# ВАЖНО про сплиты/изменения:
#   GMKN 1:100 (апр 2024) → 10 акций/лот;
#   PLZL 1:10 (май 2024)  → 1 акция/лот;
#   VKCO после редомициляции — 1 акция/лот.
_LOT_SIZES: dict[str, int] = {
    "SBER": 10, "SBERP": 10, "GAZP": 10, "LKOH": 1, "NVTK": 1,
    "ROSN": 1,  "TATN": 1,   "TATNP": 1, "CHMF": 1, "MGNT": 1,
    "YDEX": 1,  "ALRS": 10,  "PLZL": 1,  "GMKN": 10, "IRAO": 100,
    "RUAL": 10, "VTBR": 10000, "SNGSP": 100, "SNGS": 100,
    "UPRO": 1000, "OGKB": 1000,  "MSNG": 10000, "FEES": 10000,
    "HYDR": 1000, "AFLT": 10, "MTSS": 10, "RTKM": 10, "VKCO": 1,
    "MOEX": 10, "CBOM": 100, "BSPB": 1,  "PHOR": 1,  "ENPG": 1,
    "FLOT": 1,  "PIKK": 1,   "POSI": 1,  "SMLT": 1,
    "MVID": 10, "SELG": 100, "ETLN": 10, "ASTR": 1,  "LENT": 1,
    "AKRN": 1,  "NMTP": 100, "MAGN": 100, "X5": 1,
    "FIXR": 1000,   # новая рос. акция Fix Price после редомициляции (цена ~0.45 ₽)
}


# Тикеры, по которым нельзя выставить рыночную заявку через Tinkoff
# (делистинг, редомициляция, смена ISIN). В дашборде/прогнозе остаются,
# но в блоке «ИНСТРУКЦИИ ДЛЯ АВТОЗАЯВОК» маркируются N/A.
# Источник: ручной список + env TINKOFF_UNAVAILABLE_TICKERS (через пробел).
# FIXP убран: пайплайн переведён на торгуемый FIXR (новая рос. акция). Старый
# FIXP (US-GDR, ISIN US33835G2057) не торгуется через API, но в TICKERS его
# больше нет, поэтому в дашборде он не появляется.
_DEFAULT_UNAVAILABLE: set[str] = set()


def _unavailable_tickers() -> set[str]:
    extra = os.getenv("TINKOFF_UNAVAILABLE_TICKERS", "").upper().split()
    return _DEFAULT_UNAVAILABLE | set(extra)


# Бумаги, по которым брокер не даёт маржинальный шорт (shortEnabledFlag=false в
# InstrumentsService/ShareBy). Заявка SELL без позиции по ним отклоняется, а
# сигнал intraday_short по ним заведомо неисполним.
# Базовый список сверен с API по всем 46 тикерам config.TICKERS: ровно эти три.
# Расширяется из окружения: NON_SHORTABLE_TICKERS="AKRN CBOM MVID XXXX".
# Источник истины при выставлении — живой флаг Instrument.short_enabled
# (см. services/place_orders.py); этот список нужен, чтобы отсечь сигнал раньше,
# ещё до похода в API.
_DEFAULT_NON_SHORTABLE: set[str] = {"AKRN", "CBOM", "MVID"}


def non_shortable_tickers() -> set[str]:
    """Тикеры, по которым шорт запрещён: базовый список + NON_SHORTABLE_TICKERS."""
    extra = os.getenv("NON_SHORTABLE_TICKERS", "").upper().split()
    return _DEFAULT_NON_SHORTABLE | set(extra)


def _drop_blocked_shorts(rows: list[dict]) -> list[dict]:
    """Убирает SHORT-сигналы по нешортабельным бумагам.

    Вызывается ДО отбора лучшей дневной стратегии, поэтому бумага не выпадает
    из дашборда целиком — по ней остаётся LONG-кандидат, если он есть.
    """
    blocked = non_shortable_tickers()
    if not blocked:
        return rows
    kept = []
    for r in rows:
        if r.get("direction") == "SHORT" and r["ticker"].upper() in blocked:
            log.info("[SKIP SHORT] %s: шорт недоступен у брокера", r["ticker"])
            continue
        kept.append(r)
    return kept


def _lot_size(ticker: str) -> tuple[int, bool]:
    """Возвращает (размер_лота, точно_известен). False → дефолт 1, надо проверить."""
    tk = ticker.upper()
    if tk in _LOT_SIZES:
        return _LOT_SIZES[tk], True
    return 1, False


def _limit_entry_price(direction: str, anchor: float | None,
                       f_low: float | None, f_high: float | None,
                       frac: float) -> tuple[float | None, str]:
    """
    Цена лимитной заявки от прогнозного коридора.

    Идея: заходить не «по рынку», а по лучшей цене внутри прогноза:
      SHORT → ближе к ВЕРХНЕЙ границе (F.High): шорт на откате вверх;
      LONG  → ближе к НИЖНЕЙ границе (F.Low):   лонг на откате вниз.

    Параметр frac ∈ [0, 1] — насколько близко к экстремуму:
      0.0 → строго на anchor (как раньше, по рынку);
      1.0 → строго на F.High / F.Low (максимально агрессивно, риск не залиться);
      0.8 (default) → 80% пути от anchor к экстремуму — баланс цена/исполняемость.

    Возвращает (price, note). Если коридора нет или его сторона «не лучше»
    anchor (например, anchor уже выше F.High при шорте), фолбэк на anchor.
    """
    if anchor is None or anchor <= 0:
        return None, "—"
    if direction == "SHORT":
        if f_high is None or f_high <= anchor:
            return anchor, "anchor"
        return anchor + frac * (f_high - anchor), f"→{frac:.0%}·F.High"
    # LONG
    if f_low is None or f_low >= anchor:
        return anchor, "anchor"
    return anchor - frac * (anchor - f_low), f"→{frac:.0%}·F.Low"


def _take_profit_price(direction: str, entry: float | None,
                       f_low: float | None, f_high: float | None,
                       frac: float) -> float | None:
    """Цена take-profit от прогнозного коридора (анализ диапазона).

    Цель — ПРОТИВОПОЛОЖНАЯ входу граница прогноза (прибыль на возврате цены):
      SHORT → вниз к F.Low:  entry − frac·(entry − F.Low);
      LONG  → вверх к F.High: entry + frac·(F.High − entry).

    frac ∈ [0,1]: 1.0 — ровно на дальней границе (полный диапазон),
    0.8 — ближе ко входу (профит-заявка исполнится с большей вероятностью).

    None, если коридора нет или его сторона не «в прибыль» относительно входа.
    """
    if entry is None or entry <= 0:
        return None
    if direction == "SHORT":
        if f_low is None or f_low >= entry:
            return None
        return entry - frac * (entry - f_low)
    # LONG
    if f_high is None or f_high <= entry:
        return None
    return entry + frac * (f_high - entry)


@dataclass
class Order:
    """Структурированная торговая заявка (вход + связанный стоп).

    Это «полуфабрикат» для services/place_orders.py: вход = ЛИМИТКА внутри
    прогнозного коридора, стоп = STOP_LOSS от entry. Цены ещё НЕ округлены к
    min_price_increment инструмента — округление делает брокерский клиент,
    т.к. ему доступна спецификация. Здесь храним «как считала модель».

    Поля entry_price / stop_price / quantity_lots / total_rub могут быть None,
    если входных данных не хватило (нет anchor, нет коридора, тикер N/A).
    Такие заявки печатаются «—» в таблице и пропускаются в place_orders.
    """
    ticker:          str
    strategy:        str
    direction:       str            # "LONG" | "SHORT"
    anchor_price:    Optional[float]  # спот / якорь прогноза
    f_low:           Optional[float]
    f_high:          Optional[float]
    down_pct:        Optional[float]  # Downside q0.1 (нетто %, <0)
    entry_price:     Optional[float]  # лимитная цена входа
    better_pct:      Optional[float]  # насколько entry выгоднее спота
    stop_price:      Optional[float]
    stop_pct:        Optional[float]  # расстояние стопа, % (>= MIN_STOP_PCT)
    tp_price:        Optional[float]  # take-profit (цель по диапазону)
    tp_pct:          Optional[float]  # |прибыль%| от входа до tp_price
    lot_size:        int
    lot_known:       bool
    quantity_lots:   Optional[int]
    total_rub:       Optional[float]
    unavailable:     bool             # True → не торгуется через Tinkoff

    @property
    def order_direction(self) -> str:
        """Направление ВХОДНОЙ заявки в терминах брокера (BUY/SELL)."""
        return "BUY" if self.direction == "LONG" else "SELL"

    @property
    def exit_direction(self) -> str:
        """Направление ВЫХОДНЫХ заявок — стопа и тейк-профита (противоположно
        входу). Для SHORT-позиции выход = BUY, для LONG = SELL."""
        return "SELL" if self.direction == "LONG" else "BUY"

    # обратная совместимость: стоп и тейк выходят в одну сторону
    stop_direction = exit_direction

    @property
    def is_placeable(self) -> bool:
        """Готова ли заявка к отправке брокеру."""
        return (not self.unavailable
                and self.entry_price is not None and self.entry_price > 0
                and self.stop_price  is not None and self.stop_price  > 0
                and self.quantity_lots is not None and self.quantity_lots > 0)


def _risk_parity_alloc(geoms: list[dict], budget_rub: float) -> list[float]:
    """Риск-паритетное распределение бюджета: вес ∝ 1/стоп%, чтобы каждая
    позиция несла ОДИНАКОВЫЙ рублёвый риск (position_value × стоп% = const).

    Волатильные бумаги (широкий стоп) получают меньше денег, тихие — больше.
    Веса считаются только по бумагам с валидной геометрией (entry>0, стоп%>0,
    торгуется); затем каждая аллокация ограничивается лимитом ликвидности
    MaxPos (не заходим в рынок больше, чем он переваривает)."""
    inv = [(1.0 / g["stop_pct"]) if g["sizable"] else 0.0 for g in geoms]
    tot = sum(inv)
    if tot <= 0:
        return [0.0] * len(geoms)
    alloc = [budget_rub * w / tot for w in inv]
    for i, g in enumerate(geoms):        # потолок ликвидности
        mp = g.get("max_pos")
        if mp:
            alloc[i] = min(alloc[i], mp)
    return alloc


def stop_distance_pct(down: float) -> float:
    """Расстояние стопа от входа, %: всегда положительное и не меньше MIN_STOP_PCT.

    down — Downside стратегии: q0.10 знаковой доходности минус издержки. Для
    уверенного прогноза q0.10 бывает выше издержек, и тогда down > 0. Прежняя
    формула entry·(1 + down/100) давала лонгу стоп ВЫШЕ входа (ENPG 10.09:
    вход 310,81, стоп 312,47) — при постановке он сработал бы сразу.

    Убыточный хвост (down < 0) задаёт расстояние, оптимистичный (down ≥ 0) —
    нет, и тогда действует пол. Именно max(−down, 0), а не |down|: при
    down = +3% модуль дал бы стоп в 3%, то есть расстояние, растущее вместе с
    оптимизмом модели, — ровно то, от чего пол должен защищать.
    """
    import config as _cfg
    floor = float(getattr(_cfg, "MIN_STOP_PCT", 1.0))
    return max(max(-float(down), 0.0), floor)


def build_orders(top: list[dict], position_rub: float,
                 entry_frac: float, tp_frac: float | None = None,
                 budget_rub: float | None = None) -> list[Order]:
    """Чистая функция: top-N рейтинга → список структурированных Order.

    Используется и принтером `_print_order_instructions` (для печати таблицы),
    и services/place_orders.py (для отправки в T-Invest). Логика расчёта ТА ЖЕ
    (никаких расхождений между «что показано» и «что отправлено»).

    Размер позиции (контракт ОДИНАКОВ в обоих режимах — не хватает на лот,
    бумага пропускается, а не «доливается» до целого лота):
      • budget_rub задан → РИСК-ПАРИТЕТ: бюджет делится по бумагам так, чтобы
        рублёвый риск (объём × стоп%) был одинаков; вес ∝ 1/стоп%.
      • иначе → фикс position_rub на бумагу.
    В обоих случаях число лотов округляется ВНИЗ, и при 0 лотов заявка
    помечается неисполнимой (quantity_lots=None → is_placeable False).

    Прочее (цена входа в коридоре, стоп от входа, take-profit по диапазону) —
    без изменений.

    tp_frac=None → config.LIMIT_TP_FRACTION. Значение по умолчанию не
    дублируется в сигнатурах: раньше здесь и в compute_orders стояла
    захардкоженная 1.0, и она молча побеждала конфигурацию при вызове без
    явного аргумента.
    """
    if tp_frac is None:
        import config as _cfg
        tp_frac = float(getattr(_cfg, "LIMIT_TP_FRACTION", 0.5))
    unavail = _unavailable_tickers()
    blocked_short = non_shortable_tickers()

    # ── проход 1: геометрия входа/стопа/тейка на каждую бумагу
    geoms: list[dict] = []
    for r in top:
        # Вторая линия защиты: сюда строка могла прийти в обход дашборда
        # (например из сохранённого топа) — шорт по нешортабельной бумаге
        # помечаем неторгуемым, чтобы заявка не ушла брокеру.
        if r["direction"] == "SHORT" and r["ticker"].upper() in blocked_short:
            log.info("[SKIP SHORT] %s: шорт недоступен у брокера", r["ticker"])
            r = dict(r)
            r["_short_blocked"] = True
        anchor = r.get("anchor_price")
        down   = r.get("down")
        f_low  = r.get("f_low")
        f_high = r.get("f_high")
        tk     = r["ticker"]
        lng    = (r["direction"] == "LONG")
        lot, lot_known = _lot_size(tk)
        na = tk in unavail or bool(r.get("_short_blocked"))

        entry, _src = _limit_entry_price(r["direction"], anchor, f_low, f_high, entry_frac)
        better = None
        if entry and anchor and anchor > 0:
            better = (entry - anchor) / anchor * 100.0
            if lng:
                better = -better

        if entry and entry > 0 and down is not None:
            stop_pct = stop_distance_pct(down)
            stop_p   = entry * (1.0 - stop_pct / 100.0) if lng else entry * (1.0 + stop_pct / 100.0)
        else:
            stop_p = stop_pct = None

        tp_p = _take_profit_price(r["direction"], entry, f_low, f_high, tp_frac)
        tp_pct = (abs(tp_p - entry) / entry * 100.0) if (tp_p and entry) else None

        geoms.append({
            "r": r, "tk": tk, "lng": lng, "lot": lot, "lot_known": lot_known,
            "na": na, "anchor": anchor, "down": down, "f_low": f_low, "f_high": f_high,
            "entry": entry, "better": better, "stop_p": stop_p, "stop_pct": stop_pct,
            "tp_p": tp_p, "tp_pct": tp_pct, "max_pos": r.get("max_pos"),
            # можно ли считать риск-паритет: есть вход, положительный стоп%, торгуется
            "sizable": bool(entry and entry > 0 and stop_pct and not na),
        })

    # ── проход 2: размеры позиций
    use_budget = bool(budget_rub and budget_rub > 0)
    alloc = _risk_parity_alloc(geoms, budget_rub) if use_budget else [None] * len(geoms)

    orders: list[Order] = []
    for i, g in enumerate(geoms):
        entry, lot, na = g["entry"], g["lot"], g["na"]
        if entry and entry > 0 and not na:
            if use_budget:
                lots = int(alloc[i] / (entry * lot)) if g["sizable"] else 0
                # риск-паритет не форсирует лот: не хватило на 1 лот → пропуск
                total = (lots * lot * entry) if lots > 0 else None
                lots = lots if lots > 0 else None
            else:
                # Честное квантование вниз, как и в риск-паритете. Прежний
                # max(1, ...) насильно ставил минимум один лот, даже когда он
                # дороже лимита позиции: это тихая эскалация риска —
                # незапланированная маржиналка (или отказ INSUFFICIENT_FUNDS)
                # и перекос диверсификации на дорогих лотах.
                lots = int(position_rub / (entry * lot))
                if lots <= 0:
                    log.warning("[SKIP] %s: 1 лот (%.2f ₽) превышает лимит "
                                "позиции (%.2f ₽)", g["tk"], entry * lot, position_rub)
                    lots = total = None
                else:
                    total = lots * lot * entry
        else:
            lots = total = None

        orders.append(Order(
            ticker=g["tk"], strategy=g["r"]["strategy"], direction=g["r"]["direction"],
            anchor_price=g["anchor"], f_low=g["f_low"], f_high=g["f_high"], down_pct=g["down"],
            entry_price=entry, better_pct=g["better"],
            stop_price=g["stop_p"], stop_pct=g["stop_pct"],
            tp_price=g["tp_p"], tp_pct=g["tp_pct"],
            lot_size=lot, lot_known=g["lot_known"],
            quantity_lots=lots, total_rub=total,
            unavailable=na,
        ))
    return orders


def _print_order_instructions(top: list[dict], position_rub: float,
                              entry_frac: float) -> None:
    """Печатает блок «ИНСТРУКЦИИ ДЛЯ АВТОЗАЯВОК» из подготовленных Order."""
    import config as _cfg
    tp_frac = float(getattr(_cfg, "LIMIT_TP_FRACTION", 0.5))
    W = 176
    pos_k = position_rub / 1000.0
    orders = build_orders(top, position_rub, entry_frac, tp_frac=tp_frac)
    print(f"\n  ИНСТРУКЦИИ ДЛЯ АВТОЗАЯВОК  (лимитные, цель ≈ {pos_k:.0f}K ₽/бумагу,"
          f" вход = {entry_frac:.0%} к F.High/F.Low, профит = {tp_frac:.0%} к дальней границе)")
    print("  " + "─" * W)
    print(f"  {'№':>3}  {'Ticker':<7}  {'Стратегия':<16}  {'Тип':<9}"
          f"  {'Спот':>9}  {'Цена входа':>11}  {'Лучше%':>7}"
          f"  {'Стоп-цена':>10}  {'Стоп%':>6}  {'Профит':>10}  {'Профит%':>8}  {'R:R':>5}"
          f"  {'Лот':>5}  {'Лотов':>6}  {'Сумма ₽':>10}")
    print("  " + "─" * W)
    any_unsure_lot = any(not o.lot_known for o in orders)
    any_na         = any(o.unavailable  for o in orders)
    for i, o in enumerate(orders, 1):
        tk_cell = (o.ticker + "*") if (o.unavailable or not o.lot_known) else o.ticker
        spot_s    = f"{o.anchor_price:>8.2f}" if o.anchor_price else f"{'—':>9}"
        entry_s   = f"{o.entry_price:>10.2f}" if o.entry_price  else f"{'—':>10}"
        better_s  = f"{o.better_pct:>+6.2f}%" if o.better_pct is not None else f"{'—':>7}"
        stop_s    = f"{o.stop_price:>9.2f}"   if o.stop_price   else f"{'—':>9}"
        stoppct_s = f"{o.stop_pct:>5.1f}%"    if o.stop_pct is not None else f"{'—':>6}"
        tp_s      = f"{o.tp_price:>9.2f}"     if o.tp_price  else f"{'—':>10}"
        tppct_s   = f"{o.tp_pct:>7.1f}%"      if o.tp_pct is not None else f"{'—':>8}"
        rr        = (o.tp_pct / o.stop_pct) if (o.tp_pct and o.stop_pct) else None
        rr_s      = f"{rr:>5.1f}" if rr is not None else f"{'—':>5}"
        lot_s     = (f"{o.lot_size}(?)" if not o.lot_known else str(o.lot_size))
        lots_s    = (f"{'N/A':>6}"  if o.unavailable
                     else (f"{o.quantity_lots:>6}" if o.quantity_lots else f"{'—':>6}"))
        total_s   = (f"{'N/A':>10}" if o.unavailable
                     else (f"{o.total_rub:>10,.0f}" if o.total_rub else f"{'—':>10}"))

        print(f"  {i:>3}  {tk_cell:<7}  {o.strategy:<16}  {'Лимитная':<9}"
              f"  {spot_s}  {entry_s}  {better_s}"
              f"  {stop_s}  {stoppct_s}  {tp_s}  {tppct_s}  {rr_s}"
              f"  {lot_s:>5}  {lots_s}  {total_s}")
    print("  " + "─" * W)
    print(f"  Цена входа = спот + {entry_frac:.0%}·(экстремум прогноза − спот):"
          " SHORT тянется к F.High, LONG — к F.Low.")
    print("  «Лучше%» — насколько вход выгоднее спот-цены в сторону позиции.")
    print("  Стоп = Downside q0.1 модели (нетто) от ВХОДА (LONG ниже, SHORT выше).")
    print(f"  Профит = {tp_frac:.0%} пути от входа к ДАЛЬНЕЙ границе коридора"
          " (SHORT → F.Low, LONG → F.High). R:R = Профит% / Стоп%.")
    print(f"  Объём = ⌊{pos_k:.0f}K ÷ (вход × лот)⌋ лотов."
          " Если лимитка не залилась — заявку снять, в рынок не лезть.")
    if any_na:
        print("  * N/A — недоступно для торговли через Tinkoff (делистинг/смена ISIN);"
              " расширить через env TINKOFF_UNAVAILABLE_TICKERS.")
    if any_unsure_lot:
        print("  * (?) — лот-размер не зашит в таблице; проверьте спецификацию"
              " инструмента в Tinkoff/MOEX перед подачей заявки.")


def _print_best_trades(sorted_rows, top_n: int = 10,
                       position_rub: float = 10_000.0,
                       entry_frac: float = 0.8) -> None:
    """ЛУЧШИЕ СДЕЛКИ НА ЗАВТРА: топ-N по итоговому рейтингу FinalScore +
    автоматические торговые инструкции (тип заявки, цена, стоп, объём)."""
    cand = [r for r in sorted_rows
            if r["exp_pnl"] is not None and r["selected"] is not False]
    top = cand[:top_n]
    if not top:
        return

    print(f"\n  ЛУЧШИЕ СДЕЛКИ НА ЗАВТРА (топ-{len(top)} по итоговому рейтингу FinalScore)")
    print("  " + "-" * 74)
    for i, r in enumerate(top, 1):
        label, _, _ = _verdict_meta(r["verdict"])
        exp  = r["exp_pnl"]
        prob = r["prob_profit"]
        exp_txt  = f"{exp:+.2f}%"         if exp  is not None else "—"
        prob_txt = f"{prob * 100:.0f}%"   if prob is not None else "—"
        score    = r.get("final_score")
        score_txt = f"  score={score:.2f}" if score is not None else ""
        print(f"  {i:>2}. {r['ticker']:<6}{r['strategy']:<16}({r['direction']})"
              f"   [{label}]  ExpPnL={exp_txt}  ProbProfit={prob_txt}{score_txt}")
    print("  " + "-" * 74)

    _print_order_instructions(top, position_rub, entry_frac)
