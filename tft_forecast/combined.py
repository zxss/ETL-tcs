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

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

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


def _score_row(r, strict: bool):
    """
    Считает FinalScore (0..1) с учётом рыночного контекста и риск-фильтров,
    проставляет флаги предупреждений и допустимость сделки (strict-фильтр).
    Возвращает (final_score, allowed, flags).
    """
    long = r["direction"] == "LONG"
    regime = r["regime"]
    rs = r["rs"]
    vs = r["vol_spike"]
    atr = r["atr_pctl"]
    gdp = r["gap_down_prob"]
    flags = []

    # — суб-скоры (0..1) —
    exp = r["exp_pnl"]
    exp_score = _clamp01(0.5 + (exp or 0.0) / 2.0)          # ±1% → 0..1
    prob_score = r["prob_profit"] if r["prob_profit"] is not None else 0.5

    vmeta = _verdict_meta(r["verdict"])[1]
    verdict_score = {0: 1.0, 1: 0.5, 2: 0.0}.get(vmeta, 0.4)
    fdr_score = 1.0 if r["fdr"] else (0.0 if r["fdr"] is not None else 0.5)
    pbo_score = _clamp01(1.0 - r["pbo"]) if r["pbo"] is not None else 0.5
    validation_score = (verdict_score + fdr_score + pbo_score) / 3.0

    liq_score = (r["liq_score"] or 50) / 100.0

    rs_eff = (rs if long else -rs) if rs is not None else 0.0
    rs_score = _clamp01(0.5 + rs_eff / 10.0)                # ±5% → 0..1

    if regime is None:
        regime_score = 0.5
    elif long:
        regime_score = {"BULL": 1.0, "NEUTRAL": 0.5, "BEAR": 0.2}.get(regime, 0.5)
    else:
        regime_score = {"BEAR": 1.0, "NEUTRAL": 0.5, "BULL": 0.2}.get(regime, 0.5)

    if vs is None:
        vol_score = 0.5
    elif vs > 4.0:
        vol_score = 0.2
    elif vs > 2.5:
        vol_score = 0.4
    elif vs < 0.8:
        vol_score = 0.4
    else:
        vol_score = 0.7

    final = (0.30 * exp_score + 0.15 * prob_score + 0.15 * validation_score +
             0.15 * liq_score + 0.10 * rs_score + 0.10 * regime_score +
             0.05 * vol_score)

    # — множительные риск-штрафы и флаги —
    # Market Regime penalty (−30%)
    if long and regime == "BEAR":
        final *= 0.70
    if (not long) and regime == "BULL":
        final *= 0.70
    # High Risk Short: SHORT при сильной бумаге (RS > +5%) → −25%
    if (not long) and rs is not None and rs > 5.0:
        final *= 0.75
        flags.append("High Risk Short")
    # Volume climax / аномальный объём
    if vs is not None and vs > 2.5:
        flags.append("⚠ Volume Climax")
    if vs is not None and vs > 4.0:
        final *= 0.70   # ExpPnL Confidence × 0.7
    # Overnight gap risk — только long_overnight
    if r["strategy"] == "long_overnight" and gdp is not None:
        if gdp > 0.40:
            flags.append("⚠ High Overnight Risk")
        if gdp > 0.50:
            final *= 0.70   # OvernightScore × 0.7

    # — жёсткий рыночный фильтр —
    allowed = True
    if strict:
        if (long and regime == "BEAR") or ((not long) and regime == "BULL"):
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


def select_top_rows(val_rows, forecasts, tickers, strats, *,
                    show_all: bool = False, top_n: int = 10,
                    strict: bool | None = None) -> list[dict]:
    """Та же пайплайн-логика, что в print_combined → _print_best_trades,
    но без печати. Возвращает топ-N строк-кандидатов (сортированы по FinalScore).

    Используется services/place_orders.py: для выставления заявок нужна
    ТА ЖЕ выборка, что показана пользователю в блоке «ЛУЧШИЕ СДЕЛКИ».
    """
    if strict is None:
        import config as _cfg
        strict = bool(getattr(_cfg, "STRICT_MARKET_FILTER", False))
    fc = dict(forecasts or {})
    fc.pop("__meta__", None)
    rows = _build_rows(val_rows, fc, tickers, strats)
    if not rows:
        return []
    rows = _apply_selection(rows, show_all)
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
    _print_best_trades(rows, top_n=best_top_n, position_rub=best_position,
                       entry_frac=entry_frac)
    _print_market_summary(meta)


# ── НЕДЕЛЬНЫЙ ДАШБОРД (формат дневного) ───────────────────────────────────────

_DOW_RU = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def _next_trading_day(d):
    import datetime as dt
    d = d + dt.timedelta(days=1)
    while d.weekday() >= 5:          # сб/вс → следующий будний
        d += dt.timedelta(days=1)
    return d


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
    total = len(rows)
    if top_n and top_n > 0:
        rows = rows[:top_n]

    # Окно прогноза от ТЕКУЩЕГО дня недели (следующие H торговых дней).
    h = int(fc[rows[0]["ticker"]].get("WeekHorizon", 5))
    as_of = meta.get("as_of") or dt.datetime.now(dt.timezone(dt.timedelta(hours=3)))
    today = as_of.date() if hasattr(as_of, "date") else as_of
    start = _next_trading_day(today)
    end = start
    for _ in range(h - 1):
        end = _next_trading_day(end)

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
    stop_pct:        Optional[float]  # |down_pct|
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


def build_orders(top: list[dict], position_rub: float,
                 entry_frac: float, tp_frac: float = 1.0) -> list[Order]:
    """Чистая функция: top-N рейтинга → список структурированных Order.

    Используется и принтером `_print_order_instructions` (для печати таблицы),
    и services/place_orders.py (для отправки в T-Invest sandbox). Логика
    расчёта ТА ЖЕ (никаких расхождений между «что показано» и «что отправлено»).

    Логика:
      Тип       — Лимитная (LIMIT).
      Цена входа — ВНУТРИ прогнозного коридора, ближе к экстремуму в свою
                  пользу: SHORT — к F.High, LONG — к F.Low (см. _limit_entry_price).
      Стоп-цена  — от ВХОДА (не от anchor), чтобы сохранить риск ≈ Downside%:
        LONG  → entry × (1 + Down/100)   [Down < 0 → стоп ниже входа]
        SHORT → entry × (1 − Down/100)   [Down < 0 → стоп выше входа]
      Объём лотов → floor(position_rub / (entry × lot_size)), мин. 1 лот.
      Сумма ₽    → лотов × lot_size × entry.
    """
    unavail = _unavailable_tickers()
    orders: list[Order] = []
    for r in top:
        anchor = r.get("anchor_price")
        down   = r.get("down")
        f_low  = r.get("f_low")
        f_high = r.get("f_high")
        tk     = r["ticker"]
        lng    = (r["direction"] == "LONG")
        lot, lot_known = _lot_size(tk)
        na = tk in unavail

        entry, _src = _limit_entry_price(r["direction"], anchor, f_low, f_high, entry_frac)

        if entry and anchor and anchor > 0:
            better = (entry - anchor) / anchor * 100.0
            if lng:
                better = -better
        else:
            better = None

        if entry and entry > 0 and not na:
            lots  = max(1, int(position_rub / (entry * lot)))
            total = lots * lot * entry
        else:
            lots = total = None

        if entry and entry > 0 and down is not None:
            stop_p   = entry * (1.0 + down / 100.0) if lng else entry * (1.0 - down / 100.0)
            stop_pct = abs(down)
        else:
            stop_p = stop_pct = None

        # take-profit от диапазона прогноза (противоположная входу граница)
        tp_p = _take_profit_price(r["direction"], entry, f_low, f_high, tp_frac)
        tp_pct = (abs(tp_p - entry) / entry * 100.0) if (tp_p and entry) else None

        orders.append(Order(
            ticker=tk, strategy=r["strategy"], direction=r["direction"],
            anchor_price=anchor, f_low=f_low, f_high=f_high, down_pct=down,
            entry_price=entry, better_pct=better,
            stop_price=stop_p, stop_pct=stop_pct,
            tp_price=tp_p, tp_pct=tp_pct,
            lot_size=lot, lot_known=lot_known,
            quantity_lots=lots, total_rub=total,
            unavailable=na,
        ))
    return orders


def _print_order_instructions(top: list[dict], position_rub: float,
                              entry_frac: float) -> None:
    """Печатает блок «ИНСТРУКЦИИ ДЛЯ АВТОЗАЯВОК» из подготовленных Order."""
    import config as _cfg
    tp_frac = float(getattr(_cfg, "LIMIT_TP_FRACTION", 1.0))
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
