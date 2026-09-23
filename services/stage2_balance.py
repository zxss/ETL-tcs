"""
Баланс демо-счёта для Этапа 2 (STAGE2-DEMO-TZ.md §9).

Баланс снимается ЧЕТЫРЕ раза в день — на каждой фазе. Это позволяет отделить
внутридневной ход от переноса через ночь: разница PREP→CLEANUP — работа
интрадея, CLEANUP→OVERNIGHT следующего дня — стоимость переноса.

Важно про статус баланса: по §1 он отслеживается как ДИАГНОСТИКА ИСПОЛНЕНИЯ,
а не как оценка стратегии. Отчёт намеренно не содержит Sharpe, CAGR и Profit
Factor: переаудит дал Deflated Sharpe 0,251 при концентрации 119% прибыли в
5 днях из 151, поэтому на пятнадцати днях любая оценка доходности статистически
бессмысленна.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import config
from services import account_status

log = logging.getLogger("stage2.balance")

MSK = dt.timezone(dt.timedelta(hours=3))
PHASES = ("PREP", "CLOSE", "ORDER", "CLEANUP", "OVERNIGHT")


def _money(m: dict | None) -> float:
    """MoneyValue/Quotation {units, nano} → float."""
    if not m:
        return 0.0
    return int(m.get("units", 0) or 0) + int(m.get("nano", 0) or 0) / 1e9


def capture(broker, account_id: str, *, run_id: str, phase: str,
            sandbox: bool | None = None, exclude_uids=()) -> dict:
    """Снимок баланса на текущей фазе → структура для balance.json.

    Источник — OperationsService/GetPortfolio через account_status.snapshot().
    exclude_uids — паи фонда казначейства: это припаркованный кэш, а не
    торговая позиция, в счётчик открытых позиций они не входят.
    """
    snap = account_status.snapshot(broker, account_id, sandbox=sandbox)
    pf = snap.get("portfolio", {}) or {}
    skip = set(exclude_uids or ())
    positions = [p for p in (snap.get("positions", {}) or {}).get("securities", []) or []
                 if int(p.get("balance", 0) or 0) != 0
                 and p.get("instrumentUid") not in skip]
    return {
        "run_id": run_id,
        "phase": phase,
        "captured_at": dt.datetime.now(MSK).isoformat(timespec="seconds"),
        "account_id": account_id,
        "total_portfolio_rub": round(_money(pf.get("totalAmountPortfolio")), 4),
        "free_cash_rub": round(_money(pf.get("totalAmountCurrencies")), 4),
        "shares_value_rub": round(_money(pf.get("totalAmountShares")), 4),
        "etf_value_rub": round(_money(pf.get("totalAmountEtf")), 4),
        "unrealised_pnl_rub": round(_money(pf.get("expectedYield")), 4),
        "open_positions": len(positions),
        "active_orders": len(snap.get("orders", []) or []),
        "active_stop_orders": len(snap.get("stops", []) or []),
    }


# ── Дневная сводка ────────────────────────────────────────────────────────────

def _stage2_dir() -> str:
    return getattr(config, "STAGE2_DIR",
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "audit", "stage2-demo"))


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def collect_day(trading_day: dt.date, run_dirs: dict[str, str],
                *, prev_closing: float | None = None) -> dict:
    """Сводит балансы четырёх фаз в audit/stage2-demo/balance/YYYY-MM-DD.json.

    run_dirs: {"PREP": путь_к_каталогу, ...}. Отсутствующие фазы допустимы —
    их значения будут None, а сводка получит пометку в incomplete_phases.
    """
    by_phase: dict[str, float | None] = {}
    raw: dict[str, dict] = {}
    for ph in PHASES:
        d = run_dirs.get(ph)
        b = _read_json(os.path.join(d, "balance.json")) if d else None
        raw[ph] = b or {}
        by_phase[ph] = b.get("total_portfolio_rub") if b else None

    known = [v for v in by_phase.values() if v is not None]
    opening = prev_closing if prev_closing is not None else (known[0] if known else None)
    closing = known[-1] if known else None

    day_change = (closing - opening) if (opening is not None and closing is not None) else None
    day_change_pct = (day_change / opening * 100.0) if (day_change is not None and opening) else None

    # Внутридневной ход — от постановки заявок до закрытия интрадея.
    intraday = (by_phase["CLEANUP"] - by_phase["ORDER"]
                if by_phase.get("CLEANUP") is not None and by_phase.get("ORDER") is not None
                else None)
    # Перенос через ночь — что произошло между cleanup и финальным снимком.
    carry = (by_phase["OVERNIGHT"] - by_phase["CLEANUP"]
             if by_phase.get("OVERNIGHT") is not None and by_phase.get("CLEANUP") is not None
             else None)

    final = raw.get("OVERNIGHT") or raw.get("CLEANUP") or {}
    account_id = next((b.get("account_id") for b in raw.values() if b.get("account_id")), None)
    return {
        "trading_day": trading_day.isoformat(),
        "account_id": account_id,
        "opening_balance_rub": opening,
        "closing_balance_rub": closing,
        "day_change_rub": round(day_change, 4) if day_change is not None else None,
        "day_change_pct": round(day_change_pct, 4) if day_change_pct is not None else None,
        "by_phase": by_phase,
        "intraday_pnl_rub": round(intraday, 4) if intraday is not None else None,
        "overnight_carry_rub": round(carry, 4) if carry is not None else None,
        "unrealised_pnl_rub": final.get("unrealised_pnl_rub"),
        "positions_overnight": final.get("open_positions"),
        "positions_overnight_value_rub": final.get("shares_value_rub"),
        "incomplete_phases": [p for p in PHASES if by_phase.get(p) is None],
    }


def save_day(summary: dict) -> str:
    out_dir = os.path.join(_stage2_dir(), "balance")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{summary['trading_day']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


# ── Отчёт balance_report.md ───────────────────────────────────────────────────

def _fmt(v, width=10, digits=2, dash="—"):
    """Русский формат: пробел как разделитель тысяч, запятая — десятичный."""
    if v is None:
        return f"{dash:>{width}}"
    txt = f"{v:,.{digits}f}".replace(",", "\u00a0").replace(".", ",")
    return f"{txt:>{width}}"


def build_report(*, start_balance: float | None = None,
                 target_days: int | None = None,
                 execution: dict | None = None) -> str:
    """Пересобирает balance_report.md по всем дневным сводкам.

    execution — необязательный блок статистики исполнения из execution_audit
    (заявок, заливок, проскальзывание факт/расчёт).
    """
    base = _stage2_dir()
    bal_dir = os.path.join(base, "balance")
    days = sorted(f for f in os.listdir(bal_dir) if f.endswith(".json")) \
        if os.path.isdir(bal_dir) else []
    summaries = [s for s in (_read_json(os.path.join(bal_dir, f)) for f in days) if s]

    start = start_balance if start_balance is not None \
        else float(getattr(config, "STAGE2_START_BALANCE_RUB", 100000))
    target = target_days if target_days is not None \
        else int(getattr(config, "STAGE2_TARGET_DAYS", 15))
    account = next((s.get("account_id") for s in summaries if s.get("account_id")), "—")

    L = []
    L.append(f"ОТЧЁТ ПО ТЕСТОВОМУ БАЛАНСУ — {getattr(config, 'STAGE2_TEST_ID', 'stage2-demo')}")
    start_txt = f"{start:,.2f}".replace(",", "\u00a0").replace(".", ",")
    L.append(f"Счёт: SANDBOX {account}   Стартовый баланс: {start_txt} ₽")
    L.append(f"Торговых дней пройдено: {len(summaries)} из {target}")
    L.append("")
    L.append(f"{'День':<12}{'Открытие':>12}{'PREP':>10}{'CLOSE':>10}{'ORDER':>10}"
             f"{'CLEANUP':>10}{'OVERNIGHT':>11}{'Δ день':>11}{'Δ нараст.':>11}{'Поз.':>6}")

    cum_base = summaries[0].get("opening_balance_rub") if summaries else start
    for s in summaries:
        ph = s.get("by_phase", {})
        closing = s.get("closing_balance_rub")
        cum_pct = ((closing / cum_base - 1) * 100.0) if (closing and cum_base) else None
        L.append(
            f"{s['trading_day']:<12}"
            f"{_fmt(s.get('opening_balance_rub'), 12)}"
            f"{_fmt(ph.get('PREP'), 10, 0)}{_fmt(ph.get('CLOSE'), 10, 0)}"
            f"{_fmt(ph.get('ORDER'), 10, 0)}"
            f"{_fmt(ph.get('CLEANUP'), 10, 0)}{_fmt(ph.get('OVERNIGHT'), 11, 0)}"
            f"{_fmt(s.get('day_change_rub'), 11)}"
            f"{(f'{cum_pct:>10.2f}%'.replace('.', ',') if cum_pct is not None else '         —')}"
            f"{(s.get('positions_overnight') if s.get('positions_overnight') is not None else '—'):>6}"
        )

    total = None
    if summaries and summaries[-1].get("closing_balance_rub") is not None and cum_base:
        total = summaries[-1]["closing_balance_rub"] - cum_base
        L.append(f"{'ИТОГО':<12}{'':>63}{_fmt(total, 11)}"
                 f"{(total / cum_base * 100.0):>10.2f}%".replace(".", ","))

    def _sum(key):
        vals = [s.get(key) for s in summaries if s.get(key) is not None]
        return sum(vals) if vals else None

    L += ["", "РАЗЛОЖЕНИЕ",
          f"  Внутридневной P&L      {_fmt(_sum('intraday_pnl_rub'), 12)} ₽",
          f"  Перенос через ночь     {_fmt(_sum('overnight_carry_rub'), 12)} ₽",
          f"  Нереализованный P&L    {_fmt(summaries[-1].get('unrealised_pnl_rub') if summaries else None, 12)} ₽"]

    if execution:
        placed = execution.get("orders") or 0
        filled = execution.get("filled") or 0
        rate = (filled / placed * 100.0) if placed else None
        fact = execution.get("slippage_fact_pct")
        calc = execution.get("slippage_expected_pct")
        gap = ((fact / calc - 1) * 100.0) if (fact and calc) else None
        L += ["", "ИСПОЛНЕНИЕ",
              f"  Заявок выставлено                     {placed:>6}",
              f"  Исполнено                             {filled:>6}"
              + (f"   ({rate:.1f}%)" if rate is not None else ""),
              f"  Среднее проскальзывание, факт       {(f'{fact:.3f}%' if fact is not None else '—'):>8}",
              f"  Среднее проскальзывание, расчёт     {(f'{calc:.3f}%' if calc is not None else '—'):>8}"]
        if gap is not None:
            verdict = "в пределах порога 20%" if abs(gap) <= 20 else "ПРЕВЫШЕН порог 20%"
            L.append(f"  Расхождение                          {gap:>+6.0f}%   {verdict}")

    L += ["", "Sharpe, CAGR и Profit Factor намеренно отсутствуют: критерий этого",
          "теста функциональный, см. §1 ТЗ."]
    return "\n".join(L) + "\n"


def save_report(**kw) -> str:
    base = _stage2_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "balance_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report(**kw))
    return path
