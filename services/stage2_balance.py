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
from services.broker.base import POSITION_SECTIONS

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
    # Позиции — по ВСЕМ разделам ответа, а не только securities: срочные
    # инструменты лежат в futures/options и раньше в счётчик не попадали.
    pos_raw = snap.get("positions", {}) or {}
    positions = [p for section in POSITION_SECTIONS for p in (pos_raw.get(section) or [])
                 if float(p.get("balance", 0) or 0) != 0
                 and p.get("instrumentUid") not in skip]
    # Контур берём у клиента, а не из аргумента: отчёт по боевому счёту
    # «Робот» подписывался как SANDBOX, и это ровно та ошибка, из-за которой
    # боевой отчёт можно прочитать как демонстрационный.
    is_sandbox = "Sandbox" in str(getattr(broker, "ORDERS_METHOD", ""))
    return {
        "run_id": run_id,
        "phase": phase,
        "captured_at": dt.datetime.now(MSK).isoformat(timespec="seconds"),
        "account_id": account_id,
        "sandbox": is_sandbox,
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
        "sandbox": next((b.get("sandbox") for b in raw.values()
                         if b.get("sandbox") is not None), None),
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


def _load_snapshots(stage_dir: str, account_id: str | None) -> list[dict]:
    """balance.json всех фаз по времени снимка (только этот счёт)."""
    import glob
    out = []
    for f in glob.glob(os.path.join(stage_dir, "runs", "*", "balance.json")):
        b = _read_json(f)
        if not b or b.get("total_portfolio_rub") is None or not b.get("captured_at"):
            continue
        if account_id and b.get("account_id") and b["account_id"] != account_id:
            continue
        try:
            t = dt.datetime.fromisoformat(str(b["captured_at"]))
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone(dt.timedelta(hours=3)))
        out.append({**b, "captured": t})
    return sorted(out, key=lambda x: x["captured"])


def decompose(stage_dir: str, account_id: str | None) -> dict | None:
    """Казначейство отдельно от торговли по снимкам фаз и treasury_ledger.

    Метод аудита r3 (audit/stage2_decompose.py) в телескопической форме:
        казначейство = Δ стоимости паёв − чистые покупки паёв (mode=broker);
        торговля      = Δ счёта − казначейство.
    Считается по первому и последнему снимку, поэтому смещение времени операций
    с паями внутри отдельных интервалов в итог не попадает.

    Зачем (аудит 26.09.2026, F6). Прежние строки врали: «внутридневной P&L» =
    CLEANUP − ORDER показывал на r4 +7 715 ₽ при НУЛЕ внутридневных сделок — это
    переоценка паёв днём; «перенос через ночь» мерил 18:20→18:35; настоящая ночь
    (выход в CLOSE утром) и основной доход паёв падали в «Казначейство и прочее».
    Правильно на r4: торговля +1 483 ₽, казначейство +14 117 ₽ из +15 600 ₽.
    """
    snaps = _load_snapshots(stage_dir, account_id)
    if len(snaps) < 2:
        return None
    first, last = snaps[0], snaps[-1]
    flows, n = 0.0, 0
    try:
        import database
        with database.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ts, side, amount_rub FROM treasury_ledger "
                            "WHERE mode = 'broker' AND ts > %s AND ts <= %s "
                            "AND (%s::text IS NULL OR account_id = %s)",
                            (first["captured"], last["captured"],
                             account_id, account_id))
                for _ts, side, amount in cur.fetchall():
                    amt = float(amount or 0.0)
                    flows += amt if str(side).upper() == "BUY" else -amt
                    n += 1
    except Exception as e:  # noqa: BLE001 — без реестра разложения нет
        log.warning("Реестр казначейства недоступен, разложение не построено: %s", e)
        return None
    etf0 = float(first.get("etf_value_rub") or 0.0)
    etf1 = float(last.get("etf_value_rub") or 0.0)
    total = float(last["total_portfolio_rub"]) - float(first["total_portfolio_rub"])
    treasury = (etf1 - etf0) - flows
    return {"from": first["captured"], "to": last["captured"], "total": total,
            "treasury": treasury, "trading": total - treasury, "fund_ops": n}


def _benchmark_block(summaries: list[dict], base: float | None,
                     total: float | None) -> list[str]:
    """Безрисковая планка: что те же деньги дали бы по ключевой ставке.

    Итог контура сам по себе ничего не говорит: свободные деньги лежат в фонде
    денежного рынка, фонд идёт за ставкой ЦБ. Поэтому «плюс за N дней» имеет
    смысл только рядом со ставкой за те же N дней. Ставка считается как 365
    календарных дней в году: это доходность «лежать в деньгах», она начисляется
    и в выходные, а не только в торговые сессии.

    Без таблицы ставки блок не печатается — выдумывать планку нельзя.
    """
    if not summaries or not base or total is None:
        return []
    first, last = summaries[0].get("trading_day"), summaries[-1].get("trading_day")
    if not first or not last:
        return []
    try:
        d0 = dt.date.fromisoformat(str(first)[:10])
        d1 = dt.date.fromisoformat(str(last)[:10])
    except ValueError:
        return []
    try:
        import database
        rate = database.get_key_rate_avg(d0, d1)
        asof_date, rate_now = database.get_key_rate(d1)
    except Exception:  # noqa: BLE001 — отчёт важнее бенчмарка
        return []
    if not rate:
        return []
    # Календарные дни, а не торговые: деньги в фонде работают каждый день.
    held = max(1, (d1 - d0).days + 1)
    rf_pct = float(rate) * held / 365.0
    rf_rub = base * rf_pct / 100.0
    got_pct = total / base * 100.0
    alpha = total - rf_rub
    verdict = "ВЫШЕ ставки" if alpha > 0 else "НИЖЕ ставки"

    def _pct(x: float, nd: int = 3) -> str:
        return f"{x:.{nd}f}".replace(".", ",")

    out = ["", "БЕЗРИСКОВАЯ ПЛАНКА"]
    if rate_now is not None:
        out.append(f"  Ключевая ставка ЦБ     {_pct(rate_now, 2):>12} %   "
                   f"(с {asof_date}, средняя за период {_pct(float(rate), 2)} %)")
    out += [
        f"  Ставка за {held:>3} дн.       {_fmt(rf_rub, 12)} ₽   ({_pct(rf_pct)} %)",
        f"  Результат контура      {_fmt(total, 12)} ₽   ({_pct(got_pct)} %)",
        f"  Сверх ставки           {_fmt(alpha, 12)} ₽   {verdict}",
    ]
    return out


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
    sandbox = next((s.get("sandbox") for s in summaries if s.get("sandbox") is not None), None)
    if sandbox is None:
        # Сводки, записанные до 24.09.2026, признака контура не несут. Запасное
        # определение по номеру счёта: у боевого он числовой (9999999999), у
        # песочницы — UUID. Совпадение с PROD_ACCOUNT_ID решает однозначно.
        prod_acc = str(getattr(config, "PROD_ACCOUNT_ID", "") or "")
        sandbox = False if (prod_acc and account == prod_acc) else \
            (not str(account).replace(" ", "").isdigit() if account != "—" else None)
    env = "SANDBOX" if sandbox else ("PROD (РЕАЛЬНЫЕ ДЕНЬГИ)" if sandbox is False else "СЧЁТ")

    L = []
    L.append(f"ОТЧЁТ ПО ТЕСТОВОМУ БАЛАНСУ — {config.STAGE2_TEST_ID}")
    start_txt = f"{start:,.2f}".replace(",", "\u00a0").replace(".", ",")
    L.append(f"Счёт: {env} {account}   Стартовый баланс: {start_txt} ₽")
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

    unreal = summaries[-1].get("unrealised_pnl_rub") if summaries else None
    d = decompose(base, account if account != "—" else None)
    L += ["", "РАЗЛОЖЕНИЕ (снимки фаз + реестр казначейства)"]
    if d:
        L += [f"  Торговля               {_fmt(d['trading'], 12)} ₽",
              f"  Казначейство (паи)     {_fmt(d['treasury'], 12)} ₽",
              f"  Итого по снимкам       {_fmt(d['total'], 12)} ₽   "
              f"({d['from']:%d.%m %H:%M} → {d['to']:%d.%m %H:%M}, "
              f"операций с паями {d['fund_ops']})",
              f"  из торговли не закрыто {_fmt(unreal, 12)} ₽   (нереализованный P&L)"]
    else:
        L.append("  недоступно: нет снимков фаз или реестра казначейства")

    L += _benchmark_block(summaries, cum_base, total)

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
