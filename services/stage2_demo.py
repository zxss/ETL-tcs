"""
Оркестратор Этапа 2 — функциональный тест на демо-счёте (STAGE2-DEMO-TZ.md).

Четыре фазы торгового дня, каждая запускается отдельным заданием cron:

    PREP      09:45  считает сигналы и ЗАМОРАЖИВАЕТ план на диске; заявок не ставит
    ORDER     10:05  исполняет замороженный план; ничего не пересчитывает
    CLEANUP   18:20  закрывает внутридневные позиции, снимает незалившиеся лимитки
    OVERNIGHT 18:35  пересчитывает только long_overnight и ставит ночные заявки

Почему план замораживается (§3): между PREP и ORDER проходит 20 минут, за
которые меняются якорная цена, реализованная часть дневного хода и результат
переобучения TFT. Если ORDER пересчитает сигналы, требование «заявка
соответствует сигналу» станет непроверяемым. Поэтому ORDER читает plan.json и
отказывается работать, если конфигурация или датасет изменились.

Критерий теста функциональный (§1): проверяется работоспособность контура,
сохранность данных и воспроизводимость, а не прибыльность.

Запуск:
    python3 -m services.stage2_demo prep|order|cleanup|overnight
    python3 -m services.stage2_demo status
    python3 -m services.stage2_demo resume        # снять блокировку после FAIL
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                  # noqa: E402
import database                                # noqa: E402
from services import calendar as trading_cal   # noqa: E402
from services import stage2_balance            # noqa: E402

log = logging.getLogger("stage2")

MSK = dt.timezone(dt.timedelta(hours=3))
PHASES = ("PREP", "ORDER", "CLEANUP", "OVERNIGHT")

PASS, PASS_WARN, FAIL = "PASS", "PASS_WITH_WARNINGS", "FAIL"


# ── Пути и состояние ─────────────────────────────────────────────────────────

def base_dir() -> str:
    return getattr(config, "STAGE2_DIR",
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "audit", "stage2-demo"))


def _p(*parts) -> str:
    return os.path.join(base_dir(), *parts)


def _read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_state() -> dict:
    st = _read_json(_p("state.json"))
    if st:
        return st
    return {
        "test_id": getattr(config, "STAGE2_TEST_ID", "stage2-demo-15d"),
        "started_at": None,
        "completed_trading_days": 0,
        "target_trading_days": int(getattr(config, "STAGE2_TARGET_DAYS", 15)),
        "status": "running",
        "days": [],
        "halted_reason": None,
    }


def save_state(st: dict) -> None:
    _write_json(_p("state.json"), st)


def new_run_id(phase: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(MSK)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{phase}"


def make_run_dir(run_id: str) -> str:
    """Каталог запуска. exist_ok=False намеренно: коллизия RUN_ID — ошибка,
    а не повод перезаписать чужой результат (§10)."""
    path = _p("runs", run_id)
    os.makedirs(path, exist_ok=False)
    return path


# ── Хеши воспроизводимости (§3) ──────────────────────────────────────────────

_CONFIG_KEYS = (
    "TRADING_MODE", "TRADING_STRATEGIES", "VALIDATION_STRATS", "SCORE_MODE",
    "APPLY_RISK_PENALTIES", "SELLER_MOMENTUM_SHORT_ENABLED", "SHORT_IMOEX_MAX_TREND",
    "OVERNIGHT_MAX_MARKET_ATR_PCTL", "OVERNIGHT_MIN_EDGE_X_COST",
    "LIMIT_ENTRY_FRACTION", "LIMIT_TP_FRACTION", "ORDER_FILL_WAIT_SEC",
    "INTRADAY_SQUARE_OFF_ENABLED", "INTRADAY_SQUARE_OFF_TIME",
    "FIXED_POSITION_OVERFLOW_MODE", "BEST_TRADES_TOP_N", "BEST_TRADES_POSITION_RUB",
    "VALIDATION_COST_RT", "TFT_COST_RT", "VALIDATION_FULL_UNIVERSE",
    "TFT_EPOCHS", "TFT_HIDDEN", "WEEK_HORIZON_DAYS", "INCLUDE_WEEKEND_TRADING",
    "STAGE2_TARGET_DAYS", "STAGE2_ALLOW_DATASET_DRIFT",
)


def config_snapshot() -> dict:
    """Значимые параметры config — включая дефолты, а не только то, что в .env."""
    out = {}
    for k in _CONFIG_KEYS:
        v = getattr(config, k, None)
        out[k] = list(v) if isinstance(v, (list, tuple, set)) else v
    return out


def config_hash(snap: dict | None = None) -> str:
    snap = snap if snap is not None else config_snapshot()
    blob = json.dumps(snap, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


_DATASET_SQL = """
SELECT ticker, date, open, high, low, close, volume
FROM market_data
ORDER BY ticker, date;
"""


def dataset_fingerprint(conn) -> dict:
    """SHA-256 по всем строкам market_data + границы датасета.

    Хеш считается потоково, чтобы не поднимать 33 тысячи строк в память списком.
    """
    h = hashlib.sha256()
    rows = 0
    with conn.cursor(name="stage2_ds") as cur:      # серверный курсор
        cur.itersize = 10000
        cur.execute(_DATASET_SQL)
        for tk, d, o, hi, lo, c, v in cur:
            h.update(f"{tk}|{d}|{o}|{hi}|{lo}|{c}|{v}\n".encode("utf-8"))
            rows += 1
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM market_data")
        max_date = cur.fetchone()[0]
        cur.execute("SELECT MAX(ts) FROM market_data_5m")
        max_ts = cur.fetchone()[0]
    return {
        "market_data_max_date": max_date.isoformat() if max_date else None,
        "market_data_rows": rows,
        "market_data_5m_max_ts": max_ts.isoformat() if max_ts else None,
        "dataset_hash": "sha256:" + h.hexdigest(),
    }


# ── Результат фазы ───────────────────────────────────────────────────────────

class PhaseResult:
    """Итог фазы: статус, критические провалы и предупреждения."""

    def __init__(self, phase: str, run_id: str, run_dir: str):
        self.phase, self.run_id, self.run_dir = phase, run_id, run_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.critical_passed = 0
        self.skipped: str | None = None
        self.data: dict = {}

    def check(self, ok: bool, message: str, *, critical: bool = True) -> bool:
        if ok:
            if critical:
                self.critical_passed += 1
            return True
        (self.errors if critical else self.warnings).append(message)
        return False

    @property
    def verdict(self) -> str:
        if self.errors:
            return FAIL
        return PASS_WARN if self.warnings else PASS

    def exit_code(self) -> int:
        return 1 if self.errors else 0


# ── Предполётные проверки (§8) ───────────────────────────────────────────────

def preflight(res: PhaseResult, *, env: str, prod_flag: bool) -> None:
    """Критические проверки контура. Провал любой → FAIL, торговля не идёт."""
    res.check(env == "SANDBOX", f"контур не SANDBOX: {env}")
    res.check(str(getattr(config, "TRADING_MODE", "")).lower() != "prod",
              f"TRADING_MODE={getattr(config, 'TRADING_MODE', None)}")
    res.check(not prod_flag, "передан флаг --prod")
    res.check(not getattr(config, "PROD_ACCOUNT_ID", ""),
              "PROD_ACCOUNT_ID заполнен — боевой счёт должен быть закрыт")


def guard_halted(state: dict) -> str | None:
    """Тест остановлен критическим FAIL — фазы не выполняются до resume."""
    if state.get("status") == "halted" and getattr(config, "STAGE2_HALT_ON_FAIL", True):
        return state.get("halted_reason") or "остановлен предыдущим FAIL"
    return None


def guard_finished(state: dict) -> bool:
    return state.get("status") == "finished" or \
        int(state.get("completed_trading_days", 0)) >= int(state.get("target_trading_days", 15))


# ── Общий каркас фазы ────────────────────────────────────────────────────────

def _today(now: dt.datetime | None = None) -> dt.date:
    return (now or dt.datetime.now(MSK)).date()


def _run_dirs_for_day(day: dt.date) -> dict[str, str]:
    """Каталоги всех фаз этого торгового дня: {PHASE: путь}."""
    runs = _p("runs")
    if not os.path.isdir(runs):
        return {}
    prefix = day.strftime("%Y%m%d")
    out: dict[str, str] = {}
    for name in sorted(os.listdir(runs)):
        if not name.startswith(prefix) or "-" not in name:
            continue
        phase = name.rsplit("-", 1)[-1]
        if phase in PHASES:
            out[phase] = os.path.join(runs, name)      # последний по времени
    return out


# ── Уведомления в Telegram (§ отчётность) ────────────────────────────────────

_VERDICT_ICON = {PASS: "\u2705", PASS_WARN: "\u26a0\ufe0f", FAIL: "\u26d4"}

# Что показывать по каждой фазе: ключ в res.data → подпись.
_PHASE_FIELDS = {
    "PREP": (("signals_count", "сигналов"), ("plan_orders", "заявок в плане")),
    "ORDER": (("orders_count", "принято"), ("placed_count", "выставлено"),
              ("rejected_count", "отклонено"), ("fills_count", "заливок")),
    "CLEANUP": (("intraday_before", "было интрадея"), ("positions_closed", "закрыто"),
                ("orders_cancelled", "снято лимиток")),
    "OVERNIGHT": (("overnight_orders", "ночных принято"), ("placed_count", "выставлено")),
}


def _rub(value) -> str:
    """Денежный формат: пробел-разделитель тысяч, запятая-разделитель дробной."""
    if value is None:
        return "\u2014"
    return f"{float(value):+,.2f}".replace(",", "\u00a0").replace(".", ",") + "\u00a0\u20bd"


def _notify_phase(res: "PhaseResult", state: dict) -> None:
    """Карточка итога фазы в Telegram. Ошибки отправки игнорируются.

    Вызывается из _finish — единственного места, через которое проходят все
    четыре фазы, поэтому добавлять уведомление в каждую фазу отдельно не нужно.
    """
    try:
        from services import notify
        if not notify.enabled():
            return

        icon = _VERDICT_ICON.get(res.verdict, "")
        lines = [f"{icon} <b>{notify.esc(res.phase)}</b> \u2014 {notify.esc(res.verdict)}",
                 f"<code>{notify.esc(res.run_id)}</code>"]

        facts = [f"{cap}: <b>{notify.esc(res.data[key])}</b>"
                 for key, cap in _PHASE_FIELDS.get(res.phase, ())
                 if res.data.get(key) is not None]
        if facts:
            lines.append("")
            lines += facts

        day = res.data.get("day_summary")
        if day:
            lines.append("")
            lines.append("\u2500\u2500 <b>итог дня</b> \u2500\u2500")
            lines.append(f"день {notify.esc(day.get('completed'))} из "
                         f"{notify.esc(day.get('target'))}")
            if day.get("closing") is not None:
                lines.append(f"баланс: <b>{notify.esc(_rub(day['closing']).lstrip('+'))}</b>")
            lines.append(f"за день: <b>{notify.esc(_rub(day.get('change')))}</b>"
                         + (f" ({notify.esc(_pct(day.get('change_pct')))})"
                            if day.get("change_pct") is not None else ""))
            if day.get("cum_pct") is not None:
                lines.append(f"нарастающим: <b>{notify.esc(_pct(day['cum_pct']))}</b>")
            if day.get("positions") is not None:
                lines.append(f"позиций в ночь: {notify.esc(day['positions'])}")

        for e in res.errors[:5]:
            lines.append(f"\u26d4 {notify.esc(e)}")
        for w in res.warnings[:5]:
            lines.append(f"\u26a0\ufe0f {notify.esc(w)}")
        if state.get("status") == "halted":
            lines.append("")
            lines.append("<b>ТЕСТ ОСТАНОВЛЕН.</b> Снять блокировку: "
                         "<code>python3 -m services.stage2_demo resume</code>")

        # Рутинный PASS приходит без звука, всё остальное — со звуком.
        quiet = (res.verdict == PASS and res.phase != "OVERNIGHT"
                 and bool(getattr(config, "TELEGRAM_SILENT_PHASES", True)))
        notify.send("\n".join(lines), silent=quiet)
    except Exception as e:                       # noqa: BLE001 — отчётность не роняет фазу
        log.warning("уведомление не отправлено: %s", e)


def _pct(value) -> str:
    if value is None:
        return "\u2014"
    return f"{float(value):+.2f}%".replace(".", ",")


def _finish(res: PhaseResult, state: dict) -> int:
    """Записывает итог фазы, при FAIL останавливает тест."""
    _write_json(os.path.join(res.run_dir, "run_meta.json"), {
        "run_id": res.run_id, "phase": res.phase,
        "finished_at": dt.datetime.now(MSK).isoformat(timespec="seconds"),
        "verdict": res.verdict, "errors": res.errors, "warnings": res.warnings,
        "skipped": res.skipped, **res.data,
    })
    with open(os.path.join(res.run_dir, "exit_code"), "w", encoding="utf-8") as f:
        f.write(str(res.exit_code()))

    if res.errors and getattr(config, "STAGE2_HALT_ON_FAIL", True):
        state["status"] = "halted"
        state["halted_reason"] = f"{res.phase} {res.run_id}: {res.errors[0]}"
        save_state(state)
        log.error("ТЕСТ ОСТАНОВЛЕН: %s", state["halted_reason"])
        log.error("Снять блокировку: python3 -m services.stage2_demo resume")

    for e in res.errors:
        log.error("[%s] %s", res.phase, e)
    for w in res.warnings:
        log.warning("[%s] %s", res.phase, w)
    log.info("[%s] вердикт: %s", res.phase, res.verdict)
    _notify_phase(res, state)
    return res.exit_code()


def _skip(phase: str, reason: str) -> int:
    """Фаза не выполняется (не торговый день / тест завершён) — код 0."""
    log.info("[%s] пропуск: %s", phase, reason)
    run_id = new_run_id(phase)
    try:
        d = make_run_dir(run_id)
        _write_json(os.path.join(d, "run_meta.json"), {
            "run_id": run_id, "phase": phase, "skipped": reason,
            "finished_at": dt.datetime.now(MSK).isoformat(timespec="seconds"),
            "verdict": PASS,
        })
    except OSError:
        pass
    return 0


def _common_guards(phase: str, *, now: dt.datetime | None = None):
    """Проверки, общие для всех фаз. Возвращает (state, day) или код выхода."""
    state = load_state()
    if not getattr(config, "STAGE2_ENABLED", True):
        return _skip(phase, "STAGE2_ENABLED=0")
    if guard_finished(state):
        return _skip(phase, f"тест завершён: {state['completed_trading_days']} "
                            f"из {state['target_trading_days']} дней")
    halted = guard_halted(state)
    if halted:
        return _skip(phase, f"тест остановлен ({halted})")
    day = _today(now)
    if not trading_cal.is_trading_day(day):
        return _skip(phase, "not_a_trading_day")
    return state, day


# ── Фаза PREP (§4) ───────────────────────────────────────────────────────────

def phase_prep(*, now: dt.datetime | None = None, prod: bool = False) -> int:
    """Считает сигналы и замораживает план. НИ ОДНОЙ заявки не выставляет.

    Технически: этот путь не импортирует и не вызывает place_limits /
    sync_portfolio — заявку отсюда отправить нечем.
    """
    guard = _common_guards("PREP", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.data_freshness import StaleDataError, ensure_fresh_data
    from services import run_validation
    from services.place_orders import _make_broker_and_account
    from tft_forecast import combined as cmb
    import tft_forecast

    run_id = new_run_id("PREP", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("PREP", run_id, run_dir)
    log.info("[PREP] %s, торговый день %s", run_id, day)

    conn = database.get_connection()
    try:
        # 3. свежесть данных
        try:
            last_date = ensure_fresh_data(conn, refresh=True)
        except StaleDataError as e:
            res.check(False, f"устаревшие данные: {e}")
            return _finish(res, state)

        # 4-5. снимки входа и конфигурации
        ds = dataset_fingerprint(conn)
        cfg_snap = config_snapshot()
        cfg_hash = config_hash(cfg_snap)
        _write_json(os.path.join(run_dir, "input_snapshot.json"),
                    {**ds, "last_date": str(last_date)})
        with open(os.path.join(run_dir, "config_snapshot.env"), "w", encoding="utf-8") as f:
            for k, v in sorted(cfg_snap.items()):
                f.write(f"{k}={' '.join(map(str, v)) if isinstance(v, list) else v}\n")

        # 6. расчёт
        val_rows = run_validation.run(conn, quiet=True)
        forecasts = tft_forecast.run(conn, quiet=True) or {}
        universe = [k for k in forecasts if k != "__meta__"] or config.VALIDATION_TICKERS
        top = cmb.select_top_rows(
            val_rows, forecasts, universe, config.VALIDATION_STRATS,
            top_n=int(getattr(config, "BEST_TRADES_TOP_N", 5)))
        orders = cmb.build_orders(
            top, float(getattr(config, "BEST_TRADES_POSITION_RUB", 10000)),
            float(getattr(config, "LIMIT_ENTRY_FRACTION", 0.2)),
            tp_frac=float(getattr(config, "LIMIT_TP_FRACTION", 0.5)))

        # 7. signals.json — всё, что посчитано; plan.json — только исполнимое
        _write_json(os.path.join(run_dir, "signals.json"), top)
        plan = {
            "test_id": getattr(config, "STAGE2_TEST_ID", "stage2-demo-15d"),
            "trading_day": day.isoformat(),
            "run_id": run_id,
            "created_at": dt.datetime.now(MSK).isoformat(timespec="seconds"),
            "dataset": ds,
            "config_hash": cfg_hash,
            "model": {
                "backend": (forecasts.get("__meta__", {}) or {}).get("backend", "unknown"),
                "epochs": int(getattr(config, "TFT_EPOCHS", 30)),
                "tickers": len(universe),
            },
            "orders": [_plan_order(o, r, day) for o, r in zip(orders, top) if o.is_placeable],
        }
        _write_json(os.path.join(run_dir, "plan.json"), plan)
        res.data.update(signals_count=len(top), plan_orders=len(plan["orders"]),
                        dataset_hash=ds["dataset_hash"], config_hash=cfg_hash)
        log.info("[PREP] сигналов %d, в плане %d заявок", len(top), len(plan["orders"]))

        # 8. снимок счёта и баланса
        broker, account_id, env = _make_broker_and_account(prod)
        _snapshot_account(broker, account_id, run_dir, run_id, "PREP", "before")
        _snapshot_account(broker, account_id, run_dir, run_id, "PREP", "after")

        # 9. предполётные проверки
        preflight(res, env=env, prod_flag=prod)
        res.check(bool(plan["orders"]) or True, "план пуст", critical=False)
        _write_json(os.path.join(run_dir, "preflight.json"), {
            "critical_passed": res.critical_passed,
            "errors": res.errors, "warnings": res.warnings})
    except Exception as e:                       # noqa: BLE001
        log.exception("[PREP] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        conn.close()
    return _finish(res, state)


def _plan_order(o, row: dict, day: dt.date) -> dict:
    """Order + строка дашборда → запись плана (§3, формат plan.json)."""
    return {
        "signal_id": f"{day.strftime('%Y%m%d')}-{o.ticker}-{o.strategy}",
        "ticker": o.ticker,
        "strategy_type": o.strategy,
        "direction": o.direction,
        "order_type": "LIMIT",
        "anchor_price": o.anchor_price,
        "entry_price": o.entry_price,
        "stop_price": o.stop_price,
        "tp_price": o.tp_price,
        "quantity_lots": o.quantity_lots,
        "lot_size": o.lot_size,
        "total_rub": o.total_rub,
        "exp_pnl_pct": row.get("exp_pnl"),
        "final_score": row.get("final_score"),
        "expected_cost_pct": float(getattr(config, "TFT_COST_RT", 0.128)),
        "verdict": row.get("verdict"),
    }


def _snapshot_account(broker, account_id: str, run_dir: str, run_id: str,
                      phase: str, when: str) -> dict:
    """positions_<when>.json + balance.json на текущий момент."""
    from services import account_status
    snap = account_status.snapshot(broker, account_id, sandbox=True)
    _write_json(os.path.join(run_dir, f"positions_{when}.json"), snap)
    bal = stage2_balance.capture(broker, account_id, run_id=run_id, phase=phase)
    _write_json(os.path.join(run_dir, "balance.json"), bal)
    return bal


# ── Фаза ORDER (§5) ──────────────────────────────────────────────────────────

def _find_plan(day: dt.date) -> tuple[dict | None, str | None]:
    """Свежайший plan.json за этот торговый день."""
    for phase_dir in sorted(_run_dirs_for_day(day).values(), reverse=True):
        plan = _read_json(os.path.join(phase_dir, "plan.json"))
        if plan:
            return plan, phase_dir
    return None, None


def validate_plan(plan: dict | None, day: dt.date, current_ds: dict) -> list[str]:
    """Правила исполнения плана (§3). Возвращает список причин отказа."""
    errs: list[str] = []
    if not plan:
        return ["plan.json не найден — фаза PREP не отработала"]
    if plan.get("trading_day") != day.isoformat():
        errs.append(f"план от другого дня: {plan.get('trading_day')} != {day}")
    if plan.get("config_hash") != config_hash():
        errs.append("config_hash не совпадает — конфигурацию меняли между фазами")
    plan_ds = (plan.get("dataset") or {}).get("dataset_hash")
    if plan_ds != current_ds.get("dataset_hash") and \
            not getattr(config, "STAGE2_ALLOW_DATASET_DRIFT", False):
        errs.append("dataset_hash изменился, STAGE2_ALLOW_DATASET_DRIFT=0")
    return errs


def check_order_allowed(po: dict, *, env: str, used_ids: set[str]) -> str | None:
    """Проверки перед отправкой каждой заявки (§5.3). None — можно ставить."""
    from tft_forecast.combined import non_shortable_tickers, trading_strategies

    if env != "SANDBOX" or str(getattr(config, "TRADING_MODE", "")).lower() == "prod":
        return "контур не SANDBOX"
    st = po.get("strategy_type")
    if st not in trading_strategies():
        return f"стратегия {st} вне TRADING_STRATEGIES"
    if po.get("direction") == "SHORT" and po["ticker"].upper() in non_shortable_tickers():
        return "шорт недоступен у брокера"
    lots = po.get("quantity_lots") or 0
    if lots <= 0:
        return "quantity_lots <= 0"
    cap = float(getattr(config, "BEST_TRADES_POSITION_RUB", 10000))
    if (po.get("total_rub") or 0) > cap:
        return f"сумма {po.get('total_rub')} превышает лимит позиции {cap}"
    if po.get("signal_id") in used_ids:
        return "дубль signal_id в этом торговом дне"
    return None


def phase_order(*, now: dt.datetime | None = None, prod: bool = False,
                dry_run: bool = False) -> int:
    """Исполняет ЗАМОРОЖЕННЫЙ план. Ничего не пересчитывает.

    Только intraday_* — long_overnight откладывается до вечерней фазы (§7).
    """
    guard = _common_guards("ORDER", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.place_orders import (_make_broker_and_account, _open_log,
                                       place_limits, attach_stops)
    from tft_forecast.combined import Order

    run_id = new_run_id("ORDER", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("ORDER", run_id, run_dir)
    log.info("[ORDER] %s, торговый день %s", run_id, day)

    conn = database.get_connection()
    writer, fp = _open_log()
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod)

        plan, plan_dir = _find_plan(day)
        current_ds = dataset_fingerprint(conn)
        for err in validate_plan(plan, day, current_ds):
            res.check(False, err)
        if res.errors:
            _snapshot_account(broker, account_id, run_dir, run_id, "ORDER", "after")
            return _finish(res, state)

        res.data["plan_run_id"] = plan.get("run_id")
        res.data["dataset_hash"] = current_ds["dataset_hash"]
        res.data["config_hash"] = plan.get("config_hash")
        _snapshot_account(broker, account_id, run_dir, run_id, "ORDER", "before")

        # только внутридневные: овернайт ставится вечером
        used: set[str] = set()
        accepted, rejected = [], []
        for po in plan.get("orders", []):
            if not str(po.get("strategy_type", "")).startswith("intraday"):
                rejected.append({**po, "skip_reason": "long_overnight — вечерняя фаза"})
                continue
            reason = check_order_allowed(po, env=env, used_ids=used)
            if reason:
                rejected.append({**po, "skip_reason": reason})
                log.info("[ORDER] %s пропущен: %s", po["ticker"], reason)
                continue
            used.add(po["signal_id"])
            accepted.append(po)

        orders = [_order_from_plan(po, Order) for po in accepted]
        placed = []
        if orders:
            placed = place_limits(broker, account_id, orders, dry_run=dry_run,
                                  immediate_stop=False, force=False,
                                  writer=writer, env=env) or []
            _record_intents(conn, placed, accepted, run_id=run_id, phase="ORDER",
                            env=env, day=day)

        _write_json(os.path.join(run_dir, "orders.json"),
                    {"accepted": accepted, "rejected": rejected, "placed": placed})

        # 6. подождать заливку и привязать SL/TP
        wait_s = float(getattr(config, "ORDER_FILL_WAIT_SEC", 60))
        if placed and not dry_run and wait_s > 0:
            log.info("[ORDER] жду заливки %.0f с, затем привязываю SL/TP", wait_s)
            import time
            time.sleep(wait_s)
            attach_stops(broker, account_id, dry_run=False, writer=writer, env=env)

        fills = _collect_fills(broker, account_id, placed)
        _write_json(os.path.join(run_dir, "fills.json"), fills)
        _snapshot_account(broker, account_id, run_dir, run_id, "ORDER", "after")

        res.data.update(orders_count=len(accepted), placed_count=len(placed),
                        rejected_count=len(rejected), fills_count=len(fills))
        res.check(True, "")
        res.check(len(placed) == len(accepted),
                  f"выставлено {len(placed)} из {len(accepted)} принятых", critical=False)
    except Exception as e:                       # noqa: BLE001
        log.exception("[ORDER] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
        conn.close()
    return _finish(res, state)


def _order_from_plan(po: dict, Order):
    """Запись плана → Order. Цены НЕ пересчитываются: берутся из плана."""
    return Order(
        ticker=po["ticker"], strategy=po["strategy_type"], direction=po["direction"],
        anchor_price=po.get("anchor_price"), f_low=None, f_high=None,
        down_pct=None, entry_price=po.get("entry_price"), better_pct=None,
        stop_price=po.get("stop_price"), stop_pct=None,
        tp_price=po.get("tp_price"), tp_pct=None,
        lot_size=int(po.get("lot_size") or 1), lot_known=True,
        quantity_lots=po.get("quantity_lots"), total_rub=po.get("total_rub"),
        unavailable=False,
    )


def _record_intents(conn, placed: list[dict], plan_orders: list[dict], *,
                    run_id: str, phase: str, env: str, day: dt.date) -> None:
    """Журналирует намерения в execution_audit (§5.5, §7.6)."""
    from audit import execution_audit as ea
    by_ticker = {p["ticker"]: p for p in plan_orders}
    for rec in placed:
        po = by_ticker.get(rec.get("ticker")) or {}
        strategy = rec.get("strategy") or po.get("strategy_type", "?")
        oid = rec.get("order_id")
        if not oid:
            continue
        ea.record_intent(
            conn, order_id=oid, account_env=env, asof_date=day,
            ticker=rec.get("ticker", "?"), strategy=strategy,
            side="SELL" if po.get("direction") == "SHORT" else "BUY",
            requested_price=float(po.get("entry_price") or 0.0),
            run_id=run_id, phase=phase,
            final_score=po.get("final_score"), exp_pnl_pct=po.get("exp_pnl_pct"),
            anchor_price=po.get("anchor_price"),
            expected_cost_pct=po.get("expected_cost_pct"),
            stop_price=po.get("stop_price"), target_price=po.get("tp_price"),
            qty_lots=po.get("quantity_lots"), lot_size=po.get("lot_size"),
            raw={"signal_id": po.get("signal_id"), "verdict": po.get("verdict")},
        )


def _collect_fills(broker, account_id: str, placed: list[dict]) -> list[dict]:
    """Состояние выставленных заявок после ожидания."""
    out = []
    for rec in placed:
        oid = rec.get("order_id")
        if not oid:
            continue
        try:
            st = broker.get_order_state(account_id=account_id, order_id=oid)
            out.append({"order_id": oid, "ticker": rec.get("ticker"),
                        "status": st.execution_report_status,
                        "lots_requested": st.lots_requested,
                        "lots_executed": st.lots_executed})
        except Exception as e:                   # noqa: BLE001
            out.append({"order_id": oid, "ticker": rec.get("ticker"), "error": str(e)})
    return out


# ── Фаза CLEANUP (§6) ────────────────────────────────────────────────────────

_INTRADAY = ("intraday_long", "intraday_short")


def _strategy_by_uid(account_id: str) -> dict[str, dict]:
    """instrument_uid → запись реестра ожидающих стопов (там же лежит strategy).

    Это тот же источник, по которому square_off_intraday отличает дневную
    позицию от ночной, — второго реестра стратегий в проекте нет.
    """
    from services.place_orders import _load_pending
    return {r.get("instrument_uid"): r
            for r in (_load_pending().get(account_id, []) or [])
            if r.get("instrument_uid")}


def _open_intraday(broker, account_id: str) -> list[str]:
    """Тикеры открытых ВНУТРИДНЕВНЫХ позиций по реестру стратегий."""
    reg = _strategy_by_uid(account_id)
    out = []
    for p in broker.get_positions(account_id):
        if not p.is_open:
            continue
        rec = reg.get(p.instrument_uid) or {}
        if rec.get("strategy") in _INTRADAY:
            out.append(rec.get("ticker", p.instrument_uid[:8]))
    return out


def phase_cleanup(*, now: dt.datetime | None = None, prod: bool = False,
                  dry_run: bool = False) -> int:
    """Закрывает внутридневные позиции и снимает незалившиеся дневные лимитки.

    long_overnight не трогается — он держится через ночь по замыслу.
    """
    guard = _common_guards("CLEANUP", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.place_orders import (_make_broker_and_account, _open_log,
                                       square_off_intraday)

    run_id = new_run_id("CLEANUP", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("CLEANUP", run_id, run_dir)
    log.info("[CLEANUP] %s, торговый день %s", run_id, day)

    writer, fp = _open_log()
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod)
        _snapshot_account(broker, account_id, run_dir, run_id, "CLEANUP", "before")

        before = _open_intraday(broker, account_id)
        closed = square_off_intraday(broker, account_id, dry_run=dry_run,
                                     no_confirm=True, writer=writer, env=env,
                                     force=True, now=now)
        # 3. подтверждение: повторный GetPositions обязан вернуть ноль интрадея
        after = [] if dry_run else _open_intraday(broker, account_id)
        res.check(not after,
                  f"после cleanup остались внутридневные позиции: {', '.join(after)}")

        # 4. снять дневные лимитки, не залившиеся за день
        cancelled = _cancel_stale_intraday_orders(broker, account_id, dry_run=dry_run)

        _snapshot_account(broker, account_id, run_dir, run_id, "CLEANUP", "after")
        res.data.update(intraday_before=len(before), positions_closed=closed,
                        orders_cancelled=cancelled)
        log.info("[CLEANUP] закрыто позиций %d, снято лимиток %d", closed, cancelled)
    except Exception as e:                       # noqa: BLE001
        log.exception("[CLEANUP] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
    return _finish(res, state)


def _cancel_stale_intraday_orders(broker, account_id: str, *, dry_run: bool) -> int:
    """Снимает активные лимитки по внутридневным стратегиям."""
    reg = _strategy_by_uid(account_id)
    n = 0
    try:
        active = broker.get_active_orders(account_id)
    except Exception:                            # noqa: BLE001
        return 0
    for o in active:
        uid = getattr(o, "instrument_uid", None)
        if reg.get(uid, {}).get("strategy") not in _INTRADAY:
            continue
        if dry_run:
            n += 1
            continue
        try:
            broker.cancel_order(account_id=account_id, order_id=o.order_id)
            n += 1
        except Exception as e:                   # noqa: BLE001
            log.warning("[CLEANUP] не удалось снять заявку %s: %s", o.order_id, e)
    return n


# ── Фаза OVERNIGHT (§7) ──────────────────────────────────────────────────────

def phase_overnight(*, now: dt.datetime | None = None, prod: bool = False,
                    dry_run: bool = False) -> int:
    """Пересчитывает ТОЛЬКО long_overnight и ставит ночные заявки.

    Пересчёт здесь правомерен и обязателен: цель стратегии — next_open/today_close,
    то есть вход на закрытии. Якорем должна быть цена закрытия, а не утренняя.
    """
    guard = _common_guards("OVERNIGHT", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.place_orders import (_make_broker_and_account, _open_log,
                                       place_limits, compute_orders)

    run_id = new_run_id("OVERNIGHT", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("OVERNIGHT", run_id, run_dir)
    log.info("[OVERNIGHT] %s, торговый день %s", run_id, day)

    # 1. CLEANUP этого дня обязан был завершиться успешно
    dirs = _run_dirs_for_day(day)
    cl = _read_json(os.path.join(dirs.get("CLEANUP", ""), "run_meta.json")) \
        if dirs.get("CLEANUP") else None
    if not res.check(bool(cl) and cl.get("verdict") != FAIL,
                     "CLEANUP этого дня не завершился успешно — овернайт отменён"):
        return _finish(res, state)

    conn = database.get_connection()
    writer, fp = _open_log()
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod)
        _snapshot_account(broker, account_id, run_dir, run_id, "OVERNIGHT", "before")

        # 2. пересчёт только ночной стратегии
        orders, meta = compute_orders(
            int(getattr(config, "BEST_TRADES_TOP_N", 5)),
            float(getattr(config, "BEST_TRADES_POSITION_RUB", 10000)),
            float(getattr(config, "LIMIT_ENTRY_FRACTION", 0.2)),
            quiet=True, refresh=False)
        night = [o for o in orders if o.strategy == "long_overnight" and o.is_placeable]

        used: set[str] = set()
        accepted, rejected = [], []
        for o in night:
            po = _plan_order(o, {}, day)
            reason = check_order_allowed(po, env=env, used_ids=used)
            if reason:
                rejected.append({**po, "skip_reason": reason})
                continue
            used.add(po["signal_id"])
            accepted.append(po)

        placed = []
        if accepted:
            keep = {a["ticker"] for a in accepted}
            placed = place_limits(broker, account_id,
                                  [o for o in night if o.ticker in keep],
                                  dry_run=dry_run, immediate_stop=False, force=False,
                                  writer=writer, env=env) or []
            _record_intents(conn, placed, accepted, run_id=run_id, phase="OVERNIGHT",
                            env=env, day=day)

        # 5. отдельный файл — не смешивать с интрадеем
        _write_json(os.path.join(run_dir, "overnight_orders.json"),
                    {"accepted": accepted, "rejected": rejected, "placed": placed})
        _snapshot_account(broker, account_id, run_dir, run_id, "OVERNIGHT", "after")
        res.data.update(overnight_orders=len(accepted), placed_count=len(placed))
        log.info("[OVERNIGHT] ночных заявок принято %d, выставлено %d",
                 len(accepted), len(placed))

        # 8. дневной аудит, баланс, счётчик
        state = _close_trading_day(day, state, res)
    except Exception as e:                       # noqa: BLE001
        log.exception("[OVERNIGHT] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
        conn.close()
    return _finish(res, state)


# ── Дневной аудит и счётчик (§11, §12) ───────────────────────────────────────

def _execution_stats(day: dt.date) -> dict:
    """Статистика исполнения за день из execution_audit."""
    sql = """
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE filled),
               AVG(slippage_pct) FILTER (WHERE filled),
               AVG(expected_slippage_pct)
        FROM execution_audit WHERE asof_date = %s;
    """
    try:
        with database.get_db_connection() as c:
            with c.cursor() as cur:
                cur.execute(sql, (day,))
                n, filled, fact, exp = cur.fetchone()
        return {"orders": int(n or 0), "filled": int(filled or 0),
                "slippage_fact_pct": float(fact) if fact is not None else None,
                "slippage_expected_pct": float(exp) if exp is not None else None}
    except Exception as e:                       # noqa: BLE001
        log.warning("execution_audit недоступен: %s", e)
        return {"orders": 0, "filled": 0}


def build_daily_audit(day: dt.date, res: PhaseResult) -> dict:
    """Сводит день по каталогам всех четырёх фаз."""
    dirs = _run_dirs_for_day(day)
    meta = {ph: _read_json(os.path.join(d, "run_meta.json"), {}) for ph, d in dirs.items()}
    orders = _read_json(os.path.join(dirs.get("ORDER", ""), "orders.json"), {}) \
        if dirs.get("ORDER") else {}
    fills = _read_json(os.path.join(dirs.get("ORDER", ""), "fills.json"), []) \
        if dirs.get("ORDER") else []
    night = _read_json(os.path.join(dirs.get("OVERNIGHT", ""), "overnight_orders.json"), {}) \
        if dirs.get("OVERNIGHT") else {}
    signals = _read_json(os.path.join(dirs.get("PREP", ""), "signals.json"), []) \
        if dirs.get("PREP") else []

    errors = [e for m in meta.values() for e in (m.get("errors") or [])]
    warnings = [w for m in meta.values() for w in (m.get("warnings") or [])]
    missing = [p for p in PHASES if p not in dirs]
    if missing:
        warnings.append(f"фазы не отработали: {', '.join(missing)}")

    prep = meta.get("PREP", {})
    stats = _execution_stats(day)
    audit = {
        "trading_day": day.isoformat(),
        **{f"{p.lower()}_run_id": meta.get(p, {}).get("run_id") for p in PHASES},
        "signals_count": len(signals),
        "orders_count": len(orders.get("accepted") or []),
        "fills_count": sum(1 for f in fills if (f.get("lots_executed") or 0) > 0),
        "overnight_orders": len(night.get("accepted") or []),
        "execution_audit_count": stats.get("orders", 0),
        "dataset_hash": prep.get("dataset_hash"),
        "config_hash": prep.get("config_hash"),
        "checks": {
            "critical_passed": sum(m.get("critical_passed", 0) or 0 for m in meta.values())
            or res.critical_passed,
            "critical_failed": len(errors),
            "warnings": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
        "verdict": FAIL if errors else (PASS_WARN if warnings else PASS),
    }
    _write_json(_p("daily_audit", f"{day.isoformat()}.json"), audit)
    return audit


def _close_trading_day(day: dt.date, state: dict, res: PhaseResult) -> dict:
    """Дневной аудит, сводка баланса, отчёт и счётчик торговых дней."""
    dirs = _run_dirs_for_day(day)
    audit = build_daily_audit(day, res)

    prev = None
    if state.get("days"):
        prev_sum = _read_json(_p("balance", f"{state['days'][-1]}.json"), {})
        prev = prev_sum.get("closing_balance_rub")
    summary = stage2_balance.collect_day(day, dirs, prev_closing=prev)
    stage2_balance.save_day(summary)
    stage2_balance.save_report(execution=_execution_stats(day))

    # день засчитывается только при завершившейся фазе OVERNIGHT без FAIL
    if audit["verdict"] != FAIL and day.isoformat() not in state.get("days", []):
        state.setdefault("days", []).append(day.isoformat())
        state["completed_trading_days"] = len(state["days"])
        state["started_at"] = state.get("started_at") or day.isoformat()
        if state["completed_trading_days"] >= int(state.get("target_trading_days", 15)):
            state["status"] = "finished"
            log.info("ТЕСТ ЗАВЕРШЁН: %d торговых дней пройдено",
                     state["completed_trading_days"])
        save_state(state)
        log.info("[OVERNIGHT] день засчитан: %d из %d",
                 state["completed_trading_days"], state["target_trading_days"])
    res.data["daily_verdict"] = audit["verdict"]

    # Сводка для уведомления: нарастающий итог считается от баланса открытия
    # ПЕРВОГО дня теста, а не от STAGE2_START_BALANCE_RUB — стартовая сумма в
    # конфиге декларативна и может разойтись с фактическим состоянием счёта.
    first = _read_json(_p("balance", f"{state['days'][0]}.json"), {}) \
        if state.get("days") else summary
    base = (first or {}).get("opening_balance_rub")
    closing = summary.get("closing_balance_rub")
    res.data["day_summary"] = {
        "completed": state.get("completed_trading_days", 0),
        "target": state.get("target_trading_days", 15),
        "closing": closing,
        "change": summary.get("day_change_rub"),
        "change_pct": summary.get("day_change_pct"),
        "cum_pct": ((closing / base - 1.0) * 100.0
                    if (closing is not None and base) else None),
        "positions": summary.get("positions_overnight"),
    }
    return state


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_status() -> int:
    st = load_state()
    print(f"\n  Тест: {st['test_id']}   статус: {st['status']}")
    print(f"  Торговых дней: {st['completed_trading_days']} из {st['target_trading_days']}")
    if st.get("halted_reason"):
        print(f"  ОСТАНОВЛЕН: {st['halted_reason']}")
        print("  Снять: python3 -m services.stage2_demo resume")
    if st.get("days"):
        print(f"  Дни: {', '.join(st['days'])}")
    print(f"  Расписание: PREP {config.STAGE2_PREP_TIME}  ORDER {config.STAGE2_ORDER_TIME}"
          f"  CLEANUP {config.STAGE2_CLEANUP_TIME}  OVERNIGHT {config.STAGE2_OVERNIGHT_TIME}")
    print(f"  Каталог: {base_dir()}\n")
    return 0


def cmd_resume() -> int:
    st = load_state()
    if st.get("status") != "halted":
        print(f"Тест не остановлен (статус {st.get('status')}) — снимать нечего.")
        return 0
    print(f"Снята блокировка: {st.get('halted_reason')}")
    st["status"] = "running"
    st["halted_reason"] = None
    save_state(st)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    p = argparse.ArgumentParser(
        description="Этап 2: функциональный тест на демо-счёте (STAGE2-DEMO-TZ.md)")
    p.add_argument("phase", choices=["prep", "order", "cleanup", "overnight",
                                     "status", "resume"])
    p.add_argument("--dry-run", action="store_true",
                   help="пройти фазу без отправки заявок брокеру")
    p.add_argument("--prod", action="store_true",
                   help="БОЕВОЙ контур. Для Этапа 2 запрещён — фаза откажется работать.")
    args = p.parse_args(argv)

    if args.phase == "status":
        return cmd_status()
    if args.phase == "resume":
        return cmd_resume()

    fn = {"prep": phase_prep, "order": phase_order,
          "cleanup": phase_cleanup, "overnight": phase_overnight}[args.phase]
    kw = {"prod": args.prod}
    if args.phase != "prep":
        kw["dry_run"] = args.dry_run
    return fn(**kw)


if __name__ == "__main__":
    sys.exit(main())
