"""
Оркестратор Этапа 2 — функциональный тест на демо-счёте (STAGE2-DEMO-TZ.md).

Четыре фазы торгового дня, каждая запускается отдельным заданием cron:

    PREP      08:45  считает сигналы и ЗАМОРАЖИВАЕТ план на диске; заявок не ставит
    CLOSE     09:10  закрывает ночные позиции (хотфикс 11.09)
    ORDER     09:15  исполняет замороженный план; ничего не пересчитывает
    PARK      10:05  казначейство после открытия СПБ: паи TMON@ лимиткой (15.09)
    CLEANUP   18:20  закрывает внутридневные позиции, снимает незалившиеся лимитки
    OVERNIGHT 18:35  пересчитывает только long_overnight и ставит ночные заявки

Утро привязано к открытию Мосбиржи: с 14.09.2026 аукцион 09:00, торги с 09:10
(до этого PREP 09:45, CLOSE 10:00, ORDER 10:05 под открытие в 10:00).

Казначейство (services/treasury.py): CLOSE паркует свободный кэш сверх буфера
в фонд денежного рынка (до 10:00 листинг TMON@ закрыт для API — виртуально),
PARK в 10:05 покупает паи реально лимитной заявкой и переводит виртуальные в
реальные, CLEANUP гасит минус по рублям продажей паёв, OVERNIGHT продаёт паи
ровно под ночную корзину до заявок по акциям.

Почему план замораживается (§3): между PREP и ORDER проходит полчаса, за
которые меняются якорная цена, реализованная часть дневного хода и результат
переобучения TFT. Если ORDER пересчитает сигналы, требование «заявка
соответствует сигналу» станет непроверяемым. Поэтому ORDER читает plan.json и
отказывается работать, если конфигурация или датасет изменились.

Критерий теста функциональный (§1): проверяется работоспособность контура,
сохранность данных и воспроизводимость, а не прибыльность.

Запуск:
    python3 -m services.stage2_demo prep|close|order|park|cleanup|overnight
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
PHASES = ("PREP", "CLOSE", "ORDER", "CLEANUP", "OVERNIGHT")
# Вспомогательные фазы: их каталоги учитываются (_run_dirs_for_day), но их
# отсутствие торговый день не портит — это казначейство, а не торговля.
AUX_PHASES = ("PARK",)

PASS, PASS_WARN, FAIL = "PASS", "PASS_WITH_WARNINGS", "FAIL"
# Часть действий не выполнена (брокер отбил заявки), но фаза не провалена:
# жёлтый статус, тест не останавливается. Раньше такой исход был зелёным PASS.
PARTIAL = "PARTIAL_FAILURE"


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
    "OVERNIGHT_ENTRY_MODE", "OVERNIGHT_MARKETABLE_SLIP_PCT",
    "INTRADAY_SQUARE_OFF_ENABLED", "INTRADAY_SQUARE_OFF_TIME",
    "FIXED_POSITION_OVERFLOW_MODE", "BEST_TRADES_TOP_N", "BEST_TRADES_POSITION_RUB",
    "VALIDATION_COST_RT", "TFT_COST_RT", "VALIDATION_FULL_UNIVERSE",
    "TFT_EPOCHS", "TFT_HIDDEN", "WEEK_HORIZON_DAYS", "INCLUDE_WEEKEND_TRADING",
    "STAGE2_TARGET_DAYS", "STAGE2_ALLOW_DATASET_DRIFT",
    "TREASURY_ENABLED", "TREASURY_TICKER", "TREASURY_CLASS_CODE",
    "TREASURY_CASH_BUFFER_RUB", "TREASURY_MIN_SWEEP_RUB", "TREASURY_LIMIT_MAX_PREMIUM_PCT",
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
        self.partial: list[str] = []

    def check(self, ok: bool, message: str, *, critical: bool = True) -> bool:
        if ok:
            if critical:
                self.critical_passed += 1
            return True
        (self.errors if critical else self.warnings).append(message)
        return False

    def partial_fail(self, message: str) -> None:
        """Часть действий не выполнена, но фаза не провалена (PARTIAL_FAILURE)."""
        self.partial.append(message)

    @property
    def verdict(self) -> str:
        if self.errors:
            return FAIL
        if self.partial:
            return PARTIAL
        return PASS_WARN if self.warnings else PASS

    def exit_code(self) -> int:
        return 1 if self.errors else 0


# ── Предполётные проверки (§8) ───────────────────────────────────────────────

def contour_ok(env: str) -> str | None:
    """Согласован ли контур целиком. None — торговать можно, иначе причина отказа.

    Полумеры недопустимы: боевой счёт при TRADING_MODE≠prod или песочница при
    TRADING_MODE=prod — это перепутанный профиль .env, а не конфигурация.
    """
    mode = str(getattr(config, "TRADING_MODE", "")).lower()
    prod_account = str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip()
    if env == "SANDBOX":
        return None if mode != "prod" else "TRADING_MODE=prod в песочнице"
    if env == "PROD":
        if mode != "prod":
            return f"боевой счёт при TRADING_MODE={mode or None}"
        if not prod_account:
            return "боевой счёт без PROD_ACCOUNT_ID"
        if os.getenv("ALLOW_UNATTENDED_PROD", "") != "1":
            return "боевой счёт без ALLOW_UNATTENDED_PROD=1"
        return None
    return f"неизвестный контур {env}"


def preflight(res: PhaseResult, *, env: str, prod_flag: bool,
              account_id: str | None = None) -> None:
    """Критические проверки контура. Провал любой → FAIL, торговля не идёт.

    Два режима и ничего между ними:
      * песочница (без --prod) — контур SANDBOX, TRADING_MODE не prod,
        PROD_ACCOUNT_ID пуст. Изоляция песочницы не ослаблена: профиль с
        боевыми параметрами без явного --prod по-прежнему проваливает фазу;
      * боевой (--prod) — совпасть обязаны ВСЕ признаки: контур PROD,
        TRADING_MODE=prod, PROD_ACCOUNT_ID задан И равен счёту, который реально
        открыл брокер, ALLOW_UNATTENDED_PROD=1 (осознанный автозапуск),
        отдельный STAGE2_DIR, позиция не выше PROD_MAX_POSITION_RUB.

    Проверка совпадения счёта — главная: без неё опечатка в PROD_ACCOUNT_ID
    направила бы заявки на другой реальный счёт пользователя.
    """
    mode = str(getattr(config, "TRADING_MODE", "")).lower()
    prod_account = str(getattr(config, "PROD_ACCOUNT_ID", "") or "").strip()
    if not prod_flag:
        res.check(env == "SANDBOX", f"контур не SANDBOX: {env}")
        res.check(mode != "prod", f"TRADING_MODE={mode or None} без флага --prod")
        res.check(not prod_flag, "передан флаг --prod")
        res.check(not prod_account,
                  "PROD_ACCOUNT_ID заполнен — боевой счёт должен быть закрыт")
        return

    res.check(env == "PROD", f"--prod передан, но контур {env}")
    res.check(mode == "prod", f"--prod передан, но TRADING_MODE={mode or None}")
    res.check(bool(prod_account), "PROD_ACCOUNT_ID не задан")
    res.check(bool(account_id) and account_id == prod_account,
              f"счёт брокера {account_id} ≠ PROD_ACCOUNT_ID {prod_account}")
    res.check(os.getenv("ALLOW_UNATTENDED_PROD", "") == "1",
              "ALLOW_UNATTENDED_PROD ≠ 1 — автозапуск по крону не разрешён")
    res.check(os.path.basename(os.path.normpath(base_dir())) != "stage2-demo",
              "STAGE2_DIR не отделён от песочницы — счётчики смешаются")
    cap = float(getattr(config, "PROD_MAX_POSITION_RUB", 0) or 0)
    pos = float(getattr(config, "BEST_TRADES_POSITION_RUB", 0) or 0)
    res.check(not cap or pos <= cap,
              f"позиция {pos:.0f} ₽ выше предела PROD_MAX_POSITION_RUB {cap:.0f} ₽")


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
        if phase in PHASES or phase in AUX_PHASES:
            out[phase] = os.path.join(runs, name)      # последний по времени
    return out


def _superseded_run_dirs(day: dt.date, latest: dict[str, str]) -> list[tuple[str, str]]:
    """Прогоны фаз этого дня, после которых фаза запускалась ещё раз."""
    runs = _p("runs")
    if not os.path.isdir(runs):
        return []
    prefix = day.strftime("%Y%m%d")
    out = []
    for name in sorted(os.listdir(runs)):
        if not name.startswith(prefix) or "-" not in name:
            continue
        phase = name.rsplit("-", 1)[-1]
        path = os.path.join(runs, name)
        if phase in latest and latest[phase] != path:
            out.append((phase, path))
    return out


# ── Уведомления в Telegram (§ отчётность) ────────────────────────────────────

_VERDICT_ICON = {PASS: "✅", PASS_WARN: "⚠️", PARTIAL: "\U0001f7e1", FAIL: "⛔"}

# Что показывать по каждой фазе: ключ в res.data → подпись.
_PHASE_FIELDS = {
    "PREP": (("signals_count", "сигналов"), ("plan_orders", "заявок в плане")),
    "ORDER": (("orders_count", "принято"), ("placed_count", "выставлено"),
              ("rejected_count", "отклонено"), ("fills_count", "заливок")),
    "CLEANUP": (("intraday_before", "было интрадея"), ("positions_closed", "закрыто"),
                ("orders_cancelled", "снято лимиток"), ("stops_cancelled", "снято стопов")),
    "OVERNIGHT": (("overnight_orders", "ночных принято"), ("placed_count", "выставлено"),
                  ("treasury_released_lots", "продано паёв фонда")),
    "CLOSE": (("overnight_before", "ночных позиций"), ("positions_closed", "закрыто"),
              ("orders_cancelled", "снято ночных лимиток"),
              ("registry_reconciled", "сверено записей реестра"),
              ("treasury_parked_rub", "припарковано в фонд, ₽")),
    "PARK": (("treasury_parked_lots", "куплено паёв"),
             ("treasury_parked_rub", "припарковано в фонд, ₽"),
             ("treasury_converted_lots", "переведено из виртуальных, паёв"),
             ("treasury_limit_price", "лимит цены, ₽")),
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
        label = str(getattr(config, "STAGE2_NOTIFY_LABEL", "") or "").strip()
        tag = f"\U0001f3c6 <b>{notify.esc(label)}</b>\n" if label else ""
        lines = [f"{tag}{icon} <b>{notify.esc(res.phase)}</b> \u2014 {notify.esc(res.verdict)}",
                 f"<code>{notify.esc(res.run_id)}</code>"]

        facts = [f"{cap}: <b>{notify.esc(res.data[key])}</b>"
                 for key, cap in _PHASE_FIELDS.get(res.phase, ())
                 if res.data.get(key) is not None]
        if facts:
            lines.append("")
            lines += facts
        if res.data.get("treasury"):
            from services.treasury import card_line
            lines.append(notify.esc(card_line(res.data["treasury"])))

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
            if day.get("trades_target"):
                pace = day.get("trades_total") or 0
                left_days = max(0, int(day.get("target") or 0) - int(day.get("completed") or 0))
                need = day["trades_target"] - pace
                behind = " ⚠️ отстаём" if left_days and need > left_days else ""
                lines.append(f"сделок в зачёте: <b>{notify.esc(pace)} / {notify.esc(day['trades_target'])}</b>"
                             + notify.esc(behind))

        for e in res.errors[:5]:
            lines.append(f"\u26d4 {notify.esc(e)}")
        for x in res.partial[:5]:
            lines.append(f"\U0001f7e1 {notify.esc(x)}")
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
        "partial": res.partial,
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
    for x in res.partial:
        log.warning("[%s] ЧАСТИЧНЫЙ ПРОВАЛ: %s", res.phase, x)
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


def _start_date() -> dt.date | None:
    """STAGE2_START_DATE — день, с которого считается тест. Пусто — без ограничения.
    Опечатка в дате роняет фазу, а не пропускает проверку: лучше не торговать,
    чем торговать вне теста."""
    raw = str(getattr(config, "STAGE2_START_DATE", "") or "").strip()
    return dt.date.fromisoformat(raw) if raw else None


def _common_guards(phase: str, *, now: dt.datetime | None = None):
    """Проверки, общие для всех фаз. Возвращает (state, day) или код выхода."""
    state = load_state()
    if not getattr(config, "STAGE2_ENABLED", True):
        return _skip(phase, "STAGE2_ENABLED=0")
    start = _start_date()
    if start and _today(now) < start:
        # До старта не создаём даже каталог пропуска: аудировать нечего, а
        # метки «до начала» засоряли бы новый прогон.
        log.info("[%s] тест начинается %s — фаза не выполняется", phase, start)
        return 0
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
        preflight(res, env=env, prod_flag=prod, account_id=account_id)
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
    bal = stage2_balance.capture(broker, account_id, run_id=run_id, phase=phase,
                                 exclude_uids=_treasury_uids(broker))
    _write_json(os.path.join(run_dir, "balance.json"), bal)
    return bal


# ── Казначейство (ТЗ Treasury, services/treasury.py) ─────────────────────────

def _treasury_enabled() -> bool:
    return bool(getattr(config, "TREASURY_ENABLED", True))


def _treasury(broker, account_id: str, env: str, conn, writer, run_id: str):
    """Сервис казначейства с виртуальным реестром на соединении фазы."""
    from services.treasury import TreasuryService, VirtualLedger
    ledger = None
    if conn is not None:
        ledger = VirtualLedger(conn)
        try:
            ledger.init()
        except Exception as e:                   # noqa: BLE001
            try:
                conn.rollback()
            except Exception:                    # noqa: BLE001
                pass
            log.warning("treasury_ledger: схема недоступна: %s", e)
            ledger = None
    return TreasuryService(broker, account_id, env=env, ledger=ledger,
                           writer=writer, run_id=run_id)


def _treasury_brief(st: dict) -> dict:
    """Срез состояния казначейства для run_meta, дневного аудита и карточки."""
    return {k: st.get(k) for k in ("ticker", "tmon_lots", "tmon_price", "tmon_value_rub",
                                   "free_cash_rub", "cash_rub", "virtual_lots", "mode")}


def _treasury_uids(broker) -> set[str]:
    if not _treasury_enabled():
        return set()
    from services.treasury import treasury_uids
    return treasury_uids(broker)


def _treasury_lots_at(day: dt.date, phase: str) -> int | None:
    """Паёв фонда по итогу фазы этого дня (из её run_meta); None — нет данных."""
    d = _run_dirs_for_day(day).get(phase)
    meta = _read_json(os.path.join(d, "run_meta.json"), {}) if d else {}
    return ((meta or {}).get("treasury") or {}).get("tmon_lots")


def _money_rub(broker, account_id: str) -> float | None:
    """Свободные рубли на счёте; None — контур их не отдаёт."""
    try:
        return float(broker.get_money_rub(account_id))
    except Exception as e:                       # noqa: BLE001
        log.warning("свободные рубли недоступны: %s", e)
        return None


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

    bad = contour_ok(env)
    if bad:
        return bad
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
        preflight(res, env=env, prod_flag=prod, account_id=account_id)

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
        placed, rep = [], []
        if orders:
            placed = place_limits(broker, account_id, orders, dry_run=dry_run,
                                  immediate_stop=False, force=False,
                                  writer=writer, env=env, report=rep) or []
            ensure_execution_audit(conn)
            miss = _record_intents(conn, placed, accepted, run_id=run_id,
                                   phase="ORDER", env=env, day=day)
            res.check(miss == 0,
                      f"журнал исполнения: не записано намерений {miss} из "
                      f"{len(placed)}", critical=False)

        _write_json(os.path.join(run_dir, "orders.json"),
                    {"accepted": accepted, "rejected": rejected, "placed": placed})

        # 6. подождать заливку и привязать SL/TP
        wait_s = float(getattr(config, "ORDER_FILL_WAIT_SEC", 60))
        if placed and not dry_run and wait_s > 0:
            log.info("[ORDER] жду заливки %.0f с, затем привязываю SL/TP", wait_s)
            import time
            time.sleep(wait_s)
            attach_stops(broker, account_id, dry_run=False, writer=writer, env=env,
                         conn=conn)

        fills = _collect_fills(broker, account_id, placed)
        _write_json(os.path.join(run_dir, "fills.json"), fills)
        if not dry_run and fills:
            done, miss = _record_fills(conn, fills, accepted)
            res.check(miss == 0,
                      f"журнал исполнения: не записано заливок {miss}",
                      critical=False)
            log.info("[ORDER] в журнал исполнения записано заливок %d", done)
        _snapshot_account(broker, account_id, run_dir, run_id, "ORDER", "after")

        res.data.update(orders_count=len(accepted), placed_count=len(placed),
                        rejected_count=len(rejected), fills_count=len(fills))
        res.check(True, "")
        _check_placement(res, rep, placed, accepted)
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


def ensure_execution_audit(conn) -> bool:
    """Создаёт execution_audit, если её нет. Идемпотентно.

    Таблица есть в DDL database.init_db, но init_db вызывается только из
    load_instruments и в кроновом пути Этапа 2 не участвует — сама она бы не
    появилась никогда. Поднимаем её там, где в неё пишут.
    """
    from audit import execution_audit as ea
    try:
        ea.init(conn)
        return True
    except Exception as e:                       # noqa: BLE001
        conn.rollback()
        log.warning("execution_audit: схема недоступна: %s", e)
        return False


def _record_intents(conn, placed: list[dict], plan_orders: list[dict], *,
                    run_id: str, phase: str, env: str, day: dt.date) -> int:
    """Журналирует намерения в execution_audit (§5.5, §7.6).

    Возвращает число НЕзаписанных строк — вызывающий обязан отразить это в
    вердикте фазы: журнал исполнения, который молча не пишется, оставляет
    критерий ТЗ про проскальзывание непроверяемым, а по артефактам это
    неотличимо от дня без сделок.
    """
    from audit import execution_audit as ea
    failed = 0
    by_ticker = {p["ticker"]: p for p in plan_orders}
    for rec in placed:
        po = by_ticker.get(rec.get("ticker")) or {}
        strategy = rec.get("strategy") or po.get("strategy_type", "?")
        oid = rec.get("order_id")
        if not oid:
            continue
        ok = ea.record_intent(
            conn, order_id=oid, account_env=env, asof_date=day,
            ticker=rec.get("ticker", "?"), strategy=strategy,
            side="SELL" if po.get("direction") == "SHORT" else "BUY",
            requested_price=float(po.get("entry_price") or 0.0),
            run_id=run_id, phase=phase,
            final_score=po.get("final_score"), exp_pnl_pct=po.get("exp_pnl_pct"),
            anchor_price=po.get("anchor_price"),
            expected_cost_pct=po.get("expected_cost_pct"),
            expected_slippage_pct=float(getattr(config, "EXPECTED_SLIPPAGE_PCT", 0.024)),
            stop_price=po.get("stop_price"), target_price=po.get("tp_price"),
            qty_lots=po.get("quantity_lots"), lot_size=po.get("lot_size"),
            raw={"signal_id": po.get("signal_id"), "verdict": po.get("verdict")},
        )
        failed += 0 if ok else 1
    return failed


def _record_fills(conn, fills: list[dict], plan_orders: list[dict]) -> tuple[int, int]:
    """Замыкает намерение фактом: цена заливки, комиссия, проскальзывание.

    Возвращает (записано, не записано). Без этого вызова строка журнала вечно
    остаётся с filled=FALSE, и обе половины критерия «факт против расчёта»
    оказываются пустыми — сравнивать нечего.
    """
    from audit import execution_audit as ea
    by_ticker = {p["ticker"]: p for p in plan_orders}
    now = dt.datetime.now(MSK)
    done = failed = 0
    for f in fills:
        price = f.get("executed_price")
        lots = int(f.get("lots_executed") or 0)
        # Частичное исполнение тоже фиксируем: заливка на половину объёма —
        # это факт со своей ценой, а не «не исполнено».
        if not price or lots <= 0:
            continue
        po = by_ticker.get(f.get("ticker")) or {}
        requested = float(po.get("entry_price") or 0.0)
        lot_size = int(po.get("lot_size") or 0)
        ok = ea.record_fill(
            conn, order_id=f.get("order_id"), filled_price=float(price),
            filled_at=now, requested_price=requested,
            side="SELL" if po.get("direction") == "SHORT" else "BUY",
            fee_rub=f.get("executed_commission"),
            qty_shares=(lots * lot_size) or None)
        done += 1 if ok else 0
        failed += 0 if ok else 1
    return done, failed


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
                        "lots_executed": st.lots_executed,
                        "executed_price": st.executed_price,
                        "executed_commission": st.executed_commission,
                        # Сырой ответ — страховка: если я ошибся в том, какое
                        # поле API считать ценой заливки, пересчитать можно
                        # будет по файлу, не потеряв день.
                        "raw": st.raw})
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
    conn = None
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod, account_id=account_id)
        if res.errors:
            return _finish(res, state)           # вне SANDBOX заявок не шлём
        _snapshot_account(broker, account_id, run_dir, run_id, "CLEANUP", "before")
        try:
            conn = database.get_connection()
        except Exception as e:                   # noqa: BLE001
            log.warning("[CLEANUP] БД недоступна — журнал исполнения и виртуальный "
                        "реестр не учтены: %s", e)

        # Строгий порядок (аудит r4): снять лимитки и стопы → фактические позиции
        # брокера → рыночное закрытие → дождаться нуля у брокера. Закрывается всё,
        # кроме паёв казначейства и открытых ночных записей: на 18:20 законных
        # позиций, кроме интрадея, у контура нет.
        rep: dict = {}
        exclude = _treasury_uids(broker) if _treasury_enabled() else set()
        closed = square_off_intraday(broker, account_id, dry_run=dry_run,
                                     no_confirm=True, writer=writer, env=env,
                                     force=True, now=now, close_unregistered=True,
                                     exclude_uids=exclude, conn=conn, report=rep)
        if not dry_run:
            not_flat = rep.get("not_flat") or []
            leftover = _open_non_treasury(broker, account_id, exclude)
            res.check(not not_flat and not leftover,
                      "после cleanup остались позиции: "
                      + ", ".join(sorted(set(not_flat) | set(leftover))))
            if rep.get("orphans"):
                res.partial_fail("закрыты позиции вне реестра стратегий: "
                                 + ", ".join(rep["orphans"]))

        # Казначейство (ТЗ Treasury, §5 п.1–2): после закрытия интрадея
        # маржинального долга нет. Рубли ушли в минус — паи продаются до буфера.
        if _treasury_enabled() and not dry_run:
            try:
                tr = _treasury(broker, account_id, env, conn, writer, run_id)
                st = tr.get_treasury_state()
                # сверка с последним утренним шагом казначейства: PARK, иначе CLOSE
                ref = _treasury_lots_at(day, "PARK")
                if ref is None:
                    ref = _treasury_lots_at(day, "CLOSE")
                if ref is not None:
                    res.check(st["tmon_lots"] == ref,
                              f"паи {tr.ticker} изменились за день без участия "
                              f"казначейства: {ref} → {st['tmon_lots']}", critical=False)
                if st["free_cash_rub"] < 0:
                    tr.restore_buffer()
                    st = tr.get_treasury_state()
                res.data["treasury"] = _treasury_brief(st)
            except Exception as e:               # noqa: BLE001
                res.check(False, f"казначейство: {e}", critical=False)
        if not dry_run:
            cash = _money_rub(broker, account_id)
            if cash is not None:
                res.check(cash >= 0, f"маржинальный долг после CLEANUP: {cash:.2f} ₽")

        _snapshot_account(broker, account_id, run_dir, run_id, "CLEANUP", "after")
        _write_json(os.path.join(run_dir, "square_off.json"), rep)
        res.data.update(intraday_before=len(rep.get("closed", [])) + len(rep.get("not_flat", [])),
                        positions_closed=closed,
                        orders_cancelled=rep.get("cancelled_orders", 0),
                        stops_cancelled=rep.get("cancelled_stops", 0))
        log.info("[CLEANUP] закрыто позиций %d, снято лимиток %d, стопов %d", closed,
                 rep.get("cancelled_orders", 0), rep.get("cancelled_stops", 0))
    except Exception as e:                       # noqa: BLE001
        log.exception("[CLEANUP] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
        if conn is not None:
            conn.close()
    return _finish(res, state)


def _open_non_treasury(broker, account_id: str, exclude: set[str]) -> list[str]:
    """Открытые позиции, кроме паёв казначейства и открытых ночных записей реестра."""
    reg = _strategy_by_uid(account_id)
    out = []
    for p in broker.get_positions(account_id):
        if not p.is_open or p.instrument_uid in exclude:
            continue
        rec = reg.get(p.instrument_uid) or {}
        if rec.get("strategy") in _OVERNIGHT and not rec.get("closed"):
            continue
        out.append(rec.get("ticker") or p.instrument_uid[:8])
    return out


# ── Фаза CLOSE: утреннее закрытие овернайта (хотфикс 11.09) ─────────────────

_OVERNIGHT = ("long_overnight",)
_TERMINAL = ("EXECUTION_REPORT_STATUS_FILL", "EXECUTION_REPORT_STATUS_REJECTED",
             "EXECUTION_REPORT_STATUS_CANCELLED")


def _check_placement(res: PhaseResult, report: list, placed: list, accepted: list) -> None:
    """Отказ брокера по части заявок — PARTIAL_FAILURE, а не зелёный PASS.

    10.09 брокер отбил две ночные заявки из четырёх (HTTP 429), а фаза
    отчиталась PASS: сбой был виден только в cron.log. Пропуск по защите от
    задвоения ошибкой не считается — это намеренное решение.
    """
    failed = [tk for tk, st in report if str(st).startswith("error")]
    if failed:
        res.partial_fail(f"выставлено {len(placed)} из {len(accepted)}: отказ по "
                         f"{', '.join(failed)} — см. журнал заявок")


def _open_overnight(broker, account_id: str) -> list[str]:
    """Тикеры открытых НОЧНЫХ позиций по реестру стратегий."""
    reg = _strategy_by_uid(account_id)
    out = []
    for p in broker.get_positions(account_id):
        if not p.is_open:
            continue
        rec = reg.get(p.instrument_uid) or {}
        if rec.get("strategy") in _OVERNIGHT:
            out.append(rec.get("ticker", p.instrument_uid[:8]))
    return out


def _wait_terminal(broker, account_id: str, order_id: str, *, timeout_s: float = 20.0):
    """Опрашивает GetOrderState до конечного статуса.

    Именно GetOrderState, а не ответ на постановку: в ответе PostOrder нет
    averagePositionPrice, а executedOrderPrice там — цена за штуку, тогда как в
    GetOrderState — сумма заявки. 11.09 на этом разночтении упал учёт ручной
    продажи ENPG.
    """
    import time
    deadline = time.monotonic() + timeout_s
    st = broker.get_order_state(account_id=account_id, order_id=order_id)
    while st.execution_report_status not in _TERMINAL and time.monotonic() < deadline:
        time.sleep(1.0)
        st = broker.get_order_state(account_id=account_id, order_id=order_id)
    return st


def _parse_api_ts(raw) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00")) if raw else None
    except ValueError:
        return None


def _journal_overnight_exit(broker, account_id: str, conn, rec: dict, exit_state,
                            shares: float) -> None:
    """Замыкает строку execution_audit при утреннем закрытии: заливка входа (если
    не записана — время по операциям брокера) и выход по рыночной заявке CLOSE."""
    from services import exec_journal
    if not exit_state.executed_price:
        log.warning("[CLOSE] %s: нет цены выхода — журнал не замкнут", rec.get("ticker"))
        return
    exec_journal.journal_exit(
        broker, account_id, conn, rec, "overnight_close",
        exit_price=exit_state.executed_price, exit_fee=exit_state.executed_commission,
        exit_at=exec_journal.parse_ts(exit_state.raw.get("orderDate")))


def close_overnight_positions(broker, account_id: str, conn, *, dry_run: bool,
                              writer, env: str) -> dict:
    """Закрывает по рынку позиции long_overnight, открытые накануне вечером.

    Для каждой позиции — тот же порядок, что в square_off_intraday:
      1. снять её SL/TP — иначе оставшийся стоп при касании откроет обратную;
      2. продать по рынку и дождаться исполнения по GetOrderState;
      3. замкнуть строку журнала исполнения (вход + выход);
      4. пометить запись реестра закрытой.

    Позиции без записи в реестре и позиции других стратегий не трогаются:
    закрыть по ошибке чужую позицию хуже, чем не закрыть свою.

    Запись long_overnight без позиции — незалившаяся ночная лимитка или позиция,
    уже закрытая стопом ночью. Висящая лимитка снимается (иначе залилась бы
    днём и дала позицию без выхода), оставшиеся SL/TP снимаются (тейк продажи
    на пустой позиции при открытии открыл бы шорт), запись закрывается.
    """
    from services import exec_journal
    from services.broker.base import NotSupportedError
    from services.broker.tinkoff_base import new_order_id
    from services.place_orders import (_fired_leg, _load_pending, _logrow, _save_pending,
                                       _stop_history)

    out = {"overnight_before": 0, "closed": [], "failed": [],
           "orders_cancelled": 0, "registry_reconciled": 0}
    pending = _load_pending()
    recs = [r for r in pending.get(account_id, [])
            if not r.get("closed") and r.get("strategy") in _OVERNIGHT]
    if not recs:
        return out

    positions = {p.instrument_uid: p for p in broker.get_positions(account_id) if p.is_open}
    try:
        stop_orders = broker.get_active_stop_orders(account_id)
    except NotSupportedError:
        stop_orders = []
    try:
        active = {o.order_id for o in broker.get_active_orders(account_id)}
    except Exception:                            # noqa: BLE001
        active = set()
    history = {} if dry_run else _stop_history(broker, account_id, recs)

    def _cancel_stops(tk: str, uid: str) -> bool:
        ok = True
        for s in [s for s in stop_orders if s.instrument_uid == uid]:
            try:
                broker.cancel_stop_order(account_id=account_id, stop_order_id=s.stop_order_id)
                _logrow(writer, env=env, account_id=account_id, ticker=tk,
                        action="overnight_close_cancel_stop", order_id=s.stop_order_id,
                        status="cancelled", info=s.kind)
            except Exception as e:               # noqa: BLE001
                out["failed"].append(f"{tk}: не снят {s.kind} ({e})")
                ok = False
        return ok

    for r in recs:
        tk, uid = r.get("ticker", "?"), r.get("instrument_uid")
        pos = positions.get(uid)

        if pos is None:
            if dry_run:
                continue
            if not _cancel_stops(tk, uid):
                continue
            if r.get("order_id") in active:
                try:
                    broker.cancel_order(account_id=account_id, order_id=r["order_id"])
                    out["orders_cancelled"] += 1
                    _logrow(writer, env=env, account_id=account_id, ticker=tk,
                            action="overnight_cancel_stale", order_id=r["order_id"],
                            status="cancelled")
                except Exception as e:           # noqa: BLE001
                    out["failed"].append(f"{tk}: не снята ночная лимитка ({e})")
                    continue
            # Позиции нет: лимитка не залилась или позицию ночью закрыл стоп/тейк
            # (17.09 SNGS: стоп в 07:06). Во втором случае вход и выход — факт,
            # и он обязан попасть в журнал исполнения.
            fired = _fired_leg(r, history)
            reason = ("stop" if fired[0] == "STOP_LOSS" else "target") if fired \
                else "closed_externally"
            if exec_journal.journal_fill(broker, account_id, conn, r):
                exec_journal.journal_exit(
                    broker, account_id, conn, r, reason,
                    exit_at=exec_journal.parse_ts(fired[1].activated_at) if fired else None)
                out.setdefault("journaled_exits", []).append(f"{tk}:{reason}")
            r["closed"] = True
            r["closed_reason"] = reason if fired else "overnight_no_position"
            out["registry_reconciled"] += 1
            continue

        out["overnight_before"] += 1
        lot = int(r.get("lot") or 1) or 1
        shares = abs(float(pos.balance_shares))
        lots = int(shares // lot)
        edir = "SELL" if pos.balance_shares > 0 else "BUY"
        if lots <= 0:
            out["failed"].append(f"{tk}: остаток {shares:.0f} шт меньше лота {lot}")
            continue
        if dry_run:
            log.info("[CLOSE][DRY] %s: market %s %d лот", tk, edir, lots)
            continue
        if not _cancel_stops(tk, uid):
            continue                             # без снятых стопов не закрываем

        oid = new_order_id()
        try:
            inst = broker.find_instrument_by_uid(uid)
            posted = broker.post_market_order(account_id=account_id, instrument=inst,
                                              direction=edir, quantity_lots=lots, order_id=oid)
            # Состояние — по id заявки у брокера, а не по нашему ключу
            # идемпотентности: по ключу GetOrderState отвечает 404 (песочница,
            # 14.09). Иначе первое же утреннее закрытие упало бы и остановило тест.
            oid = getattr(posted, "order_id", None) or oid
            st = _wait_terminal(broker, account_id, oid)
        except Exception as e:                   # noqa: BLE001
            out["failed"].append(f"{tk}: рыночная заявка не прошла ({e})")
            _logrow(writer, env=env, account_id=account_id, ticker=tk, direction=edir,
                    action="overnight_close", qty_lots_api=lots, status="error",
                    info=str(e))
            continue
        _logrow(writer, env=env, account_id=account_id, ticker=tk, direction=edir,
                action="overnight_close", order_id=oid, qty_lots_api=lots,
                qty_shares=shares, price=st.executed_price,
                status=st.execution_report_status)
        if not st.is_filled:
            out["failed"].append(f"{tk}: закрытие не исполнено "
                                 f"({st.execution_report_status}, "
                                 f"{st.lots_executed}/{st.lots_requested})")
            continue
        _journal_overnight_exit(broker, account_id, conn, r, st, shares)
        r["closed"], r["closed_reason"] = True, "overnight_close"
        out["closed"].append(tk)

    if not dry_run:
        _save_pending(pending)
    return out


def phase_close(*, now: dt.datetime | None = None, prod: bool = False,
                dry_run: bool = False) -> int:
    """Утром закрывает позиции long_overnight — выход стратегии.

    Стратегия определена как next_open / today_close − 1: вход на закрытии,
    выход на открытии. Вход был реализован (OVERNIGHT 18:35), выход — нет, и
    позиция ENPG 10.09 провисела бы бессрочно. Эта фаза и есть выход.

    Новых заявок на вход не выставляет. Любая незакрытая ночная позиция — FAIL:
    тест останавливается до разбора, потому что открывать новые позиции поверх
    зависшей — ровно тот дефект, который здесь чинится.
    """
    guard = _common_guards("CLOSE", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.place_orders import _make_broker_and_account, _open_log

    run_id = new_run_id("CLOSE", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("CLOSE", run_id, run_dir)
    log.info("[CLOSE] %s, торговый день %s", run_id, day)

    conn = database.get_connection()
    writer, fp = _open_log()
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod, account_id=account_id)
        if res.errors:
            return _finish(res, state)           # вне SANDBOX заявок не шлём
        _snapshot_account(broker, account_id, run_dir, run_id, "CLOSE", "before")

        out = close_overnight_positions(broker, account_id, conn, dry_run=dry_run,
                                        writer=writer, env=env)
        for f in out["failed"]:
            res.check(False, f"ночная позиция не закрыта: {f}")
        after = [] if dry_run else _open_overnight(broker, account_id)
        res.check(not after, f"после CLOSE остались ночные позиции: {', '.join(after)}")

        # SWEEP (ТЗ Treasury, задача 2): выручка от ночных бумаг и весь свободный
        # кэш сверх буфера — в фонд денежного рынка. Продажи выше уже дождались
        # исполнения. Сбой парковки торговлю не останавливает: кэш просто лежит.
        if _treasury_enabled():
            try:
                tr = _treasury(broker, account_id, env, conn, writer, run_id)
                if dry_run:
                    log.info("[CLOSE][DRY] парковка: свободно %.2f ₽, буфер %.0f ₽",
                             tr.get_treasury_state()["free_cash_rub"], tr.buffer)
                else:
                    park = tr.park_idle_cash()
                    res.data["treasury_parked_rub"] = park["amount_rub"]
                res.data["treasury"] = _treasury_brief(tr.get_treasury_state())
            except Exception as e:               # noqa: BLE001
                res.check(False, f"казначейство: парковка не выполнена ({e})",
                          critical=False)

        _write_json(os.path.join(run_dir, "overnight_close.json"), out)
        _snapshot_account(broker, account_id, run_dir, run_id, "CLOSE", "after")
        res.data.update(overnight_before=out["overnight_before"],
                        positions_closed=len(out["closed"]),
                        orders_cancelled=out["orders_cancelled"],
                        registry_reconciled=out["registry_reconciled"])
        log.info("[CLOSE] ночных позиций %d, закрыто %d, снято лимиток %d",
                 out["overnight_before"], len(out["closed"]), out["orders_cancelled"])
    except Exception as e:                       # noqa: BLE001
        log.exception("[CLOSE] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
        conn.close()
    return _finish(res, state)


def phase_park(*, now: dt.datetime | None = None, prod: bool = False,
               dry_run: bool = False) -> int:
    """Казначейство после открытия СПБ (10:05, решение пользователя 15.09).

    TMON@ — единственный листинг фонда, доступный через API, — торгуется на СПБ
    с 10:00: в 09:10 CLOSE получает «через API: нет» и паркует виртуально. Здесь
    паи покупаются реально, строго лимитной заявкой не дороже последней сделки +
    TREASURY_LIMIT_MAX_PREMIUM_PCT, а виртуальные паи из CLOSE после исполнения
    покупки списываются. Первая минута торгов пропускается: 15.09 в 10:00
    проходили сделки 161,00–165,20 при цене 164,3.

    Сбой казначейства торговлю не останавливает: паи не куплены — рубли лежат на
    счёте, виртуальный учёт сохраняется. Критичны только предполётные проверки.
    """
    guard = _common_guards("PARK", now=now)
    if isinstance(guard, int):
        return guard
    state, day = guard

    from services.broker.base import BrokerError
    from services.place_orders import _make_broker_and_account, _open_log

    run_id = new_run_id("PARK", now)
    run_dir = make_run_dir(run_id)
    res = PhaseResult("PARK", run_id, run_dir)
    log.info("[PARK] %s, торговый день %s", run_id, day)
    if not _treasury_enabled():
        res.skipped = "TREASURY_ENABLED=0"
        return _finish(res, state)

    conn = database.get_connection()
    writer, fp = _open_log()
    try:
        broker, account_id, env = _make_broker_and_account(prod)
        preflight(res, env=env, prod_flag=prod, account_id=account_id)
        if res.errors:
            return _finish(res, state)           # вне SANDBOX заявок не шлём
        _snapshot_account(broker, account_id, run_dir, run_id, "PARK", "before")
        try:
            tr = _treasury(broker, account_id, env, conn, writer, run_id)
            if dry_run:
                st = tr.get_treasury_state()
                log.info("[PARK][DRY] через API: %s, рублей %.2f, виртуальных паёв %d",
                         "да" if tr.tradable() else "нет", st["cash_rub"], st["virtual_lots"])
            elif not tr.tradable():
                res.check(False, f"{tr.instrument().ticker}: листинг недоступен через API — "
                                 "паи не куплены, виртуальный учёт сохранён", critical=False)
            else:
                try:
                    out = tr.park_limit()
                except BrokerError as e:
                    res.partial_fail(f"покупка паёв не исполнена: {e}")
                else:
                    res.data.update(treasury_parked_lots=out["lots"],
                                    treasury_parked_rub=out["amount_rub"],
                                    treasury_converted_lots=out["converted_lots"],
                                    treasury_limit_price=out["limit_price"])
                    if out["reason"]:
                        log.info("[PARK] %s", out["reason"])
                    if out["requested_lots"] and out["lots"] < out["requested_lots"]:
                        res.partial_fail(f"паёв куплено {out['lots']} из {out['requested_lots']}")
            res.data["treasury"] = _treasury_brief(tr.get_treasury_state())
        except Exception as e:                   # noqa: BLE001
            res.check(False, f"казначейство: {e}", critical=False)
        _snapshot_account(broker, account_id, run_dir, run_id, "PARK", "after")
    except Exception as e:                       # noqa: BLE001
        log.exception("[PARK] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}", critical=False)
    finally:
        fp.close()
        conn.close()
    return _finish(res, state)


def cmd_protect(*, prod: bool = False) -> int:
    """Монитор защиты позиций: SL/TP по факту заливки, OCO, аварийный выход.

    Ночная лимитка выставляется в 18:35, а заливается когда угодно до конца
    вечерней сессии (ENPG 10.09 — в 21:48). Ставить стоп вместе с заявкой
    нельзя: условная SELL без позиции при срабатывании открыла бы шорт. Поэтому
    стоп ставится ПО ФАКТУ заливки.

    С аудита r4 (17.09) монитор работает всё торговое время, а не только
    вечером: утром до CLOSE срабатывают ночные стопы, и парную ногу надо снять
    сразу (SNGS: тейк висел 07:06–09:10); днём заливаются внутридневные
    лимитки, которым нужен стоп до 18:20. Каждый прогон: OCO по истории
    условных заявок, заливки в журнал исполнения, стоп либо аварийный выход по
    рынку, если цена уже за стопом.

    Работает и на остановленном тесте: остановка запрещает новые сделки, а не
    защиту уже открытых. Каталога запуска нет, сообщение в Telegram — только
    если что-то сделано или позиция осталась без защиты.
    """
    if not getattr(config, "STAGE2_ENABLED", True):
        return 0
    from services.place_orders import (_make_broker_and_account, _open_log,
                                       _load_pending, attach_stops)
    broker, account_id, env = _make_broker_and_account(prod)
    bad = contour_ok(env)
    if bad:
        log.error("[PROTECT] контур не согласован: %s", bad)
        return 1
    if env == "PROD" and account_id != str(getattr(config, "PROD_ACCOUNT_ID", "")).strip():
        log.error("[PROTECT] счёт брокера %s ≠ PROD_ACCOUNT_ID", account_id)
        return 1
    before = {r.get("order_id"): bool(r.get("stop_placed"))
              for r in _load_pending().get(account_id, []) if not r.get("closed")}
    conn = None
    try:
        conn = database.get_connection()
    except Exception as e:                       # noqa: BLE001
        log.warning("[PROTECT] БД недоступна — заливки в журнал не пишутся: %s", e)
    rep: dict = {}
    writer, fp = _open_log()
    try:
        attach_stops(broker, account_id, dry_run=False, writer=writer, env=env,
                     conn=conn, report=rep)
    finally:
        fp.close()
        if conn is not None:
            conn.close()
    open_uids = {p.instrument_uid for p in broker.get_positions(account_id) if p.is_open}
    recs = [r for r in _load_pending().get(account_id, []) if not r.get("closed")]
    newly = [r.get("ticker") for r in recs
             if r.get("stop_placed") and not before.get(r.get("order_id"), False)]
    naked = [r.get("ticker") for r in recs
             if r.get("instrument_uid") in open_uids and not r.get("stop_placed")]
    log.info("[PROTECT] стопов поставлено %d, без стопа %d, OCO закрыто %d, "
             "аварийных выходов %d (сбоев %d), заливок в журнал %d",
             len(newly), len(naked), len(rep["oco_closed"]), len(rep["breach_exits"]),
             len(rep["breach_failed"]), rep["fills_journaled"])
    if newly or naked or rep["oco_closed"] or rep["breach_exits"] or rep["breach_failed"] \
            or rep["closed_externally"]:
        try:
            from services import notify
            if notify.enabled():
                lines = ["\U0001f6e1 <b>PROTECT</b>"]
                for cap, items in (("стоп поставлен", newly),
                                   ("OCO: сработала нога, парная снята", rep["oco_closed"]),
                                   ("закрыто без известной ноги", rep["closed_externally"]),
                                   ("цена за стопом — закрыто по рынку", rep["breach_exits"])):
                    if items:
                        lines.append(f"{cap}: " + notify.esc(", ".join(items)))
                if rep["breach_failed"]:
                    lines.append("⛔ аварийный выход НЕ выполнен: "
                                 + notify.esc(", ".join(rep["breach_failed"])))
                if naked:
                    lines.append("⛔ без стопа: " + notify.esc(", ".join(naked)))
                notify.send("\n".join(lines),
                            silent=not (naked or rep["breach_failed"] or rep["breach_exits"]))
        except Exception as e:                   # noqa: BLE001
            log.warning("уведомление не отправлено: %s", e)
    return 1 if (naked or rep["breach_failed"]) else 0


# ── Фаза OVERNIGHT (§7) ──────────────────────────────────────────────────────

def reprice_marketable(broker, orders: list, *, slip_pct: float, position_rub: float) -> tuple[list, list[str]]:
    """Ночные лонги — вход по текущей цене с запасом slip_pct (решение 22.09.2026).

    Прогнозный вход вечерней фазы стоит ниже рынка и опирается на вчерашний бар,
    поэтому не исполнялся. Здесь цена берётся у брокера в момент постановки:
      вход  = last · (1 ± slip)   (BUY — выше, SELL — ниже: заявка сразу исполнима);
      стоп/тейк — те же проценты от НОВОГО входа, что считала модель;
      лоты  — под ту же сумму позиции: лимит позиции, либо урезанная MaxPos сумма.
    Нет цены — заявка остаётся как была (прогнозная лимитка), это пишется в лог.
    """
    import dataclasses
    out, notes = [], []
    for o in orders:
        try:
            inst = broker.find_instrument(o.ticker)
            last = broker.get_last_price(inst.instrument_uid)
        except Exception as e:                   # noqa: BLE001
            last = None
            notes.append(f"{o.ticker}: цена недоступна ({type(e).__name__}) — прогнозная лимитка")
        if not last or last <= 0 or not o.entry_price or not o.quantity_lots:
            if last is not None and last <= 0:
                notes.append(f"{o.ticker}: нулевая цена — прогнозная лимитка")
            out.append(o)
            continue
        sign = 1.0 if o.direction == "LONG" else -1.0
        entry = last * (1.0 + sign * slip_pct / 100.0)
        stop = entry * (1.0 - sign * o.stop_pct / 100.0) if o.stop_pct else None
        tp = entry * (1.0 + sign * o.tp_pct / 100.0) if o.tp_pct else None
        lot = int(o.lot_size or 1)
        was = float(o.total_rub or 0.0)
        # был ли размер урезан MaxPos: тогда держим ту же сумму, иначе — лимит позиции
        cap = position_rub if was >= position_rub - o.entry_price * lot else was
        lots = int(cap / (entry * lot)) if entry > 0 else 0
        if lots <= 0:
            notes.append(f"{o.ticker}: 1 лот по {entry:.2f} дороже допустимых {cap:.0f} ₽ — пропуск")
            out.append(dataclasses.replace(o, quantity_lots=None, total_rub=None))
            continue
        better = (o.anchor_price - entry) / o.anchor_price * 100.0 * sign if o.anchor_price else None
        log.info("[OVERNIGHT] %s: вход по рынку %.4f (последняя %.4f %+.2f %%; прогнозная лимитка была %.4f), "
                 "%d лот(ов) на %.0f ₽", o.ticker, entry, last, sign * slip_pct, o.entry_price,
                 lots, lots * lot * entry)
        out.append(dataclasses.replace(o, entry_price=entry, stop_price=stop, tp_price=tp,
                                       better_pct=better, quantity_lots=lots,
                                       total_rub=lots * lot * entry))
    return out, notes


def phase_overnight(*, now: dt.datetime | None = None, prod: bool = False,
                    dry_run: bool = False) -> int:
    """Пересчитывает ТОЛЬКО long_overnight и ставит ночные заявки.

    Пересчёт здесь правомерен и обязателен: цель стратегии — next_open/today_close,
    то есть вход на закрытии. Якорем должна быть цена закрытия, а не утренняя.

    Стоп здесь НЕ ставится: условная SELL-заявка без позиции при срабатывании
    открыла бы шорт. SL/TP ставит вечерний монитор `protect` по факту заливки,
    выход — утренняя фаза CLOSE (хотфикс 11.09).
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
        preflight(res, env=env, prod_flag=prod, account_id=account_id)
        _snapshot_account(broker, account_id, run_dir, run_id, "OVERNIGHT", "before")

        # 2. пересчёт только ночной стратегии
        orders, meta = compute_orders(
            int(getattr(config, "BEST_TRADES_TOP_N", 5)),
            float(getattr(config, "BEST_TRADES_POSITION_RUB", 10000)),
            float(getattr(config, "LIMIT_ENTRY_FRACTION", 0.2)),
            quiet=True, refresh=False)
        night = [o for o in orders if o.strategy == "long_overnight" and o.is_placeable]
        mode = str(getattr(config, "OVERNIGHT_ENTRY_MODE", "forecast")).lower()
        res.data["entry_mode"] = mode
        if night and mode == "marketable":
            night, notes = reprice_marketable(
                broker, night,
                slip_pct=float(getattr(config, "OVERNIGHT_MARKETABLE_SLIP_PCT", 0.1)),
                position_rub=float(getattr(config, "BEST_TRADES_POSITION_RUB", 10000)))
            night = [o for o in night if o.is_placeable]
            for n in notes:
                res.check(False, f"вход по рынку: {n}", critical=False)

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

        # UNPARK (ТЗ Treasury, задача 3): паи фонда продаются ровно под ночную
        # корзину и ДО заявок по акциям. Не хватило паёв — корзина урезается под
        # фактический кэш: держать акции ночью на заёмные деньги нельзя.
        tr = None
        if accepted and _treasury_enabled():
            from services.treasury import fit_budget, order_cost_rub
            required = sum(order_cost_rub(a) for a in accepted)
            available = 0.0
            try:
                tr = _treasury(broker, account_id, env, conn, writer, run_id)
                if dry_run:
                    log.info("[OVERNIGHT][DRY] под корзину нужно %.2f ₽ — паи не продаются",
                             required)
                else:
                    tr.release_cash_for_overnight(required)
                    res.data["treasury_released_lots"] = tr.last_release.get("sold_lots", 0)
                    available = tr.get_treasury_state()["free_cash_rub"] - tr.buffer
            except Exception as e:               # noqa: BLE001
                res.check(False, f"казначейство: кэш под ночные покупки не высвобожден "
                                 f"({e})", critical=False)
                cash = _money_rub(broker, account_id)
                available = (cash or 0.0) - float(getattr(config, "TREASURY_CASH_BUFFER_RUB",
                                                          1000.0))
            if not dry_run:
                accepted, cut = fit_budget(accepted, available)
                for c in cut:
                    rejected.append({**c, "skip_reason": f"не хватило кэша после продажи "
                                                         f"паёв: доступно {available:.2f} ₽"})
                if cut:
                    res.check(False, "ночная корзина урезана под кэш: без "
                              + ", ".join(c["ticker"] for c in cut), critical=False)

        placed = []
        if accepted:
            keep = {a["ticker"] for a in accepted}
            rep: list = []
            placed = place_limits(broker, account_id,
                                  [o for o in night if o.ticker in keep],
                                  dry_run=dry_run, immediate_stop=False, force=False,
                                  writer=writer, env=env, report=rep) or []
            ensure_execution_audit(conn)
            miss = _record_intents(conn, placed, accepted, run_id=run_id,
                                   phase="OVERNIGHT", env=env, day=day)
            res.check(miss == 0,
                      f"журнал исполнения: не записано намерений {miss} из "
                      f"{len(placed)}", critical=False)
            _check_placement(res, rep, placed, accepted)

        # Критерий приёмки казначейства (§5 п.3): ночь без плеча — свободных
        # рублей после постановки ночных заявок не меньше нуля.
        if not dry_run:
            cash = _money_rub(broker, account_id)
            if cash is not None:
                res.check(cash >= 0, f"после OVERNIGHT свободных рублей {cash:.2f} ₽ < 0 "
                                     f"— ночные лонги в плечо")
        if _treasury_enabled():
            try:
                tr = tr or _treasury(broker, account_id, env, conn, writer, run_id)
                st = tr.get_treasury_state()
                res.data["treasury"] = _treasury_brief(st)
                if not dry_run and st["virtual_lots"]:
                    res.check(st["free_cash_rub"] >= 0,
                              f"после OVERNIGHT свободный кэш за вычетом виртуальной "
                              f"парковки {st['free_cash_rub']:.2f} ₽ < 0")
            except Exception as e:               # noqa: BLE001
                res.check(False, f"казначейство: состояние недоступно ({e})", critical=False)

        # 5. отдельный файл — не смешивать с интрадеем
        _write_json(os.path.join(run_dir, "overnight_orders.json"),
                    {"accepted": accepted, "rejected": rejected, "placed": placed})
        _snapshot_account(broker, account_id, run_dir, run_id, "OVERNIGHT", "after")
        res.data.update(overnight_orders=len(accepted), placed_count=len(placed))
        log.info("[OVERNIGHT] ночных заявок принято %d, выставлено %d",
                 len(accepted), len(placed))

        # 8. дневной аудит, баланс, счётчик
        state = _close_trading_day(day, state, res, env)
    except Exception as e:                       # noqa: BLE001
        log.exception("[OVERNIGHT] сбой: %s", e)
        res.check(False, f"{type(e).__name__}: {e}")
    finally:
        fp.close()
        conn.close()
    return _finish(res, state)


# ── Дневной аудит и счётчик (§11, §12) ───────────────────────────────────────

def _execution_stats(day: dt.date, env: str = "SANDBOX") -> dict:
    """Статистика исполнения за день из execution_audit.

    Фильтр по account_env — execution_audit общая таблица для всех контуров;
    без него дневная сводка песочницы и боевого профиля, запущенных
    параллельно в один день, смешались бы в одну цифру (решение пользователя
    23.09.2026 о параллельном турнирном контуре).
    """
    sql = """
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE filled),
               AVG(slippage_pct) FILTER (WHERE filled),
               -- Тот же FILTER, что и у факта: сравнивать половины критерия
               -- по разным множествам строк — значит сравнивать разное.
               AVG(expected_slippage_pct) FILTER (WHERE filled)
        FROM execution_audit WHERE asof_date = %s AND account_env = %s;
    """
    try:
        with database.get_db_connection() as c:
            with c.cursor() as cur:
                cur.execute(sql, (day, env))
                n, filled, fact, exp = cur.fetchone()
        return {"available": True,
                "orders": int(n or 0), "filled": int(filled or 0),
                "slippage_fact_pct": float(fact) if fact is not None else None,
                "slippage_expected_pct": float(exp) if exp is not None else None}
    except Exception as e:                       # noqa: BLE001
        log.warning("execution_audit недоступен: %s", e)
        # available=False — не то же самое, что orders=0. Ноль без этого флага
        # неотличим от честного дня без сделок, и через две недели по
        # артефактам уже не понять, был журнал сломан или торговли не было.
        return {"available": False, "orders": 0, "filled": 0}


def _trades_since(start_day: dt.date, env: str = "SANDBOX") -> int:
    """Накопительный счётчик залитых сделок для STAGE2_TRADE_TARGET (§12а).

    Считает СТРОКИ execution_audit, а не позиции: вход и выход ночного лонга —
    две строки, два fill. Если турнир считает круглый оборот за одну сделку,
    это число нужно делить на два — решение о трактовке за организаторами
    турнира, здесь просто честный счётчик filled-заявок.
    """
    sql = "SELECT COUNT(*) FROM execution_audit WHERE filled AND asof_date >= %s AND account_env = %s;"
    try:
        with database.get_db_connection() as c:
            with c.cursor() as cur:
                cur.execute(sql, (start_day, env))
                (n,) = cur.fetchone()
        return int(n or 0)
    except Exception as e:                       # noqa: BLE001
        log.warning("счётчик сделок недоступен: %s", e)
        return 0


def build_daily_audit(day: dt.date, res: PhaseResult, env: str = "SANDBOX") -> dict:
    """Сводит день по каталогам всех фаз."""
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

    # Фаза, которая сейчас закрывает день (OVERNIGHT), свой run_meta ещё не
    # записала: без подстановки её вердикт и run_id в аудит не попадали
    # (10.09: overnight_run_id = null), а частичный отказ брокера терялся.
    if res is not None and not meta.get(res.phase):
        meta[res.phase] = {"run_id": res.run_id, "verdict": res.verdict,
                           "errors": res.errors, "warnings": res.warnings,
                           "partial": res.partial,
                           "critical_passed": res.critical_passed}

    errors = [e for m in meta.values() for e in (m.get("errors") or [])]
    warnings = [w for m in meta.values() for w in (m.get("warnings") or [])]
    partial = [x for m in meta.values() for x in (m.get("partial") or [])]
    # Фаза, перезапущенная в тот же день, в сводку попадает последним прогоном;
    # сбой первого прогона раньше пропадал бесследно (16.09: PARK 10:05
    # PARTIAL_FAILURE, повтор 10:16 PASS → в аудите partial: []).
    for ph, d in _superseded_run_dirs(day, dirs):
        m = _read_json(os.path.join(d, "run_meta.json"), {})
        for x in (m.get("errors") or []) + (m.get("partial") or []):
            warnings.append(f"{ph} {os.path.basename(d)} ({m.get('verdict')}), "
                            f"перезапущена: {x}")
    missing = [p for p in PHASES if p not in dirs]
    if missing:
        warnings.append(f"фазы не отработали: {', '.join(missing)}")

    prep = meta.get("PREP", {})
    stats = _execution_stats(day, env)

    placed_today = len(orders.get("accepted") or []) + len(night.get("accepted") or [])
    if not stats.get("available"):
        warnings.append("журнал исполнения недоступен — проскальзывание за день "
                        "проверить нечем")
    elif placed_today and not stats.get("orders"):
        warnings.append(f"заявок за день {placed_today}, а в журнале исполнения "
                        f"пусто — расхождение источников")
    audit = {
        "trading_day": day.isoformat(),
        **{f"{p.lower()}_run_id": meta.get(p, {}).get("run_id") for p in PHASES},
        "signals_count": len(signals),
        "orders_count": len(orders.get("accepted") or []),
        "fills_count": sum(1 for f in fills if (f.get("lots_executed") or 0) > 0),
        "overnight_orders": len(night.get("accepted") or []),
        "execution_audit_available": bool(stats.get("available")),
        "execution_audit_count": stats.get("orders", 0),
        "execution_audit_filled": stats.get("filled", 0),
        "slippage_fact_pct": stats.get("slippage_fact_pct"),
        "slippage_expected_pct": stats.get("slippage_expected_pct"),
        "treasury": res.data.get("treasury") if res is not None else None,
        "dataset_hash": prep.get("dataset_hash"),
        "config_hash": prep.get("config_hash"),
        "checks": {
            "critical_passed": sum(m.get("critical_passed", 0) or 0 for m in meta.values())
            or res.critical_passed,
            "critical_failed": len(errors),
            "partial": len(partial),
            "warnings": len(warnings),
        },
        "errors": errors,
        "partial": partial,
        "warnings": warnings,
        "verdict": (FAIL if errors else PARTIAL if partial
                    else PASS_WARN if warnings else PASS),
    }
    _write_json(_p("daily_audit", f"{day.isoformat()}.json"), audit)
    return audit


def _close_trading_day(day: dt.date, state: dict, res: PhaseResult, env: str = "SANDBOX") -> dict:
    """Дневной аудит, сводка баланса, отчёт и счётчик торговых дней."""
    dirs = _run_dirs_for_day(day)
    audit = build_daily_audit(day, res, env)

    prev = None
    if state.get("days"):
        prev_sum = _read_json(_p("balance", f"{state['days'][-1]}.json"), {})
        prev = prev_sum.get("closing_balance_rub")
    summary = stage2_balance.collect_day(day, dirs, prev_closing=prev)
    stage2_balance.save_day(summary)
    stage2_balance.save_report(execution=_execution_stats(day, env))

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
    trade_target = int(getattr(config, "STAGE2_TRADE_TARGET", 0) or 0)
    if trade_target:
        start_day = dt.date.fromisoformat(state["days"][0]) if state.get("days") else day
        res.data["day_summary"]["trades_total"] = _trades_since(start_day, env)
        res.data["day_summary"]["trades_target"] = trade_target
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
    print(f"  Расписание: PREP {config.STAGE2_PREP_TIME}  CLOSE {config.STAGE2_CLOSE_TIME}"
          f"  ORDER {config.STAGE2_ORDER_TIME}"
          f"  CLEANUP {config.STAGE2_CLEANUP_TIME}  OVERNIGHT {config.STAGE2_OVERNIGHT_TIME}")
    if _start_date():
        print(f"  Старт теста: {_start_date()}")
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
    p.add_argument("phase", choices=["prep", "close", "order", "park", "cleanup", "overnight",
                                     "protect", "status", "resume"])
    p.add_argument("--dry-run", action="store_true",
                   help="пройти фазу без отправки заявок брокеру")
    p.add_argument("--prod", action="store_true",
                   help="БОЕВОЙ контур. Для Этапа 2 запрещён — фаза откажется работать.")
    args = p.parse_args(argv)

    if args.phase == "status":
        return cmd_status()
    if args.phase == "resume":
        return cmd_resume()
    from services.place_orders import RegistryBusy, RegistryError, registry_lock

    if args.phase == "protect":
        # Занят реестр — идёт фаза; следующий прогон монитора через 5 минут.
        try:
            with registry_lock(wait=False):
                return cmd_protect(prod=args.prod)
        except RegistryBusy:
            log.info("[PROTECT] реестр занят фазой — прогон пропущен")
            return 0
        except RegistryError as e:
            log.error("[PROTECT] %s", e)
            _alert(f"⛔ <b>PROTECT</b>: {e}")
            return 1

    fn = {"prep": phase_prep, "close": phase_close, "order": phase_order,
          "park": phase_park, "cleanup": phase_cleanup,
          "overnight": phase_overnight}[args.phase]
    kw = {"prod": args.prod}
    if args.phase != "prep":
        kw["dry_run"] = args.dry_run
    if args.phase == "prep":
        return fn(**kw)                          # PREP не трогает ни реестр, ни заявки
    # Фазы, которые меняют заявки и реестр, не пересекаются между собой и с protect.
    try:
        with registry_lock():
            return fn(**kw)
    except RegistryBusy as e:
        log.error("[%s] %s — фаза не выполнена", args.phase.upper(), e)
        _alert(f"⛔ <b>{args.phase.upper()}</b> не выполнена: {e}")
        return 1


def _alert(text: str) -> None:
    """Сообщение в Telegram вне карточки фазы (сбой до её начала)."""
    try:
        from services import notify
        if notify.enabled():
            notify.send(text, silent=False)
    except Exception as e:                       # noqa: BLE001
        log.warning("уведомление не отправлено: %s", e)


if __name__ == "__main__":
    sys.exit(main())
