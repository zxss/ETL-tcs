"""
Оркестратор стратегического исследования (ТЗ 18.09.2026, шаги 2–4).

  python -m research.strategic.run_strategic --stage dev                 # шаг 2–3
  python -m research.strategic.run_strategic --stage holdout --confirm-one-shot

Отложенная выборка защищена: прогон возможен только для гипотез, прошедших гейт
разработки, и только с явным подтверждением — она одноразовая.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.strategic import h1_factor_breadth as h1     # noqa: E402
from research.strategic import h2_aggregated_ofi as h2     # noqa: E402
from research.strategic import h3_meta_labeling as h3      # noqa: E402
from research.strategic import h4_totm_inelastic as h4     # noqa: E402
from research.strategic import h5_cash_and_carry as h5     # noqa: E402

log = logging.getLogger("research.strategic.run")

RULES_PATH = os.path.join(ROOT, "research", "strategic", "rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "strategic")
TRIALS_PATH = os.path.join(ROOT, "audit", "r4_research", "trials.jsonl")
MODULES = {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5}
TITLES = {"h1": "H1. Факторный охват (Ridge, дециль, 5 дней)",
          "h2": "H2. Дисбаланс потока заявок — прокси по барам",
          "h3": "H3. Мета-разметка поверх пробоя диапазона",
          "h4": "H4. Календарный приток конца месяца",
          "h5": "H5. Cash-and-Carry (вердикт по прошлому прогону)"}


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        pass
    path = os.path.join(ROOT, "REVISION")            # исследовательская копия — не репозиторий
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()[:12] or "unknown"
    return "unknown"


def count_trials(path: str = TRIALS_PATH) -> int:
    if not os.path.exists(path):
        return 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                total += int(json.loads(line).get("trials", 0))
            except ValueError:
                continue
    return total


def register_trials(stage: str, n: int) -> None:
    os.makedirs(os.path.dirname(TRIALS_PATH), exist_ok=True)
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": 7,
           "stage": f"strategic-{stage}", "trials": n, "revision": revision()}
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def holdout_guard(rows: list[str], stage: str, confirm: bool, out_dir: str) -> list[str]:
    """К отложенной выборке допускаются только прошедшие гейт разработки."""
    if stage != "holdout":
        return rows
    if not confirm:
        raise SystemExit("Отложенная выборка одноразовая: нужен флаг --confirm-one-shot")
    dev_path = os.path.join(out_dir, "dev", "results.json")
    if not os.path.exists(dev_path):
        raise SystemExit(f"Нет результатов разработки ({dev_path}) — сначала шаг 2–3")
    with open(dev_path, encoding="utf-8") as f:
        dev = json.load(f)
    allowed = [r for r in rows if dev.get(r, {}).get("passed")]
    blocked = [r for r in rows if r not in allowed]
    if blocked:
        log.warning("[GUARD] не допущены к отложенной выборке: %s", ", ".join(blocked))
    if not allowed:
        raise SystemExit("Гейт разработки не прошла ни одна гипотеза — отложенная выборка не тратится")
    return allowed


def to_markdown(results: dict, stage: str, rules: dict, trials: int) -> str:
    title = "разработка" if stage == "dev" else "отложенная выборка (единственный прогон)"
    s = rules["samples"][stage]
    head = (f"# Стратегические гипотезы — {title}\n\n"
            f"Сформировано {dt.datetime.now():%d.%m.%Y %H:%M}. Код `{revision()}`, "
            f"правила `{rules['version']}`. Период {s['from']} … {s['to']}. "
            f"Испытаний в реестре: {trials}.\n\n")
    cols = ("| гипотеза | сделок | нетто сверх фонда, %/сделку | t | сверх фонда, % год | "
            "IR | PBO | ёмкость, млн ₽ | гейт |\n|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for key, res in results.items():
        m = res.get("summary") or {}
        pbo = (res.get("pbo") or {}).get("pbo")
        cap = res.get("capacity_rub")
        fmt = lambda v, f="{:+.2f}": "—" if v is None else f.format(v)          # noqa: E731
        body += (f"| {TITLES[key]} | {m.get('trades', 0)} | {fmt(m.get('net_excess_pct_trade'), '{:+.3f}')} | "
                 f"{fmt(m.get('t'))} | {fmt(m.get('excess_annual_pct'))} | {fmt(m.get('ir'))} | "
                 f"{'—' if pbo is None else f'{pbo:.0%}'} | "
                 f"{'—' if not cap else f'{cap / 1e6:.1f}'} | "
                 f"{'ДА' if res.get('passed') else 'нет'} |\n")
    detail = "\n## Почему не пройдено\n\n"
    for key, res in results.items():
        if res.get("failed"):
            detail += f"* **{TITLES[key]}** — " + "; ".join(res["failed"]) + "\n"
    grids = "\n## Сетки конфигураций (для PBO)\n\n```\n" + json.dumps(
        {k: v.get("grid") for k, v in results.items() if v.get("grid")},
        ensure_ascii=False, indent=1, default=str) + "\n```\n"
    return head + cols + body + detail + grids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Стратегические гипотезы H1–H5")
    ap.add_argument("--stage", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--rows", default="h1,h2,h3,h4,h5")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--confirm-one-shot", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--days", type=int, default=0,
                    help="дымовой прогон: обрезать выборку до последних N дней")
    ap.add_argument("--no-register", action="store_true",
                    help="не писать испытание в реестр (только для дымовых прогонов)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import database
    rules = load_rules()
    rows = [r.strip() for r in a.rows.split(",") if r.strip()]
    unknown = [r for r in rows if r not in MODULES]
    if unknown:
        raise SystemExit(f"неизвестные модули: {unknown}")
    rows = holdout_guard(rows, a.stage, a.confirm_one_shot, a.out)
    if a.days:
        s = rules["samples"][a.stage]
        s["from"] = (dt.date.fromisoformat(s["to"]) - dt.timedelta(days=a.days)).isoformat()
        log.warning("[SMOKE] выборка обрезана до %s … %s — результат не засчитывается",
                    s["from"], s["to"])

    out_dir = os.path.join(a.out, a.stage + a.suffix)
    os.makedirs(out_dir, exist_ok=True)
    trials = count_trials()
    conn = database.get_connection()
    results, all_trades = {}, []
    try:
        for key in rows:
            log.info("=== %s (%s) ===", TITLES[key], a.stage)
            res = MODULES[key].run(conn, a.stage, rules, n_trials=trials)
            tr = res.pop("trades", None)
            if tr is not None and len(tr):
                all_trades.append(tr)
            results[key] = res
            m = res["summary"]
            log.info("[%s] сделок %s, t %s, сверх фонда %s %% год, гейт %s", key.upper(),
                     m.get("trades"), m.get("t"), m.get("excess_annual_pct"), res["passed"])
    finally:
        conn.close()

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    report = to_markdown(results, a.stage, rules, trials)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    if a.no_register or a.days:
        log.warning("[SMOKE] испытание в реестр НЕ записано")
    else:
        register_trials(a.stage + a.suffix,
                        sum(len(r.get("grid") or {"x": 1}) for r in results.values()))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
