"""
Запуск ретро-теста структурных моделей (ТЗ 17.09.2026).

  python -m research.structural.run_structural --stage dev
  python -m research.structural.run_structural --stage holdout   # строго один раз, после dev

Результаты: audit/r4_research/structural/{stage}/ (results.json, trades.csv, report.md);
после отложенной выборки — сводный RESEARCH-MACRO-STRUCTURAL-REPORT.md в корне.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research import event_study_news as es              # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402
from research.structural import common as cmn            # noqa: E402
from research.structural import test_cash_and_carry as m_cc   # noqa: E402
from research.structural import test_dividend_gap as m_div    # noqa: E402
from research.structural import test_index_rebalance as m_idx  # noqa: E402
from research.structural import test_totm as m_totm      # noqa: E402

log = logging.getLogger("research.structural.run")

# (ключ строки, модуль, вариант, название)
ROWS = [("totm_1a", "totm", "1A", "1А. TOTM — ночи конца месяца"),
        ("totm_1b", "totm", "1B", "1Б. TOTM — удержание 4 сессии"),
        ("cc", "cash_and_carry", "C&C", "2. Cash-and-Carry (базис) — СПРАВОЧНО, заглядывание вперёд"),
        ("cc2", "cash_and_carry", "C&C-2", "2′. Cash-and-Carry без заглядывания вперёд"),
        ("div_a", "dividend_gap", "A", "3А. Дивидендный гэп — корзина А (cash cows)"),
        ("div_b", "dividend_gap", "B", "3Б. Дивидендный гэп — корзина Б (прочие)"),
        ("index", "index_rebalance", "additions", "4. Индексный ребаланс — включения")]


def holdout_guard(rules: dict) -> None:
    if os.path.exists(os.path.join(cmn.OUT_DIR, "holdout", "results.json")):
        raise SystemExit("отложенная выборка уже прогонялась — повтор запрещён")
    for name in ("dev", "dev_cc2"):
        if not os.path.exists(os.path.join(cmn.OUT_DIR, name, "results.json")):
            raise SystemExit(f"сначала выборка разработки: нет {name}")


def dev_results() -> dict:
    """Разработка: основные строки из dev, строка cc2 — из dev_cc2 (перезаверение)."""
    with open(os.path.join(cmn.OUT_DIR, "dev", "results.json"), encoding="utf-8") as f:
        dev = json.load(f)
    p = os.path.join(cmn.OUT_DIR, "dev_cc2", "results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            dev["results"]["cc2"] = json.load(f)["results"]["cc2"]
    return dev


def _f(x, nd=2):
    return es._f(x, nd)


def stage_report(stage: str, res: dict, extra: dict, meta: dict, rows: list | None = None) -> str:
    rows = rows or ROWS
    L = [f"# Структурные модели — {'разработка' if stage == 'dev' else 'отложенная выборка (единственный прогон)'}", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правила `{meta['rules']}`. "
         f"Период {meta['from']} … {meta['to']}. Испытаний в реестре: {meta['trials_total']}.", "",
         "| модель | сделок | в год | удержание, дн | нетто сверх фонда, %/сделку (t) | медиана | доля плюсовых | "
         "сверх фонда на задействованный капитал, % год | вклад, % капитала 5 млн/год | IR | критерий |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _, _, name in rows:
        s = res[key]
        if not s.get("trades"):
            L.append(f"| {name} | 0 | — | — | — | — | — | — | — | — | нет |")
            continue
        L.append(f"| {name} | {s['trades']} | {s['trades_per_year']:.1f} | {s['avg_hold_days']:.1f} | "
                 f"{_f(s['net_excess_pct_trade'], 3)} ({_f(s['t'], 1)}) | {_f(s['median_net_excess_pct'], 3)} | "
                 f"{_f(s['win_rate'])} | {_f(s['excess_employed_annual_pct'])} | "
                 f"{_f(s['contribution_pct_capital_year'])} | {_f(s['ir'])} | {'да' if s['passed'] else 'нет'} |")
    L += ["", "## Подробности модулей", "", "```", json.dumps(extra, ensure_ascii=False, indent=1, default=str), "```", ""]
    return "\n".join(L)


def final_report(dev: dict, hold: dict) -> str:
    L = ["# Ретро-тест структурных и низкооборотных моделей (Hurdle > TMON)", "",
         f"Разработка: {dev['meta']['from']} … {dev['meta']['to']} (код `{dev['meta']['revision']}`); "
         f"отложенная: {hold['meta']['from']} … {hold['meta']['to']} (код `{hold['meta']['revision']}`, один прогон). "
         f"Правила `{dev['meta']['rules']}` (research/structural/rules.json). Испытаний в реестре: {hold['meta']['trials_total']}.", "",
         "Критерий (ТЗ): избыточная годовая доходность сверх TMON ≥ +2,0 % при t ≥ 2,0 — на задействованный капитал; "
         "вклад в % капитала 5 млн зависит от размера позиции и приведён справочно.", "",
         "| Модель | Сделок / год (разр. / отл.) | Среднее удержание, дн | Net PnL на сделку сверх фонда, % | "
         "Годовой вклад к TMON, % капитала | Сверх фонда на задействованный капитал, % год | IR к TMON | t | Holdout подтверждён? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, _, _, name in ROWS:
        a, b = dev["results"].get(key, {}), hold["results"].get(key, {})

        def pair(field, nd=2):
            return f"{_f(a.get(field), nd)} / {_f(b.get(field), nd)}"
        confirmed = bool(a.get("passed") and b.get("passed"))
        L.append(f"| {name} | {_f(a.get('trades_per_year'), 1)} / {_f(b.get('trades_per_year'), 1)} | "
                 f"{pair('avg_hold_days', 1)} | {pair('net_excess_pct_trade', 3)} | {pair('contribution_pct_capital_year')} | "
                 f"{pair('excess_employed_annual_pct')} | {pair('ir')} | {pair('t', 1)} | {'ДА' if confirmed else 'нет'} |")
    L += ["", "Числа в ячейках — «разработка / отложенная».", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ретро-тест структурных моделей")
    ap.add_argument("--stage", choices=("dev", "holdout"), required=True)
    ap.add_argument("--rows", default="", help="какие строки считать (ключи через запятую); по умолчанию все")
    ap.add_argument("--suffix", default="", help="суффикс каталога результатов, например cc2")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rules = cmn.load_rules()
    keys = [k for k in a.rows.split(",") if k] or [r[0] for r in ROWS]
    rows = [r for r in ROWS if r[0] in keys]
    if not rows:
        raise SystemExit(f"неизвестные строки: {a.rows}")
    out_name = a.stage + (f"_{a.suffix}" if a.suffix else "")
    if a.stage == "holdout":
        holdout_guard(rules)
    elif os.path.exists(os.path.join(cmn.OUT_DIR, out_name, "results.json")):
        raise SystemExit(f"{out_name} уже прогонялся — повтор не делается")
    d_from, d_to = cmn.period(rules, a.stage)
    import database
    conn = database.get_connection()
    try:
        cal = cmn.union_daily(conn, [cmn.INDEX], dt.date(2021, 12, 1), dt.date(2026, 9, 18))
        ctx = cmn.Context(cmn.trading_days(cal))
        rev = es._revision()
        total = ll.register(f"structural-{out_name}", len(rows) if a.stage == "dev" else 0, rev, sprint=6)
        need = {r[1] for r in rows}
        frames, extra = [], {}
        for name, mod in (("totm", m_totm), ("cash_and_carry", m_cc), ("dividend_gap", m_div),
                          ("index_rebalance", m_idx)):
            if name not in need:
                continue
            log.info("модуль %s", name)
            tr, ex = mod.run(conn, rules, a.stage, ctx)
            frames.append(tr)
            extra[name] = ex
    finally:
        conn.close()
    trades = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) \
        else pd.DataFrame(columns=cmn.TRADE_FIELDS)
    cap = rules["account"]["capital_rub"]
    res = {}
    for key, module, variant, _ in rows:
        sub = trades[(trades["module"] == module) & (trades["variant"] == variant)] if len(trades) else trades
        res[key] = cmn.summarize(sub, d_from, d_to, cap, rules["criteria"])
    meta = {"created": dt.datetime.now().strftime("%d.%m.%Y %H:%M"), "revision": rev, "rules": rules["version"],
            "from": str(d_from), "to": str(d_to), "trials_total": total}
    out = os.path.join(cmn.OUT_DIR, out_name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": res, "extra": extra}, f, ensure_ascii=False, indent=1, default=str)
    trades.to_csv(os.path.join(out, "trades.csv"), index=False, float_format="%.5f")
    text = stage_report(a.stage, res, extra, meta, rows)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    if a.stage == "holdout":
        final = final_report(dev_results(), {"meta": meta, "results": res})
        with open(cmn.REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(final)
        print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
