"""
Офлайн-разложение баланса Этапа 2: доход казначейства (паи TMON) отдельно от
торгового результата. Только чтение — снимки balance.json фаз и treasury_ledger;
боевой контур r3 этот модуль не импортирует и не меняется.

Зачем. services/stage2_balance.collect_day (контур r3, не трогаем до конца
теста) даёт два искажения:
  * «перенос через ночь» = OVERNIGHT − CLEANUP ТОГО ЖЕ дня (18:20 → 18:35).
    Настоящая ночь long_overnight (18:35 D → выход CLOSE 09:10 D+1) в
    разложение не попадает: аудит r3 нашёл +52,63 ₽ за ночь 10→11.09 вне
    разложения;
  * паи TMON сидят внутри total_portfolio_rub: доход фонда (≈0,04 % в день от
    ≈1,95 млн ₽, то есть сотни рублей) неотличим от торговли (десятки рублей).

Метод. Снимки всех фаз выстраиваются по времени. На каждом интервале между
соседними снимками:
  казначейство = Δ стоимости паёв (etf_value_rub)
                 − чистые покупки паёв (BUY − SELL, treasury_ledger, mode=broker,
                   время операции внутри интервала);
  торговля      = Δ total_portfolio_rub − казначейство.
Сумма по интервалам в точности равна изменению счёта (телескоп) — это проверка
сходимости. Интервалы относятся к торговому дню D:
  утро      PREP(D) → CLOSE(D), если ночи до этого не было (первый день);
  интрадей  CLOSE(D) → ORDER → CLEANUP(D): intraday_short, 09:10–18:20;
  вечер     CLEANUP(D) → OVERNIGHT(D): закрытие интрадея и вход в ночь;
  ночь      OVERNIGHT(D) → CLOSE(D+1): long_overnight до выхода в 09:10,
            выходные — внутри ночи пятницы. Если у следующего дня нет снимка
            CLOSE (архив до хотфикса 11.09), ночь кончается первым снимком D+1.
Комиссии сделок с паями, если брокер их берёт, в balance.json не видны и
попадают в торговлю; интервалы, где двигались только паи, помечены. Реестр
virtual (фонд вне портфеля брокера) не отражается в снимках — такие строки
выводятся предупреждением.

Запуск (на сервере, только чтение):
    python -m audit.stage2_decompose --dir /opt/etl-tcs/audit/stage2-demo
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEGMENTS = ("утро", "интрадей", "вечер", "ночь")
JUMP_PCT = 0.5          # |доход паёв| за интервал > 0,5 % их стоимости — подозрительно


def _read(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_snapshots(stage_dir: str) -> list[dict]:
    """balance.json всех прогонов фаз, по времени снимка."""
    out = []
    for f in glob.glob(os.path.join(stage_dir, "runs", "*", "balance.json")):
        b = _read(f)
        if not b or b.get("total_portfolio_rub") is None or not b.get("captured_at"):
            continue
        b = dict(b)
        b["captured"] = dt.datetime.fromisoformat(b["captured_at"])
        b["run_dir"] = os.path.basename(os.path.dirname(f))
        out.append(b)
    return sorted(out, key=lambda b: b["captured"])


def load_ledger(conn, account_id: str | None = None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT ts, account_id, mode, side, amount_rub, reason, run_id "
                    "FROM treasury_ledger ORDER BY ts")
        rows = cur.fetchall()
    keys = ("ts", "account_id", "mode", "side", "amount_rub", "reason", "run_id")
    out = [dict(zip(keys, r)) for r in rows]
    return [r for r in out if account_id is None or r["account_id"] == account_id]


def _etf(b: dict) -> float:
    v = b.get("etf_value_rub")
    return float(v) if v is not None else 0.0


def net_purchases(ledger: list[dict], t0: dt.datetime, t1: dt.datetime) -> tuple[float, int]:
    """Чистые покупки паёв (BUY − SELL, ₽) с t0 < ts ≤ t1, только mode=broker."""
    s, n = 0.0, 0
    for r in ledger:
        if r.get("mode") != "broker" or not (t0 < r["ts"] <= t1):
            continue
        amt = float(r["amount_rub"])
        s += amt if str(r["side"]).upper() == "BUY" else -amt
        n += 1
    return s, n


def intervals(snaps: list[dict], ledger: list[dict]) -> list[dict]:
    """Интервалы между соседними снимками: сегмент, день, казначейство, торговля."""
    close_days = {b["captured"].date() for b in snaps if b.get("phase") == "CLOSE"}
    out, owner = [], None
    for a, b in zip(snaps, snaps[1:]):
        da, db = a["captured"].date(), b["captured"].date()
        pa, pb = a.get("phase"), b.get("phase")
        if pa == "OVERNIGHT":
            owner = da
        if owner is not None:
            seg, day = "ночь", owner
            if pb == "CLOSE" or (db > owner and db not in close_days):
                owner = None
        elif pa == "CLEANUP" and pb == "OVERNIGHT":
            seg, day = "вечер", da
        elif db == da and (pa in ("CLOSE", "ORDER") or pb in ("ORDER", "CLEANUP")):
            seg, day = "интрадей", da
        else:
            seg, day = "утро", db
        flows, n_flows = net_purchases(ledger, a["captured"], b["captured"])
        d_total = float(b["total_portfolio_rub"]) - float(a["total_portfolio_rub"])
        treasury = (_etf(b) - _etf(a)) - flows
        hours = (b["captured"] - a["captured"]).total_seconds() / 3600.0
        avg_etf = (_etf(a) + _etf(b) - flows) / 2.0 if (_etf(a) or _etf(b)) else 0.0
        out.append({"from": a["run_dir"], "to": b["run_dir"], "segment": seg, "day": day,
                    "hours": hours, "d_total": d_total, "treasury": treasury,
                    "trading": d_total - treasury, "flows": flows, "n_flows": n_flows,
                    "avg_etf": avg_etf,
                    "shares_only_moved": abs(float(a.get("shares_value_rub") or 0)) < 1e-9
                    and abs(float(b.get("shares_value_rub") or 0)) < 1e-9})
    return out


def by_day(iv: list[dict]) -> dict:
    days: dict[dt.date, dict] = {}
    for r in iv:
        d = days.setdefault(r["day"], {s: 0.0 for s in SEGMENTS} | {"treasury": 0.0,
                                                                     "etf_days": 0.0})
        d[r["segment"]] += r["trading"]
        d["treasury"] += r["treasury"]
        d["etf_days"] += r["avg_etf"] * r["hours"] / 24.0
    for d in days.values():
        d["trading"] = sum(d[s] for s in SEGMENTS)
        d["tmon_pct_year"] = (d["treasury"] / d["etf_days"] * 365 * 100) if d["etf_days"] else None
    return dict(sorted(days.items()))


def checks(snaps: list[dict], iv: list[dict], ledger: list[dict]) -> list[str]:
    msgs = []
    if len(snaps) >= 2:
        span = float(snaps[-1]["total_portfolio_rub"]) - float(snaps[0]["total_portfolio_rub"])
        got = sum(r["trading"] + r["treasury"] for r in iv)
        ok = abs(span - got) < 0.01
        msgs.append(f"{'✓' if ok else '✗'} сходимость: Δ счёта {span:+.2f} ₽ = торговля + "
                    f"казначейство {got:+.2f} ₽")
    virt = [r for r in ledger if r.get("mode") != "broker"]
    if virt:
        msgs.append(f"⚠ в treasury_ledger {len(virt)} строк mode≠broker — фонд вне портфеля, "
                    "доход таких паёв здесь не виден")
    for r in iv:
        if r["avg_etf"] and abs(r["treasury"]) / r["avg_etf"] * 100 > JUMP_PCT:
            msgs.append(f"⚠ {r['from']} → {r['to']}: доход паёв {r['treasury']:+.2f} ₽ "
                        f"({r['treasury'] / r['avg_etf'] * 100:+.2f} % стоимости) — проверить "
                        "привязку операций казначейства по времени")
    used = sum(r["n_flows"] for r in iv)
    brk = sum(1 for r in ledger if r.get("mode") == "broker"
              and snaps and snaps[0]["captured"] < r["ts"] <= snaps[-1]["captured"])
    if brk != used:
        msgs.append(f"✗ операций казначейства в периоде {brk}, привязано к интервалам {used}")
    elif brk:
        msgs.append(f"✓ все {brk} операций казначейства привязаны к интервалам")
    return msgs


def _f(v, digits=2):
    if v is None:
        return "—"
    return f"{v:+,.{digits}f}".replace(",", " ").replace(".", ",")


def report(snaps, iv, days, msgs, stage_dir, r3_days: dict) -> str:
    L = ["# Разложение баланса: казначейство отдельно от торговли", "",
         f"Каталог: `{stage_dir}`. Снимков {len(snaps)}, интервалов {len(iv)}. "
         "Только чтение, контур r3 не затронут.", "",
         "Казначейство = Δ стоимости паёв − чистые покупки паёв; торговля = Δ счёта − казначейство. "
         "Ночь — OVERNIGHT(D) → CLOSE(D+1), относится к дню входа D.", "",
         "| день | Δ счёта | казначейство | TMON, % годовых | торговля | утро | интрадей | вечер | ночь |",
         "|---|---|---|---|---|---|---|---|---|"]
    tot = {k: 0.0 for k in ("treasury", "trading", *SEGMENTS)}
    for d, v in days.items():
        dtot = v["treasury"] + v["trading"]
        L.append(f"| {d} | {_f(dtot)} | {_f(v['treasury'])} | "
                 f"{_f(v['tmon_pct_year'], 1) if v['tmon_pct_year'] is not None else '—'} | "
                 f"{_f(v['trading'])} | " + " | ".join(_f(v[s]) for s in SEGMENTS) + " |")
        for k in tot:
            tot[k] += v[k]
    L.append(f"| **итого** | {_f(tot['treasury'] + tot['trading'])} | {_f(tot['treasury'])} | | "
             f"{_f(tot['trading'])} | " + " | ".join(_f(tot[s]) for s in SEGMENTS) + " |")
    if r3_days:
        L += ["", "## Сравнение со сводкой r3 (services/stage2_balance)", "",
              "| день | r3: интрадей | r3: «перенос» (18:20→18:35) | здесь: интрадей | здесь: ночь |",
              "|---|---|---|---|---|"]
        for d, v in days.items():
            s = r3_days.get(d.isoformat()) or {}
            L.append(f"| {d} | {_f(s.get('intraday_pnl_rub'))} | {_f(s.get('overnight_carry_rub'))} | "
                     f"{_f(v['интрадей'])} | {_f(v['ночь'])} |")
    L += ["", "## Проверки", ""] + [f"- {m}" for m in msgs]
    L += ["", "## Интервалы", "", "| с | по | сегмент | день | часов | Δ счёта | операции паёв | казначейство | торговля |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in iv:
        L.append(f"| {r['from']} | {r['to']} | {r['segment']} | {r['day']} | {r['hours']:.1f} | "
                 f"{_f(r['d_total'])} | {_f(r['flows'])} ({r['n_flows']}) | {_f(r['treasury'])} | "
                 f"{_f(r['trading'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Разложение баланса Этапа 2: TMON отдельно от торговли")
    ap.add_argument("--dir", default="/opt/etl-tcs/audit/stage2-demo")
    ap.add_argument("--no-db", action="store_true", help="без treasury_ledger (архивы до казначейства)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    snaps = load_snapshots(a.dir)
    if len(snaps) < 2:
        print(f"снимков {len(snaps)} — разлагать нечего")
        return 0
    account = next((s.get("account_id") for s in snaps if s.get("account_id")), None)
    ledger = []
    if not a.no_db:
        import database
        conn = database.get_connection()
        try:
            ledger = load_ledger(conn, account)
        finally:
            conn.close()
    iv = intervals(snaps, ledger)
    days = by_day(iv)
    r3 = {os.path.basename(f)[:-5]: _read(f) or {}
          for f in glob.glob(os.path.join(a.dir, "balance", "*.json"))}
    text = report(snaps, iv, days, checks(snaps, iv, ledger), a.dir, r3)
    out = a.out or os.path.join(ROOT, "audit", "stage2_decomposition",
                                f"{os.path.basename(a.dir.rstrip('/'))}-{dt.datetime.now():%Y%m%d-%H%M}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
