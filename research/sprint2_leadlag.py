"""
Спринт 2 (ТЗ 50D): товарный lead-lag, гипотезы 2А′ и 2Б′ (утверждены пользователем
15.09.2026). Параметры — research/sprint2_rules.json; правила и код фиксируются
коммитом до прогона.

2А′ — «Brent до открытия → дрейф после открытия»: ход ближнего фьючерса Brent от
закрытия вечерней сессии предыдущего буднего дня до 09:50; при |ход| ≥ 1 % — лонг
(рост) или шорт (падение) LKOH, ROSN, TATN от первой цены основной сессии (10:00)
до 11:00 (2A_11) и до 12:00 (2A_12). Статистика — знак × ход против IMOEX.
2Б′ — «сырьевой фильтр ночи»: если ближний фьючерс (Brent для нефтяных, золото GD
для PLZL/SELG) с 10:00 до 18:30 вырос — ночной лонг от 18:30 до первой сделки
следующего торгового дня. IMOEX ночью не считается, поэтому статистика — нетто:
ход с дивидендом − издержки − рост пая фонда за ночь.

Фьючерс: ближний по объёму предыдущего дня среди допущенных; контракт выбывает за
2 торговых дня до экспирации (будни, праздники не учитываются); доходность —
внутри одного контракта. t — по датам (среднее по корзине за дату).

Ворота разработки и подтверждение на отложенной одинаковые (gate в правилах).
Отложенная выборка — один прогон и только гипотез, прошедших ворота на разработке
(список — из dev/results.json); повторный прогон запрещён.

Запуск (на сервере): python -m research.sprint2_leadlag --stage dev | holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import session_timing as st                # noqa: E402
from research import short_rule as sr                    # noqa: E402

log = logging.getLogger("research.sprint2_leadlag")

RULES_PATH = os.path.join(ROOT, "research", "sprint2_rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "sprint2")
MAX_STALE = dt.timedelta(minutes=30)


def _t(s: str) -> dt.time:
    return dt.time.fromisoformat(s)


def _at(d: dt.date, t: dt.time) -> np.datetime64:
    return np.datetime64(dt.datetime.combine(d, t), "ns")


def _day(b: es.Bars, i: int) -> dt.date:
    return pd.Timestamp(b.t[i]).date()


# ── Ближний контракт ─────────────────────────────────────────────────────────

def eligible_until(expirations: dict, roll_bdays: int) -> dict:
    """Последний день, когда контракт может быть ближним: за roll_bdays торговых дней
    до экспирации он уже заменён следующим."""
    out = {}
    for tk, e in expirations.items():
        out[tk] = None if e is None else pd.Timestamp(
            np.busday_offset(np.datetime64(e, "D"), -(roll_bdays + 1), roll="backward")).date()
    return out


# ── Цены по барам ────────────────────────────────────────────────────────────

def close_on_day(b: es.Bars, d: dt.date, bar: dt.time) -> float | None:
    """close последнего бара дня d, начавшегося не позже bar и не раньше bar − 30 мин."""
    j = int(np.searchsorted(b.t, _at(d, bar), "right")) - 1
    if j < 0 or _day(b, j) != d or _at(d, bar) - b.t[j] > np.timedelta64(MAX_STALE):
        return None
    return float(b.c[j])


def open_at(b: es.Bars, d: dt.date, t: dt.time) -> tuple[dt.time, float] | None:
    """open первого бара дня d с началом ≥ t (ожидание ≤ 30 мин)."""
    i = b.entry(dt.datetime.combine(d, t))
    if i is None or _day(b, i) != d:
        return None
    return pd.Timestamp(b.t[i]).time(), float(b.o[i])


def prev_weekday_close(b: es.Bars, d: dt.date) -> tuple[dt.date, float] | None:
    """close последнего бара предыдущего буднего дня (выходные сессии пропускаются)."""
    j = int(np.searchsorted(b.t, _at(d, dt.time()), "left")) - 1
    while j >= 0:
        dd = _day(b, j)
        if dd.weekday() < 5:
            return dd, float(b.c[j])
        j = int(np.searchsorted(b.t, _at(dd, dt.time()), "left")) - 1
    return None


def first_open_on(b: es.Bars, d: dt.date) -> float | None:
    i = int(np.searchsorted(b.t, _at(d, dt.time()), "left"))
    if i >= len(b.t) or _day(b, i) != d:
        return None
    return float(b.o[i])


# ── 2А′ ──────────────────────────────────────────────────────────────────────

def signal_2a(fb: es.Bars, d: dt.date, sig: dict) -> dict | None:
    pc = prev_weekday_close(fb, d)
    p = close_on_day(fb, d, _t(sig["to_bar"]))
    if pc is None or p is None or pc[1] <= 0:
        return None
    chg = (p / pc[1] - 1.0) * 100.0
    direction = (1 if chg > 0 else -1) if abs(chg) >= sig["abs_threshold_pct"] else 0
    return {"prev_day": pc[0], "chg": chg, "dir": direction}


def outcome_intraday(sb: es.Bars, idx: es.Bars, d: dt.date, entry_t: dt.time,
                     exit_bar: dt.time) -> dict | None:
    e = open_at(sb, d, entry_t)
    if e is None or e[0] > exit_bar:
        return None
    t0, p0 = e
    p1 = close_on_day(sb, d, exit_bar)
    x0 = idx.price_at(dt.datetime.combine(d, t0))
    x1 = close_on_day(idx, d, exit_bar)
    if p1 is None or x1 is None or not (p0 > 0 and x0 and x0 == x0):
        return None
    move = (p1 / p0 - 1.0) * 100.0
    return {"t0": t0, "p0": p0, "move": move, "ar": move - (x1 / x0 - 1.0) * 100.0}


def rows_2a(h: dict, front_bars, sbars: dict, idx: es.Bars, days: list[dt.date], cost) -> tuple[list, dict]:
    """front_bars(d) → Bars ближнего контракта на день d или None."""
    rows, cnt = [], {"days": len(days), "no_signal_data": 0, "signal_days": 0}
    for d in days:
        fb = front_bars(d)
        s = signal_2a(fb, d, h["signal"]) if fb is not None else None
        if s is None:
            cnt["no_signal_data"] += 1
            continue
        if s["dir"] == 0:
            continue
        cnt["signal_days"] += 1
        for tk in h["stocks"]:
            b = sbars.get(tk)
            o = outcome_intraday(b, idx, d, _t(h["entry_time"]), _t(h["exit_bar"])) if b else None
            if o is None or not sr.quantum_ok(o["p0"]):
                continue
            c = float(cost(tk))
            rows.append({"date": d, "ticker": tk, "chg": s["chg"], "dir": s["dir"], "move": o["move"],
                         "ar": o["ar"], "signed_ar": s["dir"] * o["ar"], "cost": c,
                         "net": s["dir"] * o["move"] - c})
    return rows, cnt


# ── 2Б′ ──────────────────────────────────────────────────────────────────────

def signal_2b(fb: es.Bars, d: dt.date, sig: dict) -> float | None:
    o = open_at(fb, d, _t(sig["from_time"]))
    p = close_on_day(fb, d, _t(sig["to_bar"]))
    if o is None or p is None or o[1] <= 0:
        return None
    return (p / o[1] - 1.0) * 100.0


def outcome_overnight(sb: es.Bars, d: dt.date, d_next: dt.date, entry_bar: dt.time,
                      divs: list) -> dict | None:
    p0 = close_on_day(sb, d, entry_bar)
    p0_late = close_on_day(sb, d, (dt.datetime.combine(d, entry_bar) + dt.timedelta(minutes=5)).time())
    p1 = first_open_on(sb, d_next) if d_next else None
    if p0 is None or p1 is None or p0 <= 0:
        return None
    div = es.dividends_between(divs, d, d_next)
    out = {"p0": p0, "move": ((p1 + div) / p0 - 1.0) * 100.0, "div": div,
           "move_1835": ((p1 + div) / p0_late - 1.0) * 100.0 if p0_late else float("nan")}
    return out


def rows_2b(h: dict, front_bars, sbars: dict, days: list[dt.date], tdays: list[dt.date],
            divs: dict, cost, hurdle, carry) -> tuple[list, dict]:
    """Все дни и бумаги (любой знак сырья): отбор делает сводка."""
    rows, cnt = [], {"days": len(days), "no_signal_data": 0, "up_days": 0}
    for d in days:
        fb = front_bars(d)
        s = signal_2b(fb, d, h["signal"]) if fb is not None else None
        if s is None:
            cnt["no_signal_data"] += 1
            continue
        cnt["up_days"] += int(s > 0)
        d_next = es.next_day(tdays, d)
        for tk in h["stocks"]:
            b = sbars.get(tk)
            o = outcome_overnight(b, d, d_next, _t(h["entry_bar"]), divs.get(tk, [])) if b else None
            if o is None or not sr.quantum_ok(o["p0"]):
                continue
            c, hu = float(cost(tk)), float(hurdle(d, d_next))
            nights = (d_next - d).days
            rows.append({"date": d, "ticker": tk, "sig": s, "move": o["move"], "div": o["div"], "cost": c,
                         "hurdle": hu, "net_long": o["move"] - c - hu,
                         "net_long_1835": o["move_1835"] - c - hu,
                         "net_short": -o["move"] - c - float(carry(nights))})
    return rows, cnt


# ── Статистика ───────────────────────────────────────────────────────────────

def gate_ok(stat: dict, net: dict, gate: dict) -> bool:
    return (stat.get("t") is not None and stat["t"] >= gate["t_min"]
            and (stat.get("dates") or 0) >= gate["dates_min"]
            and (net.get("mean") is not None and net["mean"] > gate["net_min"]))


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def summarize(h: dict, df: pd.DataFrame, days: list[dt.date], n_trials: int, gate: dict) -> dict:
    if h["family"] == "2A":
        sel = {"основное": (df, "signed_ar", "net")}
    else:
        up = df[df["sig"] > 0] if len(df) else df
        down = df[df["sig"] <= 0] if len(df) else df
        sel = {"основное": (up, "net_long", "net_long"),
               "справочно: лонг, сырьё ≤ 0": (down, "net_long", "net_long"),
               "справочно: лонг каждый день": (df, "net_long", "net_long"),
               "справочно: шорт, сырьё ≤ 0": (down, "net_short", "net_short"),
               "справочно: вход 18:35": (up, "net_long_1835", "net_long_1835")}
    out = {}
    for name, (x, stat_col, net_col) in sel.items():
        if not len(x):
            out[name] = {"stat": {"n": 0, "dates": 0}, "net": {"n": 0, "dates": 0}}
            continue
        stat, net = es.by_date(x[stat_col], x["date"]), es.by_date(x[net_col], x["date"])
        daily = x.groupby("date")[net_col].mean().reindex(days).fillna(0.0)
        out[name] = {"stat": stat, "net": net, "hit": float((x[net_col] > 0).mean()),
                     "ir": es.information_ratio(daily), "dsr": es.deflated_sharpe(daily, max(2, n_trials))}
    out["gate"] = gate_ok(out["основное"]["stat"], out["основное"]["net"], gate)
    return out


def register(stage: str, n: int, revision: str, path: str = es.TRIALS_PATH, sprint: int = 2) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "sprint": sprint,
                            "stage": stage, "trials": n, "revision": revision}, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as f:
        return sum(int(json.loads(line).get("trials", 0)) for line in f)


# ── Отчёт ────────────────────────────────────────────────────────────────────

_f = es._f


def report(stage: str, res: dict, meta: dict) -> str:
    L = [f"# Спринт 2 — товарный lead-lag, {'разработка' if stage == 'dev' else 'отложенная выборка (единственный прогон)'}",
         "", f"Сформировано {meta['created']}. Код `{meta['revision']}`, правила `{meta['rules']}`. "
         f"Период {meta['from']} … {meta['to']} ({meta['stock_table']}; фьючерсы research_fut_5m). "
         f"Испытаний в реестре: {meta['trials_total']}.", "",
         f"Ворота: t ≥ {meta['gate']['t_min']:g} по датам, дат ≥ {meta['gate']['dates_min']}, "
         f"среднее нетто > {meta['gate']['net_min']:g}.", "",
         "| гипотеза | вариант | сделок | дат | статистика, % (t) | нетто, % (t) | доля плюсовых | IR | DSR |"
         + (" p Холма |" if stage == "holdout" else ""),
         "|---|---|---|---|---|---|---|---|---|" + ("---|" if stage == "holdout" else "")]
    for hid, r in res.items():
        for name, x in r["summary"].items():
            if name == "gate":
                continue
            s, n = x["stat"], x["net"]
            L.append(f"| {hid} | {name} | {s.get('n', 0)} | {s.get('dates', 0)} | {_f(s.get('mean'))} "
                     f"({_f(s.get('t'), 1)}) | {_f(n.get('mean'))} ({_f(n.get('t'), 1)}) | {_f(x.get('hit'), 2)} | "
                     f"{_f(x.get('ir'), 2)} | {_f(x.get('dsr'), 2)} |"
                     + (f" {_f(r.get('p_holm'), 3) if name == 'основное' else ''} |" if stage == "holdout" else ""))
    L += ["", "| гипотеза | дней | без данных сигнала | дней с сигналом | итог |", "|---|---|---|---|---|"]
    for hid, r in res.items():
        c = r["counts"]
        sig = c.get("signal_days", c.get("up_days"))
        L.append(f"| {hid} | {c['days']} | {c['no_signal_data']} | {sig} | {r['verdict']} |")
    L += ["", "2А′: статистика — знак × (ход бумаги − ход IMOEX) от 10:00 до выхода; нетто — знак × ход − издержки.",
          "2Б′: статистика и нетто — ход с дивидендом − издержки − рост пая фонда за ночь (IMOEX ночью не считается).",
          "Строки «справочно» не отбираются и не проверяются на отложенной выборке.", ""]
    return "\n".join(L)


# ── Прогон ───────────────────────────────────────────────────────────────────

def run(stage: str, out: str | None = None) -> int:
    with open(RULES_PATH, encoding="utf-8") as f:
        rules = json.load(f)
    out = out or os.path.join(OUT_DIR, stage)
    hyps = rules["hypotheses"]
    if stage == "holdout":
        if os.path.exists(os.path.join(out, "results.json")):
            raise SystemExit("отложенная выборка уже прогонялась — повтор запрещён правилами")
        with open(os.path.join(OUT_DIR, "dev", "results.json"), encoding="utf-8") as f:
            passed = json.load(f)["passed"]
        hyps = [h for h in hyps if h["id"] in passed]
        if not hyps:
            raise SystemExit("ни одна гипотеза не прошла ворота разработки — отложенная выборка не нужна")
    smp = rules["samples"][stage]
    d_from, d_to = dt.date.fromisoformat(smp["from"]), dt.date.fromisoformat(smp["to"])
    lo, hi = d_from - dt.timedelta(days=20), d_to + dt.timedelta(days=10)
    exp, root_of = st.read_contracts()
    roots = sorted({h["driver"] for h in hyps})
    fut_tickers = [tk for tk, r in root_of.items() if r in roots]
    stocks = sorted({tk for h in hyps for tk in h["stocks"]})
    import database
    conn = database.get_connection()
    try:
        daily = st.load_daily(conn, rules["futures"]["table"], fut_tickers, lo, hi)
        daily["root"] = daily["ticker"].map(root_of)
        front = st.front_contracts(daily, eligible_until(exp, rules["futures"]["roll_business_days_before_expiration"]))
        front = front[(front["d"] >= d_from) & (front["d"] <= d_to)]
        fbars = es.load_bars(conn, rules["futures"]["table"], sorted(set(front["front"])), lo, hi)
        sbars = es.load_bars(conn, smp["stock_table"], stocks, lo, hi)
        idx = es.load_bars(conn, smp["stock_table"], [es.INDEX], lo, hi).get(es.INDEX)
    finally:
        conn.close()
    fmap = {(r.root, r.d): r.front for r in front.itertuples()}
    tdays = sorted(d for d in idx.day_close if d.weekday() < 5)
    days = [d for d in tdays if d_from <= d <= d_to]
    spreads = cm.load_spreads()
    hurdle = es.Hurdle(es.HURDLE_PATH)
    divs = es.load_dividends(es.DIV_PATH, tdays)
    pos = rules["costs"]["position_rub"]
    cost = lambda tk: cm.round_trip(tk, rules["costs"]["scenario"], spreads)          # noqa: E731
    rev = es._revision()
    total = register(f"sprint2-{stage}", len(hyps) if stage == "dev" else 0, rev)
    res, frames = {}, []
    for h in hyps:
        fb = lambda d, r=h["driver"]: fbars.get(fmap.get((r, d)))                    # noqa: E731
        if h["family"] == "2A":
            rows, cnt = rows_2a(h, fb, sbars, idx, days, cost)
        else:
            rows, cnt = rows_2b(h, fb, sbars, days, tdays, divs, cost, hurdle.growth,
                                lambda n: cm.carry_pct(pos, n) if n else 0.0)
        df = pd.DataFrame(rows)
        summ = summarize(h, df, days, total, rules["gate"])
        res[h["id"]] = {"counts": cnt, "summary": summ}
        frames.append(df.assign(hypothesis=h["id"]))
    if stage == "holdout":
        adj = holm({k: r["summary"]["основное"]["stat"].get("p") for k, r in res.items()})
        for k, r in res.items():
            r["p_holm"] = adj.get(k)
    for k, r in res.items():
        r["verdict"] = ("ворота пройдены" if r["summary"]["gate"] else "ворота не пройдены") if stage == "dev" \
            else ("подтверждено" if r["summary"]["gate"] and (r.get("p_holm") or 1) < 0.05 else "не подтверждено")
    now = dt.datetime.now()
    meta = {"created": now.strftime("%d.%m.%Y %H:%M"), "revision": rev, "rules": rules["version"],
            "from": str(d_from), "to": str(d_to), "stock_table": smp["stock_table"],
            "trials_total": total, "gate": rules["gate"],
            "hurdle_from": {k: str(v[0][0]) for k, v in hurdle.series.items()},
            "rolls": int((front.groupby("root")["front"].apply(lambda s: (s != s.shift()).sum() - 1)).sum())}
    os.makedirs(out, exist_ok=True)
    text = report(stage, res, meta)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    passed = [k for k, r in res.items() if r["summary"]["gate"]]
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "passed": passed, "results": res}, f, ensure_ascii=False, indent=1, default=str)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(out, "trades.csv"), index=False, float_format="%.5f")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Спринт 2: товарный lead-lag (2А′, 2Б′)")
    ap.add_argument("--stage", choices=("dev", "holdout"), required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(a.stage, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
