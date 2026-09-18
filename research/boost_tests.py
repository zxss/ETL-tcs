"""
Тесты идей повышения доходности (запрос пользователя 17.09.2026). Правила —
research/boost_rules.json, фиксируются коммитом до прогона; один прогон на обе
выборки.

Счёт 5 000 000 ₽. Фонд (TMON@, до 25.02.2025 LQDT) работает на весь капитал;
внутридневные шорты фонд не трогают (обеспечение паями); многодневные лонги берут
деньги из фонда — упущенный доход фонда вычитается. Итог = доход фонда + торговля.

Базовый поток шортов — условия прода без модели (research/short_rule, как Спринт 4),
отбор топ-5 по падению вчера (как TOP_N прода).
  T0 база: 100 000 ₽, стоп 1 %, выход 18:20.
  T1 риск-сайзинг: риск 0,5 % капитала при стопе 1,5×ATR14; позиция ≤ 1 % оборота
     и ≤ 700 000 ₽; портфель ≤ 2 500 000 ₽.
  T2 ATR: 100 000 ₽, стоп 1,5×ATR14, тейк 2,5×ATR14.
  T3 фьючерс: в дни шортов T0 — шорт ближнего MX на ту же сумму, 10:05 → 18:15.
  T4 лонг на перепроданности: RSI(14) < 30 и IMOEX выше EMA200 на A′, топ-3,
     100 000 ₽, 3 сессии, с дивидендами, минус доход фонда на сумму позиции.
  T4M те же лонги под залог паёв (запрос пользователя): фонд на весь капитал,
     вместо упущенного дохода фонда — плата за перенос по тарифу за ночи.
Маржа под 100 % залог: покупательная способность = капитал × (1 − дисконт);
превышение по дням считается и выводится, сделки не режутся.

Запуск (на сервере): python -m research.boost_tests
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import cost_model as cm                    # noqa: E402
from research import event_study_news as es              # noqa: E402
from research import session_calendar as sc              # noqa: E402
from research import session_timing as st                # noqa: E402
from research import short_rule as sr                    # noqa: E402
from research import sprint2_leadlag as ll               # noqa: E402
from research import sprint4_gatekeeper as sg            # noqa: E402

log = logging.getLogger("research.boost_tests")

RULES_PATH = os.path.join(ROOT, "research", "boost_rules.json")
OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "boost")
MSK = dt.timezone(dt.timedelta(hours=3))
TESTS = ("T0", "T1", "T2", "T3", "T4", "T4M")

BARS_SQL = """
SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS tm, open, high, low, close
FROM {table}
WHERE ticker = ANY(%(tk)s) AND close > 0 AND ts >= %(f)s AND ts < %(t)s
  AND EXTRACT(ISODOW FROM ts AT TIME ZONE 'Europe/Moscow') < 6
  AND (ts AT TIME ZONE 'Europe/Moscow')::time BETWEEN TIME '09:15' AND TIME '18:15'
"""


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Чистые функции (тестируются) ─────────────────────────────────────────────

def short_exit(o, h, lo, c, stop_pct: float | None = None, tp_pct: float | None = None) -> tuple[float, str]:
    """Выход шорта по 5-минутному пути: стоп по max(уровень, open бара), тейк по
    min(уровень, open бара); оба в одном баре — стоп (консервативно); иначе close."""
    entry = float(o[0])
    stop = entry * (1.0 + stop_pct / 100.0) if stop_pct else None
    tp = entry * (1.0 - tp_pct / 100.0) if tp_pct else None
    for i in range(len(o)):
        if stop is not None and h[i] >= stop:
            return max(stop, float(o[i])), "stop"
        if tp is not None and lo[i] <= tp:
            return min(tp, float(o[i])), "tp"
    return float(c[-1]), "time"


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Уайлдера."""
    d = close.diff()
    up, down = d.clip(lower=0.0), (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ad = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(ad > 0, 100.0).where(au.notna())


def lots_notional(position_rub: float, price: float, lot: int) -> float:
    """Сумма, кратная лоту, не больше позиции; 0 — лот дороже позиции."""
    n = math.floor(position_rub / (price * lot))
    return n * price * lot if n > 0 else 0.0


def risk_position(capital: float, risk_pct: float, stop_pct: float, cap_rub: float,
                  adv_rub: float | None, adv_share: float) -> float:
    """Позиция при риске risk_pct % капитала и стопе stop_pct %, с лимитами."""
    if not stop_pct or stop_pct <= 0:
        return 0.0
    pos = capital * risk_pct / 100.0 / (stop_pct / 100.0)
    caps = [pos, cap_rub]
    if adv_rub and adv_rub == adv_rub and adv_rub > 0:
        caps.append(adv_rub * adv_share)
    return max(0.0, min(caps))


def exposure_by_day(tr: pd.DataFrame, days: list[dt.date]) -> pd.Series:
    """Суммарная позиция по дням: сделка занимает капитал с дня входа по день выхода."""
    out = pd.Series(0.0, index=days)
    if tr.empty:
        return out
    idx = pd.Index(days)
    for r in tr.itertuples(index=False):
        lo, hi = idx.searchsorted(r.date, "left"), idx.searchsorted(r.exit_date, "right")
        if hi > lo:
            out.iloc[lo:hi] += r.notional
    return out


def max_drawdown(pnl: pd.Series) -> float:
    """Максимальная просадка накопленного P&L (₽, отрицательное число или 0)."""
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    return float((cum - cum.cummax().clip(lower=0.0)).min())


# ── Данные ───────────────────────────────────────────────────────────────────

def load_paths(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> dict:
    """(ticker, day) → (opens, highs, lows, closes) от бара входа (10:05, допуск 10 мин) до 18:15."""
    df = pd.read_sql(BARS_SQL.format(table=table), conn, params={"tk": tickers, **sg._window(d_from, d_to)})
    df["tm"] = pd.to_datetime(df["tm"])
    df["d"] = df["tm"].dt.date
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    out = {}
    for (tk, d), g in df.sort_values(["ticker", "tm"]).groupby(["ticker", "d"], sort=False):
        start = np.datetime64(dt.datetime.combine(d, sc.short_entry(d)), "ns")
        tm = g["tm"].to_numpy(dtype="datetime64[ns]")
        m = tm >= start
        if not m.any() or tm[m][0] - start > np.timedelta64(sr.ENTRY_TOL):
            continue
        gg = g[m]
        out[(tk, d)] = tuple(gg[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    return out


def index_ema_above(conn, span: int) -> dict:
    """Дата → IMOEX выше EMA(span) по объединённой истории обеих таблиц."""
    parts = []
    for table in ("research_bars_5m", "market_data_5m"):
        d = sg.daily_from_5m(conn, table, [sr.INDEX], dt.date(2021, 12, 1), dt.date(2026, 9, 20))
        parts.append(d)
    ix = pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]
    ema = ix.ewm(span=span, adjust=False).mean()
    ok = pd.Series(np.arange(1, len(ix) + 1) >= span, index=ix.index)
    return {d: bool(c > e) for d, c, e, k in zip(ix.index, ix, ema, ok) if k}


# ── Прогон периода ───────────────────────────────────────────────────────────

def run_period(conn, smp: dict, rules: dict, lots: dict, blocked: set, spreads: dict, hurdle,
               divs_path: str, mx_front, ema200: dict) -> tuple[pd.DataFrame, list[dt.date]]:
    table = smp["stock_table"]
    d_from, d_to = dt.date.fromisoformat(smp["from"]), dt.date.fromisoformat(smp["to"])
    tickers = list(sr.UNIVERSE) + [sr.INDEX]
    daily = sg.adjust_daily_splits(sg.daily_from_5m(conn, table, tickers, d_from - dt.timedelta(days=500),
                                                    d_to + dt.timedelta(days=10)))
    feats, mkt, above = sr.build_features(daily, lots)
    tdates = sr.trading_dates(daily)
    rs = {}
    for tk, g in daily[daily["ticker"] != sr.INDEX].groupby("ticker"):
        g = g.sort_values("date")
        rs.update({(tk, d): v for d, v in zip(g["date"], rsi(g["close"], rules["tests"]["T4"]["rsi_period"]))})
    paths = load_paths(conn, table, tickers, d_from, d_to + dt.timedelta(days=10))
    exec_days = [d for d in sr.exec_days_from(paths) if d_from <= d <= d_to]
    by_date = {d: g for d, g in feats.groupby("date")}
    divs = es.load_dividends(divs_path, tdates)
    q = sg.QUANTUM[table]
    T = rules["tests"]
    cap = rules["account"]["capital_rub"]
    rows = []
    for A in exec_days:
        Ap = sr.prev_date(tdates, A)
        f = by_date.get(Ap)
        if Ap is None or f is None:
            continue
        m = mkt.get(Ap)
        gate = sr.gates_open(None if m is None or (isinstance(m, float) and math.isnan(m)) else float(m), above.get(Ap))
        # ── шорты T0–T3 ──
        if gate is True:
            c = f[f["ret1"].notna() & (f["ret1"] < 0) & ~f["ticker"].isin(blocked)]
            picks = sr.select(c, "drop5")
            t0_notional, t1_used = 0.0, 0.0
            for r in picks.itertuples(index=False):
                p = paths.get((r.ticker, A))
                if p is None:
                    continue
                entry, lot = float(p[0][0]), lots.get(r.ticker, 1)
                if not (entry > 0 and q / entry * 100.0 <= sr.MAX_QUANTUM_PCT):
                    continue
                cost = cm.round_trip(r.ticker, "base", spreads)
                atr = float(r.atr_pct) if r.atr_pct == r.atr_pct else float("nan")
                base = {"date": A, "exit_date": A, "ticker": r.ticker, "side": "short", "cost_pct": cost}
                n0 = lots_notional(T["T0"]["position_rub"], entry, lot)
                if n0:
                    px, why = short_exit(*p, stop_pct=T["T0"]["stop_pct"])
                    g = -(px / entry - 1.0) * 100.0
                    rows.append({**base, "test": "T0", "notional": n0, "gross_pct": g, "net_pct": g - cost,
                                 "pnl_rub": n0 * (g - cost) / 100.0, "exit": why})
                    t0_notional += n0
                if atr == atr and atr > 0:
                    sp = T["T1"]["stop_atr_k"] * atr
                    want = risk_position(cap, T["T1"]["risk_pct_of_capital"], sp, T["T1"]["position_cap_rub"],
                                         r.adv_rub, T["T1"]["position_cap_adv_share"])
                    want = min(want, T["T1"]["portfolio_cap_rub"] - t1_used)
                    n1 = lots_notional(want, entry, lot) if want > 0 else 0.0
                    if n1:
                        px, why = short_exit(*p, stop_pct=sp)
                        g = -(px / entry - 1.0) * 100.0
                        rows.append({**base, "test": "T1", "notional": n1, "gross_pct": g, "net_pct": g - cost,
                                     "pnl_rub": n1 * (g - cost) / 100.0, "exit": why})
                        t1_used += n1
                    n2 = lots_notional(T["T2"]["position_rub"], entry, lot)
                    if n2:
                        px, why = short_exit(*p, stop_pct=T["T2"]["stop_atr_k"] * atr, tp_pct=T["T2"]["tp_atr_k"] * atr)
                        g = -(px / entry - 1.0) * 100.0
                        rows.append({**base, "test": "T2", "notional": n2, "gross_pct": g, "net_pct": g - cost,
                                     "pnl_rub": n2 * (g - cost) / 100.0, "exit": why})
            if t0_notional > 0:
                b = mx_front(A)
                e = ll.open_at(b, A, sc.short_entry(A)) if b is not None else None
                x = ll.close_on_day(b, A, sc.SHORT_EXIT_BAR) if b is not None else None
                if e and x and e[1] > 0:
                    g = -(x / e[1] - 1.0) * 100.0
                    c3 = T["T3"]["cost_rt_pct"]
                    rows.append({"date": A, "exit_date": A, "ticker": "MX", "side": "short", "cost_pct": c3,
                                 "test": "T3", "notional": t0_notional, "gross_pct": g, "net_pct": g - c3,
                                 "pnl_rub": t0_notional * (g - c3) / 100.0, "exit": "time"})
        # ── лонг T4 ──
        t4 = T["T4"]
        if ema200.get(Ap) is True:
            i = tdates.index(A) if A in tdates else None
            A2 = tdates[i + t4["hold_sessions"] - 1] if i is not None and i + t4["hold_sessions"] - 1 < len(tdates) else None
            cands = [(rs.get((tk, Ap)), tk) for tk in f["ticker"]]
            cands = sorted((v, tk) for v, tk in cands if v is not None and v == v and v < t4["rsi_max"])
            taken = 0
            for v, tk in cands:
                if taken >= t4["top_n"] or A2 is None:
                    break
                p0, p2 = paths.get((tk, A)), paths.get((tk, A2))
                if p0 is None or p2 is None:
                    continue
                entry, exitp, lot = float(p0[0][0]), float(p2[3][-1]), lots.get(tk, 1)
                if not (entry > 0 and q / entry * 100.0 <= sr.MAX_QUANTUM_PCT):
                    continue
                n4 = lots_notional(t4["position_rub"], entry, lot)
                if not n4:
                    continue
                div = es.dividends_between(divs.get(tk, []), A, A2)
                g = ((exitp + div) / entry - 1.0) * 100.0
                cost = cm.round_trip(tk, "base", spreads)
                fund = hurdle(A, A2)
                long_row = {"date": A, "exit_date": A2, "ticker": tk, "side": "long", "cost_pct": cost,
                            "notional": n4, "gross_pct": g, "exit": "time", "rsi": v}
                rows.append({**long_row, "test": "T4", "financing_pct": fund, "net_pct": g - cost - fund,
                             "pnl_rub": n4 * (g - cost - fund) / 100.0})
                carry = cm.carry_pct(n4, (A2 - A).days) if (A2 - A).days > 0 else 0.0
                rows.append({**long_row, "test": "T4M", "financing_pct": carry, "net_pct": g - cost - carry,
                             "pnl_rub": n4 * (g - cost - carry) / 100.0})
                taken += 1
    return pd.DataFrame(rows), exec_days


# ── Метрики ──────────────────────────────────────────────────────────────────

def metrics(tr: pd.DataFrame, exec_days: list[dt.date], d_from: dt.date, d_to: dt.date, capital: float,
            fund_pct_year: float, n_trials: int, buying_power: float = float("inf")) -> dict:
    cal = (d_to - d_from).days + 1
    if tr.empty:
        return {"trades": 0, "fund_pct_year": fund_pct_year, "account_pct_year": fund_pct_year}
    daily = tr.groupby("exit_date")["pnl_rub"].sum()
    series = daily.reindex(exec_days).fillna(0.0)
    wins, losses = tr.loc[tr["net_pct"] > 0, "net_pct"], tr.loc[tr["net_pct"] <= 0, "net_pct"]
    active = daily[daily != 0]
    t = float(active.mean() / (active.std(ddof=1) / math.sqrt(len(active)))) if len(active) > 2 and active.std(ddof=1) else None
    load = exposure_by_day(tr, exec_days)
    load = load[load > 0] if (load > 0).any() else load
    trade_pct_year = float(tr["pnl_rub"].sum()) / capital * 100.0 * 365.0 / cal
    return {"trades": int(len(tr)), "active_days": int(len(active)), "net_pct_trade": float(tr["net_pct"].mean()),
            "win_rate": float((tr["net_pct"] > 0).mean()),
            "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() else None,
            "pnl_rub": float(tr["pnl_rub"].sum()),
            "pnl_rub_year": float(tr["pnl_rub"].sum()) * 365.0 / cal,
            "trade_pct_year": trade_pct_year, "fund_pct_year": fund_pct_year,
            "account_pct_year": fund_pct_year + trade_pct_year,
            "max_dd_rub": max_drawdown(series), "max_dd_pct": max_drawdown(series) / capital * 100.0,
            "t_active_days": t, "dsr": es.deflated_sharpe(series / capital, max(2, n_trials)),
            "load_avg_rub": float(load.mean()), "load_max_rub": float(load.max()),
            "buying_power_breach_days": int((load > buying_power).sum()),
            "financing_pct_trade": float(tr["financing_pct"].mean()) if "financing_pct" in tr and tr["financing_pct"].notna().any() else None,
            "exits": tr["exit"].value_counts().to_dict()}


def report(res: dict, meta: dict, rules: dict) -> str:
    f = es._f
    L = ["# Тесты идей повышения доходности — обе выборки, один прогон", "",
         f"Сформировано {meta['created']}. Код `{meta['revision']}`, правила `{rules['version']}`. "
         f"Счёт {rules['account']['capital_rub']:,} ₽. Испытаний в реестре: {meta['trials_total']}.".replace(",", " "), "",
         "Итог счёта = доход фонда (на весь капитал) + торговый P&L. Нетто сделки — после издержек "
         "(для T4 — и после упущенного дохода фонда). t — по активным дням.", ""]
    for per in ("dev", "holdout"):
        s = rules["samples"][per]
        fund = res[per]["T0"].get("fund_pct_year")
        L += [f"## {'2024–2026' if per == 'dev' else '2022–2024'} ({s['from']} … {s['to']}); фонд {f(fund, 2)} % в год", "",
              "| тест | сделок | дней | нетто/сделку, % | доля плюсовых | payoff | торговля ₽/год | торговля % капитала/год | "
              "итог счёта %/год | просадка торговли, % | t | DSR | позиция ср./макс., ₽ | дней сверх плеча | финансирование, %/сделку | выходы |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in TESTS:
            m = res[per][k]
            name = f"{k} {rules['tests'][k]['name']}"
            if not m.get("trades"):
                L.append(f"| {name} | 0 | — | — | — | — | — | — | {f(m.get('account_pct_year'), 2)} | — | — | — | — | — | — | — |")
                continue
            exits = "; ".join(f"{a}:{b}" for a, b in m["exits"].items())
            L.append(f"| {name} | {m['trades']} | {m['active_days']} | {f(m['net_pct_trade'])} | {f(m['win_rate'], 2)} | "
                     f"{f(m['payoff'], 2)} | {m['pnl_rub_year']:+,.0f} | {f(m['trade_pct_year'], 2)} | "
                     f"{f(m['account_pct_year'], 2)} | {f(m['max_dd_pct'], 2)} | {f(m['t_active_days'], 1)} | "
                     f"{f(m['dsr'], 2)} | {m['load_avg_rub']:,.0f} / {m['load_max_rub']:,.0f} | "
                     f"{m['buying_power_breach_days']} | {f(m['financing_pct_trade'])} | ".replace(",", " ") + exits + " |")
        L.append("")
    mg = rules["account"]["margin"]
    L += [f"Маржа под 100 % залог паями: дисконт {mg['collateral_discount_pct']} %, покупательная способность "
          f"{mg['buying_power_rub']:,} ₽ (допущение, не условия брокера). «Дней сверх плеча» — дни, когда суммарная "
          "позиция превышала её; сделки при этом не резались.".replace(",", " "),
          "Финансирование: T4 — упущенный доход фонда за удержание; T4M — плата за перенос по тарифу «Премиум».", ""]
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rules = load_rules()
    if os.path.exists(os.path.join(OUT_DIR, "results.json")):
        raise SystemExit("прогон уже выполнен — повтор запрещён правилами")
    lots, blocked = sr.load_lots_and_blocked()
    spreads = cm.load_spreads()
    hurdle = es.Hurdle(es.HURDLE_PATH)
    import database
    conn = database.get_connection()
    try:
        exp, root_of = st.read_contracts()
        mx = [tk for tk, r in root_of.items() if r == rules["tests"]["T3"]["root"]]
        fd = st.load_daily(conn, "research_fut_5m", mx, dt.date(2022, 7, 1), dt.date(2026, 9, 12))
        fd["root"] = fd["ticker"].map(root_of)
        front = st.front_contracts(fd, ll.eligible_until(exp, 2))
        fmap = dict(zip(front["d"], front["front"]))
        fbars = es.load_bars(conn, "research_fut_5m", sorted(set(front["front"])), dt.date(2022, 7, 1), dt.date(2026, 9, 12))
        ema200 = index_ema_above(conn, rules["tests"]["T4"]["index_ema"])
        rev = es._revision()
        total = ll.register("boost-tests", len(TESTS), rev, sprint=5)
        res, frames = {}, []
        for per in ("dev", "holdout"):
            smp = rules["samples"][per]
            d_from, d_to = dt.date.fromisoformat(smp["from"]), dt.date.fromisoformat(smp["to"])
            tr, days = run_period(conn, smp, rules, lots, blocked, spreads, hurdle.growth, es.DIV_PATH,
                                  lambda d: fbars.get(fmap.get(d)), ema200)
            fund_year = hurdle.growth(d_from, d_to) * 365.0 / ((d_to - d_from).days + 1)
            res[per] = {k: metrics(tr[tr["test"] == k] if len(tr) else tr, days, d_from, d_to,
                                   rules["account"]["capital_rub"], fund_year, total,
                                   rules["account"]["margin"]["buying_power_rub"]) for k in TESTS}
            frames.append(tr.assign(period=per))
    finally:
        conn.close()
    meta = {"created": dt.datetime.now().strftime("%d.%m.%Y %H:%M"), "revision": rev, "trials_total": total}
    os.makedirs(OUT_DIR, exist_ok=True)
    text = report(res, meta, rules)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": res}, fh, ensure_ascii=False, indent=1, default=str)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False, float_format="%.5f")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
