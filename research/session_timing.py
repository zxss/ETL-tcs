"""
Тайминг сессий для Спринта 2 (ТЗ 50D, товарный lead-lag): когда реально
начинаются сделки у акций и фьючерсов и где лежит объём. Только описательная
статистика по времени и объёму — доходности гипотез здесь не считаются
(гипотезы 2А/2Б регистрируются после этого замера).

Ряды:
  акции — research_bars_5m (до 2024-05-20, отложенная выборка) и market_data_5m (дальше);
  фьючерсы — research_fut_5m, ближний контракт по объёму ПРЕДЫДУЩЕГО дня среди
  не истекших (выбор известен до открытия дня — без заглядывания вперёд);
  спот CNYRUB_TOM — как есть.
Бар — время начала 5-минутки по Москве; сделка аукциона открытия 09:59:xx
попадает в бар 09:55. «Первая сделка» дня — первый бар с объёмом.

Склейка: доходность окна всегда считается внутри одного контракта, выбранного
на день заранее; разрыв между контрактами в день смены (rolls.csv) показывает,
почему наивная склейка цен недопустима.

Отчёт: audit/r4_research/sprint2/timing/ (report.md, monthly.csv, lead_monthly.csv,
rolls.csv, front.csv). Запуск (на сервере): python -m research.session_timing
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import futures_loader as fl                # noqa: E402
from research import session_calendar as sc              # noqa: E402

log = logging.getLogger("research.session_timing")

OUT_DIR = os.path.join(ROOT, "audit", "r4_research", "sprint2", "timing")
HOLDOUT_END = dt.date(2024, 5, 20)
STOCKS = ("LKOH", "ROSN", "TATN", "NVTK", "PLZL", "SELG", "GMKN", "CHMF", "NLMK", "SBER")
STOCK_GROUP = "АКЦИИ"
# Корзины времени по началу бара (Москва); конец None — до конца суток.
BUCKETS = (("night", "00:00", "07:00"), ("m0700", "07:00", "09:00"), ("m0900", "09:00", "09:50"),
           ("auction", "09:50", "10:00"), ("main", "10:00", "19:00"), ("evening", "19:00", None))
PRE_AUCTION = ("night", "m0700", "m0900")                # всё до 09:50

DAILY_SQL = """
SELECT ticker, d, min(t) AS first_bar, max(t) AS last_bar, sum(volume) AS vol, count(*) AS bars,
       (array_agg(close ORDER BY ts DESC))[1] AS last_close, {buckets}
FROM (SELECT ticker, ts, close, volume,
             (ts AT TIME ZONE 'Europe/Moscow')::date AS d,
             (ts AT TIME ZONE 'Europe/Moscow')::time AS t
      FROM {table}
      WHERE ticker = ANY(%(tickers)s) AND ts >= %(f)s AND ts < %(t)s AND volume > 0) s
GROUP BY ticker, d
"""


def _bucket_sql() -> str:
    parts = []
    for name, lo, hi in BUCKETS:
        cond = f"t >= '{lo}'" + (f" AND t < '{hi}'" if hi else "")
        parts.append(f"coalesce(sum(volume) FILTER (WHERE {cond}), 0) AS v_{name}")
    return ", ".join(parts)


def _minutes(t: dt.time) -> int:
    return t.hour * 60 + t.minute


def hhmm(m) -> str:
    return "" if m is None or pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def _q(s: pd.Series, q: float):
    s = s.dropna()
    return s.quantile(q, interpolation="lower") if len(s) else float("nan")


def load_daily(conn, table: str, tickers: list[str], d_from: dt.date, d_to: dt.date) -> pd.DataFrame:
    msk = dt.timezone(dt.timedelta(hours=3))
    df = pd.read_sql(DAILY_SQL.format(table=table, buckets=_bucket_sql()), conn, params={
        "tickers": list(tickers),
        "f": dt.datetime.combine(d_from, dt.time(), msk),
        "t": dt.datetime.combine(d_to + dt.timedelta(days=1), dt.time(), msk)})
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    for c in ["vol", "bars", "last_close"] + [f"v_{b}" for b, *_ in BUCKETS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["first_m"] = [_minutes(t) for t in df["first_bar"]]
    df["last_m"] = [_minutes(t) for t in df["last_bar"]]
    df["v_pre"] = df[[f"v_{b}" for b in PRE_AUCTION]].sum(axis=1)
    return df


def front_contracts(daily: pd.DataFrame, expirations: dict) -> pd.DataFrame:
    """Ближний контракт корня на день d: максимум объёма за предыдущий торговый день
    корня среди не истекших к d. Первый день ряда — по объёму того же дня."""
    rows = []
    for root, g in daily.groupby("root"):
        vol = g.pivot_table(index="d", columns="ticker", values="vol", aggfunc="sum").sort_index()
        prev = None
        for d in vol.index:
            alive = [tk for tk in vol.columns if expirations.get(tk) is None or expirations[tk] >= d]
            today = vol.loc[d].reindex(alive).dropna()
            today = today[today > 0]
            ref = prev.reindex(alive).dropna() if prev is not None else today
            ref = ref[ref > 0]
            if ref.empty:
                ref = today
            prev = vol.loc[d]
            if ref.empty:
                continue
            front = ref.idxmax()
            total = float(today.sum())
            rows.append({"root": root, "d": d, "front": front,
                         "front_share": float(today.get(front, 0.0)) / total if total > 0 else float("nan"),
                         "same_day_max": today.idxmax() if not today.empty else None})
    return pd.DataFrame(rows, columns=["root", "d", "front", "front_share", "same_day_max"])


def rolls(front: pd.DataFrame, daily: pd.DataFrame, expirations: dict) -> pd.DataFrame:
    """Смены ближнего контракта: остаток до экспирации старого, «назад» (новый истекает
    раньше старого — дребезг), разрыв цен новый/старый по закрытию предыдущего дня."""
    close = {(r.ticker, r.d): r.last_close for r in daily.itertuples()}
    out = []
    for root, g in front.sort_values("d").groupby("root"):
        prev_front = prev_d = None
        for r in g.itertuples():
            if prev_front is not None and r.front != prev_front:
                old_exp, new_exp = expirations.get(prev_front), expirations.get(r.front)
                c_old, c_new = close.get((prev_front, prev_d)), close.get((r.front, prev_d))
                basis = ((float(c_new) / float(c_old) - 1) * 100
                         if pd.notna(c_old) and pd.notna(c_new) and c_old else float("nan"))
                out.append({"root": root, "d": r.d, "old": prev_front, "new": r.front,
                            "days_to_exp_old": (old_exp - r.d).days if old_exp else None,
                            "backward": bool(old_exp and new_exp and new_exp < old_exp),
                            "basis_pct": basis})
            prev_front, prev_d = r.front, r.d
    return pd.DataFrame(out, columns=["root", "d", "old", "new", "days_to_exp_old", "backward", "basis_pct"])


def _period(d: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(d).dt.to_period(freq).astype(str)


def timing_table(daily: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """По серии и периоду (только будни): медиана и 10-й перцентиль первой сделки,
    медиана последней, доля дней со сделками до аукциона открытия, медианные доли
    дневного объёма по корзинам; отдельно — число дней выходных сессий."""
    df = daily.copy()
    df["period"] = _period(df["d"], freq)
    df["weekend"] = pd.to_datetime(df["d"]).dt.weekday >= 5
    df["pre_auction"] = [m < _minutes(sc.session(d).opening_auction) for m, d in zip(df["first_m"], df["d"])]
    vol = df["vol"].where(df["vol"] > 0)
    shares = [f"v_{b}" for b, *_ in BUCKETS] + ["v_pre"]
    for c in shares:
        df["share" + c[1:]] = df[c] / vol
    wk = df[~df["weekend"]]
    agg = wk.groupby(["series", "period"]).agg(
        days=("d", "nunique"),
        first_med=("first_m", lambda s: _q(s, 0.5)),
        first_p10=("first_m", lambda s: _q(s, 0.1)),
        last_med=("last_m", lambda s: _q(s, 0.5)),
        pre_auction=("pre_auction", "mean"),
        **{"share" + c[1:]: ("share" + c[1:], "median") for c in shares})
    wkend = df[df["weekend"]].groupby(["series", "period"])["d"].nunique().rename("weekend_days")
    return agg.join(wkend).fillna({"weekend_days": 0}).reset_index()


def lead_by_day(stock_daily: pd.DataFrame, fut_daily: pd.DataFrame) -> pd.DataFrame:
    """На каждый будний день: медиана первой сделки по корзине акций против первой
    сделки ряда фьючерса. lead_first > 0 — фьючерс торгуется раньше акций (минут);
    lead_auction — от первой сделки фьючерса до начала аукциона открытия акций."""
    s = stock_daily.groupby("d")["first_m"].median().rename("stock_first").reset_index()
    f = fut_daily[["series", "d", "first_m"]].rename(columns={"first_m": "fut_first"})
    out = f.merge(s, on="d", how="inner")
    out = out[pd.to_datetime(out["d"]).dt.weekday < 5].copy()
    out["auction"] = [_minutes(sc.session(d).opening_auction) for d in out["d"]]
    out["lead_first"] = out["stock_first"] - out["fut_first"]
    out["lead_auction"] = out["auction"] - out["fut_first"]
    return out


def lead_table(lead: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = lead.assign(period=_period(lead["d"], freq), fut_earlier=lead["lead_first"] > 0)
    return df.groupby(["series", "period"]).agg(
        days=("d", "nunique"), lead_first_med=("lead_first", "median"),
        lead_auction_med=("lead_auction", "median"), fut_earlier=("fut_earlier", "mean")).reset_index()


def roll_stats(r: pd.DataFrame, front: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, g in front.groupby("root"):
        rr = r[r["root"] == root]
        rows.append({"root": root, "days": len(g), "rolls": len(rr), "backward": int(rr["backward"].sum()),
                     "days_to_exp_med": rr["days_to_exp_old"].median() if len(rr) else float("nan"),
                     "basis_abs_med": rr["basis_pct"].abs().median() if len(rr) else float("nan"),
                     "basis_abs_max": rr["basis_pct"].abs().max() if len(rr) else float("nan"),
                     "front_share_med": g["front_share"].median(),
                     "same_day_agree": (g["front"] == g["same_day_max"]).mean()})
    return pd.DataFrame(rows)


def read_contracts(path: str = fl.CONTRACTS_PATH) -> tuple[dict, dict]:
    exp, root = {}, {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp[row["ticker"]] = dt.date.fromisoformat(row["expiration"]) if row["expiration"] else None
            root[row["ticker"]] = row["root"]
    return exp, root


def _pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.0f} %"


def report(tq: pd.DataFrame, lq: pd.DataFrame, rs: pd.DataFrame, series: list[str]) -> str:
    t = tq.set_index(["series", "period"])
    periods = sorted(tq["period"].unique())
    lines = ["# Спринт 2 — тайминг сессий акций и фьючерсов (2022–2026)", "",
             "Описательный замер: время первой сделки и распределение объёма. Доходности гипотез не "
             "считались. Акции — корзина " + ", ".join(STOCKS) + "; фьючерс — ближний по объёму "
             "предыдущего дня. Время — начало 5-минутного бара по Москве, только будни.", "",
             "## Первая сделка (медиана) · доля дневного объёма до 09:50", "",
             "| квартал | " + " | ".join(series) + " |", "|---" * (len(series) + 1) + "|"]
    for p in periods:
        cells = []
        for s in series:
            if (s, p) in t.index:
                x = t.loc[(s, p)]
                cells.append(f"{hhmm(x['first_med'])} · {_pct(x['share_pre'])}")
            else:
                cells.append("")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lr = lq.set_index(["series", "period"])
    roots = [s for s in series if s != STOCK_GROUP]
    lines += ["", "## Опережение: первая сделка акций − первая сделка фьючерса, минут (доля дней, где фьючерс раньше)",
              "", "Положительное — фьючерс начинает раньше акций; отрицательное — акции раньше.", "",
              "| квартал | " + " | ".join(roots) + " |", "|---" * (len(roots) + 1) + "|"]
    for p in periods:
        cells = []
        for s in roots:
            if (s, p) in lr.index:
                x = lr.loc[(s, p)]
                cells.append(f"{x['lead_first_med']:+.0f} ({_pct(x['fut_earlier'])})")
            else:
                cells.append("")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines += ["", "## Склейка ближнего контракта (по объёму предыдущего дня)", "",
              "| корень | дней | смен | назад | до экспирации, дн (мед.) | разрыв |%| мед. / макс. "
              "| доля объёма ближнего (мед.) | совпадает с максимумом дня |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rs.itertuples():
        lines.append(f"| {r.root} | {r.days} | {r.rolls} | {r.backward} | {r.days_to_exp_med:.0f} | "
                     f"{r.basis_abs_med:.2f} / {r.basis_abs_max:.2f} | {_pct(r.front_share_med)} | "
                     f"{_pct(r.same_day_agree)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Тайминг сессий акций и фьючерсов (Спринт 2)")
    ap.add_argument("--from", dest="date_from", default="2022-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-09-11")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d_from, d_to = dt.date.fromisoformat(a.date_from), dt.date.fromisoformat(a.date_to)
    exp, root_of = read_contracts()
    import database
    conn = database.get_connection()
    try:
        stocks = pd.concat([
            load_daily(conn, "research_bars_5m", list(STOCKS), d_from, min(d_to, HOLDOUT_END)),
            load_daily(conn, "market_data_5m", list(STOCKS), HOLDOUT_END + dt.timedelta(days=1), d_to)])
        fut = load_daily(conn, "research_fut_5m", list(exp), d_from, d_to)
    finally:
        conn.close()
    fut["root"] = fut["ticker"].map(root_of)
    front = front_contracts(fut, exp)
    r = rolls(front, fut, exp)
    fut_front = fut.merge(front[["root", "d", "front"]], left_on=["root", "ticker", "d"],
                          right_on=["root", "front", "d"]).assign(series=lambda x: x["root"])
    stock_series = pd.concat([stocks.assign(series=stocks["ticker"]), stocks.assign(series=STOCK_GROUP)])
    all_series = pd.concat([stock_series, fut_front], ignore_index=True)
    lead = lead_by_day(stocks, fut_front)
    os.makedirs(a.out, exist_ok=True)
    tm = timing_table(all_series, "M")
    for c in ("first_med", "first_p10", "last_med"):
        tm[c] = tm[c].map(hhmm)
    tm.to_csv(os.path.join(a.out, "monthly.csv"), index=False, float_format="%.4f")
    lead_table(lead, "M").to_csv(os.path.join(a.out, "lead_monthly.csv"), index=False, float_format="%.3f")
    r.to_csv(os.path.join(a.out, "rolls.csv"), index=False, float_format="%.4f")
    front.to_csv(os.path.join(a.out, "front.csv"), index=False, float_format="%.4f")
    series = [STOCK_GROUP] + [s for s in list(fl.ROOTS) + list(fl.SPOT) if s in set(fut_front["series"])]
    text = report(timing_table(all_series, "Q"), lead_table(lead, "Q"), roll_stats(r, front), series)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    log.info("готово: %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
