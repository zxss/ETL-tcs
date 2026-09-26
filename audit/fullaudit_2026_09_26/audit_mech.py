"""Аудит: чем на самом деле является ночной сигнал в 18:38 (только чтение).

Гипотеза механизма: в 18:38 дня D последний дневной бар в базе — D-1, поэтому
модель прогнозирует ночь D-1 -> D (уже случившуюся), а _adjust_for_realized
вычитает из прогноза фактический утренний гэп дня D (первая 5-минутка D к
close D-1). Тогда ExpPnL ~ прогноз_гэпа_D - факт_гэпа_D - издержки, и ранжирование
почти целиком повторяет -гэп_D.
"""
import glob
import json
import sys
sys.path.insert(0, "/opt/etl-tcs")

import numpy as np
import pandas as pd
import database

conn = database.get_connection()
md = pd.read_sql("SELECT ticker, date, open, close FROM market_data", conn)
md["date"] = pd.to_datetime(md["date"])
md = md.sort_values(["ticker", "date"])
md["close"] = md["close"].astype(float)
md["prev_close"] = md.groupby("ticker")["close"].shift(1)
first5 = pd.read_sql(
    "SELECT DISTINCT ON (ticker, (ts AT TIME ZONE 'Europe/Moscow')::date) ticker, "
    "(ts AT TIME ZONE 'Europe/Moscow')::date AS d, open "
    "FROM market_data_5m ORDER BY ticker, (ts AT TIME ZONE 'Europe/Moscow')::date, ts", conn)
first5["d"] = pd.to_datetime(first5["d"])
first5["open"] = first5["open"].astype(float)
gap = md[["ticker", "date", "prev_close"]].merge(
    first5.rename(columns={"d": "date", "open": "open5"}), on=["ticker", "date"])
gap["gap"] = (gap.open5 / gap.prev_close - 1) * 100          # то, что вычитает поправка

fc = pd.read_sql("SELECT asof_date, ticker, exp_pnl FROM forecasts "
                 "WHERE strategy='long_overnight'", conn)
fc["asof_date"] = pd.to_datetime(fc["asof_date"])
fc["exp_pnl"] = fc["exp_pnl"].astype(float)
x = fc.merge(gap, left_on=["ticker", "asof_date"], right_on=["ticker", "date"])

print("=" * 78)
print("1) ЖУРНАЛ ПРОГНОЗОВ: связь ExpPnL(D) с уже случившимся утренним гэпом D")
rs = []
for d, s in x.groupby("asof_date"):
    s = s.dropna(subset=["exp_pnl", "gap"])
    if len(s) >= 10:
        rs.append((d.date(), s.exp_pnl.rank().corr(s.gap.rank()),
                   np.corrcoef(s.exp_pnl, s.gap)[0, 1], len(s)))
for d, sp, pe, n in rs:
    print(f"   {d}  n={n:>2}  Spearman {sp:+.2f}  Pearson {pe:+.2f}")
sp = np.array([r[1] for r in rs])
print(f"   среднее Spearman {sp.mean():+.3f} по {len(sp)} дням")

print()
print("2) РЕАЛЬНО КУПЛЕННЫЕ НОЧЬЮ БУМАГИ: где они стояли по утреннему гэпу дня D")
rows = []
for base in ("/opt/etl-tcs/audit/stage2-demo", "/opt/etl-tcs/audit/stage2-tournament"):
    for run in sorted(glob.glob(base + "/runs/*OVERNIGHT*")):
        try:
            o = json.load(open(run + "/overnight_orders.json"))
        except Exception:
            continue
        day = pd.Timestamp(run.split("/")[-1][:8])
        tks = [a["ticker"] for a in o.get("accepted", [])]
        g = gap[gap.date == day].dropna(subset=["gap"])
        if not tks or len(g) < 10:
            continue
        g = g.assign(pct=g.gap.rank(pct=True))
        for tk in tks:
            r = g[g.ticker == tk]
            if len(r):
                rows.append((base.split("-")[-1], day.date(), tk,
                             float(r.gap.iloc[0]), float(r.pct.iloc[0]), float(g.gap.median())))
for acc, d, tk, gp, pct, med in rows:
    print(f"   {acc:<10} {d}  {tk:<6} гэп {gp:+6.2f} %  перцентиль {pct:4.0%}  (медиана дня {med:+.2f} %)")
if rows:
    pcts = np.array([r[4] for r in rows])
    print(f"   купленных {len(pcts)}: средний перцентиль гэпа {pcts.mean():.0%} "
          f"(случайный выбор дал бы ~50 %), в нижней трети {np.mean(pcts <= 1/3):.0%}")
