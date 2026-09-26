"""Аудит: ex-ante проверка боевой модели по журналу forecasts (только чтение).

Для каждого прогноза, сделанного в торговый день D (19:30 МСК, данные по D
включительно), берём фактический исход следующего торгового дня N:
  ночь      = (open_N + дивиденд_с_гэпом_на_N) / close_D - 1
  день      = close_N / open_N - 1
  коридор   = попали ли low_N и high_N в [q10, q90] (ценовые)
Считаем кросс-секционный Rank IC по дням, превышение топ-5 над вселенной,
калибровку ExpPnL и покрытие коридора.
"""
import sys
sys.path.insert(0, "/opt/etl-tcs")

import numpy as np
import pandas as pd
import database

COST = 0.128
conn = database.get_connection()
fc = pd.read_sql(
    "SELECT asof_date, ticker, strategy, anchor_price, q10, q50, q90, exp_pnl, "
    "prob_profit, final_score FROM forecasts "
    "WHERE strategy IN ('long_overnight','intraday_short')", conn)
md = pd.read_sql("SELECT ticker, date, open, high, low, close FROM market_data", conn)
dv = pd.read_sql("SELECT ticker, last_buy_date, dividend_net FROM dividends", conn)
for c in ("open", "high", "low", "close"):
    md[c] = md[c].astype(float)
for c in ("anchor_price", "q10", "q50", "q90", "exp_pnl", "prob_profit", "final_score"):
    fc[c] = fc[c].astype(float)
md["date"] = pd.to_datetime(md["date"])
# Робот торгует только в будни, а контур отфильтровывает бары выходных:
# «следующий день» для пятницы — понедельник, как у контура.
md = md[md["date"].dt.weekday < 5]
fc["asof_date"] = pd.to_datetime(fc["asof_date"])
fc = fc[fc["asof_date"].dt.weekday < 5]
dv["last_buy_date"] = pd.to_datetime(dv["last_buy_date"])

# следующий торговый бар по тикеру
md = md.sort_values(["ticker", "date"])
g = md.groupby("ticker")
for c in ("date", "open", "high", "low", "close"):
    md["n_" + c] = g[c].shift(-1)
nxt = md.rename(columns={"date": "asof_date"})

df = fc.merge(nxt, on=["ticker", "asof_date"], how="inner")   # D должен быть торговым днём
df = df.dropna(subset=["n_open", "n_close"])

# дивиденд, чей гэп приходится на N: last_buy_date в [D, N)
def div_on(row):
    m = dv[(dv.ticker == row.ticker) & (dv.last_buy_date >= row.asof_date)
           & (dv.last_buy_date < row.n_date)]
    return float(m.dividend_net.sum()) if len(m) else 0.0
df["div"] = df.apply(div_on, axis=1)
df["ovn"] = ((df.n_open + df["div"]) / df.close - 1) * 100
df["intra"] = (df.n_close / df.n_open - 1) * 100

def ic_block(sub, score, real, label):
    ics = []
    for d, s in sub.groupby("asof_date"):
        s = s.dropna(subset=[score, real])
        if len(s) >= 10 and s[score].nunique() > 2:
            ics.append(s[score].rank().corr(s[real].rank()))
    ics = np.array(ics)
    if len(ics) < 2:
        print(f"  {label:<34} мало дней ({len(ics)})")
        return
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    print(f"  {label:<34} дней {len(ics):>2}  IC {ics.mean():+.3f}  t {t:+.2f}  "
          f"доля IC>0 {np.mean(ics > 0):.0%}")

print("=" * 78)
lo = df[df.strategy == "long_overnight"].copy()
lo["real"] = lo["ovn"]
print(f"LONG_OVERNIGHT: прогнозов с исходом {len(lo)}, торговых дней {lo.asof_date.nunique()}, "
      f"период {lo.asof_date.min().date()} … {lo.asof_date.max().date()}")
for sc in ("exp_pnl", "prob_profit", "final_score"):
    ic_block(lo, sc, "real", f"Rank IC {sc} → ночь")

# топ-5 по final_score после порога edge (как в контуре) против вселенной
ex, ex_raw = [], []
for d, s in lo.groupby("asof_date"):
    elig = s[s.exp_pnl > 0.5 * COST].sort_values("final_score", ascending=False).head(5)
    raw = s.sort_values("final_score", ascending=False).head(5)
    if len(elig):
        ex.append(elig.real.mean() - s.real.mean())
    ex_raw.append(raw.real.mean() - s.real.mean())
for name, arr in (("топ-5 с порогом edge − вселенная", ex), ("топ-5 без порога − вселенная", ex_raw)):
    arr = np.array(arr)
    if len(arr) >= 2:
        t = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))
        print(f"  {name:<34} дней {len(arr):>2}  {arr.mean():+.3f} %/ночь  t {t:+.2f}")
    else:
        print(f"  {name:<34} дней {len(arr)}")

print(f"  калибровка: средний ExpPnL {lo.exp_pnl.mean():+.3f} %  (нетто)")
print(f"              средняя ночь {lo.real.mean():+.3f} % брутто, "
      f"{lo.real.mean() - COST:+.3f} % нетто")
print(f"              средний ProbProfit {lo.prob_profit.mean():.3f}, "
      f"факт P(ночь > издержек) {np.mean(lo.real > COST):.3f}")
cov = ((lo.n_low >= lo.q10) & (lo.n_high <= lo.q90)).mean()
print(f"  покрытие коридора [q10,q90] следующим днём: {cov:.1%} (номинал модели ~80 %)")
print(f"  медианная ширина коридора {((lo.q90 - lo.q10) / lo.anchor_price * 100).median():.2f} %, "
      f"медианный факт. диапазон дня {((lo.n_high - lo.n_low) / lo.close * 100).median():.2f} %")

print("=" * 78)
sh = df[df.strategy == "intraday_short"].copy()
sh["real"] = -sh["intra"]
print(f"INTRADAY_SHORT: прогнозов {len(sh)}, дней {sh.asof_date.nunique()}")
for sc in ("exp_pnl", "final_score"):
    ic_block(sh, sc, "real", f"Rank IC {sc} → шорт-день")
print(f"  средний ExpPnL {sh.exp_pnl.mean():+.3f} %, факт {sh.real.mean() - COST:+.3f} % нетто")
