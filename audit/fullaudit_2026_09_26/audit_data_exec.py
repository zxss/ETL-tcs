"""Аудит: качество данных и исполнение (только чтение)."""
import sys
sys.path.insert(0, "/opt/etl-tcs")

import numpy as np
import pandas as pd
import database

conn = database.get_connection()
q = lambda s: pd.read_sql(s, conn)

print("=" * 78)
print("ДАННЫЕ: дневные свечи market_data")
md = q("SELECT ticker, date, open, high, low, close, volume, source FROM market_data "
       "WHERE ticker <> 'IMOEX'")
for c in ("open", "high", "low", "close"):
    md[c] = md[c].astype(float)
md["date"] = pd.to_datetime(md["date"])
print(f"  строк {len(md)}, тикеров {md.ticker.nunique()}, "
      f"{md.date.min().date()} … {md.date.max().date()}")
print("  источник бара:", md.source.value_counts(dropna=False).to_dict())
wk = md[md.date.dt.weekday >= 5]
print(f"  баров в сб/вс: {len(wk)} ({wk.date.nunique()} дат) — в обучении отсекаются "
      f"фильтром сессий только при INCLUDE_WEEKEND_TRADING=0")
bad_hl = md[(md.high < md[["open", "close"]].max(axis=1) - 1e-9)
            | (md.low > md[["open", "close"]].min(axis=1) + 1e-9) | (md.high < md.low)]
print(f"  OHLC противоречивы (high<max(o,c), low>min(o,c) или high<low): {len(bad_hl)}")
if len(bad_hl):
    print("   ", bad_hl[["ticker", "date", "open", "high", "low", "close", "source"]]
          .head(5).to_string(index=False).replace("\n", "\n    "))
print(f"  нулевой объём: {(md.volume <= 0).sum()}")
flat = md[(md.open == md.high) & (md.high == md.low) & (md.low == md.close)]
print(f"  «плоские» свечи o=h=l=c: {len(flat)}")
md = md.sort_values(["ticker", "date"])
md["r"] = md.groupby("ticker")["close"].pct_change() * 100
jump = md[md.r.abs() > 15]
print(f"  дневные движения > 15 %: {len(jump)}")
if len(jump):
    print("   ", jump[["ticker", "date", "r", "source"]].head(8)
          .to_string(index=False).replace("\n", "\n    "))
wd = md[md.date.dt.weekday < 5]
cnt = wd.groupby("date").ticker.nunique()
thin = cnt[cnt < 0.9 * cnt.median()]
print(f"  будних дат с неполным покрытием (<90 % тикеров): {len(thin)}")

print()
print("ИСПОЛНЕНИЕ: execution_audit")
ea = q("SELECT * FROM execution_audit")
num = ["final_score", "exp_pnl_pct", "requested_price", "expected_slippage_pct",
       "expected_cost_pct", "filled_price", "slippage_pct", "pnl_gross_rub", "pnl_net_rub",
       "fee_rub", "stop_price", "anchor_price", "exit_price"]
for c in num:
    ea[c] = pd.to_numeric(ea[c], errors="coerce")
print(f"  заявок {len(ea)}, исполнено {ea.filled.sum()}, "
      f"контур {ea.account_env.value_counts().to_dict()}")
print("  по стратегиям:", ea.groupby("strategy").filled.agg(["count", "sum"]).to_dict("index"))
print(f"  без прогноза в записи (exp_pnl_pct пуст): {ea.exp_pnl_pct.isna().sum()} из {len(ea)}; "
      f"без final_score: {ea.final_score.isna().sum()}")
f = ea[ea.filled]
print(f"  проскальзывание факт: среднее {f.slippage_pct.mean():+.3f} %, "
      f"медиана {f.slippage_pct.median():+.3f} %; расчёт {f.expected_slippage_pct.mean():+.3f} %")
print("  причины выхода:", f.exit_reason.value_counts(dropna=False).to_dict())
closed = f.dropna(subset=["pnl_net_rub"])
if len(closed):
    print(f"  закрытых с P&L: {len(closed)}, нетто {closed.pnl_net_rub.sum():+.2f} ₽, "
          f"доля в плюс {np.mean(closed.pnl_net_rub > 0):.0%}")
    print("  нетто по стратегиям:",
          closed.groupby("strategy").pnl_net_rub.agg(["count", "sum", "mean"]).round(2)
          .to_dict("index"))
st = f.dropna(subset=["stop_price", "filled_price"])
if len(st):
    dist = (abs(st.filled_price - st.stop_price) / st.filled_price * 100)
    print(f"  расстояние стопа от цены входа: медиана {dist.median():.2f} %, "
          f"доля на полу 1 %±0,05: {np.mean(abs(dist - 1.0) < 0.05):.0%}")
