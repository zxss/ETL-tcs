"""Аудит: цель модели против окна, которое реально торгуется (только чтение).

r_model = open(первая сделка D+1) / close(последняя сделка D) - 1   — то, на чём
          учится TFT и что проверяет контур валидации (дневные свечи API);
r_trade = цена 09:10 D+1 / цена 18:45 D - 1                          — то, что
          получает long_overnight (вход 18:38–18:50, выход CLOSE 09:10).
"""
import sys
sys.path.insert(0, "/opt/etl-tcs")

import numpy as np
import pandas as pd
import database

conn = database.get_connection()
SQL = """
WITH b AS (
  SELECT ticker, (ts AT TIME ZONE 'Europe/Moscow') AS t, open, close
  FROM market_data_5m
  WHERE ts >= TIMESTAMPTZ '2025-03-01'
    AND extract(isodow FROM (ts AT TIME ZONE 'Europe/Moscow')) < 6
    AND (ts AT TIME ZONE 'Europe/Moscow')::time >= TIME '06:50'
)
SELECT ticker, t::date AS d,
  (array_agg(open  ORDER BY t))[1]                                        AS first_open,
  (array_agg(close ORDER BY t DESC))[1]                                   AS last_close,
  (array_agg(close ORDER BY t DESC) FILTER (WHERE t::time <= TIME '18:45'))[1] AS c1845,
  (array_agg(open  ORDER BY t)      FILTER (WHERE t::time >= TIME '09:10'))[1] AS o0910
FROM b GROUP BY 1, 2
"""
x = pd.read_sql(SQL, conn)
for c in ("first_open", "last_close", "c1845", "o0910"):
    x[c] = x[c].astype(float)
x = x.sort_values(["ticker", "d"])
g = x.groupby("ticker")
x["n_first_open"] = g["first_open"].shift(-1)
x["n_o0910"] = g["o0910"].shift(-1)
x = x.dropna()
x["r_model"] = (x.n_first_open / x.last_close - 1) * 100
x["r_trade"] = (x.n_o0910 / x.c1845 - 1) * 100
x["r_evening"] = (x.last_close / x.c1845 - 1) * 100          # 18:45 -> 23:45
x["r_morning"] = (x.n_o0910 / x.n_first_open - 1) * 100      # 06:55 -> 09:10
x = x[(x.r_trade.abs() < 20) & (x.r_model.abs() < 20)]         # отсечь отсечки/сплиты

print(f"наблюдений (тикер-ночь): {len(x)}, тикеров {x.ticker.nunique()}, "
      f"{x.d.min()} … {x.d.max()}")
print(f"корреляция r_model и r_trade: {np.corrcoef(x.r_model, x.r_trade)[0, 1]:+.3f}")
v = x[["r_evening", "r_model", "r_morning"]].var()
tot = x.r_trade.var()
print("доля дисперсии торгуемого окна 18:45→09:10 (приближённо, без ковариаций):")
print(f"   вечерняя сессия 18:45→23:45 : {v.r_evening / tot:6.1%}")
print(f"   закрытая биржа 23:45→06:55  : {v.r_model / tot:6.1%}   ← это и есть цель модели")
print(f"   утренняя сессия 06:55→09:10 : {v.r_morning / tot:6.1%}")
print(f"доля ночей, где первая сделка D+1 равна закрытию D (r_model = 0): "
      f"{np.mean(x.r_model.abs() < 1e-9):.0%}")
print(f"средние: r_model {x.r_model.mean():+.3f} %, r_trade {x.r_trade.mean():+.3f} %, "
      f"вечер {x.r_evening.mean():+.3f} %, утро {x.r_morning.mean():+.3f} %")
