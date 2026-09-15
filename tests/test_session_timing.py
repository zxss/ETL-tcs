"""
Тайминг сессий Спринта 2 (research/session_timing.py). Без сети и БД, синтетика.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import session_timing as st                # noqa: E402

D1, D2, D3, D4 = (dt.date(2023, 3, 6) + dt.timedelta(days=i) for i in range(4))
EXP = {"BRJ3": dt.date(2023, 3, 8), "BRK3": dt.date(2023, 4, 3), "CNYRUB_TOM": None}


def day(ticker, d, vol, close=100.0, first=dt.time(9, 0), last=dt.time(23, 45), **v):
    row = {"ticker": ticker, "d": d, "first_bar": first, "last_bar": last, "vol": vol, "bars": 10,
           "last_close": close}
    for b, *_ in st.BUCKETS:
        row[f"v_{b}"] = v.get(b, 0)
    return row


def frame(rows, root=True):
    df = st.prepare(pd.DataFrame(rows))
    if root:
        df["root"] = df["ticker"].str[:2]
    return df


class TestFront(unittest.TestCase):

    def setUp(self):
        self.daily = frame([
            day("BRJ3", D1, 100, close=80.0), day("BRK3", D1, 10, close=81.0),
            day("BRJ3", D2, 10, close=80.0), day("BRK3", D2, 100, close=82.0),
            day("BRJ3", D3, 50), day("BRK3", D3, 60),
            day("BRK3", D4, 70), day("BRJ3", D4, 500),                       # BRJ3 истёк 08.03
        ])
        self.front = st.front_contracts(self.daily, EXP)

    def test_previous_day_volume_no_lookahead(self):
        f = dict(zip(self.front["d"], self.front["front"]))
        self.assertEqual(f[D1], "BRJ3")               # первый день — объём того же дня
        self.assertEqual(f[D2], "BRJ3")               # по объёму D1, хотя в D2 больше BRK3
        self.assertEqual(f[D3], "BRK3")
        self.assertEqual(f[D4], "BRK3")               # истёкший не выбирается даже при объёме
        agree = dict(zip(self.front["d"], self.front["front"] == self.front["same_day_max"]))
        self.assertFalse(agree[D2])

    def test_roll_basis_from_previous_close(self):
        r = st.rolls(self.front, self.daily, EXP)
        self.assertEqual(len(r), 1)
        x = r.iloc[0]
        self.assertEqual((x["old"], x["new"], x["d"]), ("BRJ3", "BRK3", D3))
        self.assertEqual(x["days_to_exp_old"], 0)     # смена в день экспирации старого
        self.assertAlmostEqual(x["basis_pct"], 2.5)   # 82 / 80 на закрытии D2
        self.assertFalse(x["backward"])


class TestTiming(unittest.TestCase):

    def test_medians_shares_and_weekend(self):
        sat = dt.date(2023, 3, 11)
        df = frame([
            day("LKOH", D1, 100, first=dt.time(7, 0), m0700=20, main=80),
            day("LKOH", D2, 100, first=dt.time(9, 55), auction=10, main=90),
            day("LKOH", D3, 100, first=dt.time(10, 0), main=100),
            day("LKOH", sat, 5, first=dt.time(10, 0), main=5),
        ], root=False).assign(series="LKOH")
        t = st.timing_table(df, "M").iloc[0]
        self.assertEqual(t["days"], 3)
        self.assertEqual(st.hhmm(t["first_med"]), "09:55")
        self.assertAlmostEqual(t["pre_auction"], 1 / 3)
        self.assertAlmostEqual(t["share_pre"], 0.0)        # медиана по дням: 0,2 / 0 / 0
        self.assertEqual(t["weekend_days"], 1)

    def test_lead_sign(self):
        stocks = frame([day("LKOH", D1, 1, first=dt.time(7, 0)), day("SBER", D1, 1, first=dt.time(7, 10)),
                        day("LKOH", D2, 1, first=dt.time(9, 55))], root=False)
        fut = frame([day("BRK3", D1, 1, first=dt.time(9, 0)), day("BRK3", D2, 1, first=dt.time(9, 0))],
                    root=False).assign(series="BR")
        lead = st.lead_by_day(stocks, fut).set_index("d")
        self.assertEqual(lead.at[D1, "lead_first"], -115)  # акции (медиана 07:05) раньше фьючерса
        self.assertEqual(lead.at[D2, "lead_first"], 55)
        self.assertEqual(lead.at[D2, "lead_auction"], 50)  # до аукциона 09:50

    def test_new_schedule_auction(self):
        stocks = frame([day("LKOH", dt.date(2026, 9, 14), 1, first=dt.time(9, 0))], root=False)
        fut = frame([day("BRX6", dt.date(2026, 9, 14), 1, first=dt.time(7, 0))], root=False).assign(series="BR")
        self.assertEqual(st.lead_by_day(stocks, fut).iloc[0]["lead_auction"], 120)   # аукцион 09:00


if __name__ == "__main__":
    unittest.main()
