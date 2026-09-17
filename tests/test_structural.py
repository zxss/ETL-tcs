"""
Ретро-тест структурных моделей (research/structural). Синтетика, без сети и БД.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.structural import common as cmn            # noqa: E402
from research.structural import test_cash_and_carry as cc     # noqa: E402
from research.structural import test_dividend_gap as dg       # noqa: E402
from research.structural import test_index_rebalance as ir    # noqa: E402
from research.structural import test_totm as tt          # noqa: E402

DAYS = [d.date() for d in pd.bdate_range("2025-01-27", "2025-03-07")]


class TestTotm(unittest.TestCase):

    def test_windows_and_control(self):
        w = tt.totm_windows(DAYS)
        self.assertEqual([x["month"] for x in w], ["2025-01", "2025-02"])
        self.assertEqual((w[0]["t_m2"], w[0]["t_m1"], w[0]["t_p1"], w[0]["t_p3"]),
                         (dt.date(2025, 1, 30), dt.date(2025, 1, 31), dt.date(2025, 2, 3), dt.date(2025, 2, 5)))
        ctl = tt.control_windows(DAYS, w)
        blocked = {d for x in w for d in DAYS if x["t_m2"] <= d <= x["t_p3"]}
        for d0, d1 in ctl:
            self.assertTrue(all(d not in blocked for d in DAYS if d0 <= d <= d1))
            self.assertEqual(DAYS.index(d1) - DAYS.index(d0), 4)

    def test_month_gap_skipped(self):
        # март 2022 без торгов акциями: граница февраль→апрель — не окно TOTM
        days = [dt.date(2022, 2, 24), dt.date(2022, 2, 25), dt.date(2022, 4, 1), dt.date(2022, 4, 4),
                dt.date(2022, 4, 5), dt.date(2022, 5, 4), dt.date(2022, 5, 5), dt.date(2022, 5, 6)]
        self.assertEqual([x["month"] for x in tt.totm_windows(days)], ["2022-04"])


class TestCashAndCarry(unittest.TestCase):

    def test_implied_rate(self):
        self.assertAlmostEqual(cc.implied_rate_pct(102.0, 100.0, 0.0, 73), 10.0)
        self.assertAlmostEqual(cc.implied_rate_pct(99.0, 100.0, 3.0, 73), 10.0)   # дивиденд до экспирации


class TestDividendGap(unittest.TestCase):

    def frame(self, rows):
        return pd.DataFrame(rows, columns=["date", "open", "high", "close"]).set_index("date")

    def test_take_profit_and_skips(self):
        d = DAYS[:25]
        base = [(x, 100.0, 100.5, 100.0) for x in d[:5]] + [(x, 91.0, 92.0, 91.5) for x in d[5:]]
        base[5] = (d[5], 90.0, 91.0, 90.5)          # T_ex: гэп 10 → тейк 98
        base[6] = (d[6], 90.5, 91.0, 91.0)          # вход 91
        base[9] = (d[9], 95.0, 98.5, 97.0)          # high ≥ 98 → тейк
        r = dg.gap_trade(self.frame(base), d, d[5], max_sessions=15)
        self.assertEqual((r["entry"], r["exit"], r["reason"], r["exit_day"]), (91.0, 98.0, "тейк", d[9]))
        closed = list(base); closed[6] = (d[6], 97.0, 99.0, 99.0)
        self.assertEqual(dg.gap_trade(self.frame(closed), d, d[5], max_sessions=15)["skip"], "гэп закрылся до входа")
        up = list(base); up[5] = (d[5], 101.0, 102.0, 101.0)
        self.assertEqual(dg.gap_trade(self.frame(up), d, d[5], max_sessions=15)["skip"], "нет гэпа")

    def test_time_stop(self):
        d = DAYS[:25]
        rows = [(x, 100.0, 100.5, 100.0) for x in d]
        rows[5] = (d[5], 90.0, 91.0, 90.5)
        rows = rows[:6] + [(x, 91.0, 92.0, 91.5) for x in d[6:]]
        r = dg.gap_trade(self.frame(rows), d, d[5], max_sessions=10)
        self.assertEqual((r["reason"], r["exit_day"]), ("тайм-стоп", d[16]))


class TestImoexReviews(unittest.TestCase):

    def test_title_filter(self):
        from research.structural import fetch_imoex_reviews as fr
        yes = ["Новые базы расчета индексов Московской Биржи", "Новые базы расчета индексов Московской биржи",
               "Об изменении баз расчета индексов акций", "О внеочередном пересмотре баз расчета индексов акций",
               "Московская биржа включит акции Ленты в Индекс МосБиржи",
               "Акции ДОМ.РФ, Озон и ЦИАН войдут в Индекс МосБиржи"]
        no = ["О базе расчета Индекса МосБиржи IPO", "Новые базы расчета Индексов МосБиржи – РСПП",
              "Новые параметры базы расчета Индекса московской недвижимости ДомКлик",
              "Московская биржа включила акции ДОМ.РФ в Индекс МосБиржи создания стоимости",
              "О внеочередном пересмотре баз расчета индексов облигаций",
              "Новые параметры базы расчета Индекса МосБиржи голубых фишек"]
        self.assertTrue(all(fr.is_imoex_review(t) for t in yes))
        self.assertFalse(any(fr.is_imoex_review(t) for t in no))

    def test_renames_and_earliest_announcement(self):
        from research.structural import fetch_imoex_reviews as fr
        self.assertEqual(fr.split_renames(["T", "HEAD"], ["TCSG"]), (["HEAD"], [], ["TCSG->T"]))
        ev = [{"eff_date": "2025-12-19"}]
        news = [{"id": 1, "title": "О базе расчета Индекса МосБиржи IPO", "published": "2025-12-16 10:00:00"},
                {"id": 2, "title": "Новые базы расчета индексов Московской Биржи", "published": "2025-12-05 19:00:00"},
                {"id": 3, "title": "Акции ДОМ.РФ, Озон и ЦИАН войдут в Индекс МосБиржи", "published": "2025-12-06 09:00:00"}]
        e = fr.attach_announcements(ev, news)[0]
        self.assertEqual((e["ann_date"], e["ann_url"]), ("2025-12-05", "https://www.moex.com/n2"))


class TestIndexAndSummary(unittest.TestCase):

    def test_event_days(self):
        self.assertEqual(ir.event_days(dt.date(2025, 2, 7), dt.date(2025, 2, 21), DAYS),
                         (dt.date(2025, 2, 10), dt.date(2025, 2, 20)))
        self.assertIsNone(ir.event_days(dt.date(2025, 2, 20), dt.date(2025, 2, 21), DAYS))

    def test_summarize_criterion(self):
        rows = [cmn.trade("m", "v", "X", DAYS[i], DAYS[i + 2], 100000, 1.5 + 0.1 * (i % 3), 0.2, 0.1)
                for i in range(0, 30, 3)]
        s = cmn.summarize(pd.DataFrame(rows), DAYS[0], DAYS[-1], 5e6, {"excess_annual_min_pct": 2.0, "t_min": 2.0})
        self.assertEqual(s["trades"], 10)
        self.assertTrue(s["passed"])
        bad = cmn.summarize(pd.DataFrame(rows).assign(net_excess_pct=-0.1, pnl_excess_rub=-100.0),
                            DAYS[0], DAYS[-1], 5e6, {"excess_annual_min_pct": 2.0, "t_min": 2.0})
        self.assertFalse(bad["passed"])

    def test_rules_frozen(self):
        r = cmn.load_rules()
        self.assertEqual(set(r["modules"]), {"totm", "cash_and_carry", "dividend_gap", "index_rebalance"})
        self.assertEqual((r["criteria"]["excess_annual_min_pct"], r["modules"]["cash_and_carry"]["entry_premium_pct"]),
                         (2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
