"""
Движок стратегий: контракт плагина, реестр, диспетчер капитала, сертификатор
(ТЗ 17.09.2026). Без сети и БД.

Главное здесь — доказать, что перенос боевых стратегий в плагины НЕ изменил
поведение: из тех же orders/rows получаются те же заявки плана.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategies import allocator as alc              # noqa: E402
from services.strategies import registry as reg               # noqa: E402
from services.strategies.base import (BaseStrategy, FundingType,  # noqa: E402
                                      HoldingHorizon, TradeSignal)
from services.strategies.plugins.intraday_short import IntradayShortStrategy   # noqa: E402
from services.strategies.plugins.long_overnight import LongOvernightStrategy   # noqa: E402
from tft_forecast.combined import Order                       # noqa: E402

DAY = dt.date(2026, 9, 18)


def order(ticker, strategy, direction, total=100_000.0, lots=10, stop=2.0, tp=3.0):
    return Order(ticker=ticker, strategy=strategy, direction=direction, anchor_price=100.0,
                 f_low=95.0, f_high=105.0, down_pct=-2.0, entry_price=99.0, better_pct=1.0,
                 stop_price=101.0, stop_pct=stop, tp_price=96.0, tp_pct=tp, lot_size=10,
                 lot_known=True, quantity_lots=lots, total_rub=total, unavailable=False)


def market(orders, rows=None, plan_orders=None):
    rows = rows or [{"exp_pnl": -1.0, "final_score": 0.5, "verdict": "OK"} for _ in orders]
    plan_orders = plan_orders or [{"signal_id": f"20260918-{o.ticker}-{o.strategy}"} for o in orders]
    return {"rows": rows, "orders": orders, "plan_orders": plan_orders, "cost_rt_pct": 0.128,
            "limit_entry_fraction": 0.2, "day": DAY}


def sig(strategy_id, ticker, side, budget, score=1.0, funding=FundingType.MARGIN_FREE,
        horizon=HoldingHorizon.INTRADAY, days=1):
    return TradeSignal(strategy_id=strategy_id, ticker=ticker, side=side, priority_score=score,
                       funding=funding, horizon=horizon, target_budget_rub=budget, max_holding_days=days)


class TestPluginsKeepBehaviour(unittest.TestCase):

    def test_only_own_strategy_and_plan_order_passthrough(self):
        orders = [order("SBER", "intraday_short", "SHORT"), order("LKOH", "long_overnight", "LONG"),
                  order("GAZP", "intraday_short", "SHORT", total=None, lots=None)]
        md = market(orders)
        s = IntradayShortStrategy().generate_signals(DAY, md)
        self.assertEqual([x.ticker for x in s], ["SBER"])           # неисполнимая заявка пропущена
        self.assertEqual(s[0].side, "SELL")
        self.assertEqual(s[0].target_budget_rub, 100_000.0)
        self.assertIs(s[0].plan_order, md["plan_orders"][0])        # запись плана без изменений
        self.assertEqual((s[0].funding, s[0].horizon),
                         (FundingType.MARGIN_FREE, HoldingHorizon.INTRADAY))
        o = LongOvernightStrategy().generate_signals(DAY, md)
        self.assertEqual([x.ticker for x in o], ["LKOH"])
        self.assertEqual((o[0].side, o[0].funding), ("BUY", FundingType.CASH_UNPARK_TMON))

    def test_priority_score_from_expected_move(self):
        md = market([order("SBER", "intraday_short", "SHORT")],
                    rows=[{"exp_pnl": -0.512, "verdict": "OK"}])
        s = IntradayShortStrategy().generate_signals(DAY, md)[0]
        self.assertAlmostEqual(s.priority_score, (0.512 - 0.128) / 0.128, places=6)

    def test_should_exit_rules(self):
        self.assertEqual(IntradayShortStrategy().should_exit({}, 100.0, 0)[0], False)
        self.assertEqual(IntradayShortStrategy().should_exit({}, 100.0, 1)[0], True)
        self.assertEqual(LongOvernightStrategy().should_exit({}, 100.0, 1)[0], True)


class TestAllocator(unittest.TestCase):

    def setUp(self):
        self.a = alc.PortfolioAllocator()

    def test_caps_ticker_strategy_and_total(self):
        p = self.a.allocate([sig("s1", "SBER", "SELL", 100_000), sig("s1", "SBER", "SELL", 100_000)])
        self.assertEqual(len(p.accepted), 1)
        self.assertIn("лимит на тикер", p.rejected[0][1])
        p = self.a.allocate([sig("s1", f"T{i}", "SELL", 150_000) for i in range(3)])
        self.assertAlmostEqual(p.by_strategy["s1"], 300_000.0)      # 300k на стратегию
        self.assertEqual(len(p.accepted), 2)
        many = [sig(f"s{i}", f"T{i}", "SELL", 150_000) for i in range(8)]
        p = self.a.allocate(many)
        self.assertLessEqual(p.exposure_rub, 750_000.0)             # общий лимит
        self.assertEqual(len(p.accepted), 5)

    def test_core_buffer_limits_cash(self):
        cash = [sig(f"s{i}", f"T{i}", "BUY", 150_000, funding=FundingType.CASH_UNPARK_TMON,
                    horizon=HoldingHorizon.OVERNIGHT) for i in range(8)]
        p = self.a.allocate(cash)
        self.assertLessEqual(p.cash_needed_rub, self.a.limits.cash_cap_rub)
        self.assertGreaterEqual(5_000_000 - p.cash_needed_rub, self.a.limits.core_min_rub)

    def test_opposite_signals_cancel(self):
        p = self.a.allocate([sig("s1", "SBER", "BUY", 100_000), sig("s2", "SBER", "SELL", 100_000),
                             sig("s1", "LKOH", "BUY", 100_000)])
        self.assertEqual([a.signal.ticker for a in p.accepted], ["LKOH"])
        self.assertTrue(all("встречные" in r for _, r in p.rejected))

    def test_efficiency_ranking(self):
        small_fast = sig("s1", "AAA", "BUY", 100_000, score=1.0, days=1)
        big_slow = sig("s2", "BBB", "BUY", 100_000, score=1.0, days=10)
        p = self.a.allocate([big_slow, small_fast])
        self.assertEqual([a.signal.ticker for a in p.accepted], ["AAA", "BBB"])
        self.assertGreater(alc.efficiency(small_fast), alc.efficiency(big_slow))

    def test_adv_cap_hook_and_card(self):
        a = alc.PortfolioAllocator(adv_cap=lambda tk: 50_000 if tk == "TGKA" else None)
        p = a.allocate([sig("s1", "TGKA", "SELL", 100_000), sig("s1", "SBER", "SELL", 100_000)])
        self.assertEqual([x.signal.ticker for x in p.accepted], ["SBER"])
        card = alc.telegram_card(p, 4_900_000, 5_000_000)
        self.assertIn("s1: 1 поз.", card)
        self.assertIn("98.0 %", card)


class TestRegistryIsolation(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "strategies.json")
        shutil.copy(reg.CONFIG_PATH, self.path)
        self.r = reg.StrategyRegistry(self.path)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_live_entries_and_status_write(self):
        self.assertEqual({e["strategy_id"] for e in self.r.live_entries()},
                         {"intraday_short", "long_overnight"})
        self.assertEqual(self.r.status("cash_and_carry"), "RESEARCH")
        self.r.set_status("cash_and_carry", "CERTIFIED", metrics={"t": 2.5})
        again = reg.StrategyRegistry(self.path)
        self.assertEqual(again.status("cash_and_carry"), "CERTIFIED")
        self.assertFalse(again.entry("cash_and_carry")["is_active"])     # сертификат ≠ боевой пул
        with self.assertRaises(ValueError):
            self.r.set_status("cash_and_carry", "LIVE")

    def test_broken_plugin_does_not_stop_others(self):
        class Broken(BaseStrategy):
            strategy_id, version, is_active = "broken", "1", True

            def generate_signals(self, asof_date, market_data):
                raise RuntimeError("плагин сломался")

            def should_exit(self, position, current_price, days_held):
                return False, ""

        md = market([order("SBER", "intraday_short", "SHORT")])
        signals, fails = reg.collect_signals([Broken(), IntradayShortStrategy()], DAY, md)
        self.assertEqual([s.ticker for s in signals], ["SBER"])
        self.assertEqual(fails[0]["strategy_id"], "broken")

    def test_load_reports_failures(self):
        self.r.data["strategies"].append({"strategy_id": "ghost", "module": "services.strategies.nope",
                                          "class": "X", "status": "ACTIVE_MIX", "is_active": True})
        plugins, fails = self.r.load()
        self.assertEqual({p.strategy_id for p in plugins}, {"intraday_short", "long_overnight"})
        self.assertEqual(fails[0]["strategy_id"], "ghost")


class TestCertifier(unittest.TestCase):

    def trades(self, net, n=40, notional=100_000.0, horizon="multi_day"):
        rows = []
        for i in range(n):
            d0 = dt.date(2023, 1, 2) + dt.timedelta(days=3 * i)
            rows.append({"entry_day": d0, "exit_day": d0 + dt.timedelta(days=20), "notional": notional,
                         "net_excess_pct": net[i % len(net)],
                         "pnl_excess_rub": notional * net[i % len(net)] / 100.0, "horizon": horizon})
        return pd.DataFrame(rows)

    def test_reject_reasons_listed(self):
        from research import certifier as ct
        from services.strategies.registry import StrategyRegistry
        r = StrategyRegistry()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        with unittest.mock.patch.object(ct, "OUT_DIR", tmp):
            res = ct.certify("cash_and_carry", self.trades([0.05, 0.02, 0.08]), registry=r, n_trials=113)
        self.assertEqual(res["verdict"], "CERTIFICATION_REJECTED")
        self.assertTrue(any("избыточная доходность" in f for f in res["failed"]))

    def test_square_off_violation_blocks_pass(self):
        from research import certifier as ct
        tr = self.trades([3.0], n=40, horizon="intraday")
        m = ct.metrics(tr, 5_000_000, 113)
        self.assertEqual(m["square_off_violations"], 40)
        ok, fails = ct.verdict(m, {"t_min": 2.0, "excess_annual_min_pct": 2.0, "ir_min": 0.8,
                                   "dsr_min": 0.95, "square_off_violations_max": 0})
        self.assertFalse(ok)
        self.assertTrue(any("оставшиеся на ночь" in f for f in fails))

    def test_thresholds_come_from_registry(self):
        from research import certifier as ct
        from services.strategies.registry import StrategyRegistry
        th = ct.thresholds(StrategyRegistry())
        self.assertEqual((th["t_min"], th["excess_annual_min_pct"], th["ir_min"], th["dsr_min"]),
                         (2.0, 2.0, 0.8, 0.95))


if __name__ == "__main__":
    unittest.main()
