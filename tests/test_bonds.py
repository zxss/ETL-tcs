"""Модуль ВДО: шкала рейтингов, фильтры, ограничения портфеля, сигналы выхода."""
import unittest

from bonds.core import (build_portfolio, check_portfolio, exclusion_reason, exit_signals,
                        load_rules, rating_name, rating_num, worst_rating)

RULES = load_rules()
KEY = 14.0


def bond(i, **over):
    b = {"secid": f"B{i:03d}", "isin": f"RU{i:010d}", "issuer": f"iss{i}", "sector": f"s{i % 6}",
         "yield_pct": 20.0 + (i % 7), "duration_y": 1.0 + (i % 10) / 10, "price_pct": 98.0,
         "days_to_event": 400, "value_rub": 5e6, "bid_ask_pct": 0.3,
         "issue_size_rub": 1e9, "rating_num": rating_num("BB"), "negative_watch": False,
         "floating": False, "subordinated": False, "perpetual": False, "qual_only": False,
         "required_tests": False, "tradable": True}
    b.update(over)
    return b


class TestRatings(unittest.TestCase):
    def test_agency_spellings(self):
        for s in ("ruBB+", "BB+(RU)", "BB+.ru", "BB+|ru|", "BB+"):
            self.assertEqual(rating_num(s), 11, s)
        self.assertEqual(rating_num("ruAAA"), 1)
        self.assertEqual(rating_num("BBB-"), 10)
        self.assertEqual(rating_num("B-"), 16)
        self.assertIsNone(rating_num("отозван"))

    def test_worst_is_conservative(self):
        self.assertEqual(rating_name(worst_rating(["ruA-", "BBB+(RU)"])), "BBB+")


class TestFilters(unittest.TestCase):
    def test_good_bond_passes(self):
        self.assertIsNone(exclusion_reason(bond(1), RULES, KEY))

    def test_each_filter(self):
        cases = {"только для квалов": {"qual_only": True}, "нужен тест": {"required_tests": True},
                 "плавающий купон": {"floating": True}, "мала доходность": {"yield_pct": 15.0},
                 "доходность дистресса": {"yield_pct": 40.0}, "цена вне коридора": {"price_pct": 70.0},
                 "дюрация вне коридора": {"duration_y": 4.0}, "мала ликвидность": {"value_rub": 1e5},
                 "широкий спред": {"bid_ask_pct": 3.0}, "близко событие": {"days_to_event": 20},
                 "нет рейтинга": {"rating_num": None},
                 "рейтинг ниже порога": {"rating_num": rating_num("B+")},
                 "рейтинг выше коридора ВДО": {"rating_num": rating_num("AA")},
                 "негативный прогноз": {"negative_watch": True}}
        for reason, over in cases.items():
            self.assertEqual(exclusion_reason(bond(1, **over), RULES, KEY), reason, over)

    def test_backtest_mode_ignores_credit(self):
        self.assertIsNone(exclusion_reason(bond(1, rating_num=None), RULES, KEY, use_credit=False))


class TestPortfolio(unittest.TestCase):
    def test_constraints_hold(self):
        bonds = [bond(i) for i in range(60)]
        r = build_portfolio(bonds, RULES, KEY)
        self.assertEqual(len(r["portfolio"]), RULES["portfolio"]["n_bonds"])
        self.assertEqual(r["violations"], [])
        self.assertLessEqual(sum(b["weight"] for b in r["portfolio"]),
                             1 - RULES["portfolio"]["cash_reserve"] + 1e-9)

    def test_one_bond_per_issuer(self):
        bonds = [bond(i, issuer="один") for i in range(30)] + [bond(100 + i) for i in range(30)]
        port = build_portfolio(bonds, RULES, KEY)["portfolio"]
        self.assertEqual(sum(1 for b in port if b["issuer"] == "один"), 1)

    def test_sector_cap(self):
        bonds = [bond(i, sector="девелопмент") for i in range(40)] + [bond(100 + i) for i in range(40)]
        port = build_portfolio(bonds, RULES, KEY)["portfolio"]
        w = sum(b["weight"] for b in port if b["sector"] == "девелопмент")
        self.assertLessEqual(w, RULES["portfolio"]["max_weight_sector"] + 1e-9)

    def test_holdings_kept_within_buffer(self):
        bonds = [bond(i) for i in range(60)]
        first = build_portfolio(bonds, RULES, KEY)
        picks = [b["secid"] for b in first["portfolio"]]
        # портфель из 20: 19 лучших + выпуск, сползший на 35-е место в ранге;
        # без удержания его место занял бы лучший не удерживаемый выпуск
        tail = first["ranked_ids"][35]
        held = set(picks[:19]) | {tail}
        again = {b["secid"] for b in build_portfolio(bonds, RULES, KEY, holdings=held)["portfolio"]}
        self.assertIn(tail, again)
        self.assertNotIn(picks[19], again, "удержание не должно раздувать книгу сверх N")
        # за пределами буфера (ранг ≥ 40) удержание не спасает
        far = first["ranked_ids"][-1]
        again2 = {b["secid"] for b in build_portfolio(bonds, RULES, KEY,
                                                      holdings=set(picks[:19]) | {far})["portfolio"]}
        self.assertNotIn(far, again2)

    def test_unknown_sector_does_not_cap(self):
        """Без сектора выпуски не складываются в одну корзину «?»."""
        port = build_portfolio([bond(i, sector=None) for i in range(60)], RULES, KEY)["portfolio"]
        self.assertEqual(len(port), RULES["portfolio"]["n_bonds"])

    def test_violation_detected(self):
        port = [{"secid": "X", "issuer": "i", "sector": "s", "weight": 0.5, "duration_y": 1.0}]
        self.assertTrue(check_portfolio(port, RULES))


class TestExit(unittest.TestCase):
    def test_exit_rules(self):
        self.assertIsNone(exit_signals(bond(1), RULES, KEY))
        self.assertIn("стоп", exit_signals(bond(1, price_pct=75.0), RULES, KEY))
        self.assertIn("рейтинг", exit_signals(bond(1, rating_num=rating_num("B")), RULES, KEY))
        self.assertIn("остановка", exit_signals(bond(1, halted=True), RULES, KEY))
        self.assertIn("оферта", exit_signals(bond(1, days_to_event=10), RULES, KEY))


if __name__ == "__main__":
    unittest.main()


class TestBondReturn(unittest.TestCase):
    """Доходность выпуска: купон по сбросу НКД, амортизация, погашение, дефолт."""

    def _run(self, rows, a="2024-01-01", b="2024-01-31"):
        import datetime as dt
        from unittest import mock
        from bonds import backtest as bt
        with mock.patch.object(bt.iss, "daily", return_value=rows):
            return bt.bond_return("X", dt.date.fromisoformat(a), dt.date.fromisoformat(b), "c")

    @staticmethod
    def row(day, close, acc, face=1000.0, cv=30.0, mat="2027-01-01"):
        return {"TRADEDATE": day, "CLOSE": close, "ACCINT": acc, "FACEVALUE": face,
                "COUPONVALUE": cv, "MATDATE": mat}

    def test_flat_price_with_coupon(self):
        # 100 % номинала, НКД 29 → купон 30 выплачен, НКД 1: доход ≈ 30+1 на 1029
        r, _ = self._run([self.row("2024-01-01", 100, 29), self.row("2024-01-30", 100, 1)])
        self.assertAlmostEqual(r, (1000 + 1 + 30) / 1029 - 1, places=9)

    def test_amortization_counted(self):
        r, _ = self._run([self.row("2024-01-01", 100, 0), self.row("2024-01-30", 100, 0, face=800)])
        self.assertAlmostEqual(r, 0.0, places=9)     # 200 ₽ вернули деньгами, цена та же

    def test_maturity(self):
        r, note = self._run([self.row("2024-01-01", 100, 10, mat="2024-01-10"),
                             self.row("2024-01-09", 100, 19, mat="2024-01-10")])
        self.assertEqual(note, "погашен")
        self.assertAlmostEqual(r, (1000 + 19) / 1010 - 1, places=9)

    def test_default_haircut(self):
        r, note = self._run([self.row("2024-01-01", 90, 0), self.row("2024-01-05", 40, 0)])
        self.assertIn("дефолт", note)
        self.assertAlmostEqual(r, 400 * 0.5 / 900 - 1, places=9)


# ── живая система: расчёт заявок без брокера ─────────────────────────────────

class _FakeBroker:
    """Песочница в памяти: позиции, деньги, стакан, статус торгов."""

    def __init__(self, positions, money, book=(99.0, 99.2), status="SECURITY_TRADING_STATUS_NORMAL_TRADING"):
        self.positions, self.money, self.book, self.status = positions, money, book, status

    def _post(self, method, payload=None):
        if method.endswith("GetSandboxPositions"):
            return {"money": [{"currency": "rub", "units": str(int(self.money)), "nano": 0}],
                    "securities": [{"instrumentUid": u, "balance": str(q)} for u, q in self.positions.items()]}
        if method.endswith("GetOrderBook"):
            bid, ask = self.book
            px = lambda v: {"price": {"units": str(int(v)), "nano": int(round((v % 1) * 1e9))}}
            return {"bids": [px(bid)] if bid else [], "asks": [px(ask)] if ask else []}
        if method.endswith("GetTradingStatus"):
            return {"tradingStatus": self.status, "limitOrderAvailableFlag": True}
        raise AssertionError(method)


def _live(secid, price=99.0, accrued=10.0):
    return {"secid": secid, "uid": "u" + secid, "price_pct": price, "nominal": 1000.0,
            "accrued": accrued, "lot": 1, "min_step": 0.01}


class LiveTrader(unittest.TestCase):

    def test_unit_cost_percent_of_face_plus_accrued(self):
        from bonds.trader import unit_cost
        self.assertAlmostEqual(unit_cost(_live("A", 96.27, 26.34)), 962.7 + 26.34, places=6)

    def test_limit_price_caps_slippage_and_rounds_to_step(self):
        from bonds.trader import _limit_price
        b = _live("A", 99.0)
        br = _FakeBroker({}, 0, book=(98.0, 100.5))          # ask дальше, чем 0,5 % от last
        self.assertAlmostEqual(_limit_price(br, b, "BUY", 0.5), 99.49, places=6)
        br.book = (98.9, 99.1)
        self.assertAlmostEqual(_limit_price(br, b, "BUY", 0.5), 99.1, places=6)
        self.assertAlmostEqual(_limit_price(br, b, "SELL", 0.5), 98.9, places=6)
        br.status = "SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING"   # внебиржевой режим
        self.assertIsNone(_limit_price(br, b, "BUY", 0.5))
        br.status, br.book = "SECURITY_TRADING_STATUS_NORMAL_TRADING", (None, None)
        b["price_pct"] = None
        self.assertIsNone(_limit_price(br, b, "BUY", 0.5))

    def test_trade_to_target_sells_first_and_sizes_by_nav(self):
        from bonds.trader import trade_to_target
        rules = load_rules()
        a, b, c = _live("A"), _live("B"), _live("C")          # 1 000 ₽ за штуку
        br = _FakeBroker({"uA": 30, "uC": 10}, money=60_000)  # NAV 100 000
        target = {"weights": {"A": 0.10, "B": 0.05}}
        orders = trade_to_target(br, "acc", target, {x["uid"]: x for x in (a, b, c)},
                                 {x["secid"]: x for x in (a, b, c)}, rules, dry=True)
        got = [(o["secid"], o["side"], o["lots"]) for o in orders]
        self.assertEqual(got, [("A", "SELL", 20), ("C", "SELL", 10), ("B", "BUY", 5)])

    def test_trade_to_target_exit_blocks_rebuy(self):
        from bonds.trader import trade_to_target
        rules = load_rules()
        a = _live("A")
        br = _FakeBroker({"uA": 10}, money=90_000)
        orders = trade_to_target(br, "acc", {"weights": {"A": 0.10}}, {"uA": a}, {"A": a}, rules,
                                 dry=True, exits={"A": "цена < стоп"})
        self.assertEqual([(o["side"], o["lots"]) for o in orders], [("SELL", 10)])

    def test_order_price_in_rubles_for_sandbox(self):
        from bonds.trader import order_price_rub
        self.assertAlmostEqual(order_price_rub({"nominal": 1000.0}, 97.79), 977.9, places=6)
        self.assertAlmostEqual(order_price_rub({"nominal": 500.0}, 101.2), 506.0, places=6)
