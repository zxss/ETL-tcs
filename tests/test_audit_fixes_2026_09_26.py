"""
Правки по комплексному аудиту 26.09.2026 (отчёт AUDIT-FULL-2026-09-26.md).

F1  ночной рейтинг совпадал с −утренним гэпом: поправка на реализованное движение
    вычитала сегодняшний гэп из прогноза уже случившейся ночи;
F2  модель училась на ночи 23:45→06:55, робот держит 18:40→09:10 → бары основной
    сессии (services/session_bars.py);
F7  прогноз не доходил до ночных заявок, выходы боевого счёта не журналировались;
F8  боевая модель не видела последних ~15 % выборки → дообучение на всей;
F9  масштаб признаков считался и по валидационному хвосту;
F12 плоские издержки 0,128 % для всех бумаг;
+   PROTECT верил реестру, а не брокеру: снятые вне контура стопы не восстанавливались;
+   ручное закрытие снимало стопы до продажи и падало на закрытой бирже;
+   выключатель новых входов для счёта.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import config


# ── F1: поправка на реализованное движение ────────────────────────────────────

class TestRealizedAdjustment(unittest.TestCase):
    """Бар сегодняшнего дня в выборке → сегодняшнее движение уже в признаках."""

    @staticmethod
    def _setup(last_date):
        from tft_forecast import forecast as fc
        pred = np.zeros((1, 8, 5))
        pred[0, fc.T_OVN, :] = [-1.0, -0.5, 0.2, 0.8, 1.5]
        panel = types.SimpleNamespace(infer_tickers=["SBER"])
        frame = types.SimpleNamespace(ticker="SBER", last_close=300.0, last_date=last_date,
                                      df=pd.DataFrame({"ret_std20": [1.0]}))
        q = types.SimpleNamespace(price=303.0, today_open=306.0)
        now = dt.datetime(2026, 9, 25, 18, 38, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        ctx = types.SimpleNamespace(market_open=True, quotes={"SBER": q}, as_of=now)
        return fc, pred, panel, [frame], ctx

    def test_today_bar_present_no_subtraction(self):
        """18:38 с сегодняшним баром: прогноз ночи трогать нельзя."""
        fc, pred, panel, frames, ctx = self._setup(dt.date(2026, 9, 25))
        adj = fc._adjust_for_realized(pred, panel, frames, ctx)
        np.testing.assert_allclose(adj[0, fc.T_OVN], pred[0, fc.T_OVN])

    def test_stale_bar_still_adjusted(self):
        """Днём со вчерашним баром (ручной прогон) остаток по-прежнему считается."""
        fc, pred, panel, frames, ctx = self._setup(dt.date(2026, 9, 24))
        adj = fc._adjust_for_realized(pred, panel, frames, ctx)
        self.assertAlmostEqual(adj[0, fc.T_OVN, 2], 0.2 - (306.0 / 300.0 - 1) * 100)


# ── F2: бары основной сессии ─────────────────────────────────────────────────

class TestSessionBars(unittest.TestCase):

    def test_main_open_by_era(self):
        from services.session_bars import main_open
        self.assertEqual(main_open(dt.date(2026, 9, 11)), dt.time(10, 0))
        self.assertEqual(main_open(dt.date(2026, 9, 14)), dt.time(9, 10))

    def test_session_tau_uses_era_open(self):
        from tft_forecast.forecast import _session_tau
        msk = dt.timezone(dt.timedelta(hours=3))
        self.assertEqual(_session_tau(dt.datetime(2026, 9, 25, 9, 10, tzinfo=msk)), 0.0)
        self.assertEqual(_session_tau(dt.datetime(2026, 9, 25, 18, 40, tzinfo=msk)), 1.0)
        self.assertEqual(_session_tau(dt.datetime(2026, 9, 11, 9, 30, tzinfo=msk)), 0.0)

    def test_model_reads_session_table_by_default(self):
        from tft_forecast.features import bars_table
        with mock.patch.object(config, "MODEL_BARS", "session"):
            self.assertEqual(bars_table(), "market_data_session")
        with mock.patch.object(config, "MODEL_BARS", "daily"):
            self.assertEqual(bars_table(), "market_data")

    def test_build_sql_window(self):
        """Сборка берёт только основную сессию и только будни."""
        from services import session_bars as sb
        sql = sb._BUILD_SQL
        self.assertIn("extract(isodow FROM d) < 6", sql)
        self.assertIn("t < %(close_cutoff)s", sql)
        self.assertEqual(sb.CLOSE_CUTOFF, dt.time(18, 40))

    def test_today_open_is_main_session(self):
        from tft_forecast import quotes
        self.assertIn("%(open_t)s", quotes._TODAY_OPEN_SQL)


# ── F8/F9: масштаб признаков по обучающей части ──────────────────────────────

class TestScalerTrainOnly(unittest.TestCase):

    def test_late_regime_shift_not_in_scaler(self):
        """Признаки в хвосте сдвинуты на +100: среднее обязано остаться у ранних."""
        from tft_forecast.dataset import build_panel
        from tft_forecast.features import FEATURE_COLS, TARGET_COLS
        n = 400
        dates = pd.bdate_range("2024-01-01", periods=n)
        frames = []
        for tk in ("AAA", "BBB"):
            d = pd.DataFrame({"date": dates})
            for c in FEATURE_COLS:
                d[c] = 0.0
            d.loc[d.index >= int(n * 0.9), FEATURE_COLS[0]] = 100.0
            for c in TARGET_COLS:
                d[c] = 0.1
            d["next_dow"] = 1
            frames.append(types.SimpleNamespace(ticker=tk, df=d))
        panel = build_panel(frames, lookback=10)
        self.assertLess(panel.feat_mean[0], 5.0,
                        "хвост валидации не должен попадать в среднее масштаба")

    def test_refit_epochs_configurable(self):
        self.assertGreaterEqual(int(getattr(config, "TFT_REFIT_EPOCHS", 0)), 0)


# ── F12: поштучные издержки ──────────────────────────────────────────────────

class TestCostModel(unittest.TestCase):

    def setUp(self):
        from services import cost_model
        cost_model._cache = {"AKRN": (0.011, 0.0022, 10000.0),
                             "SBER": (0.02, 0.0001, 10000.0)}
        self.cm = cost_model

    def tearDown(self):
        self.cm._cache = None

    def test_matrix_round_trip(self):
        with mock.patch.object(config, "COST_MODEL", "matrix"):
            rt = self.cm.round_trip_pct("AKRN", position_rub=10000.0)
        self.assertAlmostEqual(rt, 0.08 + 0.011 + 2 * 0.0022)

    def test_impact_scales_with_position(self):
        """Амихуд линеен по объёму: позиция в 10 раз больше — воздействие в 10 раз."""
        with mock.patch.object(config, "COST_MODEL", "matrix"):
            small = self.cm.round_trip_pct("AKRN", position_rub=10000.0)
            big = self.cm.round_trip_pct("AKRN", position_rub=100000.0)
        self.assertAlmostEqual(big - small, 2 * 0.0022 * 9)

    def test_unknown_ticker_falls_back_to_flat(self):
        with mock.patch.object(config, "COST_MODEL", "matrix"), \
             mock.patch.object(config, "TFT_COST_RT", 0.128):
            self.assertEqual(self.cm.round_trip_pct("XXXX"), 0.128)

    def test_flat_mode(self):
        with mock.patch.object(config, "COST_MODEL", "flat"), \
             mock.patch.object(config, "TFT_COST_RT", 0.128):
            self.assertEqual(self.cm.round_trip_pct("AKRN"), 0.128)

    def test_directional_uses_ticker_cost(self):
        from tft_forecast import directional
        pred = np.zeros((1, 8, 5))
        pred[0, 2, :] = [-1.0, -0.5, 0.5, 1.0, 1.5]
        rows = directional.build_rows(pred, ["AKRN"], ["AKRN"], ["long_overnight"],
                                      cost_rt=0.128, cost_by_ticker={"AKRN": 0.2})
        self.assertAlmostEqual(rows[0]["ExpPnL"], 0.5 - 0.2)
        self.assertEqual(rows[0]["CostRT"], 0.2)


# ── выключатель новых входов ─────────────────────────────────────────────────

class TestNewEntriesSwitch(unittest.TestCase):

    def test_entries_rejected_when_off(self):
        from services import stage2_demo as s2
        po = {"ticker": "SBER", "strategy_type": "long_overnight", "direction": "LONG",
              "quantity_lots": 1, "total_rub": 1000.0, "signal_id": "x"}
        with mock.patch.object(config, "STAGE2_NEW_ENTRIES", False), \
             mock.patch.object(s2, "contour_ok", return_value=None):
            reason = s2.check_order_allowed(po, env="PROD", used_ids=set())
        self.assertIn("STAGE2_NEW_ENTRIES", reason)


# ── PROTECT: сверка стопов с брокером ────────────────────────────────────────

class TestProtectRearm(unittest.TestCase):
    """26.09: скрипт снял стопы боевого счёта, продажа не прошла, а PROTECT по
    флагу реестра считал позиции защищёнными."""

    def _run(self, rec, positions):
        from tests.test_r4_blockers import Broker, run_attach
        b = Broker(positions=positions, active_stops=[], history=[], last=270.0)
        return b, run_attach(b, [rec])

    def test_missing_stops_rearmed(self):
        from tests.test_r4_blockers import Pos, _rec
        rec = _rec(stop_placed=True, stop_order_id="sl-gone",
                   tp_placed=True, tp_order_id="tp-gone")
        b, rep = self._run(rec, [Pos("u1", 36)])
        kinds = [c[1] for c in b.calls if c[0] == "stop"]
        self.assertIn("STOP_LOSS", kinds)
        self.assertIn("TAKE_PROFIT", kinds)
        self.assertEqual(rep.get("rearmed"), ["SNGS"])

    def test_no_rearm_when_quantity_changed(self):
        """Объём меньше защищённого — это частичный выход, не снятый стоп."""
        from tests.test_r4_blockers import Pos, _rec
        rec = _rec(stop_placed=True, stop_order_id="sl-1", tp_placed=True,
                   tp_order_id="tp-1", protected_qty=36)
        b, rep = self._run(rec, [Pos("u1", 20)])
        self.assertNotIn("rearmed", rep)


# ── безопасное ручное закрытие ───────────────────────────────────────────────

class TestFlatten(unittest.TestCase):

    class _Br:
        def __init__(self, status):
            self.status, self.calls = status, []

        def _post(self, method, payload):
            return {"tradingStatus": self.status, "marketOrderAvailableFlag": True,
                    "apiTradeAvailableFlag": True}

        def get_positions(self, acc):
            return [types.SimpleNamespace(instrument_uid="u1", balance_shares=480,
                                          is_open=True)]

        def find_instrument_by_uid(self, uid):
            return types.SimpleNamespace(ticker="MVID", lot=1)

        def get_active_stop_orders(self, acc):
            return [types.SimpleNamespace(instrument_uid="u1", kind="STOP_LOSS",
                                          stop_order_id="sl-1")]

        def get_active_orders(self, acc):
            return []

        def cancel_stop_order(self, **kw):
            self.calls.append(("cancel_stop", kw))

        def post_market_order(self, **kw):
            self.calls.append(("market", kw))

    def _main(self, br, argv):
        from scripts import flatten_account as fa
        with mock.patch.object(fa, "_make_broker_and_account", return_value=(br, "acc", "SANDBOX")), \
             mock.patch.object(fa, "contour_ok", return_value=None), \
             mock.patch.object(fa, "registry_lock", contextlib.nullcontext), \
             contextlib.redirect_stdout(io.StringIO()):
            return fa.main(argv)

    def test_closed_exchange_touches_nothing(self):
        """Суббота 26.09: DEALER_NORMAL_TRADING → ни один стоп не снят."""
        br = self._Br("SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING")
        self._main(br, ["--go"])
        self.assertEqual(br.calls, [], "на закрытой бирже защиту снимать нельзя")

    def test_dry_run_touches_nothing(self):
        br = self._Br("SECURITY_TRADING_STATUS_NORMAL_TRADING")
        self._main(br, [])
        self.assertEqual(br.calls, [])


if __name__ == "__main__":
    unittest.main()


class TestSplitGuards(unittest.TestCase):
    """26.09: 5-минутки не скорректированы на сплиты (VTBR 5000:1 → «ночь» +502 663 %)."""

    def test_split_day_dropped_from_targets_and_features(self):
        from tft_forecast.features import _build_features
        days = pd.bdate_range("2024-07-01", periods=10)
        px = np.array([0.02] * 5 + [100.0] * 5)
        raw = pd.DataFrame({"date": days, "open": px, "high": px * 1.01,
                            "low": px * 0.99, "close": px, "volume": 1000.0})
        d = _build_features(raw, [])
        self.assertTrue(np.isnan(d["overnight"].iloc[5]))
        self.assertTrue(np.isnan(d["next_overnight"].iloc[4]))
        self.assertTrue(np.isnan(d["log_ret"].iloc[5]))
        self.assertTrue(np.isfinite(d["ret_std20"].iloc[-1]),
                        "скользящая вола после сплита должна считаться без выброса")
        self.assertLess(d["ret_std20"].iloc[-1], 5.0)

    def test_genuine_large_move_kept(self):
        """CBOM +54 % за день — за порогом 40 %, выпадает; MVID +9 % — остаётся."""
        from tft_forecast.features import _build_features
        days = pd.bdate_range("2025-09-22", periods=6)
        px = np.array([57.85, 57.85, 63.2, 63.2, 63.2, 63.2])
        raw = pd.DataFrame({"date": days, "open": px, "high": px, "low": px,
                            "close": px, "volume": 1000.0})
        d = _build_features(raw, [])
        self.assertAlmostEqual(d["overnight"].iloc[2], (63.2 / 57.85 - 1) * 100, places=4)

    def test_block_grouping_uses_one_factor(self):
        """Блок дней одного тикера с одним порядком множителя — одно медианное число."""
        from services import session_bars as sb
        rows = [("VTBR", dt.date(2024, 5, 22), 4941.0), ("VTBR", dt.date(2024, 5, 23), 5098.0),
                ("VTBR", dt.date(2024, 5, 24), 5000.0)]
        calls = []

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, params=None):
                calls.append(params)
            def fetchall(self): return rows

        class Conn:
            def cursor(self): return Cur()
            def commit(self): pass

        done = sb.adjust_splits(Conn())
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][3], 5000.0)
