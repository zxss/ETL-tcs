"""
Тесты уведомлений в Telegram (services/notify.py + врезка в stage2_demo).

Главное, что здесь проверяется, — НЕ доставка сообщения, а то, что сбой
доставки не может уронить торговую фазу. Сеть в тестах не трогается вообще:
всё, что стучится наружу, подменено заглушкой.
"""
from __future__ import annotations

import logging
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                     # noqa: E402
from services import notify                       # noqa: E402
from services import stage2_demo as s2            # noqa: E402


class _Resp:
    """Минимальный контекст-менеджер вместо http.client.HTTPResponse."""

    def __init__(self, payload: bytes):
        self._p = payload

    def read(self):
        return self._p

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class NotifyBase(unittest.TestCase):
    def enable(self):
        p = [mock.patch.object(config, "TELEGRAM_ENABLED", True),
             mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "111:СЕКРЕТ"),
             mock.patch.object(config, "TELEGRAM_CHAT_ID", "222")]
        for x in p:
            x.start()
            self.addCleanup(x.stop)


class TestEnabled(NotifyBase):
    def test_disabled_without_full_config(self):
        """Неполная настройка = выключено. Половина конфигурации хуже, чем ноль:
        молчаливые попытки отправки в никуда съедают таймаут в каждой фазе."""
        for token, chat, flag in (("", "222", True), ("111:X", "", True),
                                  ("111:X", "222", False)):
            with mock.patch.object(config, "TELEGRAM_BOT_TOKEN", token), \
                 mock.patch.object(config, "TELEGRAM_CHAT_ID", chat), \
                 mock.patch.object(config, "TELEGRAM_ENABLED", flag):
                self.assertFalse(notify.enabled())

    def test_send_is_noop_when_disabled(self):
        """Выключённые уведомления не должны ходить в сеть вообще."""
        with mock.patch.object(config, "TELEGRAM_ENABLED", False), \
             mock.patch("urllib.request.urlopen") as u:
            self.assertFalse(notify.send("текст"))
            u.assert_not_called()


class TestEscaping(NotifyBase):
    def test_html_special_chars(self):
        """Неэкранированный '<' валит Telegram с 400 — тексты ошибок приходят
        из внешних источников, поэтому экранируется всё подставляемое."""
        self.assertEqual(notify.esc("a < b & c"), "a &lt; b &amp; c")

    def test_none_becomes_dash(self):
        self.assertEqual(notify.esc(None), "—")


class TestTransport(NotifyBase):
    def test_posts_body_not_query_string(self):
        """Текст уходит телом POST, а не в query string: строка запроса
        попадает в логи прокси и в историю, тело — нет."""
        self.enable()
        with mock.patch("urllib.request.urlopen", return_value=_Resp(b'{"ok":true}')) as u:
            self.assertTrue(notify.send("привет"))
        req = u.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertNotIn("привет", req.full_url)
        self.assertIn("привет", req.data.decode("utf-8"))

    def test_truncates_to_telegram_limit(self):
        self.enable()
        with mock.patch("urllib.request.urlopen", return_value=_Resp(b'{"ok":true}')) as u:
            notify.send("я" * 9000)
        import json as _j
        self.assertLessEqual(len(_j.loads(u.call_args[0][0].data)["text"]), 4096)

    def test_no_retry_on_bad_token(self):
        """401/403 — повтор бессмысленен и только тормозит фазу на таймауте."""
        self.enable()
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        err.read = lambda: b'{"description":"Unauthorized"}'
        with mock.patch("urllib.request.urlopen", side_effect=err) as u:
            self.assertFalse(notify.send("текст"))
        self.assertEqual(u.call_count, 1)

    def test_token_never_reaches_the_log(self):
        """Токен не должен утечь в лог даже в тексте ошибки от Telegram."""
        self.enable()
        err = urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
        err.read = lambda: b'{"description":"bot111:\\u0421\\u0415\\u041a\\u0420\\u0415\\u0422 is bad"}'
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertLogs("notify", level=logging.WARNING) as cm:
                notify.send("текст")
        self.assertNotIn("111:СЕКРЕТ", "\n".join(cm.output))

    def test_network_failure_returns_false(self):
        self.enable()
        with mock.patch("urllib.request.urlopen", side_effect=OSError("сеть недоступна")), \
             mock.patch("time.sleep"):
            self.assertFalse(notify.send("текст"))


class TestPhaseIsNeverBrokenByNotifier(NotifyBase):
    """Ключевое свойство: отчётность не имеет права уронить торговую фазу."""

    def _result(self, verdict_errors=()):
        res = s2.PhaseResult("PREP", "20260910-094500-PREP", "/tmp")
        res.data.update(signals_count=3, plan_orders=2)
        for e in verdict_errors:
            res.check(False, e)
        return res

    def test_exception_inside_notifier_is_swallowed(self):
        with mock.patch.object(notify, "enabled", side_effect=RuntimeError("бум")):
            s2._notify_phase(self._result(), {})   # не должно бросить

    def test_send_failure_is_swallowed(self):
        self.enable()
        with mock.patch.object(notify, "send", side_effect=OSError("нет сети")):
            s2._notify_phase(self._result(), {})


class TestMessageContent(NotifyBase):
    def _capture(self, res, state):
        self.enable()
        with mock.patch.object(notify, "send") as snd:
            s2._notify_phase(res, state)
        return snd.call_args

    def test_phase_facts_and_verdict(self):
        res = s2.PhaseResult("PREP", "20260910-094500-PREP", "/tmp")
        res.data.update(signals_count=3, plan_orders=2)
        text = self._capture(res, {})[0][0]
        self.assertIn("PREP", text)
        self.assertIn(s2.PASS, text)
        self.assertIn("сигналов", text)
        self.assertIn("заявок в плане", text)

    def test_notify_label_shown_for_shared_channel(self):
        """23.09.2026: один и тот же бот/чат на песочницу и турнир — метка в
        шапке отличает карточки, отдельный канал заводить не нужно."""
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        with mock.patch.object(config, "STAGE2_NOTIFY_LABEL", "ТУРНИР"):
            text = self._capture(res, {})[0][0]
        self.assertIn("ТУРНИР", text)

    def test_notify_label_absent_by_default(self):
        """r4 не задаёт STAGE2_NOTIFY_LABEL — карточка песочницы не меняется."""
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        with mock.patch.object(config, "STAGE2_NOTIFY_LABEL", ""):
            text = self._capture(res, {})[0][0]
        self.assertNotIn("\U0001f3c6", text)

    def test_notify_label_is_escaped(self):
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        with mock.patch.object(config, "STAGE2_NOTIFY_LABEL", "<ТУРНИР>"):
            text = self._capture(res, {})[0][0]
        self.assertIn("&lt;ТУРНИР&gt;", text)

    def test_routine_pass_is_silent_but_failure_is_not(self):
        """Рутинный успех не должен будить ночью; провал — должен."""
        ok = s2.PhaseResult("PREP", "r", "/tmp")
        self.assertTrue(self._capture(ok, {})[1]["silent"])

        bad = s2.PhaseResult("PREP", "r", "/tmp")
        bad.check(False, "устаревшие данные")
        self.assertFalse(self._capture(bad, {})[1]["silent"])

    def test_overnight_always_sounds(self):
        """Дневная сводка — единственное сообщение, ради которого стоит
        посмотреть в телефон, поэтому она звучит даже при чистом PASS."""
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        self.assertFalse(self._capture(res, {})[1]["silent"])

    def test_halt_instruction_is_included(self):
        res = s2.PhaseResult("ORDER", "r", "/tmp")
        res.check(False, "план не совпал с датасетом")
        text = self._capture(res, {"status": "halted"})[0][0]
        self.assertIn("ТЕСТ ОСТАНОВЛЕН", text)
        self.assertIn("resume", text)

    def test_daily_summary_block(self):
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        res.data["day_summary"] = {"completed": 1, "target": 15, "closing": 100500.0,
                                   "change": 500.0, "change_pct": 0.5,
                                   "cum_pct": 0.5, "positions": 2}
        text = self._capture(res, {})[0][0]
        self.assertIn("итог дня", text)
        self.assertIn("1", text)
        self.assertIn("15", text)

    def test_trade_counter_shown_when_target_set(self):
        """Турнирный счётчик (23.09.2026): виден только когда STAGE2_TRADE_TARGET
        задан, чтобы не менять карточку у текущего теста stage2-demo-30d-r4."""
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        res.data["day_summary"] = {"completed": 3, "target": 24, "closing": 1041000.0,
                                   "change": 1000.0, "change_pct": 0.1, "cum_pct": 0.1,
                                   "positions": 5, "trades_total": 14, "trades_target": 100}
        text = self._capture(res, {})[0][0]
        self.assertIn("сделок в зачёте", text)
        self.assertIn("14", text)
        self.assertIn("100", text)

    def test_trade_counter_hidden_without_target(self):
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        res.data["day_summary"] = {"completed": 1, "target": 15, "closing": 100500.0,
                                   "change": 500.0, "change_pct": 0.5, "cum_pct": 0.5,
                                   "positions": 2}
        text = self._capture(res, {})[0][0]
        self.assertNotIn("сделок в зачёте", text)

    def test_trade_counter_warns_when_pace_cannot_reach_target(self):
        """3 дня из 24 позади, зачтено 2 сделки из 100 — за оставшиеся 21 день
        нужно > 1 сделки/день в среднем, предупреждение должно появиться."""
        res = s2.PhaseResult("OVERNIGHT", "r", "/tmp")
        res.data["day_summary"] = {"completed": 3, "target": 24, "closing": 1000000.0,
                                   "change": 0.0, "change_pct": 0.0, "cum_pct": 0.0,
                                   "positions": 1, "trades_total": 2, "trades_target": 100}
        text = self._capture(res, {})[0][0]
        self.assertIn("отстаём", text)

    def test_error_text_is_escaped(self):
        """Ошибка с '<' не должна ломать разметку сообщения."""
        res = s2.PhaseResult("ORDER", "r", "/tmp")
        res.check(False, "ExpPnL < порога")
        self.assertIn("&lt;", self._capture(res, {})[0][0])


class TestFormatting(NotifyBase):
    def test_russian_money_format(self):
        self.assertEqual(s2._rub(1234.5), "+1 234,50 ₽")
        self.assertEqual(s2._rub(-99.0), "-99,00 ₽")
        self.assertEqual(s2._rub(None), "—")

    def test_percent_format(self):
        self.assertEqual(s2._pct(0.5), "+0,50%")
        self.assertEqual(s2._pct(-1.25), "-1,25%")
        self.assertEqual(s2._pct(None), "—")


if __name__ == "__main__":
    unittest.main()
