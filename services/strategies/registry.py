"""
Реестр стратегий (ТЗ 17.09.2026, раздел 6, шаг 1): статусы жизненного цикла и
изолированная загрузка плагинов.

  RESEARCH → CERTIFIED → SHADOW → ACTIVE_MIX

В боевой пул попадают только записи со статусом ACTIVE_MIX и is_active=true.
Статус CERTIFIED выставляет только research/certifier.py по формальным критериям
(ТЗ 2.1) — руками статус не поднимаем.

Изоляция: сбой импорта плагина или исключение в generate_signals не мешают
остальным стратегиям; проблема пишется в лог и в отчёт сбоев.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os

from services.strategies.base import BaseStrategy, TradeSignal

log = logging.getLogger("services.strategies.registry")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "config", "strategies.json")
STATUSES = ("RESEARCH", "CERTIFIED", "SHADOW", "ACTIVE_MIX")
LIVE_STATUS = "ACTIVE_MIX"


class StrategyRegistry:
    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)

    # ── чтение ──
    @property
    def entries(self) -> list[dict]:
        return list(self.data.get("strategies", []))

    def entry(self, strategy_id: str) -> dict | None:
        return next((e for e in self.entries if e["strategy_id"] == strategy_id), None)

    def status(self, strategy_id: str) -> str | None:
        e = self.entry(strategy_id)
        return e.get("status") if e else None

    def live_entries(self) -> list[dict]:
        return [e for e in self.entries
                if e.get("status") == LIVE_STATUS and e.get("is_active") is True]

    # ── загрузка плагинов ──
    def load(self, entries: list[dict] | None = None) -> tuple[list[BaseStrategy], list[dict]]:
        """(плагины, сбои). Сбой одного не мешает остальным."""
        out, fails = [], []
        for e in (entries if entries is not None else self.live_entries()):
            try:
                mod = importlib.import_module(e["module"])
                cls = getattr(mod, e["class"])
                obj = cls()
                if not isinstance(obj, BaseStrategy):
                    raise TypeError(f"{e['class']} не наследует BaseStrategy")
                if obj.strategy_id != e["strategy_id"]:
                    raise ValueError(f"id плагина {obj.strategy_id} ≠ {e['strategy_id']} в реестре")
                out.append(obj)
            except Exception as ex:                      # noqa: BLE001 — изоляция плагина
                log.error("[STRATEGIES] плагин %s не загружен: %s", e.get("strategy_id"), ex)
                fails.append({"strategy_id": e.get("strategy_id"), "stage": "load", "error": str(ex)})
        return out, fails

    # ── запись статуса ──
    def set_status(self, strategy_id: str, status: str, *, metrics: dict | None = None,
                   note: str = "") -> dict:
        if status not in STATUSES:
            raise ValueError(f"статус {status!r} не из {STATUSES}")
        e = self.entry(strategy_id)
        if e is None:
            raise KeyError(f"стратегия {strategy_id} не зарегистрирована")
        e["status"] = status
        e["status_changed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        if status != LIVE_STATUS:
            e["is_active"] = False
        if metrics is not None:
            e["metrics"] = metrics
        if note:
            e["note"] = note
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)
        return e


def collect_signals(strategies: list[BaseStrategy], asof_date: dt.date,
                    market_data: dict) -> tuple[list[TradeSignal], list[dict]]:
    """Сигналы всех плагинов; исключение в одном не останавливает остальные."""
    signals, fails = [], []
    for s in strategies:
        try:
            got = s.generate_signals(asof_date, market_data) or []
            for sig in got:
                if sig.strategy_id != s.strategy_id:
                    raise ValueError(f"сигнал с чужим id: {sig.strategy_id} ≠ {s.strategy_id}")
            signals += got
        except Exception as ex:                          # noqa: BLE001 — изоляция плагина
            log.error("[STRATEGIES] %s: сигналы не получены: %s", s.strategy_id, ex)
            fails.append({"strategy_id": s.strategy_id, "stage": "generate_signals", "error": str(ex)})
    return signals, fails
