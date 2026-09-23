"""
Диспетчер капитала (ТЗ 17.09.2026, раздел 4).

Лимиты депозита 5 000 000 ₽:
  • в фонде TMON@ всегда ≥ 85 % капитала;
  • суммарные открытые позиции всех стратегий ≤ 15 % (750 000 ₽);
  • на одну стратегию ≤ 300 000 ₽; на один тикер (по всем стратегиям) ≤ 150 000 ₽.

Конфликты: встречные сигналы по одной бумаге взаимно аннулируются — позиция не
открывается, деньги остаются в фонде. Остальные ранжируются по удельной
эффективности priority_score / (бюджет × дни удержания) и берутся сверху вниз,
пока не упрутся в лимиты. Частичных заявок нет: не помещается — пропускаем.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from services.strategies.base import FundingType, TradeSignal

log = logging.getLogger("services.strategies.allocator")


@dataclass
class Limits:
    capital_rub: float = 5_000_000.0
    core_min_pct: float = 85.0              # минимум капитала в фонде
    satellite_cap_rub: float = 750_000.0    # все открытые позиции вместе
    per_strategy_rub: float = 300_000.0
    per_ticker_rub: float = 150_000.0

    @property
    def core_min_rub(self) -> float:
        return self.capital_rub * self.core_min_pct / 100.0

    @property
    def cash_cap_rub(self) -> float:
        """Сколько максимум можно вынуть из фонда (не опустив его ниже core)."""
        return min(self.satellite_cap_rub, self.capital_rub - self.core_min_rub)


@dataclass
class Allocation:
    signal: TradeSignal
    budget_rub: float
    efficiency: float


@dataclass
class Plan:
    accepted: list[Allocation] = field(default_factory=list)
    rejected: list[tuple] = field(default_factory=list)      # (signal, причина)
    cash_needed_rub: float = 0.0
    exposure_rub: float = 0.0
    by_strategy: dict = field(default_factory=dict)
    by_ticker: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"accepted": [{"strategy_id": a.signal.strategy_id, "ticker": a.signal.ticker,
                              "side": a.signal.side, "budget_rub": a.budget_rub,
                              "funding": a.signal.funding.value, "horizon": a.signal.horizon.value,
                              "efficiency": a.efficiency} for a in self.accepted],
                "rejected": [{"strategy_id": s.strategy_id, "ticker": s.ticker, "reason": r}
                             for s, r in self.rejected],
                "cash_needed_rub": self.cash_needed_rub, "exposure_rub": self.exposure_rub,
                "by_strategy": self.by_strategy, "by_ticker": self.by_ticker}


def efficiency(sig: TradeSignal) -> float:
    """Удельная эффективность: перевес на рубль и на день удержания."""
    denom = max(1.0, sig.target_budget_rub) * max(1, sig.max_holding_days)
    return sig.priority_score / denom


class PortfolioAllocator:
    def __init__(self, limits: Limits | None = None, adv_cap=None):
        self.limits = limits or Limits()
        self.adv_cap = adv_cap              # callable(ticker) -> лимит ₽ или None

    def allocate(self, signals: list[TradeSignal]) -> Plan:
        lim = self.limits
        plan = Plan()
        alive, by_ticker_sides = [], {}
        for s in signals:
            if s.target_budget_rub <= 0:
                plan.rejected.append((s, "нулевой бюджет"))
                continue
            by_ticker_sides.setdefault(s.ticker, set()).add(s.side)
            alive.append(s)
        conflicted = {tk for tk, sides in by_ticker_sides.items() if len(sides) > 1}
        kept = []
        for s in alive:
            if s.ticker in conflicted:
                plan.rejected.append((s, "встречные сигналы — позиция заблокирована"))
            else:
                kept.append(s)
        alive = sorted(kept, key=lambda s: (-efficiency(s), s.strategy_id, s.ticker))
        for s in alive:
            want = s.target_budget_rub
            cap_ticker = lim.per_ticker_rub - plan.by_ticker.get(s.ticker, 0.0)
            cap_strategy = lim.per_strategy_rub - plan.by_strategy.get(s.strategy_id, 0.0)
            cap_total = lim.satellite_cap_rub - plan.exposure_rub
            adv = self.adv_cap(s.ticker) if self.adv_cap else None
            cash_left = lim.cash_cap_rub - plan.cash_needed_rub if s.needs_cash else float("inf")
            caps = {"лимит на тикер 150 000 ₽": cap_ticker, "лимит на стратегию 300 000 ₽": cap_strategy,
                    "общий лимит позиций 750 000 ₽": cap_total,
                    "буфер фонда 85 %": cash_left}
            if adv is not None:
                caps["лимит ликвидности ADV"] = adv
            tight = min(caps, key=lambda k: caps[k])
            if caps[tight] < want:
                plan.rejected.append((s, f"{tight}: свободно {max(0.0, caps[tight]):.0f} ₽ из {want:.0f} ₽"))
                continue
            plan.accepted.append(Allocation(s, want, efficiency(s)))
            plan.exposure_rub += want
            plan.by_strategy[s.strategy_id] = plan.by_strategy.get(s.strategy_id, 0.0) + want
            plan.by_ticker[s.ticker] = plan.by_ticker.get(s.ticker, 0.0) + want
            if s.needs_cash:
                plan.cash_needed_rub += want
        log.info("[ALLOCATOR] принято %d, отклонено %d; позиции %.0f ₽, из фонда %.0f ₽",
                 len(plan.accepted), len(plan.rejected), plan.exposure_rub, plan.cash_needed_rub)
        return plan


def telegram_card(plan: Plan, fund_value_rub: float, capital_rub: float) -> str:
    """Разбивка активного микса для утренней карточки (ТЗ 7.4)."""
    lines = ["🎯 Активный микс стратегий:"]
    by = {}
    for a in plan.accepted:
        k = a.signal.strategy_id
        v = by.setdefault(k, {"n": 0, "rub": 0.0, "funding": a.signal.funding})
        v["n"] += 1
        v["rub"] += a.budget_rub
    names = {FundingType.MARGIN_FREE: "Free Margin", FundingType.CASH_UNPARK_TMON: "Unpark TMON",
             FundingType.DERIVATIVE_COLLATERAL: "ГО под залог"}
    for k, v in sorted(by.items()):
        lines.append(f"• {k}: {v['n']} поз. ({v['rub']:,.0f} ₽) [{names[v['funding']]}]".replace(",", " "))
    if not by:
        lines.append("• сделок нет — весь капитал в фонде")
    share = fund_value_rub / capital_rub * 100.0 if capital_rub else 0.0
    lines.append(f"🛡 Казначейство: TMON@ {fund_value_rub:,.0f} ₽ ({share:.1f} %)".replace(",", " "))
    return "\n".join(lines)
