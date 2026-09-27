"""
T-Invest API — торгуемость, флаги и рейтинги облигаций (InstrumentsService/Bonds).

flags(br) → {ISIN: {...}}: можно ли купить через API, только для квалов, нужен ли
тест, плавающий купон, субординация, бессрочность, амортизация, сектор, объём
выпуска, рейтинги агентств (агентство, уровень, дата, прогноз, «на пересмотре»),
лот, номинал, uid/figi. ISIN совпадает с SECID ISS для корпоративных облигаций.
"""
from __future__ import annotations


def _q(x) -> float:
    return (int(x.get("units", 0) or 0) + int(x.get("nano", 0) or 0) / 1e9) if x else 0.0


def flags(br) -> dict:
    data = br._post("InstrumentsService/Bonds", {"instrumentStatus": "INSTRUMENT_STATUS_BASE"})
    out = {}
    for b in data.get("instruments", []) or []:
        if b.get("currency") != "rub" or b.get("classCode") != "TQCB":
            continue
        ratings = b.get("ratings") or []
        nominal = _q(b.get("nominal"))
        out[b["isin"]] = {
            "name": b.get("name"), "uid": b.get("uid"), "figi": b.get("figi"),
            "tradable": bool(b.get("apiTradeAvailableFlag") and b.get("buyAvailableFlag")),
            "sell_ok": bool(b.get("sellAvailableFlag", True)),
            "qual_only": bool(b.get("forQualInvestorFlag")),
            "required_tests": bool(b.get("requiredTests")),
            "floating": bool(b.get("floatingCouponFlag")),
            "subordinated": bool(b.get("subordinatedFlag")),
            "perpetual": bool(b.get("perpetualFlag")),
            "amortization": bool(b.get("amortizationFlag")),
            "sector": b.get("sector") or None,
            "issue_size_rub": (int(b.get("issueSize") or 0) * nominal) or None,
            "lot": int(b.get("lot") or 1), "nominal": nominal,
            "min_step": _q(b.get("minPriceIncrement")),
            "risk_level": b.get("riskLevel"),
            "ratings": [f"{r.get('agencyName')}:{r.get('ratingLevel')}" for r in ratings],
            "negative_watch": any(str(r.get("forecast", "")).lower().startswith("негатив")
                                  or r.get("isUnderWatch") for r in ratings),
        }
    return out
