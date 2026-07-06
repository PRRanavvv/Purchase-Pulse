from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .e20 import E20Assessment
from .ownership import OwnershipCost
from .pricing import PriceAssessment


@dataclass(frozen=True)
class DealScore:
    score: int
    label: str
    confidence: str
    positives: list[str]
    concerns: list[str]


def _label(score: int) -> str:
    if score >= 82:
        return "Strong deal"
    if score >= 68:
        return "Good with checks"
    if score >= 52:
        return "Fair but negotiate"
    if score >= 35:
        return "Risky"
    return "Avoid unless inspected deeply"


def calculate_deal_score(
    price_assessment: PriceAssessment,
    e20_assessment: E20Assessment,
    ownership_cost: OwnershipCost,
    model_year: int,
    km_driven: float,
    owner_count: int,
    service_history: str,
    asking_price_lakh: float,
    market_median_lakh: float | None,
) -> DealScore:
    score = 70
    positives: list[str] = []
    concerns: list[str] = []

    if price_assessment.verdict == "Fair range":
        score += 8
        positives.append("Asking price sits inside the adjusted fair band.")
    elif price_assessment.verdict == "Overpriced":
        score -= 16
        concerns.append("Asking price is above the adjusted fair band.")
    elif price_assessment.verdict.startswith("Below market"):
        score -= 5
        concerns.append("Below-market price needs inspection before it becomes a positive.")
    else:
        score -= 8
        concerns.append("No comparable market median was provided, so price confidence is limited.")

    if e20_assessment.risk_level == "Low":
        score += 8
        positives.append("E20/running-cost risk is low for the entered details.")
    elif e20_assessment.risk_level == "Medium":
        score -= 5
        concerns.append("E20/running-cost risk is medium; verify exact model guidance.")
    else:
        score -= 15
        concerns.append("E20/running-cost risk is high enough to affect negotiation.")

    current_year = date.today().year
    age = max(0, current_year - model_year) if model_year else 8
    if age <= 5:
        score += 6
        positives.append("Vehicle age is still moderate.")
    elif age >= 10:
        score -= 8
        concerns.append("Vehicle age is high; maintenance uncertainty rises.")

    if km_driven <= 60000:
        score += 6
        positives.append("Kilometres are in a comfortable range.")
    elif km_driven >= 150000:
        score -= 14
        concerns.append("Very high kilometres make inspection and service records essential.")
    elif km_driven >= 100000:
        score -= 8
        concerns.append("High kilometres should be reflected in the price.")

    yearly_km = km_driven / max(age, 1)
    if yearly_km > 25000:
        score -= 6
        concerns.append("Usage per year is unusually high.")
    elif age >= 3 and yearly_km < 2500:
        score -= 4
        concerns.append("Very low usage per year can be suspicious; verify odometer and service invoices.")

    if owner_count == 1:
        score += 5
        positives.append("Single-owner status is a positive resale signal.")
    elif owner_count >= 3:
        score -= 9
        concerns.append("Multiple previous owners lower confidence.")

    service = (service_history or "").strip().lower()
    if service == "complete":
        score += 8
        positives.append("Complete service history materially improves confidence.")
    elif service == "partial":
        score -= 2
        concerns.append("Partial service history leaves some open questions.")
    elif service in {"missing", "unknown", ""}:
        score -= 8
        concerns.append("Missing or unknown service history is a major buyer check.")

    if asking_price_lakh > 0:
        cost_ratio = ownership_cost.total_cost_lakh / asking_price_lakh
        if cost_ratio > 0.95:
            score -= 6
            concerns.append("Projected ownership cost is high relative to purchase price.")
        elif cost_ratio < 0.65:
            score += 4
            positives.append("Projected ownership cost is reasonable relative to purchase price.")

    confidence = "Medium"
    if market_median_lakh and service not in {"unknown", ""} and e20_assessment.confidence in {"Medium-High", "High"}:
        confidence = "Medium-High"
    if not market_median_lakh:
        confidence = "Low-Medium"

    score = min(100, max(0, round(score)))
    return DealScore(
        score=score,
        label=_label(score),
        confidence=confidence,
        positives=positives[:6],
        concerns=concerns[:6],
    )


def format_deal_score(score: DealScore) -> str:
    lines = [
        f"Deal score: {score.score}/100",
        f"Verdict: {score.label}",
        f"Confidence: {score.confidence}",
        "",
        "What helps:",
    ]
    lines.extend(f"- {item}" for item in (score.positives or ["No strong positives yet."]))
    lines.append("")
    lines.append("What hurts:")
    lines.extend(f"- {item}" for item in (score.concerns or ["No major concerns from the entered details."]))
    return "\n".join(lines)
