from __future__ import annotations

from .e20 import E20Assessment
from .ownership import OwnershipCost
from .pricing import PriceAssessment
from .scoring import DealScore


def negotiation_anchor(
    asking_price_lakh: float,
    price_assessment: PriceAssessment,
    deal_score: DealScore,
) -> float:
    if price_assessment.negotiate_from_lakh is not None:
        anchor = price_assessment.negotiate_from_lakh
    elif price_assessment.fair_high_lakh is not None:
        anchor = price_assessment.fair_high_lakh
    else:
        anchor = asking_price_lakh * 0.92

    if deal_score.score < 50:
        anchor *= 0.96
    elif deal_score.score >= 82:
        anchor *= 1.01

    return round(max(0.1, min(anchor, asking_price_lakh)), 2)


def generate_negotiation_message(
    vehicle_name: str,
    asking_price_lakh: float,
    price_assessment: PriceAssessment,
    e20_assessment: E20Assessment,
    ownership_cost: OwnershipCost,
    deal_score: DealScore,
) -> str:
    anchor = negotiation_anchor(asking_price_lakh, price_assessment, deal_score)
    reasons = []

    if price_assessment.verdict == "Overpriced":
        reasons.append("the asking price is above comparable market value")
    if e20_assessment.risk_level in {"Medium", "High"}:
        reasons.append("I need to account for E20/running-cost verification")
    if deal_score.concerns:
        reasons.append(deal_score.concerns[0].lower().rstrip("."))
    reasons.append(f"my estimated {ownership_cost.years}-year ownership cost is about Rs {ownership_cost.total_cost_lakh:.2f}L")

    reason_text = ", ".join(dict.fromkeys(reasons))
    name = vehicle_name or "the car"

    return (
        f"Hi, I am interested in {name}. I checked the listing against comparable prices and ownership costs. "
        f"Given that {reason_text}, my comfortable offer is around Rs {anchor:.2f}L. "
        "If the service records, insurance history, RC status, and inspection report are clean, I can move ahead quickly."
    )


def format_negotiation_block(
    vehicle_name: str,
    asking_price_lakh: float,
    price_assessment: PriceAssessment,
    e20_assessment: E20Assessment,
    ownership_cost: OwnershipCost,
    deal_score: DealScore,
) -> str:
    anchor = negotiation_anchor(asking_price_lakh, price_assessment, deal_score)
    message = generate_negotiation_message(
        vehicle_name,
        asking_price_lakh,
        price_assessment,
        e20_assessment,
        ownership_cost,
        deal_score,
    )
    return "\n".join(
        [
            f"Negotiation anchor: Rs {anchor:.2f}L",
            "",
            "Seller message:",
            message,
        ]
    )
