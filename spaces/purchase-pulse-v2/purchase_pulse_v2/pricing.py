from __future__ import annotations

from dataclasses import dataclass

from .e20 import E20Assessment


@dataclass(frozen=True)
class PriceAssessment:
    verdict: str
    fair_low_lakh: float | None
    fair_high_lakh: float | None
    negotiate_from_lakh: float | None
    notes: list[str]

    @property
    def fair_band_label(self) -> str:
        if self.fair_low_lakh is None or self.fair_high_lakh is None:
            return "Needs local market comps"
        return f"Rs {self.fair_low_lakh:.2f}L to Rs {self.fair_high_lakh:.2f}L"


def assess_price(
    asking_price_lakh: float | None,
    market_median_lakh: float | None,
    e20_assessment: E20Assessment,
    km_driven: float | None,
    owner_count: int,
    seller_type: str,
    service_history: str,
) -> PriceAssessment:
    if asking_price_lakh is None or asking_price_lakh <= 0:
        return PriceAssessment(
            verdict="Add asking price",
            fair_low_lakh=None,
            fair_high_lakh=None,
            negotiate_from_lakh=None,
            notes=["The app needs the listing price before it can judge a deal."],
        )

    if market_median_lakh is None or market_median_lakh <= 0:
        return PriceAssessment(
            verdict="Collect local comps",
            fair_low_lakh=None,
            fair_high_lakh=None,
            negotiate_from_lakh=None,
            notes=[
                "No market median was provided, so this version will not fake a fair-price estimate.",
                "Collect 5 to 10 comparable listings for the same city, year band, variant, fuel, transmission, and km range.",
                "Use their median asking price as the market-comp input.",
            ],
        )

    discount = 0.0
    notes: list[str] = []

    if e20_assessment.risk_level == "High":
        discount += 0.04
        notes.append("High E20/maintenance uncertainty adds a risk discount.")
    elif e20_assessment.risk_level == "Medium":
        discount += 0.02
        notes.append("Medium E20/maintenance uncertainty adds a small risk discount.")

    if km_driven is not None:
        if km_driven >= 150000:
            discount += 0.05
            notes.append("Very high mileage should be reflected in negotiation.")
        elif km_driven >= 100000:
            discount += 0.03
            notes.append("High mileage should slightly lower the fair band.")

    if owner_count >= 3:
        discount += 0.04
        notes.append("Multiple previous owners increase resale and condition uncertainty.")
    elif owner_count == 2:
        discount += 0.02
        notes.append("Second-owner cars usually need a small discount against single-owner comps.")

    service = (service_history or "").strip().lower()
    if service in {"no", "missing", "poor"}:
        discount += 0.03
        notes.append("Missing service history is a direct negotiation point.")
    elif service in {"unknown", ""}:
        discount += 0.015
        notes.append("Unknown service history keeps the buyer from paying top-of-market.")

    seller = (seller_type or "").strip().lower()
    if seller == "dealer":
        notes.append("Dealer listings may include margin, warranty, refurbishment, or convenience premium.")
    elif seller == "individual":
        notes.append("Individual listings often have more room for direct negotiation.")

    discount = min(discount, 0.18)
    fair_center = market_median_lakh * (1 - discount)
    fair_low = fair_center * 0.93
    fair_high = fair_center * 1.07

    if asking_price_lakh > fair_high:
        verdict = "Overpriced"
        negotiate_from = fair_center
        notes.append("Listing is above the adjusted fair band.")
    elif asking_price_lakh < fair_low:
        verdict = "Below market - inspect carefully"
        negotiate_from = asking_price_lakh
        notes.append("Price is below the adjusted band. Check for accident, loan, flood, meter rollback, or documentation issues.")
    else:
        verdict = "Fair range"
        negotiate_from = min(asking_price_lakh, fair_center)
        notes.append("Listing sits inside the adjusted fair band.")

    return PriceAssessment(
        verdict=verdict,
        fair_low_lakh=round(fair_low, 2),
        fair_high_lakh=round(fair_high, 2),
        negotiate_from_lakh=round(negotiate_from, 2),
        notes=notes,
    )


def build_buyer_report(
    make: str,
    model: str,
    variant: str,
    city: str,
    model_year: int,
    km_driven: float,
    fuel_type: str,
    transmission: str,
    owner_count: int,
    asking_price_lakh: float,
    price_assessment: PriceAssessment,
    e20_assessment: E20Assessment,
) -> str:
    vehicle_name = " ".join(part for part in [make, model, variant] if part).strip() or "Selected vehicle"
    negotiate = (
        f"Target negotiation anchor: Rs {price_assessment.negotiate_from_lakh:.2f}L"
        if price_assessment.negotiate_from_lakh is not None
        else "Target negotiation anchor: collect comps first"
    )

    decision = "Proceed with checks"
    if price_assessment.verdict == "Overpriced" or e20_assessment.risk_level == "High":
        decision = "Negotiate hard or pause"
    if price_assessment.verdict.startswith("Below market"):
        decision = "Inspect before getting excited"

    lines = [
        f"## {vehicle_name} buyer report",
        "",
        f"City: {city or 'Unknown'}",
        f"Year: {model_year}",
        f"Fuel / transmission: {fuel_type} / {transmission}",
        f"Kilometres: {km_driven:,.0f}",
        f"Owners: {owner_count}",
        f"Asking price: Rs {asking_price_lakh:.2f}L",
        "",
        f"### Decision: {decision}",
        f"- Price verdict: {price_assessment.verdict}",
        f"- Adjusted fair band: {price_assessment.fair_band_label}",
        f"- {negotiate}",
        f"- E20 risk: {e20_assessment.risk_level} ({e20_assessment.stage})",
        f"- Expected mileage impact: {e20_assessment.mileage_loss_label}",
        "",
        "### Why this decision",
    ]
    lines.extend(f"- {note}" for note in price_assessment.notes[:6])
    lines.extend(f"- {note}" for note in e20_assessment.notes[:4])
    lines.extend(
        [
            "",
            "### Ask before paying",
            "- Can the seller show full service history and insurance claim history?",
            "- Has the odometer reading been cross-checked with service invoices?",
            "- Are RC, hypothecation, challan, insurance, PUC, and NOC status clean?",
            "- Can an authorised service center inspect fuel-system and engine health?",
        ]
    )
    return "\n".join(lines)
