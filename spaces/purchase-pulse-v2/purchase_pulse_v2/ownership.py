from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .e20 import E20Assessment


@dataclass(frozen=True)
class OwnershipCost:
    years: int
    effective_mileage_kmpl: float
    fuel_cost_lakh: float
    insurance_cost_lakh: float
    service_cost_lakh: float
    depreciation_cost_lakh: float
    estimated_resale_lakh: float
    total_cost_lakh: float
    monthly_cost_rs: int
    assumptions: list[str]


def _default_service_cost_rs(model_year: int, km_driven: float, e20: E20Assessment) -> int:
    age = max(0, date.today().year - model_year) if model_year else 8
    annual = 9000
    if age >= 10:
        annual += 7000
    elif age >= 7:
        annual += 4000

    if km_driven >= 150000:
        annual += 7000
    elif km_driven >= 100000:
        annual += 4000

    if e20.risk_level == "High":
        annual += 4000
    elif e20.risk_level == "Medium":
        annual += 2000

    return annual


def _depreciation_rate(model_year: int, fuel_type: str, e20: E20Assessment) -> float:
    age = max(0, date.today().year - model_year) if model_year else 8
    rate = 0.11

    if age <= 3:
        rate = 0.13
    elif age >= 10:
        rate = 0.08

    fuel = (fuel_type or "").strip().lower()
    if "diesel" in fuel:
        rate += 0.01
    if e20.risk_level == "High":
        rate += 0.01
    if "ev" in fuel or "electric" in fuel:
        rate += 0.02

    return min(max(rate, 0.06), 0.18)


def estimate_ownership_cost(
    purchase_price_lakh: float,
    model_year: int,
    km_driven: float,
    fuel_type: str,
    e20_assessment: E20Assessment,
    years: int = 5,
    annual_km: float = 12000,
    mileage_kmpl: float = 15.0,
    fuel_price_per_litre: float = 105.0,
    annual_insurance_pct: float = 2.5,
    annual_service_cost_rs: float | None = None,
) -> OwnershipCost:
    years = max(1, int(years))
    purchase_price_lakh = max(0.0, float(purchase_price_lakh or 0.0))
    annual_km = max(0.0, float(annual_km or 0.0))
    mileage_kmpl = max(1.0, float(mileage_kmpl or 1.0))
    fuel_price_per_litre = max(0.0, float(fuel_price_per_litre or 0.0))
    annual_insurance_pct = max(0.0, float(annual_insurance_pct or 0.0))

    loss_low, loss_high = e20_assessment.expected_mileage_loss_pct
    mileage_loss_mid = (loss_low + loss_high) / 2
    effective_mileage = mileage_kmpl * (1 - mileage_loss_mid / 100)
    effective_mileage = max(1.0, effective_mileage)

    fuel_cost_lakh = (annual_km * years / effective_mileage * fuel_price_per_litre) / 100000

    insurance_cost_lakh = purchase_price_lakh * (annual_insurance_pct / 100) * years

    if annual_service_cost_rs is None or annual_service_cost_rs <= 0:
        annual_service_cost_rs = _default_service_cost_rs(model_year, km_driven, e20_assessment)
    service_cost_lakh = (annual_service_cost_rs * years) / 100000

    dep_rate = _depreciation_rate(model_year, fuel_type, e20_assessment)
    estimated_resale_lakh = purchase_price_lakh * ((1 - dep_rate) ** years)
    estimated_resale_lakh = max(0.15 * purchase_price_lakh, estimated_resale_lakh)
    depreciation_cost_lakh = purchase_price_lakh - estimated_resale_lakh

    total_cost_lakh = fuel_cost_lakh + insurance_cost_lakh + service_cost_lakh + depreciation_cost_lakh
    monthly_cost_rs = int(round(total_cost_lakh * 100000 / (years * 12)))

    assumptions = [
        f"Ownership horizon: {years} years",
        f"Annual usage: {annual_km:,.0f} km",
        f"Base mileage: {mileage_kmpl:.1f} km/l",
        f"Effective mileage after E20/risk adjustment: {effective_mileage:.1f} km/l",
        f"Fuel price: Rs {fuel_price_per_litre:.0f}/litre",
        f"Annual insurance assumption: {annual_insurance_pct:.1f}% of purchase price",
        f"Annual service buffer: Rs {annual_service_cost_rs:,.0f}",
        f"Depreciation rate assumption: {dep_rate * 100:.1f}% per year",
    ]

    return OwnershipCost(
        years=years,
        effective_mileage_kmpl=round(effective_mileage, 1),
        fuel_cost_lakh=round(fuel_cost_lakh, 2),
        insurance_cost_lakh=round(insurance_cost_lakh, 2),
        service_cost_lakh=round(service_cost_lakh, 2),
        depreciation_cost_lakh=round(depreciation_cost_lakh, 2),
        estimated_resale_lakh=round(estimated_resale_lakh, 2),
        total_cost_lakh=round(total_cost_lakh, 2),
        monthly_cost_rs=monthly_cost_rs,
        assumptions=assumptions,
    )


def format_ownership_report(cost: OwnershipCost) -> str:
    lines = [
        f"Total {cost.years}-year ownership cost: Rs {cost.total_cost_lakh:.2f}L",
        f"Estimated monthly cost: Rs {cost.monthly_cost_rs:,}",
        f"Estimated resale after {cost.years} years: Rs {cost.estimated_resale_lakh:.2f}L",
        "",
        "Cost breakdown:",
        f"- Fuel: Rs {cost.fuel_cost_lakh:.2f}L",
        f"- Insurance: Rs {cost.insurance_cost_lakh:.2f}L",
        f"- Service buffer: Rs {cost.service_cost_lakh:.2f}L",
        f"- Depreciation: Rs {cost.depreciation_cost_lakh:.2f}L",
        "",
        "Assumptions:",
    ]
    lines.extend(f"- {assumption}" for assumption in cost.assumptions)
    return "\n".join(lines)
