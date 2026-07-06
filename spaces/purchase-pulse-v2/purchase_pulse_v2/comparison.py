from __future__ import annotations

from dataclasses import dataclass

from .e20 import assess_e20_risk
from .ownership import estimate_ownership_cost
from .pricing import assess_price
from .scoring import calculate_deal_score


@dataclass(frozen=True)
class ComparedCar:
    rank: int
    name: str
    score: int
    verdict: str
    fair_band: str
    e20_risk: str
    ownership_cost_lakh: float
    notes: list[str]


def _parts(line: str) -> list[str]:
    return [part.strip() for part in line.split("|")]


def parse_comparison_lines(text: str) -> list[dict]:
    cars: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _parts(line)
        if len(parts) < 11:
            continue

        def to_float(idx: int, default: float = 0.0) -> float:
            try:
                return float(parts[idx])
            except (IndexError, TypeError, ValueError):
                return default

        def to_int(idx: int, default: int = 0) -> int:
            try:
                return int(float(parts[idx]))
            except (IndexError, TypeError, ValueError):
                return default

        cars.append(
            {
                "name": parts[0],
                "year": to_int(1),
                "km": to_float(2),
                "fuel": parts[3],
                "transmission": parts[4],
                "owners": to_int(5, 1),
                "asking": to_float(6),
                "market_median": to_float(7) or None,
                "bs_norm": parts[8],
                "service": parts[9],
                "mileage": to_float(10, 15.0),
            }
        )
    return cars


def compare_cars(
    text: str,
    years: int = 5,
    annual_km: float = 12000,
    fuel_price_per_litre: float = 105.0,
) -> list[ComparedCar]:
    compared: list[ComparedCar] = []
    for car in parse_comparison_lines(text):
        e20 = assess_e20_risk(
            model_year=car["year"],
            fuel_type=car["fuel"],
            bs_norm=car["bs_norm"],
            km_driven=car["km"],
            service_history=car["service"],
        )
        price = assess_price(
            asking_price_lakh=car["asking"],
            market_median_lakh=car["market_median"],
            e20_assessment=e20,
            km_driven=car["km"],
            owner_count=car["owners"],
            seller_type="Unknown",
            service_history=car["service"],
        )
        ownership = estimate_ownership_cost(
            purchase_price_lakh=car["asking"],
            model_year=car["year"],
            km_driven=car["km"],
            fuel_type=car["fuel"],
            e20_assessment=e20,
            years=years,
            annual_km=annual_km,
            mileage_kmpl=car["mileage"],
            fuel_price_per_litre=fuel_price_per_litre,
        )
        score = calculate_deal_score(
            price_assessment=price,
            e20_assessment=e20,
            ownership_cost=ownership,
            model_year=car["year"],
            km_driven=car["km"],
            owner_count=car["owners"],
            service_history=car["service"],
            asking_price_lakh=car["asking"],
            market_median_lakh=car["market_median"],
        )
        notes = []
        notes.extend(score.concerns[:2])
        notes.extend(score.positives[:1])
        compared.append(
            ComparedCar(
                rank=0,
                name=car["name"],
                score=score.score,
                verdict=score.label,
                fair_band=price.fair_band_label,
                e20_risk=e20.risk_level,
                ownership_cost_lakh=ownership.total_cost_lakh,
                notes=notes[:3],
            )
        )

    ranked = sorted(compared, key=lambda item: (-item.score, item.ownership_cost_lakh))
    return [
        ComparedCar(
            rank=index + 1,
            name=item.name,
            score=item.score,
            verdict=item.verdict,
            fair_band=item.fair_band,
            e20_risk=item.e20_risk,
            ownership_cost_lakh=item.ownership_cost_lakh,
            notes=item.notes,
        )
        for index, item in enumerate(ranked)
    ]


def format_comparison_table(cars: list[ComparedCar]) -> str:
    if not cars:
        return (
            "Add cars using this format, one per line:\n\n"
            "`Name | Year | Km | Fuel | Transmission | Owners | AskingL | MarketMedianL | BSNorm | ServiceHistory | MileageKmpl`"
        )

    lines = [
        "| Rank | Car | Score | Verdict | Fair band | E20 | 5-year cost | Notes |",
        "|---:|---|---:|---|---|---|---:|---|",
    ]
    for car in cars:
        notes = "; ".join(car.notes) if car.notes else "No major notes"
        lines.append(
            f"| {car.rank} | {car.name} | {car.score} | {car.verdict} | {car.fair_band} | "
            f"{car.e20_risk} | Rs {car.ownership_cost_lakh:.2f}L | {notes} |"
        )
    return "\n".join(lines)
