from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


PETROL_SIGNAL_WORDS = ("petrol", "gasoline", "hybrid")
NON_E20_FUELS = ("diesel", "electric", "ev", "cng only", "lpg only")


@dataclass(frozen=True)
class E20Assessment:
    applies: bool
    stage: str
    risk_level: str
    confidence: str
    expected_mileage_loss_pct: tuple[float, float]
    notes: list[str]
    verification_questions: list[str]

    @property
    def mileage_loss_label(self) -> str:
        low, high = self.expected_mileage_loss_pct
        if low == 0 and high == 0:
            return "Not applicable"
        if low == high:
            return f"{low:.1f}%"
        return f"{low:.1f}% to {high:.1f}%"


def _contains_any(value: str, words: Iterable[str]) -> bool:
    normalized = value.strip().lower()
    return any(word in normalized for word in words)


def is_petrol_blend_vehicle(fuel_type: str) -> bool:
    fuel = (fuel_type or "").strip().lower()
    if not fuel:
        return False
    if fuel in NON_E20_FUELS:
        return False
    if "petrol/cng" in fuel or "petrol/lpg" in fuel:
        return True
    return _contains_any(fuel, PETROL_SIGNAL_WORDS) and "diesel" not in fuel


def _stage_from_year(model_year: int | None) -> tuple[str, tuple[float, float], float, str]:
    if not model_year:
        return (
            "Unknown E20 generation",
            (2.0, 7.0),
            2.5,
            "Low confidence because model year is missing.",
        )

    if model_year <= 2022:
        return (
            "Pre-E20 transition vehicle",
            (3.0, 7.0),
            3.0,
            "Older petrol vehicles need OEM/manual verification before assuming E20 readiness.",
        )

    if 2023 <= model_year <= 2024:
        return (
            "E20 material-compliant transition vehicle",
            (1.0, 4.0),
            1.5,
            "Likely better prepared for E20 materials, but may not be fully E20-tuned.",
        )

    return (
        "E20-tuned generation vehicle",
        (0.0, 2.0),
        0.5,
        "Newer vehicles should be closest to the intended E20 rollout assumptions.",
    )


def assess_e20_risk(
    model_year: int | None,
    fuel_type: str,
    bs_norm: str = "Unknown",
    km_driven: float | None = None,
    service_history: str = "Unknown",
) -> E20Assessment:
    """Estimate buyer-facing E20 risk using conservative public-roadmap rules.

    This is a rules layer, not a diagnosis. It should be paired with OEM guidance,
    vehicle manual checks, and service-center verification for a real purchase.
    """

    if not is_petrol_blend_vehicle(fuel_type):
        return E20Assessment(
            applies=False,
            stage="Not applicable to selected fuel",
            risk_level="Low",
            confidence="High",
            expected_mileage_loss_pct=(0.0, 0.0),
            notes=[
                "E20 is a petrol blend issue. The selected fuel type is not treated as petrol-blend dependent.",
                "Still verify fuel-specific service costs and local fuel availability separately.",
            ],
            verification_questions=[
                "Confirm the exact fuel type and variant from the registration certificate or owner manual.",
            ],
        )

    stage, mileage_loss, base_score, stage_note = _stage_from_year(model_year)
    score = base_score
    notes = [stage_note]
    verification_questions = [
        "Does the owner manual mention E20, ethanol blend tolerance, or fuel compatibility?",
        "Has the fuel system, injector cleaning, fuel filter, or rubber-line inspection been done recently?",
        "Can an authorised service center confirm E20 guidance for this exact model and variant?",
    ]

    if model_year:
        age = max(0, date.today().year - model_year)
        if age >= 10:
            score += 1.0
            notes.append("Vehicle age is 10+ years, so rubber, seals, and maintenance history matter more.")
        elif age >= 7:
            score += 0.5
            notes.append("Vehicle is older enough that service history should carry extra weight.")

    if km_driven is not None:
        if km_driven >= 150000:
            score += 1.5
            notes.append("Very high mileage increases inspection importance before purchase.")
        elif km_driven >= 100000:
            score += 1.0
            notes.append("High mileage adds moderate risk unless maintenance records are strong.")

    norm = (bs_norm or "").strip().upper()
    if norm in {"BS3", "BS III", "BHARAT STAGE III", "BS2", "BS II", "BHARAT STAGE II"}:
        score += 1.0
        notes.append("Older BS norm suggests extra caution and stronger verification.")
    elif norm in {"BS4", "BS IV", "BHARAT STAGE IV"}:
        score += 0.5
        notes.append("BS4 vehicles can be fine, but the exact OEM guidance still matters.")
    elif norm in {"BS6", "BS VI", "BHARAT STAGE VI"}:
        notes.append("BS6 is a positive signal, but model-year and OEM guidance still decide confidence.")

    service = (service_history or "").strip().lower()
    if service in {"no", "missing", "poor"}:
        score += 1.0
        notes.append("Missing or poor service records should push the buyer toward inspection or negotiation.")
    elif service in {"unknown", ""}:
        score += 0.5
        notes.append("Unknown service history keeps confidence limited.")
    elif service in {"yes", "complete", "authorised", "authorized"}:
        score -= 0.5
        notes.append("Documented service history lowers practical buyer risk.")

    if score <= 1.5:
        risk_level = "Low"
    elif score <= 3.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    confidence = "Medium"
    if model_year and bs_norm and service not in {"unknown", ""}:
        confidence = "Medium-High"
    if not model_year or norm in {"UNKNOWN", ""}:
        confidence = "Low-Medium"

    return E20Assessment(
        applies=True,
        stage=stage,
        risk_level=risk_level,
        confidence=confidence,
        expected_mileage_loss_pct=mileage_loss,
        notes=notes,
        verification_questions=verification_questions,
    )


def format_e20_report(assessment: E20Assessment) -> str:
    lines = [
        f"Stage: {assessment.stage}",
        f"Risk level: {assessment.risk_level}",
        f"Confidence: {assessment.confidence}",
        f"Expected mileage impact: {assessment.mileage_loss_label}",
        "",
        "Why:",
    ]
    lines.extend(f"- {note}" for note in assessment.notes)
    lines.append("")
    lines.append("Verify before purchase:")
    lines.extend(f"- {question}" for question in assessment.verification_questions)
    return "\n".join(lines)
