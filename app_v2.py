from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from purchase_pulse_v2.comparison import compare_cars, format_comparison_table
from purchase_pulse_v2.e20 import assess_e20_risk, format_e20_report
from purchase_pulse_v2.negotiation import format_negotiation_block
from purchase_pulse_v2.ownership import estimate_ownership_cost, format_ownership_report
from purchase_pulse_v2.pricing import assess_price, build_buyer_report
from purchase_pulse_v2.rag import LocalRagAssistant
from purchase_pulse_v2.scoring import calculate_deal_score, format_deal_score


ROOT = Path(__file__).resolve().parent
RAG = LocalRagAssistant(ROOT / "knowledge")


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_buyer_check(
    make,
    model,
    variant,
    city,
    model_year,
    km_driven,
    fuel_type,
    transmission,
    owner_count,
    seller_type,
    asking_price_lakh,
    market_median_lakh,
    bs_norm,
    service_history,
    ownership_years,
    annual_km,
    mileage_kmpl,
    fuel_price_per_litre,
    annual_insurance_pct,
    annual_service_cost_rs,
):
    model_year = _coerce_int(model_year, 0)
    km_driven = _coerce_float(km_driven, 0.0) or 0.0
    owner_count = _coerce_int(owner_count, 1)
    asking_price_lakh = _coerce_float(asking_price_lakh, 0.0) or 0.0
    market_median_lakh = _coerce_float(market_median_lakh)
    ownership_years = _coerce_int(ownership_years, 5)
    annual_km = _coerce_float(annual_km, 12000.0) or 12000.0
    mileage_kmpl = _coerce_float(mileage_kmpl, 15.0) or 15.0
    fuel_price_per_litre = _coerce_float(fuel_price_per_litre, 105.0) or 105.0
    annual_insurance_pct = _coerce_float(annual_insurance_pct, 2.5) or 2.5
    annual_service_cost_rs = _coerce_float(annual_service_cost_rs)

    e20 = assess_e20_risk(
        model_year=model_year,
        fuel_type=fuel_type,
        bs_norm=bs_norm,
        km_driven=km_driven,
        service_history=service_history,
    )
    price = assess_price(
        asking_price_lakh=asking_price_lakh,
        market_median_lakh=market_median_lakh,
        e20_assessment=e20,
        km_driven=km_driven,
        owner_count=owner_count,
        seller_type=seller_type,
        service_history=service_history,
    )
    ownership = estimate_ownership_cost(
        purchase_price_lakh=asking_price_lakh,
        model_year=model_year,
        km_driven=km_driven,
        fuel_type=fuel_type,
        e20_assessment=e20,
        years=ownership_years,
        annual_km=annual_km,
        mileage_kmpl=mileage_kmpl,
        fuel_price_per_litre=fuel_price_per_litre,
        annual_insurance_pct=annual_insurance_pct,
        annual_service_cost_rs=annual_service_cost_rs,
    )
    deal_score = calculate_deal_score(
        price_assessment=price,
        e20_assessment=e20,
        ownership_cost=ownership,
        model_year=model_year,
        km_driven=km_driven,
        owner_count=owner_count,
        service_history=service_history,
        asking_price_lakh=asking_price_lakh,
        market_median_lakh=market_median_lakh,
    )

    report = build_buyer_report(
        make=make,
        model=model,
        variant=variant,
        city=city,
        model_year=model_year,
        km_driven=km_driven,
        fuel_type=fuel_type,
        transmission=transmission,
        owner_count=owner_count,
        asking_price_lakh=asking_price_lakh,
        price_assessment=price,
        e20_assessment=e20,
    )

    decision = deal_score.label
    if deal_score.score < 52 or price.verdict == "Overpriced" or e20.risk_level == "High":
        decision = "Negotiate hard or pause"
    if price.verdict.startswith("Below market"):
        decision = "Inspect before getting excited"

    price_summary = (
        f"{price.verdict}\n"
        f"Fair band: {price.fair_band_label}\n"
        f"Negotiation anchor: "
        f"{'collect comps first' if price.negotiate_from_lakh is None else f'Rs {price.negotiate_from_lakh:.2f}L'}"
    )

    vehicle_name = " ".join(part for part in [make, model, variant] if part).strip()
    e20_summary = format_e20_report(e20)
    deal_summary = format_deal_score(deal_score)
    ownership_summary = format_ownership_report(ownership)
    negotiation = format_negotiation_block(
        vehicle_name,
        asking_price_lakh,
        price,
        e20,
        ownership,
        deal_score,
    )
    report = "\n\n".join(
        [
            report,
            "### Deal score",
            deal_summary,
            "### Ownership cost",
            ownership_summary,
            "### Negotiation copilot",
            negotiation,
        ]
    )
    return decision, deal_summary, price_summary, ownership_summary, e20_summary, negotiation, report


def ask_assistant(question):
    return RAG.answer(question)


def compare_listings(comparison_text, ownership_years, annual_km, fuel_price_per_litre):
    cars = compare_cars(
        comparison_text,
        years=_coerce_int(ownership_years, 5),
        annual_km=_coerce_float(annual_km, 12000.0) or 12000.0,
        fuel_price_per_litre=_coerce_float(fuel_price_per_litre, 105.0) or 105.0,
    )
    return format_comparison_table(cars)


SCHEMA_PREVIEW = """
### Minimum dataset columns

`make, model, variant, city, state, model_year, listing_year, km_driven, fuel_type, transmission, owner_count, seller_type, asking_price_lakh, engine_cc, power_bhp, arai_mileage_kmpl, bs_norm, e20_stage, expected_mileage_loss_pct, service_history, accident_history, source_name, source_url, date_collected`

Start with 5 to 10 comparable listings for one model-city cluster. The product gets useful much faster if the first dataset is narrow but clean.
"""


COMPARISON_EXAMPLE = """Hyundai i20 Asta | 2019 | 52000 | Petrol | Manual | 1 | 5.8 | 5.4 | BS6 | Unknown | 17
Maruti Swift ZXI | 2020 | 41000 | Petrol | Manual | 1 | 5.9 | 5.7 | BS6 | Complete | 18
Honda City VX | 2018 | 76000 | Petrol | Manual | 2 | 7.0 | 6.8 | BS4 | Partial | 15"""


with gr.Blocks(title="Purchase Pulse 2.0") as demo:
    gr.Markdown(
        """
        # Purchase Pulse 2.0

        India-focused used-car buyer-risk assistant with fair-band pricing, E20 risk, and local knowledge retrieval.
        """
    )

    with gr.Tab("Buyer Check"):
        with gr.Row():
            with gr.Column():
                make = gr.Textbox(label="Make", value="Hyundai")
                model = gr.Textbox(label="Model", value="i20")
                variant = gr.Textbox(label="Variant", value="Asta")
                city = gr.Textbox(label="City", value="Delhi")
                model_year = gr.Slider(2005, 2026, value=2019, step=1, label="Model / registration year")
                km_driven = gr.Number(value=52000, label="Kilometres driven")
            with gr.Column():
                fuel_type = gr.Dropdown(
                    choices=["Petrol", "Petrol/CNG", "Petrol/Hybrid", "Diesel", "CNG only", "EV"],
                    value="Petrol",
                    label="Fuel type",
                )
                transmission = gr.Dropdown(
                    choices=["Manual", "Automatic", "AMT", "CVT", "DCT", "Unknown"],
                    value="Manual",
                    label="Transmission",
                )
                owner_count = gr.Slider(1, 5, value=1, step=1, label="Owner count")
                seller_type = gr.Dropdown(
                    choices=["Individual", "Dealer", "Platform", "Auction", "Unknown"],
                    value="Individual",
                    label="Seller type",
                )
                bs_norm = gr.Dropdown(
                    choices=["Unknown", "BS3", "BS4", "BS6"],
                    value="BS6",
                    label="Emission norm",
                )
                service_history = gr.Dropdown(
                    choices=["Complete", "Partial", "Missing", "Unknown"],
                    value="Unknown",
                    label="Service history",
                )
            with gr.Column():
                asking_price_lakh = gr.Number(value=5.8, label="Asking price (lakh INR)")
                market_median_lakh = gr.Number(
                    value=None,
                    label="Median of local comparable listings (optional, lakh INR)",
                    info="Collect 5 to 10 similar listings and enter their median asking price.",
                )
                ownership_years = gr.Slider(1, 7, value=5, step=1, label="Ownership horizon (years)")
                annual_km = gr.Number(value=12000, label="Expected annual km")
                mileage_kmpl = gr.Number(value=17, label="Expected real-world mileage (km/l)")
                fuel_price_per_litre = gr.Number(value=105, label="Fuel price (Rs/litre)")
                annual_insurance_pct = gr.Number(value=2.5, label="Annual insurance (% of purchase price)")
                annual_service_cost_rs = gr.Number(
                    value=None,
                    label="Annual service buffer (optional, Rs)",
                    info="Leave blank to let the app estimate from age, km, and E20 risk.",
                )

        check_btn = gr.Button("Generate buyer report", variant="primary")

        with gr.Row():
            decision = gr.Textbox(label="Decision", interactive=False)
            price_summary = gr.Textbox(label="Price view", lines=4, interactive=False)
        deal_summary = gr.Textbox(label="Deal score", lines=10, interactive=False)
        ownership_summary = gr.Textbox(label="Ownership cost", lines=13, interactive=False)
        e20_summary = gr.Textbox(label="E20 view", lines=10, interactive=False)
        negotiation = gr.Textbox(label="Negotiation copilot", lines=6, interactive=False)
        report = gr.Markdown(label="Buyer report")

        check_btn.click(
            run_buyer_check,
            inputs=[
                make,
                model,
                variant,
                city,
                model_year,
                km_driven,
                fuel_type,
                transmission,
                owner_count,
                seller_type,
                asking_price_lakh,
                market_median_lakh,
                bs_norm,
                service_history,
                ownership_years,
                annual_km,
                mileage_kmpl,
                fuel_price_per_litre,
                annual_insurance_pct,
                annual_service_cost_rs,
            ],
            outputs=[decision, deal_summary, price_summary, ownership_summary, e20_summary, negotiation, report],
        )

    with gr.Tab("Compare Cars"):
        gr.Markdown(
            """
            Paste up to a few listings, one per line:

            `Name | Year | Km | Fuel | Transmission | Owners | AskingL | MarketMedianL | BSNorm | ServiceHistory | MileageKmpl`
            """
        )
        comparison_text = gr.Textbox(label="Listings", value=COMPARISON_EXAMPLE, lines=6)
        with gr.Row():
            comparison_years = gr.Slider(1, 7, value=5, step=1, label="Ownership horizon")
            comparison_annual_km = gr.Number(value=12000, label="Annual km")
            comparison_fuel_price = gr.Number(value=105, label="Fuel price (Rs/litre)")
        compare_btn = gr.Button("Rank cars", variant="primary")
        comparison_output = gr.Markdown()
        compare_btn.click(
            compare_listings,
            inputs=[comparison_text, comparison_years, comparison_annual_km, comparison_fuel_price],
            outputs=[comparison_output],
        )

    with gr.Tab("Ask Assistant"):
        gr.Markdown("Ask about dataset collection, E20 risk, pricing logic, or buyer checks. Answers are grounded in local markdown notes.")
        question = gr.Textbox(
            label="Question",
            value="How should I collect a dataset for fair used-car pricing in India?",
            lines=3,
        )
        answer_btn = gr.Button("Ask")
        answer = gr.Markdown()
        answer_btn.click(ask_assistant, inputs=[question], outputs=[answer])

    with gr.Tab("Dataset"):
        gr.Markdown(SCHEMA_PREVIEW)
        gr.Markdown("Template file: `data/purchase_pulse_v2_template.csv`")


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PURCHASE_PULSE_PORT", "7861")),
    )
