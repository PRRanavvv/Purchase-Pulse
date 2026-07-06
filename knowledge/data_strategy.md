# Dataset Strategy

## The perfect dataset probably does not exist

A strong India-focused used-car buyer dataset needs columns that are usually split across many sources: listing price, location, model variant, fuel type, transmission, kilometres, owner count, service history, official specifications, BS norm, and E20 readiness.

Treat the dataset as a product asset that gets better over time.

## Starter data layers

Layer 1: public starter datasets

- Kaggle CarDekho-style datasets
- used-car price prediction datasets
- existing project CSVs

Layer 2: manually collected current listings

- 5 to 10 comparable listings per target model and city
- source URL and date collected for every row
- clean make, model, variant, fuel, transmission, year, km, owner count, and asking price

Layer 3: specifications

- OEM brochures
- manuals
- official spec pages
- trusted automotive specification pages

Layer 4: rules and context

- E20 generation rules
- BS norm
- city and state context
- running-cost assumptions

Layer 5: user-provided comparable listings

- Buyer pastes 5 to 10 similar listings into the app
- App uses the median comparable asking price as an anchor
- This gives a useful price verdict before a large dataset exists

## Minimum viable schema

Start with:

make, model, variant, city, state, model_year, listing_year, km_driven, fuel_type, transmission, owner_count, seller_type, asking_price_lakh, engine_cc, power_bhp, arai_mileage_kmpl, bs_norm, e20_stage, expected_mileage_loss_pct, service_history, accident_history, source_name, source_url, date_collected

## Data quality rules

- Preserve the original listing text in a raw field if possible.
- Normalize make/model/variant into clean fields.
- Keep source URL and collection date.
- Do not mix ex-showroom new-car prices with used-car asking prices.
- Mark whether price is asking price, dealer quote, auction result, or final sale price.

## Narrow first dataset

Start with one city and a few common petrol models. A good first target is Delhi NCR with Swift, Baleno, i20, City, and Creta listings from 2017 to 2024. This is enough to demonstrate a serious product without pretending to cover the entire Indian market.
