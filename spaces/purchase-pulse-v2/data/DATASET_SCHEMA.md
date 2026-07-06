# Purchase Pulse v2 Dataset Schema

This is the starter schema for an India-focused used-car buyer-risk dataset.

## Required listing fields

- `make`: manufacturer name, normalized
- `model`: model name, normalized
- `variant`: trim/variant when available
- `city`: listing city
- `state`: listing state
- `model_year`: vehicle model year or registration year
- `listing_year`: year when listing was collected
- `km_driven`: odometer reading in kilometres
- `fuel_type`: Petrol, Petrol/CNG, Petrol/Hybrid, Diesel, CNG, EV, etc.
- `transmission`: Manual, Automatic, AMT, CVT, DCT, etc.
- `owner_count`: number of previous owners
- `seller_type`: Individual, Dealer, Platform, Auction, Unknown
- `asking_price_lakh`: asking price in lakh INR

## Enrichment fields

- `engine_cc`: engine displacement
- `power_bhp`: power output
- `arai_mileage_kmpl`: official mileage where available
- `bs_norm`: BS4, BS6, etc.
- `e20_stage`: Pre-E20 transition, E20 material-compliant transition, E20-tuned generation, or Not applicable
- `expected_mileage_loss_pct`: current rule-based assumption
- `service_history`: Complete, Partial, Missing, Unknown
- `accident_history`: None, Minor, Major, Unknown

## Source fields

- `source_name`: source platform or dataset
- `source_url`: URL or file reference
- `date_collected`: collection date in `yyyy-mm-dd`
- `notes`: free-form audit notes

## Collection rule

For a fair-price verdict, collect 5 to 10 comparable listings per target model-city cluster. Do not train on rows that cannot distinguish asking prices from final sale prices unless the model explicitly treats them as different labels.
