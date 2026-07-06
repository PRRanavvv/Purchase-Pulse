# Purchase Pulse 2.0

Purchase Pulse 2.0 reframes the old project from a generic car price predictor into an India-focused used-car buyer-risk assistant.

The first goal is not to predict a magical exact price. The goal is to help a buyer decide:

> Should I buy this exact car at this exact price in this city right now?

## What is included in this first slice

- `app_v2.py`: Gradio MVP for buyer checks and a local RAG-style assistant
- `purchase_pulse_v2/e20.py`: rule-based E20 risk assessment
- `purchase_pulse_v2/pricing.py`: fair-band and negotiation logic using local market comps
- `purchase_pulse_v2/scoring.py`: buyer-facing deal score
- `purchase_pulse_v2/ownership.py`: five-year ownership-cost estimate
- `purchase_pulse_v2/comparison.py`: paste-based comparison and ranking for multiple cars
- `purchase_pulse_v2/negotiation.py`: seller-message and negotiation-anchor helper
- `purchase_pulse_v2/rag.py`: lightweight TF-IDF retriever over local markdown knowledge
- `knowledge/`: editable knowledge base for E20, buyer checks, and dataset strategy
- `data/purchase_pulse_v2_template.csv`: starter schema for collected listings

## How to run

```bash
python app_v2.py
```

## How the v2 app thinks

Price:

- It needs a listing asking price.
- It optionally accepts a local market median from comparable listings.
- If no market median is provided, it asks you to collect comps instead of pretending.

E20:

- Petrol-blend vehicles get a rule-based E20 risk assessment.
- Non-petrol vehicles are marked not applicable for E20.
- The output stays conservative and tells the buyer what to verify.

Deal score:

- Combines fair-price verdict, E20 risk, age, kilometres, owner count, service history, and ownership cost.
- It is a decision score, not a resale guarantee.
- Confidence drops when comparable-market data is missing.

Ownership cost:

- Estimates fuel, insurance, service buffer, depreciation, resale value, and monthly cost.
- E20 mileage impact feeds into effective mileage.

Comparison:

- Paste cars in a pipe-separated format and rank them by deal score and projected ownership cost.
- This keeps the first comparison feature useful without needing scraping.

RAG:

- The assistant retrieves from local markdown notes in `knowledge/`.
- It is currently retrieval-plus-summary, not a full LLM answer.
- Later, the retrieved chunks can be passed into an LLM for a richer answer.

## Next data milestone

Collect 100 to 200 current used-car listings for 3 to 5 high-volume petrol cars in one city first. Keep source URL and collection date for every row. Expand only after the first narrow slice works.
