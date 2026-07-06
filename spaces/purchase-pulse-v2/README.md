---
title: Purchase Pulse V2
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
---

# Purchase Pulse V2

Purchase Pulse V2 is an India-focused used-car buyer decision engine.

It expands the original Purchase Pulse price predictor into a practical buyer assistant that combines:

- used-car deal score
- fair-price band using local comparable listings
- E20 fuel and running-cost risk
- five-year ownership-cost estimate
- buyer report
- car comparison mode
- negotiation copilot
- local RAG-style assistant over project knowledge notes

This demo is intentionally transparent: it uses rule-based risk logic and comparable-market inputs instead of pretending to know every Indian used-car price from a tiny dataset.

## Suggested demo flow

1. Open the Buyer Check tab.
2. Try the default Hyundai i20 petrol example.
3. Add a local comparable median to see fair-band pricing.
4. Open Compare Cars and rank the sample listings.
5. Ask the assistant about E20 or dataset strategy.
