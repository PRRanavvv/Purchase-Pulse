# Used Car Buyer Playbook

## Core buyer decision

The product should answer one practical question: should I buy this exact car at this exact price in this city right now?

The recommendation should combine:

- price compared with local market comps
- age and kilometres driven
- fuel type and E20 readiness
- ownership count
- service history
- accident, claim, loan, and documentation checks
- expected five-year ownership cost

## Price bands over exact prices

Used-car pricing is noisy. A useful product should show a fair band, not one exact number. The fair band should widen when data is weak, variants are unclear, or the listing is far from normal market patterns.

## Minimum comparable listing rule

Before judging whether a car is overpriced, collect 5 to 10 comparable listings from the same city or nearby market. Comparable means similar make, model, variant, year, fuel, transmission, owner count, and kilometre range.

Use the median comparable price as the first market anchor.

## Negotiation logic

Strong negotiation signals:

- asking price above adjusted fair band
- missing service history
- high kilometres for age
- multiple owners
- accident or insurance-claim history
- old tyres, weak battery, pending major service
- uncertain E20 compatibility for older petrol vehicles

## Report output

A buyer report should include:

- buy, negotiate, pause, or avoid decision
- deal score and confidence
- fair price band
- negotiation anchor
- E20 risk and expected mileage impact
- five-year ownership-cost estimate
- inspection questions
- confidence level
- sources or assumptions used

## MVP feature boundary

Keep these features in the first serious version:

- deal score
- fair price band
- E20 running-cost risk
- ownership cost
- buyer report
- RAG assistant over local notes and later manuals
- comparison mode
- negotiation copilot

Do not spend early effort on city heatmaps, dealer dashboards, fleet tools, EV battery scoring, or insurance-claim intelligence. Those need too much data or a different product motion.
