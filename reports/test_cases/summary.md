# Walkthrough test cases - prototype outputs

Run 20261020144343 UTC via `http://localhost:8000`. Every value except the dataset reference row came back from the pricing service.

| Attribute | TC1 · AAV0010358 | TC2 · AAV0009223 | TC3 · AAV0000582 |
|---|---|---|---|
| Stage 1 range | 8.85–9.45% | 12.35–13.10% | 12.35–13.20% |
| Card rate | 9.35% | 12.55% | 12.60% |
| Recommended rate | 9.12% | 12.50% | 12.28% |
| Permitted band | 8.75–16.50% | 11.95–16.50% | 12.00–16.50% |
| Risk (model) | grade A, PD 0.30%, LGD 26% | grade C, PD 3.65%, LGD 44% | grade C, PD 3.79%, LGD 39% |
| Approval at recommendation | branch_manager | sales_officer | regional_manager |
| Guardrail at recommendation | refer | refer; resize required: at 12.50% the FOIR is 57% against a 55% cap - reduce the loan to Rs 2,255,000 or extend the tenure to 18 years | refer; resize required: at 12.28% the FOIR is 57% against a 55% cap - reduce the loan to Rs 1,804,000 or extend the tenure to 21 years |
| RM proposal | 8.50% (competitor_lower_rate, evidence verified) | 12.82% (competitor_lower_rate, evidence verified) | none |
| Policy verdict | block, counter-offer 8.75% (refer), needs credit_head, escalated to credit_head | refer, needs sales_officer | no override |
| Outcome replayed | rejected at 8.75% | accepted at 12.82% | accepted at 12.28% |
| Dataset reference | range 9.02–9.79%, rec. 9.20% | range 12.99–13.56%, rec. 13.23% | range 12.27–12.94%, rec. 12.63% |
