# API reference

FastAPI service, `src/aavas/api`. Interactive docs at `/docs` when it is running
(`make api` → http://localhost:8000/docs). Behaviour and sequencing are in
[flows.md](flows.md).

Base URL in development: `http://localhost:8000`. The React app reaches it through a
same-origin `/api/*` proxy (see [operations.md](operations.md#3-frontend)).

| Method | Path | Purpose |
|---|---|---|
| POST | `/pricing/pre-assessment` | Indicative band, eligibility, confidence |
| POST | `/pricing/final-assessment` | Risk-adjusted recommendation + reason codes |
| POST | `/pricing/override` | Capture the human decision |
| POST | `/pricing/outcome` | Capture what actually happened |
| GET | `/health` | Model bundle state, versions, decision count |
| GET | `/model-cards` | Model cards, including written limitations |
| GET | `/pricing/decision/{decision_id}` | One recorded decision |
| GET | `/pricing/explain/{decision_id}` | Every feature's SHAP weight in the rate, risk and conversion models; the rate bridge |
| POST | `/pricing/explain/{decision_id}/narrative` | Plain-language explanation written by Claude from those figures; `?refresh=true` rewrites it |
| GET | `/pricing/audit/{loan_id}` | Full history for one file |
| GET | `/pricing/policy/explain?question=` | Cite the policy clause behind a limit |
| GET | `/pricing/override/reasons` | Allowed reasons, evidence rules, authority table |
| GET | `/pricing/learning/override-review` | Did overrides create or destroy value? |
| GET | `/pricing/learning/questions` | Standing review questions, answered from data |

All request bodies are JSON and validated by pydantic
([api/contracts.py](../src/aavas/api/contracts.py)); a contract violation returns `422` with
the offending field path.

---

## POST /pricing/pre-assessment

Stage-1. Only pre-assessment fields are accepted — the feature contract makes verified data
structurally impossible here.

**Request** (`Stage1Request`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `loan_id` | string | ✔ | Threads the whole journey; reuse it across all four endpoints |
| `segment` | enum | ✔ | `salaried_formal`, `salaried_informal`, `self_employed_professional`, `self_employed_non_professional` |
| `city` | string | ✔ | |
| `channel` | enum | | `branch_walkin` (default), `dsa`, `connector`, `digital` |
| `property_type` | enum | | `resale_flat` (default), `builder_flat`, `self_construction`, `plot_plus_construction` |
| `declared_monthly_income` | float > 0 | ✔ | |
| `requested_loan_amount` | float > 0 | ✔ | |
| `requested_tenure_years` | int 1–30 | | default 20 |
| `applicant_age` | int 18–75 | | default 35 |
| `declared_obligations` | float ≥ 0 | | default 0 |
| `estimated_property_value` | float > 0 | | defaults to `requested_loan_amount / 0.70` |
| `bureau_score` | int | | **300–900, or −1 for new to credit**; omitted → −1 |
| `new_to_credit` | bool | | inferred from `bureau_score < 0` when omitted |
| `sales_context` | object | | see below; defaults are neutral |

`sales_context`: `competitor_rate`, `competitor_name`, `competitor_evidence_verified` (false),
`sales_target_pct` (100), `branch_target_pct` (100), `customer_negotiating` (false),
`prior_negotiation_rounds` (0), `existing_relationship` (false), `month_end`,
`days_to_month_end`.

**Response** (200)

```jsonc
{
  "loan_id": "DEMO-FLOW-001",
  "stage": "pre_assessment",
  "band": { "min_rate": 11.60, "max_rate": 12.35, "width_bps": 75.0 },
  "indicative_rate": 11.95,
  "confidence": 0.78,
  "confidence_label": "high",              // high ≥ 0.75, moderate ≥ 0.5, else low
  "eligible_amount": { "min": 1180000, "max": 1251000,
                       "binding_constraint": "property value (LTV cap)" },  // or "affordability (FOIR)" / "product maximum"
  "caveats": ["Income is self-declared. ..."],
  "completeness": { "post_assessment_fields_present": 0, "completeness_ratio": 0.0, "...": "..." },
  "notes": [],
  "model_versions": { "stage1": "v1" },
  "versions": { "app_version": "...", "policy_version": "HL-POLICY-2026.10", "...": "..." },
  "elapsed_ms": 18.4,
  "decision_id": "DEC-XXXXXXXXXXXX",
  "disclaimer": "Indicative only. ..."
}
```

**Errors** — `422` contract violation (including a bureau score outside 300–900 that is not −1);
`503` the `stage1` artefact is missing; `500` otherwise.

---

## POST /pricing/final-assessment

Stage-2. Extends `Stage1Request` with the assessment outcome. Query parameter
`include_scenarios=true` returns the full scored scenario table (large).

**Additional request fields** (`Stage2Request`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `verified_monthly_income` | float > 0 | ✔ | `income_verification_haircut` is derived from it |
| `appraised_property_value` | float > 0 | ✔ | `ltv` is derived from it |
| `sanctioned_loan_amount` | float > 0 | ✔ | |
| `tenure_years` | int 1–30 | ✔ | |
| `bureau_vintage_months` | int ≥ 0 | | default 0 |
| `delinquency_count_24m` | int ≥ 0 | | default 0 |
| `banking_behaviour_score` | float 0–100 | | default 50 |
| `deviation_count` | int ≥ 0 | | default 0 |
| `legal_flag`, `technical_flag` | int 0/1 | | default 0 |
| `risk_grade` | string | | Optional — **the risk agent re-grades the file anyway** |

**Response shape** (200) — full key map in [flows.md](flows.md#what-each-step-contributes-to-the-payload)

```jsonc
{
  "loan_id": "...", "stage": "post_assessment",
  "twin": { "...": "consolidated view + completeness + notes" },
  "application_snapshot": { "...": "the file exactly as priced" },
  "risk":        { "pd": 0.011, "lifetime_pd": 0.044, "lgd": 0.28, "risk_grade": "B",
                   "expected_loss_pct": 0.31, "breakeven_rate": 8.99, "source": "model" },
  "envelope":    { "card_rate": 12.35, "min_rate": 11.75, "max_rate": 13.35,
                   "max_affordable_rate": 13.35, "affordability_binding": false,
                   "infeasible": false, "policy_version": "HL-POLICY-2026.10" },
  "recommendation": {
    "recommended_rate": 12.79,
    "economics": { "conversion_probability": 0.737, "expected_life_years": 8.0,
                   "lifetime_value_pct": 12.09, "expected_value_inr": 175456,
                   "contractual_yield_pct": 4.89, "...": "..." },
    "value_decomposition": { "customer_rate": 12.79, "less_funding_cost": -7.90, "...": "..." },
    "binding_constraint": "conversion floor",
    "scenarios_rejected": { "below_min_conversion": 12 },
    "marginal_analysis": { "next_step_bps": 5.0, "next_step_conversion_change_pts": -1.4,
                           "next_step_value_change_inr": -820, "unconstrained_peak_rate": 12.79 },
    "comparisons": { "card_rate": { "...": "value_vs_recommended_inr" },
                     "competitor_match": { "...": "..." },
                     "anticipated_human_rate": { "...": "..." } },
    "stage2_anchor": { "anchor_rate": 12.64, "search_band": {"min_rate": 12.51, "max_rate": 12.77},
                       "residual_std_bps": 12.8, "source": "stage2_model" },
    "scenarios": { "n_scenarios": 41, "band": {"...": "..."}, "anchors": {"...": "..."},
                   "notes": ["upper end trimmed from 16.50% to 13.73%: ..."] },
    "scenario_table": [ "... only when include_scenarios=true ..." ]
  },
  "policy":      { "allowed": true, "discount_bps": -44.0, "required_approval": "sales_officer",
                   "violations": [], "warnings": [], "permissible_band": {"...": "..."} },
  "sales_context": { "competitor_gap_bps": 129.0, "competitor_claim_status": "unverified",
                     "max_discount_on_this_claim_bps": 15, "pressure_index": 0.42,
                     "override_risk": { "probability": 0.63, "expected_override_bps": 11.2,
                                        "anticipated_human_rate": 12.68 } },
  "price_sensitivity": { "conversion_at_recommended": 0.737, "points_lost_per_25bps": 1.4,
                         "sensitivity_label": "moderately price sensitive",
                         "rate_for_70pct_conversion": 12.95, "negotiation_table": [] },
  "explanation": { "headline": "...", "attribution_method": "ablation",
                   "reason_codes": [ { "code": "RISK_GRADE_UP", "feature": "risk_grade",
                                       "label": "Risk grade", "direction": "increase",
                                       "impact_bps": 68.0, "value": "C",
                                       "text": "Weaker risk grade (+68 bps)" } ],
                   "commercial_vs_risk": { "commercial_share_pct": 24.0,
                                           "commercial_net_bps": -41.0, "risk_net_bps": 68.0 },
                   "negotiation_guidance": {"...": "..."}, "policy_position": {"...": "..."},
                   "fair_pricing_review": { "flagged": false, "...": "..." },
                   "actionable_for_customer": [ {"...": "clearing 1 deviation ≈ 12 bps"} ],
                   "customer_caveats": ["..."], "disclaimer": "..." },
  "model_versions": {"stage1": "v1", "...": "..."},
  "versions": {"...": "..."},
  "human_decision_required": true,
  "notes": ["..."],
  "elapsed_ms": 312.7,
  "decision_id": "DEC-XXXXXXXXXXXX"
}
```

`human_decision_required` is always `true`. **Save `decision_id`** — it is the handle for
`/pricing/decision/{decision_id}`.

**Errors** — `422` contract violation; `500` with the loan id logged.

---

## POST /pricing/override

Records the human decision **beside** the recommendation; it never replaces it.

```jsonc
{
  "loan_id": "DEMO-FLOW-001",
  "human_rate": 12.60,
  "reason": "competitor_lower_rate",          // must be in policy.override.allowed_reasons
  "reason_detail": "customer produced a sanction letter",
  "approved_by": "r.sharma:branch_manager",   // 'name:level' or just a level
  "evidence_verified": true
}
```

**Response** (200) — the stored record plus:

| Field | Meaning |
|---|---|
| `override_id`, `decision_id`, `loan_id`, `created_at` | Record identity and the decision it overrides |
| `ai_rate`, `human_rate`, `override_bps` | Both numbers and the delta (`human_rate − ai_rate` in bps: negative = priced below the recommendation) |
| `reason`, `reason_detail`, `evidence_verified`, `approved_by`, `sales_context` | As submitted, stored for the learning loop |
| `required_approval` | Lowest authority that may sign this deviation from the **card rate** |
| `within_authority` | `true` / `false` / `null` when the approver is unknown or unparseable |
| `policy_check` | Full re-validation of the human rate against the stored application snapshot |
| `warning` | Present when `within_authority` is `false` |

**Errors** — `422` reason not allowed (response lists `allowed_reasons`); `404` no
post-assessment decision on record for that loan; `500` otherwise.

> Recording is not approving. A `200` with `within_authority: false` or
> `policy_check.allowed: false` means the price must not be released to the customer.

---

## POST /pricing/outcome

```jsonc
{
  "loan_id": "DEMO-FLOW-001",
  "disbursed": true,
  "customer_response": "accepted",     // accepted | rejected | negotiated | deferred
  "sanctioned_rate": 12.60,            // the rate actually agreed (override rate if used)
  "bt_out": false,
  "delinquent": false,
  "months_to_exit": null,              // leave unobserved future outcomes null
  "realized_yield_pct": null
}
```

Response echoes the stored outcome plus `realized_value_pct` and `vs_expected` — the
decision's expectation against what the book actually did.

---

## Supporting endpoints

### GET /health
```jsonc
{ "status": "ok",            // "degraded" when artefacts are missing
  "models_loaded": { "stage1": "v1", "stage2": "v1", "...": "..." },
  "models_missing": [],
  "versions": { "app_version": "...", "policy_version": "HL-POLICY-2026.10",
                "feature_config_version": "FEAT-2026.09", "model_config_version": "MODEL-2026.09" },
  "decisions_recorded": 128 }
```

### GET /model-cards
Per model: `name`, `version`, `learner`, `target`, `trained_at`, `n_train`, `n_test`,
`features`, `metrics`, `training_window`, `config_versions` and a written `limitations` list.

### GET /pricing/decision/{decision_id}
The stored decision, including the full payload. `404` if unknown.

### GET /pricing/audit/{loan_id}
`decisions`, `overrides`, `outcomes` and `audit_entries` for one file, in order. `404` when the
file has no history.

### GET /pricing/policy/explain?question=
`{ "question": "...", "clauses": [ { "text": "...", "metadata": {...}, "score": 0.42 } ] }` —
the policy text that governs the limit, retrieved from the indexed
`config/pricing_policy.yaml`. `question` must be ≥ 3 characters.

### GET /pricing/override/reasons
`allowed_reasons`, `evidence_required_reasons`, `max_unevidenced_discount_bps` and the full
`approval_authority` table (level → max discount / max premium bps).

### GET /pricing/learning/override-review
`naive_comparison`, `adjusted_overall`, `confounding_gap_inr`, `reason_league_table`,
`evidence_effect`, `suggested_adjustments` (capped at ±15 bps — proposals, not changes) and a
`method_note`. Reads the **training table**; `503` if it has not been built.

### GET /pricing/learning/questions
`answers` (the standing review questions answered from data) and `recommendation_quality`
(override rate, mean human adjustment, share priced up/down, interpretation). Also `503`
without a training table.

### GET /applications (staff)
Each file carries its `stage` — `in_progress`, `ready`, `pending_rate_approval`, `offer_sent`,
`accepted`, `not_accepted` — read from its documents, rate requests and offers, plus
`missing_documents`, `verified_at` (when the last document verification needs arrived; the list is
`required_for_verification` in `config/intake.yaml`) and, once `ready`, `suggested`: the
recommended `rate`, the `acceptance` likely at it and the `sanctioned_amount`. A file that changed
since it was last priced is priced again first.

### GET /applications/in-progress · POST /applications/nudges
Files short of verification plus the sample leads in `config/intake.yaml` (`sample: true`), each
with `detail`, `days_since_activity`, `last_nudge` and the `preview` of the message a nudge sends.
`POST {ref, channel}` (relationship manager) records a nudge; nothing is delivered — the service
has no SMS or WhatsApp gateway.

### POST /applications/{loan_id}/offer-decision (borrower)
`{accepted, loss_reason?}`: the customer accepts or declines the offer waiting for them; the RM's
list and the outcome record update without anyone re-keying it. `409` when no offer is waiting.

### GET /pricing/rate-requests/{request_id}/deep-dive (area manager)
The file's acceptance `curve` from below break-even to the policy ceiling, the `breakeven_rate`,
`requested_rate`, `recommended_rate`, the `ceiling_rate` the customer is still likely to accept
(`ceiling_capped_by` the quoted range or policy), the floor `build_up` (cost of funds, operating
cost, capital charge, expected credit loss) and the reasons behind the price.

### GET /pricing/area/disbursements · GET /pricing/my/disbursements
Month to date: accepted offers at their assessed loan amount against each RM's
`monthly_disbursement_target_inr` (`config/access.yaml`), pace through the month, and each RM's
offers (sent, taken, depth below the recommendation) and lower-rate requests. `area` for the area
manager, `my` for the signed-in RM.

### GET /models/portfolio — `business`
Beside the model evidence: `segments` (default rate, loss given default, expected credit loss a
year, starting and typical rate per segment), `acceptance_curves` per segment, `discount_effect`
(acceptance points a 0.25% discount wins, measured by the uplift model and read off the curve) and
the fixed `floor` components from the pricing policy.

---

## A complete demo flow

Use one loan id throughout (the OpenAPI examples use `DEMO-FLOW-001`).

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/pricing/pre-assessment \
  -H 'content-type: application/json' \
  -d '{"loan_id":"DEMO-FLOW-001","segment":"salaried_formal","city":"Jaipur",
       "declared_monthly_income":55000,"requested_loan_amount":2000000,
       "estimated_property_value":3000000,"bureau_score":750}'

curl -s -X POST 'localhost:8000/pricing/final-assessment?include_scenarios=true' \
  -H 'content-type: application/json' \
  -d '{"loan_id":"DEMO-FLOW-001","segment":"salaried_formal","city":"Jaipur",
       "declared_monthly_income":55000,"requested_loan_amount":2000000,
       "estimated_property_value":3000000,"bureau_score":750,
       "verified_monthly_income":53000,"appraised_property_value":2900000,
       "sanctioned_loan_amount":2000000,"tenure_years":20,
       "bureau_vintage_months":96,"banking_behaviour_score":75}'

curl -s localhost:8000/pricing/override/reasons

curl -s -X POST localhost:8000/pricing/override \
  -H 'content-type: application/json' \
  -d '{"loan_id":"DEMO-FLOW-001","human_rate":12.60,"reason":"competitor_lower_rate",
       "approved_by":"r.sharma:branch_manager","evidence_verified":true}'

curl -s -X POST localhost:8000/pricing/outcome \
  -H 'content-type: application/json' \
  -d '{"loan_id":"DEMO-FLOW-001","disbursed":true,"customer_response":"accepted",
       "sanctioned_rate":12.60}'

curl -s localhost:8000/pricing/audit/DEMO-FLOW-001
```

`scripts/run_api_flow.py` does the same end to end against a running server.

---

## Conventions

* **Rates** are annual percentages (`12.79` = 12.79% p.a.); **spreads and deltas** are basis
  points; **money** is INR.
* `discount_bps` is measured **from the card rate**: positive for a discount, negative for a
  premium. `override_bps` is measured **from the AI recommendation**: negative when the human
  priced below it. They are different baselines on purpose.
* One `loan_id` threads enquiry → band → assessment → recommendation → human decision →
  sanction → disbursal → behaviour → yield. Reuse it; pick a new one for a new application.
* Every decision response carries `versions` (policy, feature-config, model-config, app) and
  `model_versions`, so any answer can be reproduced later.
* Nothing returned by this API approves or disburses a loan.
