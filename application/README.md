# Aavas application UI

A standalone React + Vite frontend based on the three supplied design references. All new source, assets, dependencies, and verification artifacts are contained in `application/`. The existing Python application is not modified.

## Open the application

While the existing pricing API is running on port 8000:

```bash
cd application
npm install
npm run dev -- --host 0.0.0.0 --port 4173 --strictPort
```

- Dashboard: http://localhost:4173/#/dashboard
- Assessment workspace: http://localhost:4173/#/application
- Customer journey: http://localhost:4173/#/offer
- Model registry: http://localhost:4173/#/models
- Override learning: http://localhost:4173/#/learning
- Audit history: http://localhost:4173/#/audit
- Policy questions: http://localhost:4173/#/policy

Vite forwards same-origin `/api/*` requests to `http://127.0.0.1:8000`. This avoids changing backend CORS configuration. To target a different backend, set `AAVAS_API_URL` when starting Vite. The browser makes no direct cross-origin requests.

## One complete flow

1. Open Customer Portal. Edit Basic Information and continue to Loan & Property.
2. Enter the loan details and consent, then select **See My Indicative Offer**.
3. Open the Assessment Workspace. Use the pencil beside the applicant to review verified income, property valuation, bureau score and assessment flags.
4. Run Final Assessment. Compare rate scenarios, model attributions, risk, and policy authority.
5. Optionally select a scenario, enter an override reason, and review Approver & Evidence. **Record Override** saves the human rate alongside the AI rate. A saved override is not an approval; policy violations and insufficient authority prevent continuing to disbursed outcome in the UI.
6. **Record Outcome** records an actual customer response. Accepted and disbursed outcomes require confirmation that necessary approvals were obtained. No loan is actually approved or disbursed by this UI.
7. Open Audit Trail to inspect the API records for the same loan ID.

The initial applicant is fictional. Every new application receives a unique ID. Stage 1 requires a bureau score of 300–900 or -1 for new to credit. Form validation prevents the Swagger placeholder value of 0. Changing application data clears stale results and requires repricing.

## Data and persistence

- **Live API:** pre-assessment, final-assessment, rate scenarios, explanations, override capture, outcomes, audit, model cards, policy search, and learning reviews.
- **Illustrative:** dashboard portfolio totals, trends, funnel, branch comparisons, regional growth, rate distribution and trade-off curves. The dashboard marks these clearly. Region and branch filters operate on the included sample dataset.
- Learning review endpoints analyze the existing synthetic training dataset, not just outcomes recorded from this UI.
- Model metrics are real values returned by the model registry, but the models were trained on synthetic data.
- **Save as Draft** stores one draft, including notes, in this browser's local storage. Without saving, unsent edits and notes are lost on reload. The backend retains submitted decisions. **Export** downloads the current application as JSON.
- Staff identities and role switching are demo UI, not authentication. Document upload, actual sanction approval, and disbursal are outside the existing API's capabilities.

## Verification

```bash
npm run build
npm run test:ui
```

The UI test requires the frontend on port 4173, the existing backend on port 8000, and Playwright Chromium (`npx playwright install chromium`). It creates synthetic decision, override and outcome records while checking the complete flow, live operational screens, invalid bureau score rejection, mobile navigation, overflow, browser errors and API responses.

Screenshots, test evidence and the visual review are in `qa/` and `design-qa.md`. `backend-baseline.json` records pre-build hashes of existing Python, configuration and documentation files to verify isolation.

The build's static bundle does not independently provide the `/api` proxy. Hosting beyond local Vite requires an equivalent same-origin reverse proxy to the existing backend. Nothing has been deployed.

## Design assets

- Logo: official Aavas website asset, https://d3atsxo2rus6jj.cloudfront.net/aavas-logo.png
- Hero and sidebar home photographs: generated for this project from the user's design references.
- Icons: Phosphor Icons, https://phosphoricons.com/
- Fonts: locally bundled Roboto and Roboto Condensed through Fontsource.
- Charts: Chart.js, with canvas rendering and accessible labels.
