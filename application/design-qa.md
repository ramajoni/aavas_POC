# Aavas UI design QA

final result: passed

## Source visual truth

- Assessment workspace: `/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 08_20_50 AM.png`
- Performance dashboard: `/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 08_20_34 AM.png`
- Customer offer journey: `/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 07_56_56 AM.png`

All three source images are 1536 × 1024 pixels. Reference imagery is a design target, not an instruction to fabricate operational data or add unsupported approval APIs.

## Rendered evidence

Local app: `http://localhost:4173/`. Browser: user-authorized local Playwright Chromium.

Desktop viewport: 1536 × 1024 CSS pixels, deviceScaleFactor 1. Source and browser captures are 1536 × 1024 pixels; no density normalization needed.

- `qa/dashboard-desktop.png`: dashboard, all regions and branches, live model metrics loaded.
- `qa/application-desktop.png`: completed pre-assessment and final assessment, no human override recorded yet, selected recommendation visible.
- `qa/offer-desktop.png`: basic-information tab after receiving an indicative offer, matching the source's populated-offer state.
- `qa/audit-desktop.png`: completed lifecycle with recorded override and outcome.

Mobile viewport: 390 × 844 CSS pixels, deviceScaleFactor 1. Full-page screenshots retain their natural document height:

- `qa/dashboard-mobile.png`
- `qa/application-mobile.png`
- `qa/offer-mobile.png`

No mobile reference was supplied; mobile checks evaluate usable reflow rather than claiming pixel fidelity to a desktop mock.

## Full-view and focused comparisons

Each source image and corresponding browser screenshot were opened together in the same comparison input, at original resolution. Final comparisons inspected all three complete screens.

Focused comparisons were captured from the original source and actual browser screenshots without altering either original file:

- `qa/reference-pricing-detail.png` (1028 × 271) and `qa/implementation-pricing-detail.png` (1039 × 257): stage cards, recommendation, permitted band, reason codes, attribution label.
- `qa/reference-models-detail.png` (422 × 224) and `qa/implementation-models-detail.png` (408 × 235): model table, typography, metrics, status, grid lines.
- `qa/reference-offer-detail.png` (520 × 379) and `qa/implementation-offer-detail.png` (530 × 424): interest range, amounts, EMI, explanatory card.

Focused captures remain at 1× density. Different crop dimensions reflect actual content/layout differences, not image scaling. They were viewed in source/implementation pairs in a single comparison input.

## Findings and comparison history

1. **P2, resolved: dashboard bottom row clipped at the reference viewport.** Initial `qa/dashboard-initial.png` placed part of the live model table beneath the persistent footer. Reduced dashboard row spacing, compacted the learning list, and moved the model-card link into the card heading. `qa/dashboard-desktop.png` shows the complete table, learning entries, and regional list above the footer.
2. **P2, resolved: recommended pricing scenario initially below the scenario viewport.** All 37 scenarios originally rendered in a scroll container starting from the lowest rate. The default now shows four options around the recommendation, highlights the recommended row, and exposes all scenarios through an explicit toggle. Final evidence: `qa/application-desktop.png`.
3. **P2, resolved: assessment actions below the desktop fold.** Compacted the top spacing and placed detailed approver/evidence inputs in an expandable section. Final evidence shows Save as Draft and Record Outcome above the footer, with Record Override directly above them.
4. **P2, resolved: hero crop cut off part of the child's face.** Adjusted the desktop image width and focal position to retain all three faces and maintain whitespace behind the headline. Final evidence: `qa/offer-desktop.png`.
5. **P0, resolved during runtime verification:** Vite dependency optimization initially produced a transient duplicate React runtime during the first browser load. Once the dependencies had been optimized, repeated complete browser runs produced zero runtime or console errors. See `qa/verification.json`.

No unresolved P0/P1/P2 finding remains within the requested standalone UI scope.

## Required fidelity surfaces

- **Fonts and typography:** locally bundled Roboto and Roboto Condensed reproduce the reference's sans-serif hierarchy, compact analytical headings, and prominent rates. Native-size focused captures confirmed readable table labels and value columns. Minor weight/width differences remain compared with the raster reference, as P3 refinement.
- **Spacing and layout rhythm:** navy 216 px desktop sidebar, white utility bar, five KPI cards, three analytical rows, customer summary strip, three pricing cards, lower scenario/override split, and right-hand customer journey are retained. Cards use pale borders and compact 9–16 px radii. All primary desktop actions are visible at 1536 × 1024. Mobile reflows, uses a navigation drawer, and has no horizontal document overflow.
- **Colors and tokens:** dark navy navigation and footer, royal-blue actions, soft blue/green panels, green favourable indicators, red unfavourable indicators, and violet insights follow the references. Selected, loading, error, pending, and success states use consistent treatments.
- **Image quality and assets:** the actual Aavas logo is downloaded from the official site's asset host. Hero and sidebar home images are generated photographic assets grounded in the supplied references; no screenshots are used as the UI. Phosphor supplies icons. Chart.js supplies analytical charts. Hero faces are retained in the final crop. A subtle join between the pale hero background and image is a P3 polish opportunity.
- **Copy and content:** API-derived rates and model metrics intentionally differ from the fabricated numbers in the mock. Labels correctly distinguish conversion probability from approval probability, annual margin from lifetime value, and the ablation anchor attribution from final-price optimization. Unsupported sanction/disbursal actions are represented by recording an actual outcome, not by claiming approval. Portfolio sample figures are explicitly marked illustrative.

## Intentional implementation differences

- Portfolio charts and branch statistics are illustrative because no portfolio HTTP endpoints exist. Live model, pricing, policy, learning and audit endpoints remain connected. Filters slice the included sample branch data.
- The regional reference map is represented by a compact city comparison using labelled bars, rather than inventing a geographic data source. This is a functional adaptation, not a claim of an exact choropleth recreation.
- Avatar initials replace a generated employee portrait; identities and role switching are demo views, not authentication.
- Competitor context uses the actual applicant's reported quote, not invented live bank rates.
- The customer journey uses three real form steps. Document upload, actual sanction approval, actual disbursal, and authenticated staff management are not available from the existing backend.
- Offer confidence and API caveats add a small amount of vertical content compared with the reference.

## Interaction and technical verification

`npm run test:ui` passed with:

- live service connectivity and loaded model cards;
- branch filtering changing KPI values;
- customer details → loan details → consent → live indicative offer;
- final assessment → selected scenario → live explanation;
- value-breakdown and key-driver tabs;
- human override → outcome attestation → recorded outcome;
- audit containing both `Capture Override` and `Capture Outcome`;
- model registry, learning review and policy clause search;
- bureau score 0 rejected by native form validation;
- mobile navigation and all three primary views at 390 px;
- zero horizontal document overflow;
- zero browser errors and zero failed API responses in the recorded run.

Evidence: `qa/verification.json`, screenshots listed above. The automated flow creates synthetic records in the existing API database.

`npm run build` passed. Evidence: `qa/build.log`. The bundler reports a non-blocking large-chunk advisory for the app/chart bundle.

Backend isolation: 107 pre-existing source, config and documentation files checked against pre-build SHA-256 hashes; none changed. Evidence: `qa/backend-isolation.json` and `backend-baseline.json`.

## Implementation checklist

- [x] New frontend contained in `application/`.
- [x] Existing backend code unchanged.
- [x] Live pricing lifecycle integrated through a same-origin development proxy.
- [x] All primary views rendered and visually inspected.
- [x] Desktop and mobile interactions verified.
- [x] Invalid bureau score prevented in UI.
- [x] Data provenance and unsupported actions communicated honestly.
- [x] Local UI left running on port 4173.

## Follow-up polish

- P3: refine condensed font optical weights and small metadata sizes toward the supplied raster references.
- P3: soften the photographic hero's left join using a revised wider generated asset.
- P3: split the chart-heavy bundle if this local prototype moves toward production hosting.

final result: passed
