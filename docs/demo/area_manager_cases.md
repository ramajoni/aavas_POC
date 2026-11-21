# Area manager cases: the coverage matrix

Probed 2026-11-21 against app 0.9.3, policy HL-LAP-POLICY-2026.14, model config MODEL-2026.10. Every rate is inside the policy band (13.00-17.00%). Generated from `scripts/demo_catalogue.json` by `python scripts/demo_catalogue.py report`; do not edit by hand.

## What is covered

- Primary cells (grade x tenure band x ask position x evidence): 180 in all, **82 filled, 20 impossible under the policy, 78 not found by the probe** (possible under the policy, but no probed file produced them).
- Pending cases for the area manager: 92 (82 matrix, 10 special situations). History cases decided by the area manager: 9.
- Probe: 855 profiles priced (run time not recorded: the system clock jumped during the run, so its duration was not measured) (9 rejected by the intake rules).

Ask position: **within room** is at least 30 bps under the recommendation and above the floor; **at floor** is exactly the policy floor; **below floor** is under it (the area manager can approve only down to the floor). Evidence is read against the file's own floor and recommendation: a verified letter below our floor, between the floor and the recommendation, or at or above the recommendation; a claim the RM has not seen the letter for; or no competitor.

## Matrix

Each entry is a case id (see the case list); `x` with a note number for a cell the policy makes impossible; `?` with a note number for a cell that is possible under the policy but that no probed file produced.

### Grade A+

| Tenure | Ask position | letter < floor | letter floor..rec | letter >= rec | claim, no letter | no competitor |
|---|---|---|---|---|---|---|
| 36-60 | within room | x (1) | x (1) | x (1) | x (1) | x (1) |
| 36-60 | at floor | x (1) | x (1) | x (1) | x (1) | x (1) |
| 36-60 | below floor | x (1) | x (1) | x (1) | x (1) | x (1) |
| 61-84 | within room | M-030 | M-031 | M-029 | M-032 | M-033 |
| 61-84 | at floor | M-025 | M-026 | M-024 | M-027 | M-028 |
| 61-84 | below floor | x (2) | x (2) | x (2) | x (2) | x (2) |
| 85-120 | within room | ? (3) | ? (3) | M-041 | ? (3) | M-042 |
| 85-120 | at floor | ? (3) | ? (3) | M-034 | ? (3) | M-035 |
| 85-120 | below floor | M-037 | M-038 | M-036 | M-039 | M-040 |

### Grade A

| Tenure | Ask position | letter < floor | letter floor..rec | letter >= rec | claim, no letter | no competitor |
|---|---|---|---|---|---|---|
| 36-60 | within room | ? (4) | ? (4) | M-008 | ? (4) | M-009 |
| 36-60 | at floor | ? (4) | ? (4) | M-001 | M-002 | M-003 |
| 36-60 | below floor | M-005 | ? (4) | M-004 | M-006 | M-007 |
| 61-84 | within room | ? (5) | ? (5) | ? (5) | ? (5) | M-016 |
| 61-84 | at floor | ? (5) | ? (5) | M-010 | ? (5) | M-011 |
| 61-84 | below floor | M-013 | ? (5) | M-012 | M-014 | M-015 |
| 85-120 | within room | ? (6) | ? (6) | ? (6) | ? (6) | ? (6) |
| 85-120 | at floor | ? (6) | ? (6) | M-017 | ? (6) | M-018 |
| 85-120 | below floor | M-020 | M-021 | M-019 | M-022 | M-023 |

### Grade B

| Tenure | Ask position | letter < floor | letter floor..rec | letter >= rec | claim, no letter | no competitor |
|---|---|---|---|---|---|---|
| 36-60 | within room | M-054 | M-055 | M-053 | ? (7) | M-056 |
| 36-60 | at floor | M-044 | M-045 | M-043 | M-046 | M-047 |
| 36-60 | below floor | M-049 | M-050 | M-048 | M-051 | M-052 |
| 61-84 | within room | ? (8) | ? (8) | ? (8) | ? (8) | ? (8) |
| 61-84 | at floor | M-058 | M-059 | M-057 | M-060 | M-061 |
| 61-84 | below floor | M-063 | M-064 | M-062 | M-065 | M-066 |
| 85-120 | within room | ? (9) | ? (9) | ? (9) | ? (9) | ? (9) |
| 85-120 | at floor | ? (10) | ? (10) | ? (10) | ? (10) | ? (10) |
| 85-120 | below floor | M-068 | ? (11) | M-067 | M-069 | M-070 |

### Grade C

| Tenure | Ask position | letter < floor | letter floor..rec | letter >= rec | claim, no letter | no competitor |
|---|---|---|---|---|---|---|
| 36-60 | within room | ? (12) | ? (12) | ? (12) | ? (12) | ? (12) |
| 36-60 | at floor | ? (13) | ? (13) | ? (13) | ? (13) | ? (13) |
| 36-60 | below floor | M-072 | ? (14) | M-071 | M-073 | M-074 |
| 61-84 | within room | ? (15) | ? (15) | ? (15) | ? (15) | ? (15) |
| 61-84 | at floor | ? (16) | ? (16) | ? (16) | ? (16) | ? (16) |
| 61-84 | below floor | M-076 | ? (17) | M-075 | M-077 | M-078 |
| 85-120 | within room | ? (18) | ? (18) | ? (18) | ? (18) | ? (18) |
| 85-120 | at floor | ? (19) | ? (19) | ? (19) | ? (19) | ? (19) |
| 85-120 | below floor | M-080 | ? (20) | M-079 | M-081 | M-082 |

## Cells without a case, and why

1. The card rate for A+ at 36-60 months is 13.00%, the same as the policy floor: nothing can be conceded, and an ask more than 25 bps under the recommendation would be under 13.00%
2. The floor for A+ at 61-84 months is 13.00%, the lowest rate allowed, so an ask below it would be under 13.00%
3. Not found in 36 probed grade A+ profiles at 85-120 months: possible under the policy (the room reached 58 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
4. Not found in 59 probed grade A profiles at 36-60 months: possible under the policy (the room reached 59 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
5. Not found in 41 probed grade A profiles at 61-84 months: possible under the policy (the room reached 40 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
6. Not found in 44 probed grade A profiles at 85-120 months: possible under the policy (the room reached 39 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
7. Not found in 57 probed grade B profiles at 36-60 months: possible under the policy (the room reached 59 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
8. Not found in 41 probed grade B profiles at 61-84 months: possible under the policy (the room reached 32 bps), but no probed file combined this evidence with this ask position; a wider probe may fill it
9. Not found in 66 probed grade B profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 30 bps
10. Not found in 66 probed grade B profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 25 bps
11. Not found in 66 probed grade B profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 0 bps
12. Not found in 74 probed grade C profiles at 36-60 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 30 bps
13. Not found in 74 probed grade C profiles at 36-60 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 25 bps
14. Not found in 74 probed grade C profiles at 36-60 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 0 bps
15. Not found in 51 probed grade C profiles at 61-84 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 30 bps
16. Not found in 51 probed grade C profiles at 61-84 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 25 bps
17. Not found in 51 probed grade C profiles at 61-84 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 0 bps
18. Not found in 74 probed grade C profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 30 bps
19. Not found in 74 probed grade C profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 25 bps
20. Not found in 74 probed grade C profiles at 85-120 months: the recommendation never rose more than 0 bps above the floor, and this cell needs more than 0 bps

Outside the matrix:

- **Grade D**: Declined under standard policy before it reaches the area manager: no card rate, no offer, so no rate request can exist (risk_grade_loading_bps D: null)
- **LAP residential / LAP commercial**: The intake API accepts only the four housing property types (resale_flat, builder_flat, self_construction, plot_plus_construction)
- **An ask within 25 bps of the recommendation**: The RM may send it alone, with a reason (offer_below_recommendation_bps 25): it never reaches the area manager
- **An ask beyond the area manager's 100 bps**: The band is [card - 60 bps, card], narrower than her 100 bps, so the policy floor always binds first
- **An ask above the recommendation**: A rate request is a request for a lower rate; going above the recommendation is the area manager's uplift, never above the matrix rate
- **An ask within 1 point of break-even**: No probed profile matched; the smallest gap between a floor and break-even was 3.28 points

## Secondary values

| Dimension | Value | Cases |
|---|---|---|
| segment | salaried_formal | 17 (M-006, M-007, M-008, M-012, ...) |
| segment | salaried_informal | 22 (M-001, M-002, M-009, M-013, ...) |
| segment | self_employed_professional | 29 (M-003, M-004, M-005, M-010, ...) |
| segment | self_employed_non_professional | 33 (M-011, M-017, M-018, M-021, ...) |
| channel | branch_walkin | 25 (M-005, M-007, M-011, M-012, ...) |
| channel | dsa | 24 (M-006, M-013, M-015, M-017, ...) |
| channel | connector | 27 (M-001, M-003, M-004, M-009, ...) |
| channel | digital | 25 (M-002, M-008, M-022, M-025, ...) |
| property_type | resale_flat | 23 (M-003, M-014, M-019, M-021, ...) |
| property_type | builder_flat | 28 (M-002, M-005, M-007, M-009, ...) |
| property_type | self_construction | 23 (M-001, M-004, M-006, M-008, ...) |
| property_type | plot_plus_construction | 27 (M-010, M-011, M-017, M-025, ...) |
| reason | branch_competitive_pressure | 15 (M-004, M-010, M-017, M-022, ...) |
| reason | competitor_lower_rate | 23 (M-001, M-006, M-013, M-020, ...) |
| reason | customer_threatening_bt | 16 (M-002, M-008, M-014, M-021, ...) |
| reason | improve_s2d_conversion | 5 (M-011, M-033, M-056, M-082, ...) |
| reason | local_market_knowledge | 15 (M-005, M-012, M-019, M-024, ...) |
| reason | month_end_target | 6 (M-009, M-028, M-052, M-078, ...) |
| reason | other | 4 (M-016, M-040, M-066, S-002) |
| reason | relationship_value | 6 (M-007, M-023, M-047, M-074, ...) |
| reason | salesperson_incentive | 4 (M-015, M-035, M-061, S-001) |
| reason | strong_negotiation | 7 (M-003, M-018, M-042, M-070, ...) |
| existing | True | 40 (M-003, M-004, M-006, M-015, ...) |
| existing | False | 61 (M-001, M-002, M-005, M-007, ...) |
| negotiating | True | 82 (M-001, M-002, M-003, M-004, ...) |
| negotiating | False | 19 (M-007, M-015, M-016, M-028, ...) |

## Special situations

| Situation | Case | Detail |
|---|---|---|
| New to credit, no bureau score: the policy refers it for manual review | S-001 | Sanjay Menon, Alwar: grade B, card 16.75, floor 16.15, rec 16.15, ask 15.85 |
| Bureau score just above the 675 minimum | S-002 | Shalini Verma, Pune: grade C, card 16.00, floor 15.40, rec 15.40, ask 15.10 |
| Loan to value within 2 points of the cap for its grade and loan size | S-003 | Aditya Verma, Kota: grade B, card 15.50, floor 14.90, rec 14.90, ask 14.60 |
| Loan of Rs 70 lakh or more (the cap is Rs 75 lakh) | S-004 | Harpreet Kapoor, Nashik: grade C, card 17.00, floor 16.40, rec 16.40, ask 16.10 |
| Stressed EMI burden within 4 points of the grade's FOIR cap | S-005 | Imran Rathore, Jaipur: grade A+, card 13.50, floor 13.00, rec 13.47, ask 13.17 |
| Age at maturity close to the limit (salaried 60, self-employed 70) | S-006 | Aditya Dubey, Surat: grade A, card 15.00, floor 14.40, rec 14.40, ask 14.10 |
| Missed payments and credit deviations on the file | S-007 | Sanjay Malhotra, Alwar: grade C, card 16.50, floor 15.90, rec 15.90, ask 15.60 |
| An ask within 1 point of break-even | none | No probed profile matched; the smallest gap between a floor and break-even was 3.28 points |
| Balance-transfer threat, competitor letter seen | S-008 | Aditya Parmar, Lucknow: grade C, card 17.00, floor 16.40, rec 16.40, ask 16.10 |
| Month-end target, nothing from the customer | S-009 | Lakshmi Iyer, Indore: grade A+, card 14.00, floor 13.40, rec 13.90, ask 13.60 |
| Price-insensitive existing customer: little to gain from a cut | S-010 | Harpreet Agarwal, Surat: grade A+, card 13.50, floor 13.00, rec 13.48, ask 13.18 |

## All cases

| Case | Customer | City | Segment | Channel | Property | Tenure | Loan | Score | Letter | Existing | Negotiating | Grade | Card | Floor | Rec | Ask | Position | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M-001 | Karan Rawat | Lucknow | salaried_informal | connector | self_construction | 3y | 10.0L | 710 | letter >= rec | no | yes | A | 14.00 | 13.40 | 13.91 | 13.40 | at floor | competitor_lower_rate |
| M-002 | Farida Saxena | Jodhpur | salaried_informal | digital | builder_flat | 3y | 10.0L | 710 | claim, no letter | no | yes | A | 14.00 | 13.40 | 13.80 | 13.40 | at floor | customer_threatening_bt |
| M-003 | Rashmi Verma | Bhopal | self_employed_professional | connector | resale_flat | 5y | 15.0L | 837 | no competitor | yes | yes | A | 14.00 | 13.40 | 13.88 | 13.40 | at floor | strong_negotiation |
| M-004 | Mahesh Yadav | Nashik | self_employed_professional | connector | self_construction | 5y | 30.0L | 792 | letter >= rec | yes | yes | A | 14.00 | 13.40 | 13.47 | 13.15 | below floor | branch_competitive_pressure |
| M-005 | Lakshmi Reddy | Jaipur | self_employed_professional | branch_walkin | builder_flat | 3y | 10.0L | 790 | letter < floor | no | yes | A | 14.00 | 13.40 | 13.61 | 13.15 | below floor | local_market_knowledge |
| M-006 | Harpreet Sharma | Lucknow | salaried_formal | dsa | self_construction | 5y | 15.0L | 763 | claim, no letter | yes | yes | A | 14.00 | 13.40 | 13.40 | 13.10 | below floor | competitor_lower_rate |
| M-007 | Divya Mishra | Indore | salaried_formal | branch_walkin | builder_flat | 4y | 10.0L | 766 | no competitor | no | no | A | 14.00 | 13.40 | 13.64 | 13.15 | below floor | relationship_value |
| M-008 | Swati Solanki | Alwar | salaried_formal | digital | self_construction | 4y | 15.0L | 681 | letter >= rec | no | yes | A | 14.00 | 13.40 | 13.91 | 13.61 | within room | customer_threatening_bt |
| M-009 | Arjun Solanki | Indore | salaried_informal | connector | builder_flat | 3y | 10.0L | 710 | no competitor | no | yes | A | 14.00 | 13.40 | 13.98 | 13.68 | within room | month_end_target |
| M-010 | Ananya Verma | Kota | self_employed_professional | connector | plot_plus_construction | 7y | 30.0L | 775 | letter >= rec | no | yes | A | 14.50 | 13.90 | 14.20 | 13.90 | at floor | branch_competitive_pressure |
| M-011 | Vandana Sharma | Kota | self_employed_non_professional | branch_walkin | plot_plus_construction | 7y | 10.0L | 828 | no competitor | no | yes | A | 14.50 | 13.90 | 14.30 | 13.90 | at floor | improve_s2d_conversion |
| M-012 | Pooja Kulkarni | Indore | salaried_formal | branch_walkin | builder_flat | 6y | 20.0L | 782 | letter >= rec | no | yes | A | 14.50 | 13.90 | 13.90 | 13.60 | below floor | local_market_knowledge |
| M-013 | Karan Malhotra | Bhopal | salaried_informal | dsa | self_construction | 6y | 15.0L | 839 | letter < floor | no | yes | A | 14.50 | 13.90 | 13.90 | 13.60 | below floor | competitor_lower_rate |
| M-014 | Harpreet Dubey | Nashik | salaried_formal | connector | resale_flat | 7y | 10.0L | 814 | claim, no letter | no | yes | A | 14.50 | 13.90 | 13.90 | 13.60 | below floor | customer_threatening_bt |
| M-015 | Shalini Kapoor | Kota | self_employed_professional | dsa | builder_flat | 6y | 15.0L | 833 | no competitor | yes | no | A | 14.50 | 13.90 | 13.90 | 13.60 | below floor | salesperson_incentive |
| M-016 | Suresh Malhotra | Alwar | self_employed_professional | connector | builder_flat | 7y | 30.0L | 775 | no competitor | no | no | A | 14.50 | 13.90 | 14.28 | 13.98 | within room | other |
| M-017 | Lakshmi Gupta | Jodhpur | self_employed_non_professional | dsa | plot_plus_construction | 9y | 15.0L | 742 | letter >= rec | yes | yes | A | 15.00 | 14.40 | 14.75 | 14.40 | at floor | branch_competitive_pressure |
| M-018 | Aarav Bansal | Ahmedabad | self_employed_non_professional | connector | self_construction | 9y | 15.0L | 742 | no competitor | yes | yes | A | 15.00 | 14.40 | 14.79 | 14.40 | at floor | strong_negotiation |
| M-019 | Karan Qureshi | Pune | salaried_formal | dsa | resale_flat | 8y | 30.0L | 829 | letter >= rec | no | yes | A | 15.00 | 14.40 | 14.40 | 14.10 | below floor | local_market_knowledge |
| M-020 | Aarav Sharma | Nagpur | self_employed_professional | branch_walkin | self_construction | 10y | 20.0L | 775 | letter < floor | yes | yes | A | 15.00 | 14.40 | 14.40 | 14.10 | below floor | competitor_lower_rate |
| M-021 | Lakshmi Joshi | Alwar | self_employed_non_professional | dsa | resale_flat | 9y | 15.0L | 742 | letter floor..rec | yes | yes | A | 15.00 | 14.40 | 14.64 | 14.15 | below floor | customer_threatening_bt |
| M-022 | Yogesh Pillai | Surat | salaried_formal | digital | builder_flat | 8y | 20.0L | 829 | claim, no letter | no | yes | A | 15.00 | 14.40 | 14.40 | 14.10 | below floor | branch_competitive_pressure |
| M-023 | Sanjay Rawat | Nagpur | self_employed_professional | branch_walkin | resale_flat | 9y | 30.0L | 790 | no competitor | no | yes | A | 15.00 | 14.40 | 14.45 | 14.15 | below floor | relationship_value |
| M-024 | Karan Sharma | Lucknow | salaried_informal | connector | self_construction | 7y | 10.0L | 708 | letter >= rec | no | yes | A+ | 13.50 | 13.00 | 13.50 | 13.00 | at floor | local_market_knowledge |
| M-025 | Harpreet Pillai | Surat | self_employed_non_professional | digital | plot_plus_construction | 6y | 10.0L | 716 | letter < floor | no | yes | A+ | 13.50 | 13.00 | 13.45 | 13.00 | at floor | competitor_lower_rate |
| M-026 | Prakash Rawat | Indore | self_employed_non_professional | digital | plot_plus_construction | 6y | 15.0L | 730 | letter floor..rec | yes | yes | A+ | 13.50 | 13.00 | 13.50 | 13.00 | at floor | customer_threatening_bt |
| M-027 | Manish Reddy | Kanpur | self_employed_non_professional | dsa | self_construction | 6y | 10.0L | 716 | claim, no letter | no | yes | A+ | 13.50 | 13.00 | 13.45 | 13.00 | at floor | branch_competitive_pressure |
| M-028 | Aditya Tiwari | Ahmedabad | self_employed_non_professional | digital | self_construction | 6y | 10.0L | 716 | no competitor | no | no | A+ | 13.50 | 13.00 | 13.46 | 13.00 | at floor | month_end_target |
| M-029 | Tarun Deshmukh | Ahmedabad | self_employed_professional | digital | self_construction | 7y | 10.0L | 774 | letter >= rec | yes | yes | A+ | 13.50 | 13.00 | 13.37 | 13.07 | within room | local_market_knowledge |
| M-030 | Ananya Shah | Pune | self_employed_non_professional | dsa | resale_flat | 6y | 15.0L | 730 | letter < floor | yes | yes | A+ | 13.50 | 13.00 | 13.45 | 13.15 | within room | competitor_lower_rate |
| M-031 | Meera Ansari | Jodhpur | self_employed_non_professional | branch_walkin | plot_plus_construction | 6y | 10.0L | 716 | letter floor..rec | no | yes | A+ | 13.50 | 13.00 | 13.40 | 13.10 | within room | customer_threatening_bt |
| M-032 | Rajendra Kulkarni | Surat | salaried_informal | digital | self_construction | 7y | 10.0L | 708 | claim, no letter | no | yes | A+ | 13.50 | 13.00 | 13.50 | 13.20 | within room | branch_competitive_pressure |
| M-033 | Aarav Kulkarni | Agra | self_employed_non_professional | branch_walkin | builder_flat | 6y | 15.0L | 730 | no competitor | yes | yes | A+ | 13.50 | 13.00 | 13.50 | 13.20 | within room | improve_s2d_conversion |
| M-034 | Neha Solanki | Jaipur | self_employed_non_professional | connector | plot_plus_construction | 10y | 10.0L | 793 | letter >= rec | yes | yes | A+ | 14.00 | 13.40 | 13.78 | 13.40 | at floor | local_market_knowledge |
| M-035 | Yogesh Yadav | Rajkot | salaried_formal | connector | resale_flat | 8y | 20.0L | 829 | no competitor | no | no | A+ | 14.00 | 13.40 | 13.75 | 13.40 | at floor | salesperson_incentive |
| M-036 | Anjali Mishra | Nagpur | self_employed_non_professional | dsa | plot_plus_construction | 10y | 10.0L | 793 | letter >= rec | yes | yes | A+ | 14.00 | 13.40 | 13.58 | 13.15 | below floor | competitor_lower_rate |
| M-037 | Suresh Patel | Nashik | self_employed_non_professional | connector | resale_flat | 10y | 10.0L | 793 | letter < floor | yes | yes | A+ | 14.00 | 13.40 | 13.54 | 13.15 | below floor | customer_threatening_bt |
| M-038 | Sanjay Solanki | Bhopal | salaried_informal | dsa | resale_flat | 9y | 10.0L | 748 | letter floor..rec | yes | yes | A+ | 14.00 | 13.40 | 13.54 | 13.15 | below floor | branch_competitive_pressure |
| M-039 | Anil Chauhan | Lucknow | salaried_formal | connector | builder_flat | 9y | 10.0L | 819 | claim, no letter | yes | yes | A+ | 14.00 | 13.40 | 13.40 | 13.10 | below floor | local_market_knowledge |
| M-040 | Swati Sheikh | Pune | self_employed_professional | branch_walkin | self_construction | 8y | 40.0L | 844 | no competitor | yes | yes | A+ | 14.00 | 13.40 | 13.90 | 13.15 | below floor | other |
| M-041 | Manish Parmar | Nashik | self_employed_non_professional | connector | builder_flat | 8y | 15.0L | 811 | letter >= rec | yes | yes | A+ | 14.00 | 13.40 | 13.90 | 13.60 | within room | competitor_lower_rate |
| M-042 | Vandana Saxena | Surat | salaried_formal | dsa | builder_flat | 8y | 20.0L | 836 | no competitor | yes | yes | A+ | 14.00 | 13.40 | 13.96 | 13.66 | within room | strong_negotiation |
| M-043 | Priya Ansari | Agra | salaried_informal | digital | plot_plus_construction | 4y | 15.0L | 685 | letter >= rec | yes | yes | B | 15.00 | 14.40 | 14.70 | 14.40 | at floor | customer_threatening_bt |
| M-044 | Karan Gupta | Agra | salaried_informal | dsa | resale_flat | 4y | 15.0L | 685 | letter < floor | yes | yes | B | 15.00 | 14.40 | 14.70 | 14.40 | at floor | branch_competitive_pressure |
| M-045 | Divya Shah | Kanpur | salaried_informal | branch_walkin | builder_flat | 4y | 15.0L | 685 | letter floor..rec | yes | yes | B | 15.00 | 14.40 | 14.70 | 14.40 | at floor | local_market_knowledge |
| M-046 | Rajendra Mishra | Nagpur | salaried_informal | connector | resale_flat | 3y | 20.0L | 685 | claim, no letter | no | yes | B | 15.00 | 14.40 | 14.77 | 14.40 | at floor | competitor_lower_rate |
| M-047 | Suresh Shah | Jodhpur | salaried_informal | dsa | builder_flat | 4y | 15.0L | 685 | no competitor | yes | yes | B | 15.00 | 14.40 | 14.70 | 14.40 | at floor | relationship_value |
| M-048 | Neha Chauhan | Nashik | self_employed_non_professional | dsa | self_construction | 4y | 30.0L | 761 | letter >= rec | no | yes | B | 15.00 | 14.40 | 14.40 | 14.10 | below floor | customer_threatening_bt |
| M-049 | Vandana Bhatt | Kanpur | self_employed_non_professional | dsa | resale_flat | 4y | 30.0L | 761 | letter < floor | no | yes | B | 15.00 | 14.40 | 14.40 | 14.10 | below floor | branch_competitive_pressure |
| M-050 | Shalini Malhotra | Bhopal | self_employed_non_professional | dsa | plot_plus_construction | 5y | 20.0L | 740 | letter floor..rec | yes | yes | B | 15.00 | 14.40 | 14.55 | 14.15 | below floor | local_market_knowledge |
| M-051 | Rekha Ansari | Kanpur | self_employed_non_professional | branch_walkin | resale_flat | 5y | 20.0L | 740 | claim, no letter | yes | yes | B | 15.00 | 14.40 | 14.55 | 14.15 | below floor | competitor_lower_rate |
| M-052 | Suresh Menon | Kanpur | self_employed_non_professional | branch_walkin | plot_plus_construction | 5y | 10.0L | 824 | no competitor | yes | yes | B | 15.00 | 14.40 | 14.40 | 14.10 | below floor | month_end_target |
| M-053 | Rashmi Deshmukh | Jaipur | salaried_informal | connector | resale_flat | 3y | 20.0L | 685 | letter >= rec | no | yes | B | 15.00 | 14.40 | 14.95 | 14.65 | within room | customer_threatening_bt |
| M-054 | Harpreet Bansal | Nashik | salaried_informal | branch_walkin | self_construction | 3y | 20.0L | 685 | letter < floor | no | yes | B | 15.00 | 14.40 | 14.74 | 14.44 | within room | branch_competitive_pressure |
| M-055 | Pooja Pandey | Rajkot | salaried_informal | branch_walkin | builder_flat | 3y | 20.0L | 685 | letter floor..rec | no | yes | B | 15.00 | 14.40 | 14.71 | 14.41 | within room | local_market_knowledge |
| M-056 | Tarun Sheikh | Jabalpur | salaried_informal | branch_walkin | plot_plus_construction | 3y | 20.0L | 685 | no competitor | no | no | B | 15.00 | 14.40 | 14.99 | 14.69 | within room | improve_s2d_conversion |
| M-057 | Nikhil Saxena | Jodhpur | self_employed_non_professional | digital | builder_flat | 7y | 10.0L | 680 | letter >= rec | no | yes | B | 15.50 | 14.90 | 15.22 | 14.90 | at floor | competitor_lower_rate |
| M-058 | Rekha Rathore | Surat | self_employed_non_professional | dsa | resale_flat | 7y | 10.0L | 680 | letter < floor | no | yes | B | 15.50 | 14.90 | 15.16 | 14.90 | at floor | customer_threatening_bt |
| M-059 | Rohan Chauhan | Pune | self_employed_non_professional | digital | plot_plus_construction | 7y | 10.0L | 680 | letter floor..rec | no | yes | B | 15.50 | 14.90 | 15.20 | 14.90 | at floor | branch_competitive_pressure |
| M-060 | Arjun Mehta | Kanpur | self_employed_non_professional | connector | resale_flat | 7y | 10.0L | 680 | claim, no letter | no | yes | B | 15.50 | 14.90 | 15.22 | 14.90 | at floor | local_market_knowledge |
| M-061 | Sunita Parmar | Kota | self_employed_non_professional | connector | builder_flat | 7y | 10.0L | 680 | no competitor | no | no | B | 15.50 | 14.90 | 15.22 | 14.90 | at floor | salesperson_incentive |
| M-062 | Meera Reddy | Nagpur | self_employed_professional | digital | plot_plus_construction | 7y | 15.0L | 682 | letter >= rec | yes | yes | B | 15.50 | 14.90 | 14.90 | 14.60 | below floor | competitor_lower_rate |
| M-063 | Vivaan Qureshi | Alwar | self_employed_professional | branch_walkin | builder_flat | 7y | 40.0L | 772 | letter < floor | yes | yes | B | 15.50 | 14.90 | 14.90 | 14.60 | below floor | customer_threatening_bt |
| M-064 | Yogesh Shah | Rajkot | salaried_informal | digital | plot_plus_construction | 7y | 30.0L | 702 | letter floor..rec | no | yes | B | 15.50 | 14.90 | 15.05 | 14.65 | below floor | branch_competitive_pressure |
| M-065 | Vandana Jain | Pune | self_employed_professional | digital | builder_flat | 6y | 40.0L | 732 | claim, no letter | no | yes | B | 15.50 | 14.90 | 14.90 | 14.60 | below floor | local_market_knowledge |
| M-066 | Swati Pandey | Rajkot | self_employed_professional | dsa | plot_plus_construction | 6y | 20.0L | 720 | no competitor | no | no | B | 15.50 | 14.90 | 14.90 | 14.60 | below floor | other |
| M-067 | Kavita Jain | Pune | self_employed_professional | connector | builder_flat | 8y | 10.0L | 713 | letter >= rec | no | yes | B | 16.00 | 15.40 | 15.40 | 15.10 | below floor | competitor_lower_rate |
| M-068 | Neha Reddy | Nashik | salaried_informal | branch_walkin | plot_plus_construction | 9y | 10.0L | 741 | letter < floor | no | yes | B | 16.00 | 15.40 | 15.40 | 15.10 | below floor | customer_threatening_bt |
| M-069 | Lakshmi Mehta | Lucknow | self_employed_professional | dsa | plot_plus_construction | 8y | 10.0L | 713 | claim, no letter | no | yes | B | 16.00 | 15.40 | 15.40 | 15.10 | below floor | branch_competitive_pressure |
| M-070 | Kavita Mishra | Bhopal | salaried_informal | digital | builder_flat | 9y | 25.0L | 803 | no competitor | no | no | B | 16.00 | 15.40 | 15.40 | 15.10 | below floor | strong_negotiation |
| M-071 | Manish Kulkarni | Jabalpur | salaried_informal | dsa | resale_flat | 4y | 30.0L | 795 | letter >= rec | no | yes | C | 16.00 | 15.40 | 15.40 | 15.10 | below floor | local_market_knowledge |
| M-072 | Anjali Pillai | Rajkot | self_employed_non_professional | branch_walkin | builder_flat | 4y | 60.0L | 697 | letter < floor | no | yes | C | 16.00 | 15.40 | 15.40 | 15.10 | below floor | competitor_lower_rate |
| M-073 | Swati Mehta | Nagpur | salaried_formal | connector | resale_flat | 4y | 50.0L | 709 | claim, no letter | no | yes | C | 16.00 | 15.40 | 15.40 | 15.10 | below floor | customer_threatening_bt |
| M-074 | Imran Reddy | Lucknow | self_employed_professional | digital | resale_flat | 4y | 25.0L | 713 | no competitor | yes | yes | C | 16.00 | 15.40 | 15.40 | 15.10 | below floor | relationship_value |
| M-075 | Rajendra Reddy | Ahmedabad | self_employed_professional | digital | plot_plus_construction | 6y | 25.0L | 677 | letter >= rec | no | yes | C | 16.50 | 15.90 | 15.90 | 15.60 | below floor | branch_competitive_pressure |
| M-076 | Rashmi Kulkarni | Kota | self_employed_professional | branch_walkin | self_construction | 7y | 30.0L | 739 | letter < floor | yes | yes | C | 16.50 | 15.90 | 15.90 | 15.60 | below floor | local_market_knowledge |
| M-077 | Dinesh Rathore | Bhopal | self_employed_professional | branch_walkin | builder_flat | 6y | 40.0L | 801 | claim, no letter | no | yes | C | 16.50 | 15.90 | 15.90 | 15.60 | below floor | competitor_lower_rate |
| M-078 | Farida Patel | Nashik | self_employed_professional | connector | self_construction | 6y | 25.0L | 677 | no competitor | no | no | C | 16.50 | 15.90 | 15.90 | 15.60 | below floor | month_end_target |
| M-079 | Sunita Nair | Lucknow | self_employed_professional | dsa | builder_flat | 10y | 30.0L | 815 | letter >= rec | yes | yes | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | customer_threatening_bt |
| M-080 | Lakshmi Mishra | Pune | salaried_formal | branch_walkin | self_construction | 9y | 74.0L | 796 | letter < floor | no | yes | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | branch_competitive_pressure |
| M-081 | Prakash Verma | Ahmedabad | salaried_formal | branch_walkin | self_construction | 8y | 25.0L | 706 | claim, no letter | no | yes | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | local_market_knowledge |
| M-082 | Rohan Joshi | Jodhpur | salaried_formal | connector | self_construction | 8y | 70.0L | 786 | no competitor | yes | no | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | improve_s2d_conversion |
| S-001 | Sanjay Menon | Alwar | salaried_formal | digital | resale_flat | 10y | 15.0L | none | no competitor | no | no | B | 16.75 | 16.15 | 16.15 | 15.85 | below floor | salesperson_incentive |
| S-002 | Shalini Verma | Pune | self_employed_professional | branch_walkin | self_construction | 5y | 15.0L | 676 | no competitor | no | no | C | 16.00 | 15.40 | 15.40 | 15.10 | below floor | other |
| S-003 | Aditya Verma | Kota | self_employed_professional | digital | self_construction | 6y | 20.0L | 720 | claim, no letter | no | yes | B | 15.50 | 14.90 | 14.90 | 14.60 | below floor | competitor_lower_rate |
| S-004 | Harpreet Kapoor | Nashik | salaried_formal | dsa | plot_plus_construction | 9y | 72.0L | 690 | no competitor | yes | no | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | strong_negotiation |
| S-005 | Imran Rathore | Jaipur | self_employed_professional | branch_walkin | plot_plus_construction | 7y | 15.0L | 832 | no competitor | no | yes | A+ | 13.50 | 13.00 | 13.47 | 13.17 | within room | relationship_value |
| S-006 | Aditya Dubey | Surat | self_employed_professional | digital | plot_plus_construction | 10y | 74.0L | 804 | no competitor | no | no | A | 15.00 | 14.40 | 14.40 | 14.10 | below floor | month_end_target |
| S-007 | Sanjay Malhotra | Alwar | salaried_informal | branch_walkin | builder_flat | 7y | 15.0L | 684 | no competitor | yes | no | C | 16.50 | 15.90 | 15.90 | 15.60 | below floor | improve_s2d_conversion |
| S-008 | Aditya Parmar | Lucknow | self_employed_non_professional | connector | builder_flat | 8y | 50.0L | 688 | letter < floor | no | yes | C | 17.00 | 16.40 | 16.40 | 16.10 | below floor | customer_threatening_bt |
| S-009 | Lakshmi Iyer | Indore | self_employed_non_professional | digital | plot_plus_construction | 9y | 30.0L | 843 | no competitor | no | no | A+ | 14.00 | 13.40 | 13.90 | 13.60 | within room | month_end_target |
| S-010 | Harpreet Agarwal | Surat | self_employed_professional | connector | plot_plus_construction | 7y | 10.0L | 774 | no competitor | yes | no | A+ | 13.50 | 13.00 | 13.48 | 13.18 | within room | relationship_value |

## Decision history

With `--decide-history` these are decided by the area manager (meera) and set back in time, for the precedent and decision-memory panels.

| Case | Customer | Grade | Tenure | Card | Floor | Rec | Ask | Decision | Rate |
|---|---|---|---|---|---|---|---|---|---|
| H-001 | Anjali Patel, Surat | A+ | 61-84 | 13.50 | 13.00 | 13.41 | 13.11 | Approve as asked | 13.11 |
| H-002 | Vivaan Iyer, Jaipur | A+ | 85-120 | 14.00 | 13.40 | 13.89 | 13.40 | Approve at the floor | 13.40 |
| H-003 | Manish Kapoor, Bhopal | A | 36-60 | 14.00 | 13.40 | 13.59 | 13.15 | Counter at the floor | 13.40 |
| H-004 | Rashmi Rawat, Jaipur | A+ | 85-120 | 14.00 | 13.40 | 13.97 | 13.67 | Decline | rec stands |
| H-005 | Anil Shah, Bhopal | C | 36-60 | 16.00 | 15.40 | 15.40 | 15.10 | Decline | rec stands |
| H-006 | Rashmi Patel, Jaipur | C | 36-60 | 16.00 | 15.40 | 15.40 | 15.10 | Uplift above the recommendation | 15.55 |
| H-007 | Suresh Kulkarni, Indore | A | 36-60 | 14.00 | 13.40 | 13.49 | 13.15 | Uplift above the recommendation | 13.54 |
| H-008 | Rajendra Pandey, Jodhpur | A+ | 61-84 | 13.50 | 13.00 | 13.37 | 13.07 | Approve as asked | 13.07 |
| H-009 | Harpreet Joshi, Agra | A+ | 85-120 | 14.00 | 13.40 | 13.47 | 13.15 | Counter at the floor | 13.40 |

## How to run

```
# 1. Probe the engine (throwaway API only, never the live server)
PYTHONPATH=src python scripts/demo_catalogue.py probe --base http://127.0.0.1:8011 --scratch

# 2. Create the pending cases on a server
PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000

# 2b. Also create and decide the history cases
PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000 --decide-history

# 3. Rewrite this page from the catalogue
python scripts/demo_catalogue.py report
```

### Probing on another machine (the server's models differ from a laptop's)

Recommendations move a little between model sets, so probe where the cases will live, writing a separate catalogue, then create from it:

```
# a throwaway API on a copy of the store (works on Linux and Windows; see scripts/scratch_api.py)
PYTHONPATH=src python scripts/scratch_api.py start --source /path/to/aavas_governance.db --port 8011 --dir /tmp/scratch
PYTHONPATH=src python scripts/demo_catalogue.py probe --base http://127.0.0.1:8011 --scratch --out /tmp/catalogue.json
PYTHONPATH=src python scripts/scratch_api.py stop --dir /tmp/scratch     # stops the API, deletes the copy
PYTHONPATH=src python scripts/demo_catalogue.py create --base http://127.0.0.1:8000 --catalogue /tmp/catalogue.json
python scripts/demo_catalogue.py report --catalogue /tmp/catalogue.json --out /tmp/cases.md
```

`create` checks that every case landed in its intended cell, prints any that did not (and labels them by what was observed), and refuses to post a rate outside the policy band. It back-dates the files so the worklists show aging; pass `--no-backdate` to skip that. Run it once per server: a second run adds a second set of cases.
