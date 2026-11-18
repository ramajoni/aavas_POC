"""Policy file consistency checks (invariants INV-01..INV-11 in the policy's Part C).

    python scripts/check_policy.py [config/pricing_policy.yaml] [config/model_config.yaml]

Prints PASS / FAIL / SKIP per check and exits 1 if any check fails."""
import sys, yaml, math
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "config" / "pricing_policy.yaml"
model_cfg = sys.argv[2] if len(sys.argv) > 2 else ROOT / "config" / "model_config.yaml"
p = yaml.safe_load(open(path, encoding="utf-8"))
fails = []
def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

# INV-01 weights
sc = p["scorecard"]
for model, cats in ((m, p["models"][m]["category_weights"]) for m in p["models"]):
    for cat, w in cats.items():
        s = sum(v["weight"].get(model, 0) for k, v in sc[cat].items() if isinstance(v, dict) and "weight" in v)
        ok(f"INV-01 {model}.{cat}", s == w, f"params={s} category={w}")
    ok(f"INV-01 {model} total", sum(cats.values()) == 100, str(sum(cats.values())))
# INV-02/03/04
pr = p["pricing"]; adj = pr["tenure_adjustment"]
for g, row in pr["matrix"].items():
    base = pr["grade_base_roi"][g]
    for i, r in enumerate(row):
        if base is None: ok(f"INV-02 {g}[{i}] null", r is None); continue
        ok(f"INV-02 {g}[{i}]", math.isclose(r, base + adj[i]["add_pct"]), f"{r} vs {base}+{adj[i]['add_pct']}")
    load = p["risk_grade_loading_bps"].get(g)
    ok(f"INV-03 {g}", (load is None and base is None) or (load is not None and math.isclose(p["reference"]["card_rate_floor"] + load/100, base)))
for t, a in zip(p["tenure_loading_bps"], adj):
    ok(f"INV-04 {t['max_months']}m", t["max_months"] == a["max_months"] and math.isclose(t["bps"]/100, a["add_pct"]))
# INV-05
rates = [r for row in pr["matrix"].values() for r in row if r is not None]
ok("INV-05 matrix in band", min(rates) >= pr["roi_floor"] and max(rates) <= pr["roi_ceiling"]
   and pr["roi_floor"] == p["reference"]["card_rate_floor"] and pr["roi_ceiling"] == p["reference"]["card_rate_ceiling"])
# INV-06
for name in ("employer", "builder"):
    ok(f"INV-06 {name}", sum(f["weight"] for f in p["subscores"][name]["factors"].values()) == 100)
ok("INV-06 industry", sum(p["subscores"]["industry"]["factors"].values()) == 100)
# INV-07 band order + catch-all
def walk(node, trail=""):
    if isinstance(node, dict):
        for k, v in node.items(): yield from walk(v, f"{trail}.{k}")
    elif isinstance(node, list):
        if node and all(isinstance(b, dict) and ({"ge","le","gt","lt","else"} & set(b)) for b in node) and any("score" in b or "grade" in b or "outcome" in b or "max_ltv" in b for b in node):
            yield trail, node
        for i, v in enumerate(node): yield from walk(v, f"{trail}[{i}]")
for trail, bands in walk(p):
    ends = bands[-1].get("else") is True
    ge = [b["ge"] for b in bands if "ge" in b]; le = [b["le"] for b in bands if "le" in b]
    mono = ge == sorted(ge, reverse=True) and le == sorted(le)
    ok(f"INV-07 {trail}", ends and mono, "" if ends else "no else catch-all")
# INV-08 every action word catalogued
ACTION_KEYS = {"action", "outcome", "breach", "below", "no_score", "decision", "route", "confirmed_fraud", "unsafe",
               "prohibited_activity", "non_positive_tnw", "overdue_mandatory_return", "fails_at_120_months",
               "unresolved_calc_loop", "below_category_floor", "inconclusive_mandatory_factor"}
found = set()
def collect(node, key=None, under_mr=False):
    if isinstance(node, dict):
        for k, v in node.items():
            collect(v, k, under_mr)
    elif isinstance(node, list):
        for v in node: collect(v, key, under_mr)
    elif isinstance(node, str) and (key in ACTION_KEYS or under_mr):
        found.add(node)
for k, v in p.items():
    if k in ("actions", "model_variant_routing", "reason_codes", "invariants", "open_items", "go_live_gates"): continue
    collect(v, None, under_mr=(k == "mandatory_rules"))
collect(p.get("model_variant_routing") or {})
# categories with bare string values are actions too (e.g. missing_evidence: inconclusive)
def cat_strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "categories" and isinstance(v, dict):
                found.update(x for x in v.values() if isinstance(x, str))
            cat_strings(v)
    elif isinstance(node, list):
        for v in node: cat_strings(v)
cat_strings(p)
if "actions" in p:
    missing = sorted(found - set(p["actions"]) - {"PASS"})
    ok("INV-08 actions catalogued", not missing, f"missing={missing}")
else:
    print("SKIP INV-08 (no actions catalogue in this file)")
# INV-09 grades the PD model emits
cutoffs = yaml.safe_load(open(model_cfg, encoding="utf-8"))["risk"]["grade_cutoffs"]
ok("INV-09 PD grades covered", set(cutoffs) <= set(p["risk_grade_loading_bps"]), f"pd={sorted(cutoffs)}")
# INV-10, INV-11
c = p["constraints"]
if "stress_rate_add_bps" in c:
    ok("INV-10 stress add (legacy)", math.isclose(c["stress_rate_add_bps"]/100, pr["stress_test_add_pct"]))
ok("INV-10 no premium above the matrix", all(l["max_premium_bps"] == 0 for l in p["approval_authority"]))
ok("INV-11 tenure", c["max_tenure_years"]*12 == c["max_tenure_months"] == adj[-1]["max_months"])
# reason codes cover every scorecard parameter that carries an action
with_action = {pid for cat in sc.values() if isinstance(cat, dict) for pid, v in cat.items()
               if isinstance(v, dict) and "action" in yaml.safe_dump(v) or (isinstance(v, dict) and "prohibited_activity" in v)}
if "reason_codes" in p:
    ok("reason codes for actioned parameters", with_action <= set(p["reason_codes"]), f"missing={sorted(with_action - set(p['reason_codes']))}")
else:
    print("SKIP reason codes (no reason_codes block in this file)")
print("\nFAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
