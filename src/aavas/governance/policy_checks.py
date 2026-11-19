"""Consistency checks on the pricing policy file: invariants INV-01..INV-11 (Part C).

They read only the YAML, so they catch a bad edit before any code prices with it.
``scripts/check_policy.py`` prints them; the agentic workflow's policy step runs
them before any training starts.
"""

from __future__ import annotations

import math
from typing import Any

ACTION_KEYS = {"action", "outcome", "breach", "below", "no_score", "decision", "route", "confirmed_fraud",
               "unsafe", "prohibited_activity", "non_positive_tnw", "overdue_mandatory_return",
               "fails_at_120_months", "unresolved_calc_loop", "below_category_floor",
               "inconclusive_mandatory_factor"}
CONTRACT_BLOCKS = ("actions", "model_variant_routing", "reason_codes", "invariants", "decisions",
                   "open_items", "go_live_gates")


def run(policy: dict[str, Any], grade_cutoffs: dict[str, float]) -> list[dict[str, Any]]:
    """Every check as ``{"check", "ok", "detail"}``; ``ok`` is None for a skipped check."""
    out: list[dict[str, Any]] = []

    def check(name: str, ok: bool | None, detail: str = "") -> None:
        out.append({"check": name, "ok": ok, "detail": detail})

    sc = policy["scorecard"]
    for model, cats in ((m, policy["models"][m]["category_weights"]) for m in policy["models"]):
        for cat, weight in cats.items():
            total = sum(v["weight"].get(model, 0) for v in sc[cat].values() if isinstance(v, dict) and "weight" in v)
            check(f"INV-01 {model}.{cat}", total == weight, f"parameters {total}, category {weight}")
        check(f"INV-01 {model} total", sum(cats.values()) == 100, str(sum(cats.values())))

    pr = policy["pricing"]
    adj = pr["tenure_adjustment"]
    for grade, row in pr["matrix"].items():
        base = pr["grade_base_roi"][grade]
        for i, rate in enumerate(row):
            if base is None:
                check(f"INV-02 {grade}[{i}]", rate is None, "no base rate, so no matrix rate")
                continue
            check(f"INV-02 {grade}[{i}]", rate is not None and math.isclose(rate, base + adj[i]["add_pct"]),
                  f"{rate} vs {base} + {adj[i]['add_pct']}")
        loading = policy["risk_grade_loading_bps"].get(grade)
        check(f"INV-03 {grade}", (loading is None and base is None) or (
            loading is not None and base is not None
            and math.isclose(policy["reference"]["card_rate_floor"] + loading / 100, base)))
    for band, a in zip(policy["tenure_loading_bps"], adj):
        check(f"INV-04 {band['max_months']}m",
              band["max_months"] == a["max_months"] and math.isclose(band["bps"] / 100, a["add_pct"]))

    rates = [r for row in pr["matrix"].values() for r in row if r is not None]
    check("INV-05 matrix inside the band",
          min(rates) >= pr["roi_floor"] and max(rates) <= pr["roi_ceiling"]
          and pr["roi_floor"] == policy["reference"]["card_rate_floor"]
          and pr["roi_ceiling"] == policy["reference"]["card_rate_ceiling"])

    for name in ("employer", "builder"):
        check(f"INV-06 {name}", sum(f["weight"] for f in policy["subscores"][name]["factors"].values()) == 100)
    check("INV-06 industry", sum(policy["subscores"]["industry"]["factors"].values()) == 100)

    for trail, bands in _band_lists(policy):
        ends = bands[-1].get("else") is True
        ge = [b["ge"] for b in bands if "ge" in b]
        le = [b["le"] for b in bands if "le" in b]
        check(f"INV-07 {trail}", ends and ge == sorted(ge, reverse=True) and le == sorted(le),
              "" if ends else "no else catch-all")

    if "actions" in policy:
        missing = sorted(_action_words(policy) - set(policy["actions"]) - {"PASS"})
        check("INV-08 actions catalogued", not missing, f"missing {missing}" if missing else "")
    else:
        check("INV-08 actions catalogued", None, "no actions catalogue in this file")

    check("INV-09 risk-model grades priced", set(grade_cutoffs) <= set(policy["risk_grade_loading_bps"]),
          f"risk model grades {sorted(grade_cutoffs)}")
    con = policy["constraints"]
    if "stress_rate_add_bps" in con:
        check("INV-10 stress add-on (legacy)", math.isclose(con["stress_rate_add_bps"] / 100, pr["stress_test_add_pct"]))
    check("INV-10 no premium above the matrix", all(l["max_premium_bps"] == 0 for l in policy["approval_authority"]))
    check("INV-11 tenure limits agree",
          con["max_tenure_years"] * 12 == con.get("max_tenure_months") == adj[-1]["max_months"])

    if "reason_codes" in policy:
        actioned = {pid for cat in sc.values() if isinstance(cat, dict) for pid, v in cat.items()
                    if isinstance(v, dict) and _has_action(v)}
        missing = sorted(actioned - set(policy["reason_codes"]))
        check("Reason codes for every actioned measure", not missing, f"missing {missing}" if missing else "")
    else:
        check("Reason codes for every actioned measure", None, "no reason_codes block in this file")
    return out


def summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [r for r in results if r["ok"] is False]
    return {"checks": len(results), "passed": sum(1 for r in results if r["ok"]),
            "skipped": sum(1 for r in results if r["ok"] is None), "failed": failed, "ok": not failed}


def _band_lists(node, trail=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _band_lists(v, f"{trail}.{k}")
    elif isinstance(node, list):
        if node and all(isinstance(b, dict) and ({"ge", "le", "gt", "lt", "else"} & set(b)) for b in node) \
                and any("score" in b or "grade" in b or "outcome" in b for b in node):
            yield trail.lstrip("."), node
        for i, v in enumerate(node):
            yield from _band_lists(v, f"{trail}[{i}]")


def _action_words(policy: dict) -> set[str]:
    found: set[str] = set()

    def collect(node, key=None, mandatory=False):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "categories" and isinstance(v, dict):
                    found.update(x for x in v.values() if isinstance(x, str))
                collect(v, k, mandatory)
        elif isinstance(node, list):
            for v in node:
                collect(v, key, mandatory)
        elif isinstance(node, str) and (key in ACTION_KEYS or mandatory):
            found.add(node)

    for k, v in policy.items():
        if k not in CONTRACT_BLOCKS:
            collect(v, None, mandatory=(k == "mandatory_rules"))
    collect(policy.get("model_variant_routing") or {})
    return found


def _has_action(node) -> bool:
    if isinstance(node, dict):
        return "action" in node or "prohibited_activity" in node or any(_has_action(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_action(v) for v in node)
    return False
