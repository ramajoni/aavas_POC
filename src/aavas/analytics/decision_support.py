"""A suggested decision on a lower-rate request, with the three or four facts
behind it - so an area manager can decide in a couple of minutes.

The suggestion is the rate that earns the bank the most, given how this branch's
customers respond: the expected value of the loan (its value if booked, times the
chance this branch's customer takes it) at every rate from the RM's ask up to the
system recommendation, among the rates the manager may approve.

    best at the recommendation  -> hold the system recommendation (decline the request)
    best at the RM's ask        -> approve the ask
    best in between             -> approve that rate instead (a counter-offer)
    best above the recommendation -> uplift: decline the discount and price higher, when
                                   customers like this still accept (policy
                                   ``override.uplift``), never above the quoted range

Nothing here decides: it fills the decision form, and the manager approves or
declines with their own note.
"""

from __future__ import annotations

from typing import Any

from ..config import policy


def _pct(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.2f}%"


def _inr(value: float | None) -> str:
    """Rupees with Indian digit grouping: 1,55,622."""
    if value is None:
        return "—"
    digits = f"{abs(value):.0f}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(([head] if head else []) + groups + [tail])


def suggest(request: dict[str, Any], scen: dict[str, Any], *, floor_rate: float,
            similar: dict[str, Any] | None, evidence_verified: bool,
            uplift_ceiling: float | None = None) -> dict[str, Any]:
    ask, rec = float(request["requested_rate"]), float(scen["recommended_rate"])
    reason = request.get("reason")
    # A row the table shows only for reference (the ask below the floor) is looked at, never chosen.
    options = [r for r in scen["table"] if not r.get("outside_policy")]
    rows = {round(r["rate"], 2): r for r in scen["table"]}
    allowed = [r for r in options if max(ask, floor_rate) - 1e-9 <= r["rate"] <= rec + 1e-9]
    at_ask, at_rec = rows.get(round(ask, 2)), rows.get(round(rec, 2))
    best = max(allowed, key=lambda r: r["value_branch_inr"]) if allowed else at_rec

    if best is None or abs(best["rate"] - rec) < 0.005:
        action, rate = "hold", rec
        headline = f"Hold {_pct(rec)}: decline the request"
    elif abs(best["rate"] - ask) < 0.005:
        action, rate = "approve", ask
        headline = f"Approve {_pct(ask)}"
    else:
        action, rate = "counter", best["rate"]
        headline = f"Approve {_pct(rate)} instead of {_pct(ask)}"

    # Above the model rate: worth it only where customers like this still accept and
    # the loan is worth clearly more, and never above the range the customer was quoted.
    up_cfg = ((policy().get("override") or {}).get("uplift") or {})
    up_best = None
    if action == "hold" and at_rec and uplift_ceiling is not None and uplift_ceiling > rec + 0.005:
        drop = float(up_cfg.get("max_acceptance_drop_pts", 3)) / 100
        gain = float(up_cfg.get("min_value_gain_pct", 0.5)) / 100
        above = [r for r in options if rec + 0.005 < r["rate"] <= uplift_ceiling + 1e-9
                 and r["acceptance_model"] >= at_rec["acceptance_model"] - drop]
        up_best = max(above, key=lambda r: r["value_branch_inr"]) if above else None
        if up_best and up_best["value_branch_inr"] > at_rec["value_branch_inr"] * (1 + gain):
            action, rate = "uplift", up_best["rate"]
            headline = f"Uplift to {_pct(rate)}: decline the discount and price higher"
        else:
            up_best = None

    reasons: list[dict[str, str]] = []
    if action == "uplift":
        reasons.append({"tone": "good", "text": f"At {_pct(rate)}, {up_best['acceptance_model']:.0%} of customers like this "
                                                f"still accept, against {at_rec['acceptance_model']:.0%} at {_pct(rec)}."})
        reasons.append({"tone": "good", "text": f"The loan is worth {_inr(up_best['value_branch_inr'])} to the bank at {_pct(rate)}, "
                                                f"against {_inr(at_rec['value_branch_inr'])} at {_pct(rec)}."})
    if at_ask and at_rec:
        # The model's acceptance, the number every other screen shows; the choice of
        # rate below uses this branch's response.
        gain = round((at_ask["acceptance_model"] - at_rec["acceptance_model"]) * 100)
        acc_rec, acc_ask = f"{at_rec['acceptance_model']:.0%}", f"{at_ask['acceptance_model']:.0%}"
        reasons.append({"tone": "good" if gain >= 5 else "bad",
                        "text": (f"The cut wins almost no extra customers: {acc_rec} accept at {_pct(rec)}, {acc_ask} at {_pct(ask)}."
                                 if gain < 1 else
                                 f"Cutting to {_pct(ask)} raises the chance the customer accepts from {acc_rec} to {acc_ask} "
                                 f"({'only ' if gain < 5 else ''}+{gain} point{'s' if gain != 1 else ''}).")})
        chosen = rows.get(round(rate, 2)) or at_rec
        if action != "uplift":                   # the uplift already set its value against the model rate
            reasons.append({"tone": "good",
                            "text": f"The loan is worth {_inr(chosen['value_branch_inr'])} to the bank at {_pct(rate)}, "
                                    f"against {_inr(at_ask['value_branch_inr'])} at the RM's {_pct(ask)}."})
    history = scen.get("branch_history") or {}
    if history.get("s2d_lift_pts") is not None:
        lift, value = history["s2d_lift_pts"], history.get("value_lift_inr") or 0
        where, files = scen.get("city") or "the branch", history.get("files")
        figures = (f"customers accepted {abs(lift):.0f} points {'more' if lift >= 0 else 'less'} often and each loan "
                   f"earned {_inr(abs(value))} {'more' if value >= 0 else 'less'}")
        verdict = history.get("verdict")
        reasons.append({"tone": "good" if verdict == "value creating" else "bad" if verdict == "value destroying" else "warn",
                        "text": (f"Past cuts like this in {where} paid off: {figures} ({files} loans)." if verdict == "value creating"
                                 else f"Past cuts like this in {where} did not pay: {figures} ({files} loans)." if verdict == "value destroying"
                                 else f"Past cuts like this in {where} are not a reliable guide: {figures}, but {files} loans "
                                      "are too few to be sure.")})
    depth = (rec - ask) * 100
    if similar and similar.get("deepest_bps") is not None and depth > similar["deepest_bps"]:
        reasons.append({"tone": "bad", "text": f"A cut of {depth / 100:.2f}% is bigger than any given here before for this reason "
                                               f"(the largest so far was {similar['deepest_bps'] / 100:.2f}%)."})
    if action != "hold" and reason == "competitor_lower_rate" and not evidence_verified:
        reasons.append({"tone": "warn", "text": "The competitor quote is not yet evidenced: approve subject to the sanction letter."})

    condition = (" Subject to the competitor's sanction letter before disbursement."
                 if action != "hold" and reason == "competitor_lower_rate" and not evidence_verified else "")
    small = not (at_ask and at_rec) or (at_ask["acceptance_model"] - at_rec["acceptance_model"]) < 0.05
    note = {"hold": f"Keep the system recommendation of {_pct(rec)}: " + ("the cut wins few extra customers and makes the loan worth less to the bank."
                                                                  if small else "the extra customers it wins are worth less than the interest given up."),
            "approve": f"Approved at {_pct(rate)}: the cut is worth it for this customer.",
            "counter": f"Approved at {_pct(rate)} instead of {_pct(ask)}: the best value for the bank between the two.",
            "uplift": f"Cut declined and the rate set at {_pct(rate)}: customers like this still accept, and the loan is worth more."}[action] + (condition if action != "uplift" else "")
    return {"action": action, "rate": round(rate, 2), "headline": headline, "reasons": reasons[:4], "note": note,
            "reason": (uplift_reasons_default() if action == "uplift" else reason), "values": {"ask": at_ask and at_ask["value_branch_inr"], "model": at_rec and at_rec["value_branch_inr"],
                                         "suggested": (rows.get(round(rate, 2)) or {}).get("value_branch_inr")},
            "method": "Suggested from the rate that makes the loan worth the most to the bank, based on how customers "
                      "in this branch respond to price."}


def uplift_reasons_default() -> str | None:
    reasons = ((policy().get("override") or {}).get("uplift") or {}).get("reasons") or []
    return reasons[0] if reasons else None
