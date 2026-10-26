"""How believable a competitor's quoted rate is, as a score out of 100.

Three things decide it, each worth a stated number of points so a reviewer can
see why a quote scored what it did (``config/market_research.yaml``):

* **Evidence.** A sanction letter the relationship manager has seen starts the
  claim high; a rate the customer only mentioned starts it low.
* **The market.** What the lender advertises today, found online by the market
  research agent. A quote inside that range is believable; one under the lowest
  advertised rate is not, the more so the further under it is. Rates found only
  on news or comparison sites count for less than the lender's own website.
* **The customer.** A lender's lowest rate goes to its best borrowers. A quote at
  that floor for a customer with a low or no credit score is less likely.

The score is guidance for a person and never moves the recommendation. It says
how far the quote can justify a discount: nothing when doubtful; at most the
policy's allowance for an unevidenced claim when no letter has been seen; no
claim-specific limit when the letter is seen and the quote is believable.
"""

from __future__ import annotations

from typing import Any

from ..config import load, policy

SEGMENT_WORDS = {"salaried_informal": "informally salaried",
                 "self_employed_non_professional": "self-employed (non-professional)"}


def settings() -> dict[str, Any]:
    return load("market_research")["claim_scoring"]


def score(quote: float, evidence_seen: bool, profile: dict[str, Any] | None = None,
          research: dict[str, Any] | None = None, *, lender_label: str = "the lender",
          plural: bool = False, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """``plural`` when ``lender_label`` names several lenders (the market as a whole)."""
    cfg = cfg or settings()
    advertises = "advertise" if plural else "advertises"
    seen = "seen" if evidence_seen else "not_seen"
    quote = float(quote)
    components = [{"label": "Sanction letter seen" if evidence_seen else "Sanction letter not seen",
                   "points": int(cfg["base"][seen])}]

    research = research or {}
    lo, hi = research.get("published_min"), research.get("published_max")
    position, gap_bps = "unknown", None
    if lo is None:
        components.append({"label": "Fact check against advertised rates", "points": 0,
                           "detail": "no advertised rate to compare with"})
    else:
        secondary = research.get("source_quality") != "lender_website"
        weight = float(cfg["secondary_source_weight"]) if secondary else 1.0
        span = f"{lo:.2f}%" + (f"–{hi:.2f}%" if hi is not None and hi > lo else "")
        if quote < lo - 1e-9:
            gap_bps = round((lo - quote) * 100)
            points = next(b["points"] for b in cfg["below_floor"]
                          if b["max_bps"] is None or gap_bps <= b["max_bps"])
            position = "below_floor"
            detail = f"{gap_bps} bps under the lowest rate {lender_label} {advertises} ({lo:.2f}%)"
        elif hi is not None and quote > hi + 1e-9:
            points, position = cfg["above_range"][seen], "above_range"
            detail = f"above what {lender_label} {advertises} ({span})"
        else:
            points, position = cfg["within_range"][seen], "within"
            detail = f"within what {lender_label} {advertises} ({span})"
        if secondary:
            detail += "; found on news or comparison sites, not the lender's own"
        components.append({"label": "Fact check against advertised rates", "points": round(points * weight),
                           "detail": detail})

        near_floor = quote <= lo + float(cfg["best_rate_window_bps"]) / 100 + 1e-9
        weak = _weak_profile(profile or {}, research, cfg) if near_floor else None
        if weak:
            components.append({"label": "Customer profile", "points": int(cfg["weak_profile_points"]),
                               "detail": f"{lender_label}'s lowest rate usually needs a strong "
                                         f"profile; this customer is {weak}"})

    total = max(0, min(100, sum(c["points"] for c in components)))
    band = next(b for b in cfg["bands"] if total >= b["min"])
    unevidenced = int(policy()["override"]["max_unevidenced_discount_bps"])
    if band["key"] == "doubtful":
        cap = 0
    elif not evidence_seen:
        cap = unevidenced
    elif band["key"] in ("credible", "plausible"):
        cap = None
    else:
        cap = unevidenced
    return {
        "points": total, "band": band["key"], "label": band["label"],
        "components": components, "quote": quote, "evidence_seen": evidence_seen,
        "market_position": position, "gap_to_floor_bps": gap_bps,
        "max_discount_bps": cap,
        "guidance": _guidance(band["key"], evidence_seen, position, quote, lo, gap_bps, cap,
                              lender_label, advertises),
    }


def _weak_profile(profile: dict[str, Any], research: dict[str, Any],
                  cfg: dict[str, Any]) -> str | None:
    score = profile.get("bureau_score")
    need = research.get("best_rate_min_credit_score") or cfg["best_rate_min_credit_score"]
    if profile.get("new_to_credit") or score is None or score < 0:
        return "new to credit"
    if score < need:
        return f"at a credit score of {score}, under the {need}+ it usually asks for"
    if profile.get("segment") in cfg.get("weak_segments", ()):
        return SEGMENT_WORDS.get(profile["segment"], str(profile["segment"]).replace("_", " "))
    return None


def _guidance(band: str, seen: bool, position: str, quote: float, lo: float | None,
              gap_bps: int | None, cap: int | None, lender: str, advertises: str) -> str:
    allowance = f"{cap / 100:.2f}%" if cap else None
    if band == "doubtful":
        if position == "below_floor":
            subject = f"{lender[:1].upper()}{lender[1:]} {'do' if advertises == 'advertise' else 'does'}"
            return (f"{subject} not advertise rates this low: {quote:.2f}% is {gap_bps} bps under "
                    f"the lowest ({lo:.2f}%). "
                    + ("Check the letter is genuine and for this loan amount, tenure and customer. "
                       if seen else "Without a sanction letter, treat this quote as unlikely. ")
                    + "It cannot justify a lower rate.")
        return "This quote cannot justify a lower rate as it stands."
    if not seen:
        start = (f"The quote is in line with what {lender} {advertises}, but no"
                 if position == "within" else "No")
        return (f"{start} sanction letter has been seen. Ask for it: until then this quote can "
                f"justify at most {allowance} off the recommendation.")
    if position == "below_floor":
        return (f"The letter shows {quote:.2f}%, {gap_bps} bps under the lowest rate {lender} "
                f"{advertises} ({lo:.2f}%). Confirm it is for this loan amount and tenure before "
                f"matching it; until then it can justify at most {allowance} off.")
    if position == "unknown":
        return ("Sanction letter seen. No advertised rate was found to compare it with, so it "
                "rests on the letter alone.")
    return (f"Sanction letter seen and the rate is consistent with what {lender} {advertises}. "
            "It can support a matching offer within your pricing authority.")
