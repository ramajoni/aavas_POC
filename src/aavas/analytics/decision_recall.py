"""Which past decisions are like this one, how a manager tends to decide, and whether
today's suggestion sits with that habit or against it.

Three readings over decision-memory episodes, all deterministic:

* **nearest** - the past requests most like this one, each with the factors that
  made it alike. Similarity is a weighted match over a handful of case fields, not
  an embedding, so every recalled case can say *why* it was recalled.
* **profile** - how this manager decides: how often they cut, hold or price higher,
  how often they take the suggestion and which way they depart from it, which
  factors move their decision most, and how the customers then answered.
* **consistency** - whether the suggestion in front of them matches what they did in
  cases like this. Treating like files alike is a fair-pricing property, so a
  departure is named; it is not called wrong, because the file in front of them may
  differ in a way no field records.

Counts are shown as "3 of 4", never as a rate, because a manager's own memory is
small and a percentage on four decisions claims a precision it does not have.

Pure: episodes in, readings out. Nothing here reaches a store.
"""

from __future__ import annotations

from typing import Any

from .decision_factors import LABELS

WEIGHTS = {"reason": 3.0, "discount_bps": 2.0, "acceptance_gain_pts": 1.5, "evidence_verified": 1.5,
           "risk_grade": 1.5, "competitor": 1.0, "gap_to_competitor_bps": 1.0, "margin_pct": 1.0,
           "segment": 1.0, "city": 0.5}
SCALES = {"discount_bps": 25.0, "acceptance_gain_pts": 5.0, "gap_to_competitor_bps": 50.0, "margin_pct": 0.5}
MATCHED = {"reason": "same reason", "discount_bps": "a cut of similar size",
           "acceptance_gain_pts": "a similar gain in acceptance", "evidence_verified": "same evidence status",
           "risk_grade": "same credit grade", "competitor": "same competitor situation",
           "gap_to_competitor_bps": "a similar gap to the competitor", "margin_pct": "a similar margin",
           "segment": "same segment", "city": "same branch"}
TOP, MIN_SIMILARITY = 5, 0.6
MIN_EACH_SIDE, MIN_SPREAD = 2, 0.4          # a factor "moves" a manager only on this much evidence
SUPPORTS, ARGUES = ("for", "clear"), ("against", "caution", "verify")

DID = {"approve": "approved the ask", "counter": "approved a smaller cut", "hold": "held the recommended rate",
       "uplift": "priced above the recommendation"}
BUCKET_DID = {"cut": "gave a cut", "hold": "held the rate", "uplift": "priced higher"}
SUGGESTS = {"cut": "give a cut", "hold": "hold the rate", "uplift": "price higher"}


def bucket(action: str | None) -> str | None:
    return "cut" if action in ("approve", "counter") else action


def _of(k: int, n: int) -> str:
    return f"{k} of {n}"


# ------------------------------------------------------------------- nearest

def similarity(case: dict[str, Any], other: dict[str, Any]) -> tuple[float, list[str]]:
    """A 0-1 match over the fields both cases carry, and which fields matched."""
    total = score = 0.0
    matched: list[str] = []
    for field, weight in WEIGHTS.items():
        a, b = case.get(field), other.get(field)
        if a is None and b is None:
            continue
        if field in SCALES:
            if a is None or b is None:
                continue                      # a number only one side has says nothing either way
            s = max(0.0, 1.0 - abs(float(a) - float(b)) / SCALES[field])
        else:
            s = 1.0 if a == b else 0.0
        total += weight
        score += weight * s
        if s >= 0.75:
            matched.append(MATCHED[field])
    return (round(score / total, 3) if total else 0.0), matched


def nearest(case: dict[str, Any], episodes: list[dict[str, Any]], top: int = TOP,
            min_similarity: float = MIN_SIMILARITY) -> list[dict[str, Any]]:
    ranked = []
    for ep in episodes:
        score, matched = similarity(case, ep.get("case_features") or {})
        if score >= min_similarity:
            ranked.append((score, ep, matched))
    ranked.sort(key=lambda t: str(t[1].get("decided_at") or ""), reverse=True)    # newest first ...
    ranked.sort(key=lambda t: -t[0])                                              # ... within a score
    return [card(ep, score, matched) for score, ep, matched in ranked[:top]]


def card(ep: dict[str, Any], score: float, matched: list[str]) -> dict[str, Any]:
    """One recalled decision, as the manager reads it."""
    return {"episode_id": ep["episode_id"], "request_id": ep["request_id"], "loan_id": ep["loan_id"],
            "manager": ep["manager"], "decided_at": ep["decided_at"], "similarity": score, "matched": matched,
            "requested_rate": ep["requested_rate"], "recommended_rate": ep["recommended_rate"],
            "request_reason": ep.get("request_reason"),
            "suggested": {"action": ep.get("suggested_action"), "rate": ep.get("suggested_rate")},
            "decided": {"action": ep["action"], "rate": ep.get("decided_rate"), "text": DID.get(ep["action"], ep["action"])},
            "followed_suggestion": ep.get("followed_suggestion"), "note": ep.get("note"),
            "reason": ep.get("reason"), "outcome": ep.get("outcome")}


# ------------------------------------------------------------------- profile

def _answered(eps: list[dict[str, Any]]) -> tuple[int, int]:
    states = [(ep.get("outcome") or {}).get("state") for ep in eps]
    return sum(s == "accepted" for s in states), sum(s in ("accepted", "not_accepted") for s in states)


def tendencies(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The factors whose reading most changes whether this manager gives the cut."""
    out = []
    for key, label in LABELS.items():
        sides: dict[str, list[bool]] = {"supports": [], "argues": []}
        for ep in episodes:
            if bucket(ep["action"]) == "uplift":
                continue
            signal = next((f["signal"] for f in ep.get("factors") or [] if f["key"] == key), None)
            side = "supports" if signal in SUPPORTS else "argues" if signal in ARGUES else None
            if side:
                sides[side].append(bucket(ep["action"]) == "cut")
        yes, no = sides["supports"], sides["argues"]
        if len(yes) < MIN_EACH_SIDE or len(no) < MIN_EACH_SIDE:
            continue
        spread = sum(yes) / len(yes) - sum(no) / len(no)
        if abs(spread) >= MIN_SPREAD:
            out.append({"factor": key, "label": label, "spread": round(spread, 2),
                        "cut_when_supports": [sum(yes), len(yes)], "cut_when_argues": [sum(no), len(no)],
                        "text": f"{label}: you gave the cut in {_of(sum(yes), len(yes))} cases where it favoured "
                                f"the ask, and {_of(sum(no), len(no))} where it did not."})
    return sorted(out, key=lambda t: -abs(t["spread"]))[:3]


def profile(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """How this manager decides, from their own episodes."""
    n = len(episodes)
    if not n:
        return {"decisions": 0, "lines": [], "tendencies": []}
    tally = {b: sum(bucket(ep["action"]) == b for ep in episodes) for b in ("cut", "hold", "uplift")}
    with_suggestion = [ep for ep in episodes if ep.get("followed_suggestion") is not None]
    took = sum(bool(ep["followed_suggestion"]) for ep in with_suggestion)
    departed = [ep for ep in with_suggestion if not ep["followed_suggestion"]]
    more = sum(1 for ep in departed if ep.get("decided_rate") is not None and ep.get("suggested_rate") is not None
               and ep["decided_rate"] < ep["suggested_rate"] - 0.005)
    less = sum(1 for ep in departed if ep.get("decided_rate") is not None and ep.get("suggested_rate") is not None
               and ep["decided_rate"] > ep["suggested_rate"] + 0.005)
    cuts = [ep for ep in episodes if bucket(ep["action"]) == "cut"]
    holds = [ep for ep in episodes if ep["action"] == "hold"]
    cut_won, cut_answered = _answered(cuts)
    hold_won, hold_answered = _answered(holds)
    leaning = tendencies(episodes)

    parts = [f"{BUCKET_DID[b]} in {tally[b]}" for b in ("cut", "hold", "uplift") if tally[b]]
    lines = [f"You have decided {n} request{'s' if n != 1 else ''}: you " + ", ".join(parts) + "."]
    if with_suggestion:
        line = f"You took the suggested decision in {_of(took, len(with_suggestion))}"
        if more or less:
            line += (f"; when you departed, you went lower than suggested {more} time{'s' if more != 1 else ''}"
                     f" and higher {less} time{'s' if less != 1 else ''}")
        lines.append(line + ".")
    lines += [t["text"] for t in leaning]
    if cut_answered:
        lines.append(f"Where you gave a cut, {_of(cut_won, cut_answered)} customers who answered accepted.")
    if hold_answered:
        lines.append(f"Where you held the rate, {_of(hold_won, hold_answered)} customers who answered accepted.")
    return {"decisions": n, "actions": tally, "suggestion_taken": [took, len(with_suggestion)],
            "departed_lower": more, "departed_higher": less, "tendencies": leaning,
            "cuts_accepted": [cut_won, cut_answered], "holds_accepted": [hold_won, hold_answered], "lines": lines}


# --------------------------------------------------------------- consistency

def consistency(mine: list[dict[str, Any]], suggestion: dict[str, Any] | None) -> dict[str, Any] | None:
    """The suggestion in front of the manager against what they did in cases like it."""
    if len(mine) < 2 or not suggestion or not suggestion.get("action"):
        return None
    buckets = [bucket(c["decided"]["action"]) for c in mine]
    usual = max(set(buckets), key=buckets.count)
    k, n = buckets.count(usual), len(buckets)
    if k * 2 <= n:
        return {"tone": "flat", "usual": None,
                "text": f"Your decisions on the {n} cases most like this were split; there is no habit to compare with."}
    now = bucket(suggestion["action"])
    those = [c for c in mine if bucket(c["decided"]["action"]) == usual]
    won, answered = _answered([{"outcome": c.get("outcome")} for c in those])
    followed_up = f" {_of(won, answered)} of those customers who answered accepted." if answered else ""
    if now == usual:
        return {"tone": "flat", "usual": usual,
                "text": f"The suggestion matches how you decided cases like this: you {BUCKET_DID[usual]} in "
                        f"{_of(k, n)}.{followed_up}"}
    return {"tone": "warn", "usual": usual,
            "text": f"In cases like this you usually {BUCKET_DID[usual]} ({_of(k, n)}); today's suggestion is to "
                    f"{SUGGESTS[now]}.{followed_up} "
                    f"If you depart from your usual decision, say in the note what is different about this file."}
