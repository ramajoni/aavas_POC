"""Scenario simulation: the candidate prices worth evaluating.

The grid is not arbitrary. It spans the policy-permissible band, and always
includes the prices a human would raise in the room - the card rate, the Stage-2
anchor, a rate that matches the competitor, and the rate a salesperson is
predicted to offer anyway - so the recommendation can be defended against each of
them rather than only against its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import policy
from ..ingestion.schemas import RateBand
from . import constraints


@dataclass
class ScenarioSet:
    rates: np.ndarray
    band: RateBand
    anchors: dict[str, float]
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "n_scenarios": len(self.rates),
            "band": self.band.as_dict(),
            "anchors": {k: round(v, 2) for k, v in self.anchors.items()},
            "notes": self.notes,
        }


def generate(
    application: pd.DataFrame,
    stage2_band: RateBand | None = None,
    competitor_rate: float | None = None,
    anticipated_human_rate: float | None = None,
    step_bps: int | None = None,
    support_band: tuple[float, float] | None = None,
    quoted_band: RateBand | None = None,
) -> ScenarioSet:
    pol = policy()
    step = (step_bps or pol["optimizer"]["step_bps"]) / 100.0
    notes: list[str] = []

    envelope = constraints.permissible_band(application).iloc[0]
    low, high = float(envelope["min_rate"]), float(envelope["max_rate"])
    card = float(envelope["card_rate"])

    # The Stage-2 model describes what history priced. Restricting the search to
    # that interval means the engine can only ever reproduce historical behaviour,
    # so by default the whole permissible band is searched and the anchor is kept
    # as a comparison point instead of a constraint.
    if stage2_band is not None and pol["optimizer"].get("search_space") == "stage2_interval":
        sigmas = float(pol["optimizer"].get("stage2_interval_sigmas", 1.0))
        centre = (stage2_band.min_rate + stage2_band.max_rate) / 2
        half = (stage2_band.max_rate - stage2_band.min_rate) / 2 * sigmas
        narrowed_low = max(low, centre - half)
        narrowed_high = min(high, centre + half)
        if narrowed_low <= narrowed_high and (narrowed_high - narrowed_low) >= step:
            low, high = narrowed_low, narrowed_high
            notes.append(f"search restricted to the Stage-2 interval (+/-{sigmas:g} sigma)")
        else:
            notes.append("Stage-2 interval fell outside policy; searching the full permissible band")
    else:
        notes.append("searching the full policy-permissible band")

    # The range the customer was quoted at Stage 1 is a promise: the final price
    # moves inside it as the file is verified, never out of it.
    held = False
    if quoted_band is not None and pol["optimizer"].get("honour_indicative_band", True):
        held_low, held_high = max(low, quoted_band.min_rate), min(high, quoted_band.max_rate)
        if held_low <= held_high:
            low, high, held = held_low, held_high, True
            notes.append(f"held to the indicative range quoted at Stage 1 "
                         f"({quoted_band.min_rate:.2f}-{quoted_band.max_rate:.2f}%)")
        else:
            # Policy wins, but only as far as it must: the price goes to the
            # permissible rate nearest the promise, never anywhere in the band. A
            # verified file graded worse than its quote must not be walked up to
            # the policy ceiling because the demand curve happens to be flat.
            edge = low if quoted_band.max_rate < low else high
            notes.append(f"the indicative range quoted at Stage 1 ({quoted_band.min_rate:.2f}-"
                         f"{quoted_band.max_rate:.2f}%) lies outside the policy band ({low:.2f}-"
                         f"{high:.2f}%); priced at the nearest permissible rate, {edge:.2f}%, "
                         "and the customer must be told why")
            low = high = edge
            held = True

    # Never search where the conversion model has no data to stand on - except
    # that a held range is never trimmed away entirely: an extrapolated price
    # inside the quoted range beats a supported one outside it.
    if support_band is not None:
        support_low, support_high = support_band
        if support_high < low or support_low > high:
            notes.append(
                f"the conversion model has no data in this file's permissible band "
                f"({low:.2f}-{high:.2f}%); its price response is extrapolated and the "
                "recommendation should be treated as low confidence"
            )
        else:
            if support_high < high:
                notes.append(
                    f"upper end trimmed from {high:.2f}% to {support_high:.2f}%: beyond that the "
                    "conversion model is extrapolating, not predicting"
                )
                high = support_high
            if support_low > low:
                notes.append(f"lower end trimmed to {support_low:.2f}% for the same reason")
                low = support_low

    resize_required = bool(envelope.get("resize_required", envelope["infeasible"]))
    if resize_required:
        # Collapsing to one point would recommend a below-card rate that is still
        # unaffordable. The file is priced on its risk band instead, and every
        # scenario carries the same condition: the loan is resized first.
        notes.append("no permissible rate keeps this file inside the FOIR cap at its current size; "
                     "scenarios are priced on the risk-based band, each conditional on resizing the loan")

    rates = np.arange(low, high + 1e-9, step)

    anchors: dict[str, float] = ({"card_rate": card, "quoted_min": low, "quoted_max": high} if held
                                 else {"card_rate": card, "policy_floor": low, "policy_ceiling": high})
    if stage2_band is not None:
        anchors["stage2_anchor"] = round((stage2_band.min_rate + stage2_band.max_rate) / 2, 2)
    if competitor_rate is not None and np.isfinite(competitor_rate):
        anchors["competitor_match"] = float(competitor_rate)
    if anticipated_human_rate is not None and np.isfinite(anticipated_human_rate):
        anchors["anticipated_human_rate"] = float(anticipated_human_rate)

    # Fold the anchors into the grid so the comparison table contains them exactly.
    extra = [v for v in anchors.values() if low - 1e-9 <= v <= high + 1e-9]
    rates = np.unique(np.round(np.concatenate([rates, np.array(extra, dtype=float)]), 2))

    if bool(envelope["affordability_binding"]) and not resize_required:
        notes.append(f"upper end capped by affordability at {envelope['max_affordable_rate']:.2f}%")
    if competitor_rate is not None and np.isfinite(competitor_rate) and competitor_rate < low:
        # Worth saying out loud: no permissible price matches this competitor, so
        # the conversation has to move off price rather than chase it.
        notes.append(
            f"competitor quote {competitor_rate:.2f}% is {100 * (low - competitor_rate):.0f} bps below our "
            "permissible floor - it cannot be matched within policy"
        )

    return ScenarioSet(rates, RateBand(low, high), anchors, notes)
