"""Champion / challenger.

A pricing model does not get promoted because its AUC improved. It gets promoted
because, replayed over real decisions, it would have made more risk-adjusted
value *without* costing conversion - and a human signed off. The gates come from
``model_config.champion_challenger``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import model_config
from ..governance import audit, model_registry
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ChallengerVerdict:
    promote: bool
    reasons: list[str]
    metrics: dict

    def as_dict(self) -> dict:
        return {"promote": self.promote, "reasons": self.reasons, "metrics": self.metrics}


def shadow_compare(champion_rates: np.ndarray, challenger_rates: np.ndarray,
                   conversion_model, prepayment_model, applications: pd.DataFrame,
                   expected_loss_pct: np.ndarray) -> pd.DataFrame:
    """Replay both models over the same files and price the difference."""
    from ..optimization import yield_calculator as yc

    rows = []
    for i in range(len(applications)):
        row = applications.iloc[[i]]
        amount = float(row["sanctioned_loan_amount"].iloc[0])
        tenure = float(row["tenure_years"].iloc[0])
        record = {"loan_id": row["loan_id"].iloc[0]}
        for name, rate in (("champion", champion_rates[i]), ("challenger", challenger_rates[i])):
            p_conv = float(conversion_model.predict_at_rate(row, float(rate))[0])
            life = float(prepayment_model.expected_life_years(row, float(rate))[0])
            result = yc.compute(rate=float(rate), conversion_probability=p_conv,
                                expected_loss_pct=float(expected_loss_pct[i]),
                                expected_life_years=life, loan_amount=amount, tenure_years=tenure)
            record[f"{name}_rate"] = float(rate)
            record[f"{name}_conversion"] = p_conv
            record[f"{name}_value_inr"] = result.expected_value_inr
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate(comparison: pd.DataFrame) -> ChallengerVerdict:
    cfg = model_config()["champion_challenger"]
    reasons: list[str] = []

    n = len(comparison)
    champion_value = comparison["champion_value_inr"].mean()
    challenger_value = comparison["challenger_value_inr"].mean()
    value_gain_pct = 100 * (challenger_value - champion_value) / abs(champion_value) if champion_value else 0.0
    conversion_change_pts = 100 * (comparison["challenger_conversion"].mean()
                                   - comparison["champion_conversion"].mean())

    # Paired test: the same files scored twice, so the pairing is real.
    differences = comparison["challenger_value_inr"] - comparison["champion_value_inr"]
    standard_error = differences.std(ddof=1) / np.sqrt(max(n, 1))
    z = float(differences.mean() / standard_error) if standard_error else 0.0

    metrics = {
        "n_decisions": n,
        "champion_mean_value_inr": round(float(champion_value), 0),
        "challenger_mean_value_inr": round(float(challenger_value), 0),
        "value_gain_pct": round(float(value_gain_pct), 2),
        "conversion_change_pts": round(float(conversion_change_pts), 2),
        "paired_z": round(z, 2),
        "mean_rate_change_bps": round(float((comparison["challenger_rate"] - comparison["champion_rate"]).mean() * 100), 1),
    }

    if n < cfg["shadow_min_decisions"]:
        reasons.append(f"only {n} shadow decisions; {cfg['shadow_min_decisions']} required")
    if value_gain_pct < cfg["promote_if_better_by_pct"]:
        reasons.append(f"value gain {value_gain_pct:.2f}% is below the {cfg['promote_if_better_by_pct']}% bar")
    if conversion_change_pts < -cfg["max_conversion_drop_pct"]:
        reasons.append(f"conversion drops {abs(conversion_change_pts):.2f} pts, beyond the "
                       f"{cfg['max_conversion_drop_pct']} pt tolerance")
    if abs(z) < 1.96:
        reasons.append(f"value difference is not statistically distinguishable (z={z:.2f})")
    if cfg["require_human_signoff"]:
        reasons.append("human sign-off required before promotion")

    promote = not [r for r in reasons if "human sign-off" not in r]
    if promote and cfg["require_human_signoff"]:
        promote = False
    return ChallengerVerdict(promote=promote, reasons=reasons, metrics=metrics)


def promote(name: str, version: str, approved_by: str, verdict: ChallengerVerdict) -> bool:
    """Promotion is an explicit, audited human act."""
    if not approved_by:
        raise ValueError("promotion requires an approver")
    blocking = [r for r in verdict.reasons if "human sign-off" not in r]
    if blocking:
        log.warning("refusing to promote %s %s: %s", name, version, "; ".join(blocking))
        return False
    model_registry.promote(name, version, actor=approved_by)
    audit.record(approved_by, "promote_challenger", "model", f"{name}:{version}", verdict.as_dict())
    return True
