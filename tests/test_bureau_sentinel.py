"""bureau_score = -1 is "no score", not a score 300 points below the worst one.

The transformer reads the sentinel as missing (so it neither drives a split nor
drags the imputation median down), the new_to_credit flag carries what it meant,
and everything a person reads shows "No bureau score (new to credit)".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aavas.config import feature_config
from aavas.features.feature_pipeline import (FeatureSpec, FeatureTransformer, display_value,
                                             is_sentinel, sentinel_rules, spec)
from aavas.xai import narrative, reason_codes
from aavas.xai.loan_explanation import _applicant
from aavas.xai.shap_explainer import Attribution

NO_SCORE = "No bureau score (new to credit)"
SPEC = FeatureSpec("t", numeric=["bureau_score", "ltv"], categorical=["new_to_credit"])


def _frame():
    return pd.DataFrame({"bureau_score": [-1, -1, -1, 700, 720, 740],
                         "ltv": [0.7, 0.6, 0.8, 0.7, 0.5, 0.6],
                         "new_to_credit": [True, True, True, False, False, False]})


def test_the_sentinel_is_declared_in_the_feature_config():
    rule = sentinel_rules()["bureau_score"]
    assert rule["flag"] == "new_to_credit" and rule["display"] == NO_SCORE
    assert feature_config()["feature_config_version"] != "FEAT-2026.09"


def test_every_contract_holding_the_score_also_holds_the_flag():
    cfg = feature_config()
    for name, block in cfg.items():
        if isinstance(block, dict) and "bureau_score" in (block.get("numeric") or []):
            assert "new_to_credit" in spec(name).columns, f"{name} has the score but not the flag"


def test_the_median_is_computed_over_measured_scores_only():
    transformer = FeatureTransformer(SPEC).fit(_frame())
    assert transformer.medians_["bureau_score"] == 720.0       # not 349.5 with the -1s


def test_a_new_to_credit_file_is_imputed_not_scored_at_minus_one():
    transformer = FeatureTransformer(SPEC).fit(_frame())
    X = transformer.transform(pd.DataFrame({"bureau_score": [-1, 650], "ltv": [0.7, 0.7],
                                            "new_to_credit": [True, False]}))
    assert X["bureau_score"].tolist() == [720.0, 650.0]
    assert X["new_to_credit"].nunique() == 2                   # the flag carries the difference


def test_the_flag_is_derived_when_a_caller_sends_only_the_score():
    transformer = FeatureTransformer(SPEC).fit(_frame())
    X = transformer.transform(pd.DataFrame({"bureau_score": [-1, 700], "ltv": [0.7, 0.7]}))
    codes = {c: i for i, c in enumerate(transformer.categories_["new_to_credit"])}
    assert X["new_to_credit"].tolist() == [codes["True"], codes["False"]]


def test_an_artefact_fitted_before_sentinels_keeps_its_encoding():
    """A pickled transformer from FEAT-2026.09 has no sentinels_; it must keep
    scoring -1 as -1, the way its model was fitted."""
    transformer = FeatureTransformer(SPEC).fit(_frame())
    del transformer.sentinels_
    X = transformer.transform(pd.DataFrame({"bureau_score": [-1], "ltv": [0.7],
                                            "new_to_credit": [True]}))
    assert X["bureau_score"].tolist() == [-1.0]


def test_display_relabels_only_the_sentinel():
    assert is_sentinel("bureau_score", -1) and not is_sentinel("bureau_score", 742)
    assert display_value("bureau_score", -1) == NO_SCORE
    assert display_value("bureau_score", -1.0) == NO_SCORE
    assert display_value("bureau_score", 742) == 742
    assert display_value("ltv", -1) == -1
    assert display_value("bureau_score", None) is None


def test_reason_codes_say_no_score_rather_than_a_weak_profile():
    codes = reason_codes.build([Attribution("bureau_score", -1.0, 0.25)], top_k=5, min_bps=0)
    assert codes[0].value == NO_SCORE
    assert codes[0].text.startswith(NO_SCORE) and "Weaker bureau profile" not in codes[0].text
    assert codes[0].as_dict()["value"] == NO_SCORE


def test_a_scored_file_keeps_its_direction_language():
    codes = reason_codes.build([Attribution("bureau_score", 610.0, 0.25)], top_k=5, min_bps=0)
    assert codes[0].text.startswith("Weaker bureau profile") and codes[0].value == 610.0


def test_the_loan_explanation_shows_no_score_for_the_applicant():
    applicant = _applicant({"bureau_score": -1, "segment": "salaried_formal"})
    assert applicant["bureau_score"] == NO_SCORE


def test_the_narrative_is_given_no_score_not_minus_one():
    detail = {"applicant": {"bureau_score": -1},
              "models": {"risk": {"features": [
                  {"feature": "bureau_score", "label": "Bureau score", "value": -1,
                   "group": "customer_risk", "direction": "increase", "share_pct": 30.0}]}}}
    material = narrative.facts(detail)
    assert material["applicant"]["bureau_score"] == NO_SCORE
    assert material["risk"]["top_drivers"][0]["value"] == NO_SCORE
    assert "-1" not in str(material["applicant"])


def test_the_explainer_displays_the_sentinel(training_data):
    """End to end through PricingExplainer on a real new-to-credit file."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from aavas.features import stage2_features
    from aavas.xai.shap_explainer import PricingExplainer

    sample = training_data.head(3000)
    contract = spec("stage2")
    transformer = FeatureTransformer(contract).fit(stage2_features.build(sample))
    X = transformer.transform(stage2_features.build(sample))
    model = HistGradientBoostingRegressor(max_iter=30).fit(X, sample["sanctioned_rate"])
    ntc = training_data[training_data["bureau_score"] < 0].head(1)
    attributions = PricingExplainer(model, transformer).explain(ntc, build=stage2_features.build)
    shown = {a.feature: a.value for a in attributions}
    assert shown["bureau_score"] == NO_SCORE
    assert np.isclose(transformer.medians_["bureau_score"],
                      sample.loc[sample["bureau_score"] >= 0, "bureau_score"].median())
