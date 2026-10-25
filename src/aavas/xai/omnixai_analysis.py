"""Global and instance-level model analysis with OmniXAI.

Why a second explainability layer at all? ``xai/shap_explainer.py`` answers the
question a *decision* raises - "why this rate, for this file, in basis points" -
in the units the pricing screen shows. This module answers the question model
*governance* raises:

* global   - what does the model do across the whole book, and is that the
             behaviour policy intended? (mean |SHAP| over a sample, permutation
             importance against the holdout, and PDP/ALE curves showing the
             shape - not just the size - of each feature's effect.)
* instance - does one file's explanation survive being recomputed by a second,
             model-agnostic method? KernelSHAP and LIME are produced side by
             side, plus MACE counterfactuals: the smallest change to the file
             that would have flipped the outcome.

Three deliberate choices:

* Morris sensitivity is not used. OmniXAI's implementation asserts there are no
  categorical features, and every feature contract here has them.
* Everything is computed through the model's own ``FeatureTransformer``, so an
  explanation can never describe a feature the model was not fitted on, and
  categoricals reach the dashboard as labels ("salaried") rather than codes.
* The analysed frame is the *holdout*, split exactly the way training split it.
  Explaining a model on rows it was fitted on measures memorisation.

Entry point: :func:`analyse`. The dashboard is a separate concern, in
``scripts/run_omnixai.py``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import model_config, seed
from ..features import sales_features, stage1_features, stage2_features
from ..features.feature_pipeline import FeatureTransformer, reason_label
from ..ingestion import loader
from ..logging_utils import get_logger
from ..models import base as model_base
from ..paths import ROOT

log = get_logger(__name__)

MODELS = ("stage1", "stage2", "conversion", "prepayment", "risk", "override")

# Global explainers are cheap per feature and expensive per row; local ones are
# the reverse. Both lists are overridable per call, which is how --fast works.
GLOBAL_EXPLAINERS = ("shap_global", "pdp", "ale", "permutation")
LOCAL_EXPLAINERS = ("shap", "lime")
COUNTERFACTUAL_EXPLAINER = "mace"       # classification only

# The dashboard's What-if tab deep-copies whatever explainer it is handed, on
# every render. A KernelExplainer that has already computed shap values holds a
# Cython memoryview from that run and cannot be deep-copied, which takes the tab
# down with a 500. So What-if gets its own explainer, with the model-agnostic
# explainer that survives the copy - and it is a better fit anyway, being the
# cheap one to re-run interactively on an edited row.
WHATIF_EXPLAINERS = ("lime",)

# PDP and ALE are computed per feature. For a categorical, omnixai's ALE first
# orders the categories by a pairwise distance - quadratic in their number - and
# then scores every one of them. On `city` (16 values) that alone took longer
# than the entire rest of the analysis, to produce a 16-bar chart nobody can
# read. Such features still appear in the SHAP and permutation rankings; only
# their curve is skipped, and the report says which and why.
MAX_CURVE_CATEGORIES = 12

# The credit score, under the schema's name for it: `ingestion/schemas.py` maps an
# incoming `credit_score` onto this column and `api/contracts.py` fixes its range
# at 300-900, with -1 for a file that has no bureau history.
#
# That sentinel is why the score gets its own section rather than being read off
# the PDP summary. The smallest *value* in the book is -1, so "the model at the
# bottom of the credit score" read straight off the curve is the new-to-credit
# cohort - a different population, not a worse score. The band below is computed
# over scored files only, and the thin-file cohort is reported separately.
CREDIT_SCORE_FEATURE = "bureau_score"
CREDIT_SCORE_FLOOR = 300
CREDIT_SCORE_CEILING = 900
NEW_TO_CREDIT_SENTINEL = -1
CREDIT_SCORE_TAIL_PCT = 10          # the "at the floor" / "at the ceiling" cohorts

REPORT_DIR = ROOT / "reports" / "omnixai"


# ------------------------------------------------------------- library plumbing


def _quiet_libraries() -> None:
    """KernelSHAP and LIME log every internal step at INFO. That is useful when
    debugging an explainer and useless when explaining six models."""
    import logging

    for noisy in ("shap", "lime", "matplotlib", "numba", "urllib3", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _patch_shap_legacy_output() -> None:
    """Restore the multi-output SHAP return shape omnixai 1.3.1 was written for.

    shap <= 0.44 returned one array per class from ``KernelExplainer.shap_values``
    and omnixai indexes it that way (``shap_values[label][i]``). shap >= 0.45
    returns a single ``(n_samples, n_features, n_classes)`` array, so that index
    silently selects a *sample* instead of a class - every classification
    attribution in this repo would be wrong rather than broken. Older shap is not
    an option: only >= 0.46 supports numpy 2. So the old shape is handed back at
    the boundary, once, idempotently.
    """
    import shap

    if getattr(shap.KernelExplainer.shap_values, "_aavas_legacy_shape", False):
        return
    original = shap.KernelExplainer.shap_values

    def shap_values(self, X, **kwargs):
        values = original(self, X, **kwargs)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            return [values[..., i] for i in range(values.shape[-1])]
        return values

    shap_values._aavas_legacy_shape = True
    shap.KernelExplainer.shap_values = shap_values
    log.debug("applied the shap>=0.45 multi-output compatibility shim")


# --------------------------------------------------------------------- bundles


@dataclass
class ModelBundle:
    """A trained artefact reassembled into the four things an explainer needs:
    an estimator, the feature contract it was fitted on, the frame builder that
    feeds that contract, and the target it was fitted against."""

    name: str
    mode: str                                   # classification | regression
    estimator: Any
    transformer: FeatureTransformer
    build: Callable[[pd.DataFrame], pd.DataFrame]
    target: str
    output_label: str
    units: str
    card: model_base.ModelCard
    class_names: list[str] | None = None
    rows: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
    notes: str = ""

    @property
    def spec(self):
        return self.transformer.spec

    @property
    def is_classification(self) -> bool:
        return self.mode == "classification"

    @property
    def attribution_units(self) -> str:
        """KernelSHAP and LIME explain a classifier through the logit link, so
        their contributions are log-odds even though the prediction is a
        probability. PDP and ALE are computed on the model output directly and
        stay in probability. Conflating the two is the classic way to report an
        explanation that does not reconcile with the screen."""
        return "log-odds" if self.is_classification else self.units


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _override_build(df: pd.DataFrame) -> pd.DataFrame:
    return sales_features.build(df, reference_rate_col="ai_recommended_rate")


def load_bundle(name: str) -> ModelBundle:
    """Load a trained model exactly as its trainer left it.

    Each trainer feeds its transformer a differently prepared frame (Stage-1 its
    pre-assessment view, Stage-2 and risk the post-assessment one, the sales
    models the raw table plus derived negotiation columns). Getting that pairing
    wrong would silently explain a model on features it never saw, so the pairing
    is declared here once rather than inferred.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; expected one of {', '.join(MODELS)}")
    payload, card = model_base.load(name)

    if name == "stage1":
        quantiles = payload["quantiles"]
        median = str(sorted(quantiles)[len(quantiles) // 2])
        return ModelBundle(
            name=name, mode="regression", estimator=payload["models"][median],
            transformer=payload["transformer"], build=stage1_features.build,
            target="sanctioned_rate", output_label="Indicative rate (median quantile)",
            units="% p.a.", card=card,
            notes=f"Stage-1 is a quantile trio {quantiles}; the q={median} model is analysed "
                  f"because it is the one that anchors the band shown to the customer.",
        )
    if name == "stage2":
        return ModelBundle(
            name=name, mode="regression", estimator=payload["model"],
            transformer=payload["transformer"], build=stage2_features.build,
            target="sanctioned_rate", output_label="Anchor rate", units="% p.a.", card=card,
        )
    if name == "risk":
        return ModelBundle(
            name=name, mode="classification", estimator=payload["model"],
            transformer=payload["transformer"], build=stage2_features.build,
            target="defaulted", output_label="Lifetime PD", units="probability", card=card,
            class_names=["no default", "default"],
        )
    if name == "conversion":
        return ModelBundle(
            name=name, mode="classification", estimator=payload["model"],
            transformer=payload["transformer"], build=_identity,
            target="disbursed", output_label="P(disbursed)", units="probability", card=card,
            class_names=["lost", "disbursed"],
        )
    if name == "prepayment":
        return ModelBundle(
            name=name, mode="classification", estimator=payload["model"],
            transformer=payload["transformer"], build=_identity,
            target="bt_out_24m", output_label="P(balance transfer out in 24m)",
            units="probability", card=card, class_names=["retained", "transferred out"],
            # Prepayment is only defined on money that went out of the door.
            rows=lambda df: df[df["disbursed"].astype(bool)].copy(),
        )
    # override: the classifier is the decision-shaped half; the regressor only
    # sizes an override that the classifier has already said is likely.
    return ModelBundle(
        name=name, mode="classification", estimator=payload["classifier"],
        transformer=payload["transformer"], build=_override_build,
        target="was_overridden", output_label="P(human overrides the system price)",
        units="probability", card=card, class_names=["held", "overridden"],
        notes="The magnitude regressor is not analysed here: it is fitted on overridden "
              "files only, so a global attribution over the whole book would be misleading.",
    )


# ------------------------------------------------------------------- framing


def human_frame(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """The model's feature contract in human units.

    OmniXAI wants raw features plus a preprocess function, not an encoded
    matrix - that is what lets the dashboard show ``segment = salaried`` and
    lets MACE propose a counterfactual a human can read. Numerics are imputed
    with the *transformer's own* medians so the explained rows and the scored
    rows are the same rows.
    """
    frame = bundle.build(df)
    missing = [c for c in bundle.spec.columns if c not in frame.columns]
    if missing:
        raise KeyError(f"[{bundle.name}] feature columns absent after build: {missing}")

    out = pd.DataFrame(index=frame.index)
    for col in bundle.spec.numeric:
        series = pd.to_numeric(frame[col], errors="coerce")
        out[col] = series.fillna(bundle.transformer.medians_.get(col, 0.0)).astype(float)
    for col in bundle.spec.categorical:
        out[col] = frame[col].astype(str)
    return out[bundle.spec.columns]


def predict(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, or the regression value, for a human frame."""
    X = bundle.transformer.transform(frame)
    if bundle.is_classification:
        return bundle.estimator.predict_proba(X)[:, 1]
    return bundle.estimator.predict(X)


def _tabular(frame: pd.DataFrame, bundle: ModelBundle, target: pd.Series | None = None):
    from omnixai.data.tabular import Tabular

    data = frame if target is None else frame.assign(**{bundle.target: np.asarray(target)})
    return Tabular(
        data=data.reset_index(drop=True),
        categorical_columns=list(bundle.spec.categorical),
        target_column=None if target is None else bundle.target,
    )


def _preprocessor(bundle: ModelBundle):
    def preprocess(tabular):
        return bundle.transformer.transform(tabular.to_pd())

    return preprocess


def _counterfactuals(bundle: ModelBundle, reference: pd.DataFrame, instances,
                     max_examples: int = 3) -> tuple[Any, str]:
    """MACE counterfactuals, computed outside the main explainer on purpose.

    MACE is the one explainer here that can fail on a perfectly good model: it
    searches for a nearby file the model puts on the *other* side of the
    decision boundary, and two of these classifiers (risk, prepayment) are
    imbalanced enough that the boundary is never crossed inside a reference
    sample. That is a fact worth reporting, not an error worth aborting a whole
    analysis for, so the failure modes are caught and named.
    """
    from omnixai.explainers.tabular import MACEExplainer

    def predict_function(tabular):
        return bundle.estimator.predict_proba(bundle.transformer.transform(tabular.to_pd()))

    tab = _tabular(reference, bundle)
    predicted = np.argmax(predict_function(tab), axis=1)
    classes = sorted(set(int(p) for p in predicted))
    if len(classes) < 2:
        only = (bundle.class_names or [str(classes[0])] * 2)[classes[0]]
        return None, (f"not available: the model predicts '{only}' for every row of the "
                      f"reference population, so no counterfactual of the same shape exists "
                      f"on the other side of the 0.5 boundary")

    last = "unknown failure"
    for use_knn, how in ((True, "kNN retrieval"), (False, "simple retrieval")):
        try:
            explainer = MACEExplainer(training_data=tab, predict_function=predict_function,
                                      mode="classification", use_knn=use_knn)
            return explainer.explain(X=instances, max_number_examples=max_examples), f"MACE, {how}"
        except Exception as exc:                                  # noqa: BLE001
            log.warning("[%s] MACE with %s failed (%s: %s)", bundle.name, how,
                        type(exc).__name__, exc)
            last = f"{type(exc).__name__}: {exc}"
    return None, f"not available: {last}"


def _curve_features(bundle: ModelBundle, ranked: list[str], limit: int | None
                    ) -> tuple[list[str], list[tuple[str, int]]]:
    """The features worth drawing a PDP/ALE curve for, most important first."""
    chosen: list[str] = []
    skipped: list[tuple[str, int]] = []
    for feature in ranked:
        if feature in bundle.spec.categorical:
            n = len(bundle.transformer.categories_.get(feature, []))
            if n > MAX_CURVE_CATEGORIES:
                skipped.append((feature, n))
                continue
        chosen.append(feature)
        if limit and len(chosen) >= limit:
            break
    return chosen, skipped


def _scored_credit(frame: pd.DataFrame | None) -> pd.Series | None:
    """The credit score of every file that has one, as a positional Series.

    Files with the new-to-credit sentinel are dropped rather than clipped: a
    thin file is not a low score, and ranking the two together is what produces
    a "worst credit score" explanation of somebody with no bureau history.
    """
    if frame is None or CREDIT_SCORE_FEATURE not in frame.columns:
        return None
    values = pd.to_numeric(frame[CREDIT_SCORE_FEATURE], errors="coerce")
    values = values.reset_index(drop=True)
    scored = values[values >= CREDIT_SCORE_FLOOR]
    return scored if len(scored) else None


def _credit_score_extremes(frame: pd.DataFrame | None) -> list[tuple[str, int]]:
    """Positional indices of the lowest- and highest-scoring credit files."""
    scored = _scored_credit(frame)
    if scored is None:
        return []
    return [("lowest credit score", int(scored.idxmin())),
            ("highest credit score", int(scored.idxmax()))]


def _select_instances(scores: np.ndarray, n: int, rng: np.random.Generator,
                      frame: pd.DataFrame | None = None) -> list[tuple[str, int]]:
    """Which files to explain in detail.

    Three kinds of row, in priority order:

    * the tails of the model's own output - where a pricing model either earns
      its keep or embarrasses itself;
    * the ends of the credit score, which is the axis a credit reviewer reads
      first and is *not* the same axis: the file with the worst bureau score is
      rarely the file with the highest predicted PD, and where the two diverge
      is exactly what the section is for;
    * a random remainder, so the report is not built only from cherry-picked
      rows.
    """
    order = np.argsort(scores)
    picked: list[tuple[str, int]] = []

    def add(label: str, idx: int) -> None:
        idx = int(idx)
        for pos, (existing, taken) in enumerate(picked):
            if taken == idx:
                # The same row can be both tails at once - the worst credit score
                # in the book often is the highest predicted PD. Keep both names
                # rather than dropping one: which of them a row satisfies is the
                # finding, and explaining it twice would waste an instance.
                picked[pos] = (f"{existing} + {label}", idx)
                return
        if len(picked) < n:
            picked.append((label, idx))

    for label, idx in (("highest predicted", order[-1]),
                       ("lowest predicted", order[0]),
                       ("median predicted", order[len(order) // 2])):
        add(label, idx)
    for label, idx in _credit_score_extremes(frame):
        add(label, idx)

    pool = [i for i in range(len(scores)) if i not in {i for _, i in picked}]
    for idx in rng.choice(pool, size=max(0, min(n - len(picked), len(pool))), replace=False):
        picked.append(("randomly sampled", int(idx)))
    return picked


# -------------------------------------------------------------------- result


@dataclass
class Analysis:
    """Everything the report writer and the dashboard need, computed once."""

    bundle: ModelBundle
    explainer: Any
    whatif_explainer: Any                           # see WHATIF_EXPLAINERS
    instances: Any                                  # Tabular
    instance_labels: list[str]
    local_explanations: dict = field(default_factory=dict)
    global_explanations: dict = field(default_factory=dict)
    data_explanations: dict = field(default_factory=dict)
    prediction_explanations: Any = None
    report: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    counterfactual_status: str = "not requested"
    curve_skipped: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.bundle.name


# ------------------------------------------------------------------ analysis


def analyse(
    name: str,
    *,
    train_sample: int = 500,
    global_sample: int = 100,
    n_instances: int = 7,
    permutation_repeats: int = 5,
    permutation_sample: int = 1000,
    counterfactuals: bool = True,
    pdp_features: int | None = 12,
    df: pd.DataFrame | None = None,
) -> Analysis:
    """Run the full OmniXAI sweep for one trained model.

    ``train_sample`` sizes the reference population every explainer is defined
    against; ``global_sample`` sizes the rows KernelSHAP actually attributes.
    Both are small on purpose: KernelSHAP costs ``rows x nsamples x background``
    model evaluations, and a global importance ranking is stable long before it
    is exhaustive.
    """
    import time

    from omnixai.explainers.data import DataAnalyzer
    from omnixai.explainers.prediction import PredictionAnalyzer
    from omnixai.explainers.tabular import TabularExplainer

    _quiet_libraries()
    _patch_shap_legacy_output()

    t0 = time.time()
    timings: dict[str, float] = {}
    bundle = load_bundle(name)
    rng = np.random.default_rng(seed())

    table = loader.load_training_table() if df is None else df
    table = bundle.rows(table)
    train_df, test_df = loader.train_test_split(table)

    train_h = human_frame(bundle, train_df)
    test_h = human_frame(bundle, test_df)
    y_test = test_df[bundle.target].astype(int if bundle.is_classification else float).to_numpy()

    # Reference population for the explainers, and the holdout rows we explain.
    ref_idx = rng.choice(len(train_h), size=min(train_sample, len(train_h)), replace=False)
    ref_h = train_h.iloc[ref_idx]
    glob_idx = rng.choice(len(test_h), size=min(global_sample, len(test_h)), replace=False)
    perm_idx = rng.choice(len(test_h), size=min(permutation_sample, len(test_h)), replace=False)

    ref_tab = _tabular(ref_h, bundle)
    global_tab = _tabular(test_h.iloc[glob_idx], bundle)
    perm_tab = _tabular(test_h.iloc[perm_idx], bundle)
    preprocess = _preprocessor(bundle)

    wants_cf = counterfactuals and bundle.is_classification
    explainer_names = list(LOCAL_EXPLAINERS) + list(GLOBAL_EXPLAINERS)

    log.info("[%s] explainers=%s ref=%s holdout=%s", name, explainer_names, len(ref_h), len(test_h))
    explainer = TabularExplainer(
        explainers=explainer_names,
        mode=bundle.mode,
        data=ref_tab,
        model=bundle.estimator,
        preprocess=preprocess,
        params={
            "shap": {"nsamples": min(100, len(ref_h))},
            "shap_global": {"nsamples": min(100, len(ref_h))},
            "lime": {"discretize_continuous": True},
        },
    )
    whatif_explainer = TabularExplainer(
        explainers=list(WHATIF_EXPLAINERS), mode=bundle.mode, data=ref_tab,
        model=bundle.estimator, preprocess=preprocess,
    )
    timings["init"] = time.time() - t0

    # ---- instances: the tails, the credit-score ends, plus a random draw
    scores = predict(bundle, test_h)
    picks = _select_instances(scores, n_instances, rng, test_h)
    instance_rows = test_h.iloc[[i for _, i in picks]]
    instances = _tabular(instance_rows, bundle)
    instance_ids = [str(test_df.index[i]) for _, i in picks]
    loan_ids = ([str(test_df["loan_id"].iloc[i]) for _, i in picks]
                if "loan_id" in test_df.columns else instance_ids)

    t = time.time()
    local_explanations = explainer.explain(X=instances, params={"shap": {"silent": True}})
    timings["local"] = time.time() - t

    cf_status = ("not requested" if not counterfactuals else
                 "not applicable: MACE counterfactuals are defined for classifiers only")
    if wants_cf:
        t = time.time()
        cf_explanation, cf_status = _counterfactuals(bundle, ref_h, instances)
        if cf_explanation is not None:
            local_explanations[COUNTERFACTUAL_EXPLAINER] = cf_explanation
        timings["counterfactual"] = time.time() - t
        log.info("[%s] counterfactuals: %s", name, cf_status)

    # ---- global
    #
    # Deliberately not explain_global() in one shot: PDP and ALE cost a full
    # scoring pass per feature per grid point, so they are pointed at the
    # features the SHAP ranking has just said matter, rather than at the first N
    # columns of the contract.
    t = time.time()
    explainers = explainer.explainers
    global_explanations: "OrderedDict[str, Any]" = OrderedDict()
    global_explanations["shap_global"] = explainers["shap_global"].explain(
        X=global_tab, nsamples=min(100, len(ref_h)), silent=True)

    ranking = _importance_series(global_explanations["shap_global"])
    ranked = (list(ranking.sort_values(ascending=False).index) if ranking is not None
              else list(bundle.spec.columns))
    curve_features, curve_skipped = _curve_features(bundle, ranked, pdp_features)
    if curve_skipped:
        log.info("[%s] no PDP/ALE curve for %s (too many categories)", name,
                 ", ".join(f"{f} ({n})" for f, n in curve_skipped))

    global_explanations["pdp"] = explainers["pdp"].explain(
        features=curve_features, monte_carlo=False)
    global_explanations["ale"] = explainers["ale"].explain(
        features=curve_features, monte_carlo=False)
    global_explanations["permutation"] = explainers["permutation"].explain(
        X=perm_tab, y=y_test[perm_idx], n_repeats=permutation_repeats)
    timings["global"] = time.time() - t

    # ---- dataset and prediction analysis (the dashboard's other two tabs)
    t = time.time()
    # imbalance and chi-square are classification-only in omnixai; correlation and
    # mutual information are defined for both.
    data_analyzers = ["correlation", "mutual"]
    data_params: dict[str, dict] = {}
    if bundle.is_classification:
        data_analyzers += ["imbalance", "chi2"]
        data_params["imbalance"] = {"features": list(bundle.spec.categorical)[:4]}
    data_explanations = DataAnalyzer(
        explainers=data_analyzers, mode=bundle.mode,
        data=_tabular(test_h, bundle, target=y_test),
    ).explain(params=data_params)

    prediction_explanations = PredictionAnalyzer(
        mode=bundle.mode, test_data=_tabular(test_h, bundle), test_targets=y_test,
        model=bundle.estimator, preprocess=preprocess,
    ).explain()
    timings["dataset+prediction"] = time.time() - t
    timings["total"] = time.time() - t0

    analysis = Analysis(
        bundle=bundle, explainer=explainer, whatif_explainer=whatif_explainer, instances=instances,
        instance_labels=[f"{label} ({lid})" for (label, _), lid in zip(picks, loan_ids)],
        local_explanations=local_explanations, global_explanations=global_explanations,
        data_explanations=data_explanations, prediction_explanations=prediction_explanations,
        timings=timings, counterfactual_status=cf_status, curve_skipped=curve_skipped,
    )
    credit = (test_h[CREDIT_SCORE_FEATURE].to_numpy()
              if CREDIT_SCORE_FEATURE in test_h.columns else None)
    analysis.report = build_report(
        analysis,
        instance_meta=[{"label": label, "row": int(i), "loan_id": lid,
                        "prediction": float(scores[i]), "actual": _native(y_test[i]),
                        "credit_score": (None if credit is None else float(credit[i]))}
                       for (label, i), lid in zip(picks, loan_ids)],
        credit_frame=test_h, predictions=scores,
        n_train=len(train_h), n_test=len(test_h), n_reference=len(ref_h),
        n_global_sample=len(glob_idx), n_permutation_sample=len(perm_idx),
    )
    return analysis


# --------------------------------------------------------------- extraction


def _native(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _importance_series(explanation) -> pd.Series | None:
    """``{"features": [...], "scores": [...]}`` -> an indexed, sorted Series."""
    if explanation is None:
        return None
    data = explanation.get_explanations()
    if not data or "features" not in data:
        return None
    return pd.Series(np.abs(np.asarray(data["scores"], dtype=float)),
                     index=list(data["features"]), dtype=float)


def _curve_effects(explanation, categorical: set[str] | None = None) -> dict[str, dict]:
    """PDP/ALE curves reduced to the two numbers a reviewer reads first: how far
    the prediction moves across the feature's range, and in which direction.

    Direction is only reported for numerics. For an unordered categorical the
    first and last categories are an arbitrary pair, so the peak and trough
    categories are what the report carries instead.
    """
    if explanation is None:
        return {}
    categorical = categorical or set()
    out: dict[str, dict] = {}
    for feature, payload in explanation.get_explanations().items():
        values = list(payload["values"])
        scores = np.asarray(payload["scores"], dtype=float)
        if scores.ndim == 2:                       # classification: (n_values, n_classes)
            scores = scores[:, -1]
        if scores.size == 0:
            continue
        swing = float(scores.max() - scores.min())
        direction = float(scores[-1] - scores[0])
        if feature in categorical:
            reads_as = "varies by category"
        elif direction > 0:
            reads_as = "increases"
        elif direction < 0:
            reads_as = "decreases"
        else:
            reads_as = "flat"
        out[feature] = {
            "swing": swing,
            "direction": reads_as,
            "net_change": direction,
            # The curve itself, not only its ends: a feature with a sentinel value
            # (see CREDIT_SCORE_FEATURE) has to be re-read over the range that
            # means something, which cannot be done from at_low/at_high alone.
            "points": [{"value": _native(v), "score": float(sc)}
                       for v, sc in zip(values, scores)],
            "at_low": {"value": _native(values[0]), "score": float(scores[0])},
            "at_high": {"value": _native(values[-1]), "score": float(scores[-1])},
            "peak": {"value": _native(values[int(scores.argmax())]), "score": float(scores.max())},
            "trough": {"value": _native(values[int(scores.argmin())]), "score": float(scores.min())},
            "monotone": bool(np.all(np.diff(scores) >= -1e-12) or np.all(np.diff(scores) <= 1e-12)),
        }
    return out


def _credit_score_band(bundle: ModelBundle, frame: pd.DataFrame | None,
                       predictions: np.ndarray | None, pdp: dict,
                       instances: list[dict]) -> dict | None:
    """What the model does at the bottom and the top of the credit score.

    Two readings, because they answer different questions and can disagree:

    * **observed** - the files that actually sit in the bottom and top deciles
      of the score, and what the model gave them. Confounded by everything that
      travels with a credit score (income, LTV, vintage), which is the point:
      it is what the book looks like.
    * **holding everything else fixed** - the PDP re-read over the scored range
      only. The raw curve starts at the -1 sentinel, so its low end is the
      new-to-credit cohort rather than a bad score; that point is dropped here
      and reported separately.

    The two explained files - lowest and highest scored - are named so a
    reviewer can jump from the band to the attribution behind either end.
    """
    if frame is None or predictions is None or CREDIT_SCORE_FEATURE not in frame.columns:
        return None
    values = pd.to_numeric(frame[CREDIT_SCORE_FEATURE], errors="coerce").reset_index(drop=True)
    preds = pd.Series(np.asarray(predictions, dtype=float))
    if len(values) != len(preds):
        return None

    scored_mask = values >= CREDIT_SCORE_FLOOR
    scored = values[scored_mask]
    if scored.empty:
        return None

    lo_cut = float(np.percentile(scored, CREDIT_SCORE_TAIL_PCT))
    hi_cut = float(np.percentile(scored, 100 - CREDIT_SCORE_TAIL_PCT))

    def cohort(mask: pd.Series) -> dict:
        p, v = preds[mask], values[mask]
        return {"files": int(mask.sum()),
                "score_min": float(v.min()) if len(v) else None,
                "score_max": float(v.max()) if len(v) else None,
                "mean_prediction": float(p.mean()) if len(p) else None,
                "median_prediction": float(p.median()) if len(p) else None}

    floor = cohort(scored_mask & (values <= lo_cut))
    ceiling = cohort(scored_mask & (values >= hi_cut))
    thin = cohort(~scored_mask)
    gap = (None if floor["mean_prediction"] is None or ceiling["mean_prediction"] is None
           else ceiling["mean_prediction"] - floor["mean_prediction"])

    # ---- the curve, re-read over the scored range only
    raw = pdp.get(CREDIT_SCORE_FEATURE) or {}
    points = [pt for pt in raw.get("points", [])
              if isinstance(pt.get("value"), (int, float))
              and pt["value"] >= CREDIT_SCORE_FLOOR]
    dropped = [pt for pt in raw.get("points", [])
               if isinstance(pt.get("value"), (int, float))
               and pt["value"] < CREDIT_SCORE_FLOOR]
    curve = None
    if len(points) >= 2:
        curve_scores = np.array([pt["score"] for pt in points], dtype=float)
        diffs = np.diff(curve_scores)
        net = float(curve_scores[-1] - curve_scores[0])
        # How far the curve moves *against* its own trend. A curve can be
        # non-monotone by 0.003 or by half its swing, and the flag alone cannot
        # tell a reviewer which - one is grid noise, the other is a policy
        # question.
        trend = 1.0 if net >= 0 else -1.0
        max_reversal = float(max(0.0, float((-diffs * trend).max()))) if diffs.size else 0.0
        swing = float(curve_scores.max() - curve_scores.min())
        curve = {
            "points": points,
            "at_min": points[0], "at_max": points[-1],
            "net_change": net,
            "direction": "decreases" if net < 0 else "increases" if net > 0 else "flat",
            "swing": swing,
            "monotone": bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)),
            "max_reversal": max_reversal,
            "max_reversal_share": (round(max_reversal / swing, 4) if swing else 0.0),
            "peak": max(points, key=lambda pt: pt["score"]),
            "trough": min(points, key=lambda pt: pt["score"]),
            "sentinel_points_dropped": [pt["value"] for pt in dropped],
        }

    # Labels are merged when one row is two extremes at once, so match on the
    # phrase rather than on equality.
    named = {key: next((i for i in instances if key in i["label"]), None)
             for key in ("lowest credit score", "highest credit score")}
    band = {
        "feature": CREDIT_SCORE_FEATURE,
        "label": reason_label(CREDIT_SCORE_FEATURE)["label"],
        "valid_range": [CREDIT_SCORE_FLOOR, CREDIT_SCORE_CEILING],
        "sentinel": NEW_TO_CREDIT_SENTINEL,
        "tail_pct": CREDIT_SCORE_TAIL_PCT,
        "output": bundle.output_label,
        "units": bundle.units,
        "observed": {"scored_files": int(scored_mask.sum()),
                     "score_min": float(scored.min()), "score_max": float(scored.max()),
                     "floor_cohort": floor, "ceiling_cohort": ceiling,
                     "new_to_credit": thin,
                     "new_to_credit_pct": round(100 * float((~scored_mask).mean()), 2),
                     "gap": gap},
        "curve": curve,
        "instances": {k: v for k, v in named.items() if v is not None},
    }
    band["reading"] = _credit_score_reading(bundle, band)
    return band


def _credit_score_reading(bundle: ModelBundle, band: dict) -> str:
    """The band in one paragraph, in the model's own units."""
    units = bundle.units
    obs = band["observed"]
    floor, ceiling, thin = obs["floor_cohort"], obs["ceiling_cohort"], obs["new_to_credit"]
    pct = band["tail_pct"]

    parts = [f"Scored files run {obs['score_min']:.0f} to {obs['score_max']:.0f}."]
    if floor["mean_prediction"] is not None and ceiling["mean_prediction"] is not None:
        parts.append(
            f"The bottom {pct}% of the score ({floor['score_min']:.0f}-{floor['score_max']:.0f}, "
            f"{floor['files']:,} files) average {_fmt(floor['mean_prediction'], units)} "
            f"{units}; the top {pct}% ({ceiling['score_min']:.0f}-{ceiling['score_max']:.0f}, "
            f"{ceiling['files']:,} files) average {_fmt(ceiling['mean_prediction'], units)}, "
            f"a gap of {_fmt(abs(obs['gap']), units)}.")

    curve = band.get("curve")
    if curve:
        shape = ("monotonically" if curve["monotone"] else
                 f"but not monotonically - it turns at {curve['peak']['value']:.0f}, the largest "
                 f"move against that trend being {_fmt(curve['max_reversal'], units)} "
                 f"({curve['max_reversal_share']:.0%} of the swing)")
        parts.append(
            f"Holding every other feature fixed, {band['output'].lower()} "
            f"{curve['direction']} {_fmt(abs(curve['net_change']), units)} from "
            f"{_fmt(curve['at_min']['score'], units)} at {curve['at_min']['value']:.0f} to "
            f"{_fmt(curve['at_max']['score'], units)} at {curve['at_max']['value']:.0f}, {shape}.")
        if curve["sentinel_points_dropped"]:
            parts.append("The raw curve's low end is the new-to-credit sentinel and has been "
                         "excluded from this band.")
    if thin["files"]:
        parts.append(
            f"{thin['files']:,} new-to-credit files ({obs['new_to_credit_pct']}%) carry no score "
            f"at all and average {_fmt(thin['mean_prediction'], units)} - they are a separate "
            f"population, not the bottom of this band.")
    return " ".join(parts)


def _explained_class(explanation, index: int, bundle: ModelBundle) -> str | None:
    """Which class the local attributions are about: omnixai explains the top
    predicted label, which for a low-probability file is the *negative* class."""
    if explanation is None or not bundle.is_classification:
        return None
    label = explanation.get_explanations(index=index).get("target_label")
    if label is None:
        return None
    names = bundle.class_names or []
    label = int(label)
    return names[label] if 0 <= label < len(names) else str(label)


def _local_rows(explanation, index: int, top_k: int) -> list[dict]:
    if explanation is None:
        return []
    item = explanation.get_explanations(index=index)
    scores = np.asarray(item["scores"], dtype=float)
    order = np.argsort(-np.abs(scores))[:top_k]
    return [{"feature": item["features"][i],
             "label": reason_label(str(item["features"][i]))["label"],
             "value": _native(item["values"][i]),
             "contribution": float(scores[i])} for i in order]


def _counterfactual_rows(explanation, index: int, bundle: ModelBundle | None = None) -> list[dict]:
    """The changed columns between the file and its nearest counterfactual."""
    if explanation is None:
        return []
    names = (bundle.class_names or []) if bundle else []
    item = explanation.get_explanations(index=index)
    query, cfs = item.get("query"), item.get("counterfactual")
    if cfs is None or len(cfs) == 0:
        return []
    out = []
    q = query.iloc[0]
    for _, cf in cfs.iterrows():
        changes = {col: {"from": _native(q[col]), "to": _native(cf[col])}
                   for col in cfs.columns
                   if col != "label" and not _equalish(q[col], cf[col])}
        if changes:
            label = _native(cf.get("label"))
            readable = names[int(label)] if isinstance(label, (int, float)) and 0 <= int(label) < len(names) else label
            out.append({"label": readable, "changes": changes})
    return out


def _equalish(a, b) -> bool:
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-9))
    return str(a) == str(b)


def build_report(analysis: Analysis, *, instance_meta: list[dict],
                 credit_frame: pd.DataFrame | None = None,
                 predictions: np.ndarray | None = None, **counts) -> dict:
    """Reduce the explanation objects to plain numbers: what the markdown
    report prints, what the JSON persists, and what a regression test could
    assert on later."""
    bundle = analysis.bundle
    g = analysis.global_explanations

    shap_global = _importance_series(g.get("shap_global"))
    permutation = _importance_series(g.get("permutation"))
    categorical = set(bundle.spec.categorical)
    pdp = _curve_effects(g.get("pdp"), categorical)
    ale = _curve_effects(g.get("ale"), categorical)

    frame = pd.DataFrame(index=list(bundle.spec.columns))
    frame["label"] = [reason_label(c)["label"] for c in frame.index]
    frame["kind"] = ["categorical" if c in bundle.spec.categorical else "numeric" for c in frame.index]
    if shap_global is not None:
        frame["shap_global"] = shap_global
        frame["shap_share_pct"] = 100 * shap_global / shap_global.sum()
    if permutation is not None:
        frame["permutation"] = permutation
    frame["pdp_swing"] = pd.Series({k: v["swing"] for k, v in pdp.items()})
    frame["pdp_direction"] = pd.Series({k: v["direction"] for k, v in pdp.items()})
    frame["ale_swing"] = pd.Series({k: v["swing"] for k, v in ale.items()})
    frame["monotone_in_pdp"] = pd.Series({k: v["monotone"] for k, v in pdp.items()})

    rank_col = "shap_global" if "shap_global" in frame else "permutation"
    frame = frame.sort_values(rank_col, ascending=False) if rank_col in frame else frame
    frame.insert(0, "rank", range(1, len(frame) + 1))

    # Rank agreement between two methods that share no machinery is the cheapest
    # honest check that a global ranking means anything.
    agreement = None
    if shap_global is not None and permutation is not None:
        common = shap_global.index.intersection(permutation.index)
        if len(common) > 2:
            agreement = float(pd.Series(shap_global[common]).corr(
                pd.Series(permutation[common]), method="spearman"))

    instances = []
    for pos, meta in enumerate(instance_meta):
        instances.append({
            **meta,
            "explained_class": _explained_class(
                analysis.local_explanations.get("shap"), pos, bundle),
            "shap": _local_rows(analysis.local_explanations.get("shap"), pos, 6),
            "lime": _local_rows(analysis.local_explanations.get("lime"), pos, 6),
            "counterfactuals": _counterfactual_rows(
                analysis.local_explanations.get(COUNTERFACTUAL_EXPLAINER), pos, bundle),
        })
    for item in instances:
        shap_feats = [r["feature"] for r in item["shap"][:3]]
        lime_feats = [r["feature"] for r in item["lime"][:3]]
        item["top3_agreement"] = len(set(shap_feats) & set(lime_feats))

    credit_score = _credit_score_band(bundle, credit_frame, predictions, pdp, instances)

    return {
        "model": bundle.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": bundle.mode,
        "learner": bundle.card.learner,
        "model_version": bundle.card.version,
        "target": bundle.target,
        "output": bundle.output_label,
        "units": bundle.units,
        "attribution_units": bundle.attribution_units,
        "class_names": bundle.class_names,
        "counterfactual_status": analysis.counterfactual_status,
        "curve_skipped": [{"feature": f, "categories": n} for f, n in analysis.curve_skipped],
        "notes": bundle.notes,
        "counts": counts,
        "card_metrics": {k: round(float(v), 5) for k, v in bundle.card.metrics.items()
                         if isinstance(v, (int, float, np.floating))},
        "limitations": list(bundle.card.limitations),
        "global_importance": frame.reset_index().rename(columns={"index": "feature"}).to_dict("records"),
        "rank_agreement_spearman": agreement,
        "pdp": pdp,
        "ale": ale,
        "credit_score": credit_score,
        "instances": instances,
        "timings_sec": {k: round(v, 2) for k, v in analysis.timings.items()},
        "config_versions": bundle.card.config_versions,
    }


# ---------------------------------------------------------------- reporting


def _fmt(value: float, units: str) -> str:
    if units == "probability":
        return f"{value:.4f}"
    return f"{value:.3f}"


def _sentence_end(units: str) -> str:
    """`% p.a.` already carries a full stop; `log-odds` does not."""
    return units if units.endswith(".") else f"{units}."


def _cell(value, units: str) -> str:
    """A table cell for a value that may be absent or NaN (a feature can be
    ranked by one method and skipped by another)."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    value = float(value)
    return "-" if value != value else _fmt(value, units)


def write_report(analysis: Analysis, directory: Path | str | None = None) -> Path:
    """Persist the analysis as JSON, a ranked CSV and a readable markdown brief."""
    report = analysis.report
    out = Path(directory) if directory else REPORT_DIR / analysis.name
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    importance = pd.DataFrame(report["global_importance"])
    importance.to_csv(out / "global_importance.csv", index=False)
    (out / "report.md").write_text(markdown_report(analysis), encoding="utf-8")
    log.info("[%s] report -> %s", analysis.name, out)
    return out


def markdown_report(analysis: Analysis) -> str:
    report, bundle = analysis.report, analysis.bundle
    units = report["units"]
    lines: list[str] = []
    add = lines.append

    add(f"# OmniXAI analysis - `{report['model']}`")
    add("")
    add(f"*{report['output']}* | learner `{report['learner']}` | artefact `{report['model_version']}` "
        f"| target `{report['target']}` | generated {report['generated_at']}")
    add("")
    counts = report["counts"]
    add(f"Holdout rows analysed: **{counts['n_test']:,}** (trained on {counts['n_train']:,}). "
        f"Explainer reference population {counts['n_reference']}, KernelSHAP sample "
        f"{counts['n_global_sample']}, permutation sample {counts['n_permutation_sample']:,}.")
    if report["notes"]:
        add("")
        add(f"> {report['notes']}")
    add("")

    add("## Headline training metrics")
    add("")
    add("| metric | value |")
    add("| --- | ---: |")
    for key, value in list(report["card_metrics"].items())[:10]:
        add(f"| {key} | {value:g} |")
    add("")

    add("## Global analysis")
    add("")
    agreement = report["rank_agreement_spearman"]
    if agreement is not None:
        add(f"Spearman rank agreement between KernelSHAP and permutation importance: "
            f"**{agreement:.2f}**. Two methods sharing no machinery ranking features the same "
            f"way is what makes the ranking below worth acting on.")
        add("")
    add(f"| # | feature | reads as | mean abs SHAP ({report['attribution_units']}) | share % "
        f"| permutation | PDP swing ({units}) | PDP direction | monotone |")
    add("| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in report["global_importance"][:15]:
        add("| {rank} | `{feature}` | {label} | {shap} | {share} | {perm} | {swing} | {direction} | {mono} |".format(
            rank=row["rank"], feature=row["feature"], label=row.get("label", ""),
            shap=_cell(row.get("shap_global"), units),
            share=_cell(row.get("shap_share_pct"), "share"),
            perm=_cell(row.get("permutation"), units),
            swing=_cell(row.get("pdp_swing"), units),
            direction=_cell(row.get("pdp_direction"), units),
            mono=_cell(row.get("monotone_in_pdp"), units),
        ))
    add("")
    attribution = report["attribution_units"]
    caveat = (" KernelSHAP explains a classifier through the logit link, so its contributions "
              "do not add up to a probability." if attribution == "log-odds" else "")
    add(f"`PDP swing` is how far the average prediction moves across that feature's range "
        f"(in {units}); direction is the sign from the feature's lowest value to its highest. "
        f"SHAP and permutation columns are in {_sentence_end(attribution)}{caveat}")
    if report.get("curve_skipped"):
        add("")
        add("No curve was computed for "
            + ", ".join(f"`{s['feature']}` ({s['categories']} categories)"
                        for s in report["curve_skipped"])
            + f": above {MAX_CURVE_CATEGORIES} categories the curve costs more than it "
              "explains. They are still ranked above.")
    add("")

    curves = sorted(report["pdp"].items(), key=lambda kv: -kv[1]["swing"])[:5]
    if curves:
        add("### Shape of the strongest effects")
        add("")
        for feature, effect in curves:
            if effect["direction"] == "varies by category":
                add(f"- **{reason_label(feature)['label']}** (`{feature}`): spans "
                    f"{_fmt(effect['swing'], units)} {units} across categories, lowest at "
                    f"`{effect['trough']['value']}`, highest at `{effect['peak']['value']}`.")
            else:
                add(f"- **{reason_label(feature)['label']}** (`{feature}`): "
                    f"{effect['direction']} the prediction by "
                    f"{_fmt(abs(effect['net_change']), units)} {units} from "
                    f"{effect['at_low']['value']:g} to {effect['at_high']['value']:g}; "
                    f"{'monotone' if effect['monotone'] else 'non-monotone'}, "
                    f"peak at {effect['peak']['value']:g}.")
        add("")

    band = report.get("credit_score")
    if band:
        obs = band["observed"]
        add("## Credit score - both ends of the band")
        add("")
        add(band["reading"])
        add("")
        add(f"| cohort | files | score range | mean {report['output'].lower()} | "
            f"median |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for key, title in (("floor_cohort", f"bottom {band['tail_pct']}% of the score"),
                           ("ceiling_cohort", f"top {band['tail_pct']}% of the score"),
                           ("new_to_credit", "new to credit (no score)")):
            cohort = obs[key]
            if not cohort["files"]:
                continue
            span = ("-" if cohort["score_min"] is None
                    else f"sentinel {band['sentinel']}" if cohort["score_min"] < 0
                    else f"{cohort['score_min']:.0f}-{cohort['score_max']:.0f}")
            add(f"| {title} | {cohort['files']:,} | {span} "
                f"| {_fmt(cohort['mean_prediction'], units)} "
                f"| {_fmt(cohort['median_prediction'], units)} |")
        add("")
        curve = band.get("curve")
        if curve:
            add(f"Holding every other feature fixed (PDP over scored files only): "
                f"**{_fmt(curve['at_min']['score'], units)}** {units} at "
                f"{curve['at_min']['value']:.0f} -> "
                f"**{_fmt(curve['at_max']['score'], units)}** at "
                f"{curve['at_max']['value']:.0f}, "
                f"{'monotone' if curve['monotone'] else 'NON-MONOTONE'}.")
            add("")
        for key in ("lowest credit score", "highest credit score"):
            item = (band.get("instances") or {}).get(key)
            if item:
                add(f"- **{key}**: `{item['loan_id']}`, score "
                    f"{item['credit_score']:.0f}, prediction "
                    f"{_fmt(item['prediction'], units)} {units} - attributions under "
                    f"*Instance analysis* below.")
        add("")

    add("## Instance analysis")
    add("")
    for item in report["instances"]:
        add(f"### {item['label']} - `{item['loan_id']}`")
        add("")
        actual = item["actual"]
        explained = f"explained class `{item['explained_class']}` | " if item.get("explained_class") else ""
        add(f"Prediction **{_fmt(item['prediction'], units)}** {units} | {explained}"
            f"actual `{actual}` | top-3 feature agreement between SHAP and LIME: "
            f"{item['top3_agreement']}/3")
        add("")
        add(f"Contributions below are in {_sentence_end(report['attribution_units'])}")
        add("")
        add("| feature | value | SHAP | LIME |")
        add("| --- | ---: | ---: | ---: |")
        lime_by_feature = {r["feature"]: r["contribution"] for r in item["lime"]}
        for row in item["shap"]:
            lime_value = lime_by_feature.get(row["feature"])
            lime_cell = f"{lime_value:+.4f}" if lime_value is not None else "-"
            add(f"| {row['label']} (`{row['feature']}`) | {row['value']} "
                f"| {row['contribution']:+.4f} | {lime_cell} |")
        add("")
        if not item["counterfactuals"] and report.get("counterfactual_status", "").startswith("not"):
            add(f"Counterfactuals: {report['counterfactual_status']}.")
            add("")
        if item["counterfactuals"]:
            add("Nearest counterfactual (MACE) - the smallest change that flips the outcome:")
            add("")
            for cf in item["counterfactuals"]:
                changes = "; ".join(f"`{k}` {v['from']} -> {v['to']}" for k, v in cf["changes"].items())
                add(f"- to class `{cf['label']}`: {changes}")
            add("")

    add("## Limitations carried from the model card")
    add("")
    for limitation in report["limitations"]:
        add(f"- {limitation}")
    add("")
    add(f"Compute: {report['timings_sec']}")
    add("")
    return "\n".join(lines)


def console_summary(analysis: Analysis, top: int = 8) -> str:
    report = analysis.report
    units = report["units"]
    lines = [
        "",
        "=" * 88,
        f"{report['model'].upper():<12} {report['output']}  [{report['learner']}]",
        "=" * 88,
    ]
    agreement = report["rank_agreement_spearman"]
    if agreement is not None:
        lines.append(f"  SHAP vs permutation rank agreement (Spearman): {agreement:.2f}")
    lines.append(f"  attributions in {report['attribution_units']}, PDP in {units}")
    lines.append(f"  {'#':>2}  {'feature':<28} {'mean|SHAP|':>11} {'share%':>7} {'PDP swing':>10}  direction")
    for row in report["global_importance"][:top]:
        shap_value = row.get("shap_global")
        swing = row.get("pdp_swing")
        share = row.get("shap_share_pct")
        share = f"{share:.1f}" if isinstance(share, float) and share == share else "-"
        lines.append(
            f"  {row['rank']:>2}  {row['feature']:<28} {_cell(shap_value, units):>11} "
            f"{share:>7} {_cell(swing, units):>10}  {_cell(row.get('pdp_direction'), units)}"
        )
    band = report.get("credit_score")
    if band:
        obs, curve = band["observed"], band.get("curve")
        lines.append(f"  credit score {obs['score_min']:.0f}-{obs['score_max']:.0f}: "
                     f"bottom {band['tail_pct']}% -> {_fmt(obs['floor_cohort']['mean_prediction'], units)}, "
                     f"top {band['tail_pct']}% -> {_fmt(obs['ceiling_cohort']['mean_prediction'], units)} "
                     f"({obs['new_to_credit']['files']:,} new-to-credit held out)")
        if curve:
            lines.append(f"    holding all else fixed: {_fmt(curve['at_min']['score'], units)} at "
                         f"{curve['at_min']['value']:.0f} -> {_fmt(curve['at_max']['score'], units)} at "
                         f"{curve['at_max']['value']:.0f}"
                         f"{'' if curve['monotone'] else '  NON-MONOTONE'}")
    lines.append("  instances:")
    for item in report["instances"]:
        drivers = ", ".join(f"{r['feature']}{r['contribution']:+.3f}" for r in item["shap"][:3])
        explained = f" [{item['explained_class']}]" if item.get("explained_class") else ""
        lines.append(f"    {item['label']:<22} {item['loan_id']:<14} "
                     f"pred={_fmt(item['prediction'], units)} actual={item['actual']}{explained}  {drivers}")
    lines.append(f"  compute: {report['timings_sec']}")
    return "\n".join(lines)
