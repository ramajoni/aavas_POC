"""The feature-engineering funnel, measured rather than drawn.

The walkthrough's feature slide is a funnel - raw attributes, derived candidates,
passed screening, approved in review, used in the model. This module is that
funnel as a measurement of *this* pipeline over the training table, so every
count on it is a count of things that can be listed by name, and every number
next to a name is one a test actually computed.

What each stage counts:

1. **Raw attributes** - every column of the loan record the model could see at
   decision time, *before* any contract is consulted. Each dataset column is
   walked and given a disposition. Keys, the time index, outcomes, synthetic
   latent truth, pricing decisions and anything computed from them,
   post-assessment fields for Stage 1, a model's own output, and any column the
   dataset contract (``ingestion.schemas``) does not declare are excluded *with
   their reason*. That list is the leakage-control evidence, not a side effect.
2. **Derived candidates** - everything the discovery generators build from the
   numeric raw attributes: ratios, gaps, logs, interactions and the trend family.
   No 400-candidate cap; the funnel counts what the generators produce.
3. **Passed screening** - raw attributes and derived candidates that clear five
   statistical screens: coverage, degeneracy, stability, minimum signal against
   the model's target, and redundancy.
4. **Approved in review** - screened features a person has approved, where the
   approval is an authored entry in ``feature_config.reason_code_labels``. This
   stage reads what people wrote and counts it. It never approves anything and
   never writes a label - discovery can be automated, admission cannot.
5. **In the model** - the contract's own columns, each reconciled against the
   stages above. A contract feature that fails a screen is shown failing it.

The funnel is not monotone end to end and is not meant to be. Derived candidates
outnumber raw attributes because generation is an expansion; and "in the model"
can exceed "screened and approved" when a contract feature fails a screen, which
the reconciliation names rather than the count hiding.

Runs from ``scripts/run_feature_funnel.py`` (writes ``reports/features/``) and
``GET /features/funnel``, which serves a saved report only while it is fresh -
see ``freshness``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from ..config import feature_config, model_config
from ..ingestion import schemas
from ..logging_utils import get_logger
from ..paths import CONFIG_DIR, REPORTS_DIR, resolve
from . import discovery, profiling
from .feature_pipeline import spec

log = get_logger(__name__)

REPORT_DIR = REPORTS_DIR / "features"
DEFAULT_SAMPLE = 6000
FAMILIES = ["ratio", "gap", "log", "interaction", "trend"]
SLIDE_MODELS = ["stage2", "risk", "conversion", "stage1"]

DISPLAY_NAMES = {
    "stage1": "Indicative pricing", "stage2": "Risk-adjusted pricing", "risk": "Risk (PD)",
    "lgd": "Risk (LGD)", "conversion": "Acceptance", "uplift": "Pricing uplift",
    "prepayment": "Prepayment", "override": "Override", "sales_context": "Sales context",
}

# Contracts whose TARGETS key is not the contract name.
TARGET_ALIASES = {"override": "override_flag"}
# Only observable on a loan that exists; the trainers fit these on disbursed files.
POST_DISBURSAL_TARGETS = {"defaulted", "bt_out_24m", "delinquent_12m"}

# ------------------------------------------------------------ the dispositions

EXCLUSIONS = {
    "identifier": "a key, not an attribute of the customer or the loan - a model fed it "
                  "memorises rows",
    "time_index": "orders the out-of-time split and the stability windows; a date is a "
                  "leakage surface, not a feature",
    "outcome": "observed after the decision - a target, or a consequence of the price",
    "latent": "synthetic ground truth that exists only because the data is generated; it "
              "may evaluate a model, never feed one",
    "decision": "a pricing decision or the human response to one - what the model helps "
                "set, not an input to it",
    "decision_derived": "computed from a decision or an outcome, so it carries that "
                        "decision with it",
    "post_assessment": "collected during assessment; Stage 1 prices before any of it exists",
    "own_output": "derived from this model's own prediction, so using it would be circular",
    "unclassified": "not declared in the dataset contract (ingestion.schemas), so when it "
                    "becomes known is undocumented - excluded until someone declares it",
}

# Two identity columns are locations. The stage sees them at decision time and the
# contracts price on city, so they are attributes, not keys. Every other identity
# column but the time index (loan, branch and sales-officer ids) is an identifier.
LOCATIONS = {"state", "city"}

# Columns the loader adds (``loader.add_derived_columns``) and what each is
# computed from. A derived column inherits the most restrictive disposition of its
# inputs, so a gap built on the sanctioned rate is a decision, whatever its name.
LOADER_LINEAGE = {
    "offered_rate": ["sanctioned_rate"],
    "competitor_gap_bps": ["sanctioned_rate", "competitor_rate"],
    "recommended_gap_bps": ["ai_recommended_rate", "competitor_rate"],
    "was_overridden": ["override_bps"],
    "override_direction": ["override_bps"],
    "total_discount_bps": ["policy_card_rate", "sanctioned_rate"],
    "application_month": ["application_date"],
}
# In the dataset, outside the schema groups (``schemas.ALL_COLUMNS`` extras).
EXTRA_COLUMNS = {
    "stage1_min_rate": ("decision", "the Stage-1 band quoted at enquiry - an earlier "
                                    "pricing decision, not an attribute of the customer"),
    "stage1_max_rate": ("decision", "the Stage-1 band quoted at enquiry - an earlier "
                                    "pricing decision, not an attribute of the customer"),
    "sanctioned": ("outcome", "credit's sanction, taken after the price"),
}

# A decision column a particular model legitimately sees, because the decision is
# taken *before* the thing that model predicts.
UPSTREAM_DECISIONS = {
    "override": {"ai_recommended_rate": "set by the engine before a person decides; the "
                                        "override model is asked whether they will move it"},
    "uplift": {"ai_recommended_rate": "set by the engine before the RM weighs a discount; "
                                      "the uplift model prices the discount against it"},
}

# Columns a trainer rebuilds before fitting, and what the rebuilt column is computed
# from. The override trainer (and the uplift contract) measure the competitor gap
# at the *recommended* rate via ``sales_features.build``, not at the sanctioned
# rate the loader uses - a different, earlier quantity, so its lineage differs.
REBUILT_LINEAGE = {
    "override": {"competitor_gap_bps": ["ai_recommended_rate", "competitor_rate"]},
    "uplift": {"competitor_gap_bps": ["ai_recommended_rate", "competitor_rate"]},
}
# A model's own output, where it appears in the record as a column.
OWN_OUTPUTS = {
    "risk": {"risk_grade": "assigned from the PD this model produces (see the risk block in "
                           "feature_config.yaml)"},
}
# The columns the optimiser sets per candidate price (``models.base.expand_over_rates``).
# A model whose contract constrains its response to one of them monotonically is a
# response curve over price: the price is the variable under simulation, not a leak
# of the answer. Both conditions are required, so naming a decision column in a
# contract is not enough to let it through.
SCENARIO_COLUMNS = {"offered_rate", "competitor_gap_bps"}

# Read as missing before any test or generator runs - a ratio built on -1 is
# arithmetic on a code. ``feature_config.sentinels`` is the source of truth (the
# transformer reads the same block); this default applies only if it is absent.
DEFAULT_SENTINELS = {"bureau_score": {"below": 0, "flag": "new_to_credit"}}

# --------------------------------------------------------------- the screens

WEAK = "weak_signal"
SCREENS = {
    discovery.THIN: "coverage",
    discovery.DEGENERATE: "degeneracy",
    discovery.UNSTABLE: "stability (PSI)",
    WEAK: "minimum signal",
    discovery.REDUNDANT: "redundancy",
}
SCREEN_WHY = {
    discovery.THIN: discovery.REFUSALS[discovery.THIN],
    discovery.DEGENERATE: discovery.REFUSALS[discovery.DEGENERATE],
    discovery.UNSTABLE: discovery.REFUSALS[discovery.UNSTABLE],
    WEAK: "no measurable association with the target - noise, not signal",
    discovery.REDUNDANT: discovery.REFUSALS[discovery.REDUNDANT],
}
SCREEN_ORDER = [discovery.THIN, discovery.DEGENERATE, discovery.UNSTABLE, WEAK,
                discovery.REDUNDANT]

# The categorical floor is the numeric one, applied to a bias-corrected Cramer's V
# so a many-level category cannot clear it on noise alone.
EXTRA_THRESHOLDS = {"min_cramers_v": discovery.DEFAULTS["min_abs_target_corr"]}


@dataclass
class Item:
    """One raw attribute or derived candidate on its way through the funnel."""

    name: str
    kind: str                    # "raw" | "derived"
    dtype: str                   # "numeric" | "categorical"
    family: str                  # raw group, or the generator family
    recipe: str
    inputs: list[str]
    in_contract: bool = False
    tests: dict[str, Any] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    labelled: bool = False

    @property
    def screened(self) -> bool:
        return not self.failed

    @property
    def signal(self) -> float:
        value = self.tests.get("signal")
        return abs(value) if value is not None else -1.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "dtype": self.dtype,
                "family": self.family, "recipe": self.recipe, "inputs": self.inputs,
                "in_contract": self.in_contract, "tests": self.tests,
                "failed": self.failed, "passed_screening": self.screened,
                "reason_label": self.labelled}


# ------------------------------------------------------------------- helpers


def _round(value: Any, places: int = 4) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, places) if np.isfinite(value) else None


def resolve_target(model: str) -> str | None:
    return schemas.TARGETS.get(model) or schemas.TARGETS.get(TARGET_ALIASES.get(model, ""))


def contracts() -> list[str]:
    """Every block in feature_config.yaml that declares columns."""
    return [name for name, block in feature_config().items()
            if isinstance(block, dict) and ("numeric" in block or "categorical" in block)]


def _schema_group(column: str) -> str | None:
    if column in schemas.LATENT_COLUMNS:
        return "latent"
    if column in schemas.OUTCOME_COLUMNS:
        return "outcome"
    if column in schemas.DECISION_COLUMNS:
        return "decision"
    if column in schemas.IDENTITY_COLUMNS:
        return "identity"
    if column in schemas.PRE_ASSESSMENT_COLUMNS:
        return "pre_assessment"
    if column in schemas.POST_ASSESSMENT_COLUMNS:
        return "post_assessment"
    if column in schemas.SALES_CONTEXT_COLUMNS:
        return "sales_context"
    if column in EXTRA_COLUMNS:
        return "extra"
    if column in LOADER_LINEAGE:
        return "loader_derived"
    return None


# ------------------------------------------------------- 1. raw attributes


def model_view(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """The frame as this model's trainer builds it before fitting.

    Copied from the trainers (the same table ``scripts/run_automl.py`` keeps)
    rather than re-derived: the funnel has to describe the columns a model is
    actually fed, and for the override model ``competitor_gap_bps`` is rebuilt at
    the recommended rate - measuring the loader's column instead would report a
    leak the model does not have.
    """
    from . import sales_features, stage1_features, stage2_features

    if model == "stage1":
        return stage1_features.build(frame)
    if model in ("stage2", "risk", "lgd"):
        return stage2_features.build(frame)
    if model in REBUILT_LINEAGE:
        return sales_features.build(frame, reference_rate_col="ai_recommended_rate")
    return frame


def classify_columns(frame: pd.DataFrame, model: str) -> tuple[list[dict], list[dict]]:
    """Every dataset column, as a raw attribute or as an exclusion with its reason.

    The two lists partition ``frame.columns`` exactly: nothing is dropped without
    a recorded reason, and nothing reaches the generators without a disposition.
    """
    contract = spec(model)
    time_column = model_config()["data"].get("time_column", "application_date")
    target = resolve_target(model)
    upstream = UPSTREAM_DECISIONS.get(model, {})
    own = OWN_OUTPUTS.get(model, {})
    lineage = {**LOADER_LINEAGE, **REBUILT_LINEAGE.get(model, {})}

    def disposition(column: str, seen: frozenset = frozenset()) -> dict[str, Any]:
        group = _schema_group(column)
        if column in own:
            return {"reason": "own_output", "group": group, "detail": own[column]}
        if group == "latent":
            return {"reason": "latent", "group": group}
        if group == "outcome":
            return {"reason": "outcome", "group": group}
        if group == "decision":
            if column in upstream:
                return {"keep": "upstream_decision", "group": group, "detail": upstream[column]}
            return {"reason": "decision", "group": group}
        if group == "identity":
            if column == time_column:
                return {"reason": "time_index", "group": group}
            if column in LOCATIONS:
                return {"keep": "location", "group": group,
                        "detail": "listed under identity in schemas, kept as a location"}
            return {"reason": "identifier", "group": group}
        if group == "post_assessment" and model == "stage1":
            return {"reason": "post_assessment", "group": group}
        if group in ("pre_assessment", "post_assessment", "sales_context"):
            return {"keep": group, "group": group}
        if group == "extra":
            reason, detail = EXTRA_COLUMNS[column]
            return {"reason": reason, "group": group, "detail": detail}
        if group == "loader_derived" and column not in seen:
            inputs = lineage[column]
            rebuilt = column in REBUILT_LINEAGE.get(model, {})
            parents = {c: disposition(c, seen | {column}) for c in inputs}
            blocked = {c: d for c, d in parents.items() if "reason" in d}
            if not blocked:
                return {"keep": "derived_attribute", "group": group,
                        "detail": (f"rebuilt by the {model} trainer from {', '.join(inputs)}"
                                   if rebuilt else
                                   f"computed by the loader from {', '.join(inputs)}")}
            if column in SCENARIO_COLUMNS and column in contract.monotone:
                return {"keep": "scenario_input", "group": group,
                        "detail": (f"the price under simulation: the optimiser sets it per "
                                   f"candidate rate, and the {model} contract constrains the "
                                   f"response to it (monotone {contract.monotone[column]:+d})")}
            codes = sorted({d["reason"] for d in blocked.values()})
            reason = "time_index" if codes == ["time_index"] else "decision_derived"
            return {"reason": reason, "group": group,
                    "detail": "computed from " + ", ".join(
                        f"{c} ({d['reason']})" for c, d in blocked.items())}
        return {"reason": "unclassified", "group": None,
                "detail": "in the data file, in no column group of ingestion.schemas"}

    raw, excluded = [], []
    for column in frame.columns:
        d = disposition(column)
        if "reason" in d:
            excluded.append({"column": column, "group": d["group"], "reason": d["reason"],
                             "why": EXCLUSIONS[d["reason"]], "detail": d.get("detail"),
                             "is_target": column == target})
            continue
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        raw.append({"column": column, "dtype": "numeric" if numeric else "categorical",
                    "group": d["keep"], "detail": d.get("detail"),
                    "in_contract": column in contract.columns})
    return raw, excluded


def sentinel_rules() -> tuple[dict[str, dict], str]:
    declared = feature_config().get("sentinels")
    if declared:
        return dict(declared), "feature_config.yaml (sentinels)"
    return dict(DEFAULT_SENTINELS), "funnel default - feature_config.yaml declares no sentinels"


def _clean_sentinels(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rules, source = sentinel_rules()
    out, notes = frame.copy(), []
    for column, rule in rules.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        hit = pd.Series(False, index=out.index)
        if "below" in rule:
            hit |= values < float(rule["below"])
        if "values" in rule:
            hit |= values.isin([float(v) for v in rule["values"]])
        if hit.any():
            out[column] = values.mask(hit)
        notes.append({"column": column, "rule": {k: rule[k] for k in ("below", "values", "flag")
                                                 if k in rule},
                      "rows_read_as_missing": int(hit.sum()), "source": source,
                      "why": "a code, not a measurement - read as missing before any test or "
                             "generator, so no candidate is arithmetic on it"})
    return out, notes


def _population(frame: pd.DataFrame, target: str | None) -> tuple[pd.Series, str]:
    if target is None or target not in frame.columns:
        return pd.Series(True, index=frame.index), "all rows (no target, so no signal test)"
    rows = frame[target].notna()
    rule = f"rows where {target} is observed"
    if target in POST_DISBURSAL_TARGETS and "disbursed" in frame.columns:
        rows &= frame["disbursed"].astype(bool)
        rule = f"disbursed loans only - {target} is only observable on a loan that exists"
    return rows, rule


# ---------------------------------------------------- 2. derived candidates


def derive(frame: pd.DataFrame, numeric: list[str], families: list[str],
           time: pd.Series | None) -> tuple[list[discovery.Candidate], dict[str, Any]]:
    """Run every generator family over the numeric raw attributes. No cap."""
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    p = len(numeric)
    possible = {"ratio": p * (p - 1), "gap": p * (p - 1) // 2, "log": p,
                "interaction": p * (p - 1) // 2, "trend": p}
    out: list[discovery.Candidate] = []
    by_family: dict[str, Any] = {}
    for family in families:
        generator = discovery.GENERATORS.get(family)
        if generator is None:
            raise ValueError(f"unknown generator family {family!r}")
        built = (discovery.gen_trend(values, numeric, time=time) if family == "trend"
                 else generator(values, numeric))
        by_family[family] = {
            "built": len(built), "possible": possible.get(family),
            "rule": discovery.FUNCTION_REGISTRY.get(family, {}).get("applies_to"),
            "produces": discovery.FUNCTION_REGISTRY.get(family, {}).get("produces")}
        out.extend(built)

    # A generated name that equals a raw attribute, or repeats an earlier one, is
    # the same column twice; it is counted once.
    seen, unique, dropped = set(numeric), [], 0
    for candidate in out:
        if candidate.name in seen:
            dropped += 1
            continue
        seen.add(candidate.name)
        unique.append(candidate)
    return unique, {"by_family": by_family, "duplicates_dropped": dropped}


# ------------------------------------------------------------ 3. screening


def _rank_block(columns: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(columns)
    if not names:
        return names, np.empty((0, 0)), np.empty((0, 0))
    ranks = pd.DataFrame(columns).rank(method="average").to_numpy(float)
    mask = np.isfinite(ranks)
    centred = ranks - np.nanmean(np.where(mask, ranks, np.nan), axis=0)
    return names, np.where(mask, centred, 0.0), mask.astype(float)


def _masked_corr(xa: np.ndarray, ma: np.ndarray, xb: np.ndarray, mb: np.ndarray,
                 min_rows: int = 50) -> np.ndarray:
    """Pearson over pairwise-complete rows, every column of A against every column
    of B, in six matrix products. On rank-transformed inputs this is Spearman (exact
    where both columns are complete; ranks are taken per column otherwise)."""
    n = ma.T @ mb
    sa, sb = xa.T @ mb, ma.T @ xb
    saa, sbb = (xa * xa).T @ mb, ma.T @ (xb * xb)
    sab = xa.T @ xb
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sab - sa * sb / n
        va, vb = saa - sa * sa / n, sbb - sb * sb / n
        r = cov / np.sqrt(va * vb)
    r[(n < min_rows) | ~(va > 1e-9) | ~(vb > 1e-9)] = np.nan
    return r


def _target_bins(target: pd.Series) -> pd.Series:
    values = pd.to_numeric(target, errors="coerce")
    if values.nunique() <= 10:
        return values
    return pd.qcut(values, 10, labels=False, duplicates="drop")


def _numeric_tests(values: pd.Series, target: pd.Series | None) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    std = float(clean.std(skipna=True)) if clean.notna().sum() > 1 else float("nan")
    signal = (discovery._target_correlation(clean, target) if target is not None
              else float("nan"))
    return {"coverage": _round(clean.notna().mean()),
            "std": _round(std, 6),
            "psi": _round(discovery._psi_halves(clean)),
            "signal": _round(signal),
            "signal_test": "spearman" if target is not None else None}


def _categorical_tests(values: pd.Series, target_bins: pd.Series | None) -> dict[str, Any]:
    present = values.dropna()
    return {"coverage": _round(values.notna().mean()),
            "levels": int(present.astype(str).nunique()),
            "psi": profiling.stability_categorical(values)["psi"],
            "signal": (profiling.cramers_v(values, target_bins, bias_corrected=True)
                       if target_bins is not None else None),
            "signal_test": "cramers_v_bias_corrected" if target_bins is not None else None}


def _first_screens(item: Item, rules: dict[str, float], has_target: bool) -> list[str]:
    tests, failed = item.tests, []
    if (tests.get("coverage") or 0.0) < rules["min_coverage"]:
        failed.append(discovery.THIN)
    if item.dtype == "numeric":
        std = tests.get("std")
        if std is None or std < rules["min_std"]:
            failed.append(discovery.DEGENERATE)
    elif (tests.get("levels") or 0) < 2:
        failed.append(discovery.DEGENERATE)
    psi_value = tests.get("psi")
    if psi_value is not None and psi_value > rules["max_psi"]:
        failed.append(discovery.UNSTABLE)
    if has_target:
        floor = rules["min_abs_target_corr"] if item.dtype == "numeric" else rules["min_cramers_v"]
        signal = tests.get("signal")
        # Not measurable is not passed: a candidate with no evidence of signal has
        # not shown it carries any.
        if signal is None or abs(signal) < floor:
            failed.append(WEAK)
    return failed


def _redundancy(items: list[Item], block: tuple[list[str], np.ndarray, np.ndarray],
                rules: dict[str, float], has_target: bool) -> None:
    """Greedy, so the stronger of a correlated pair is the one kept.

    Contract features go first and are checked only against *stronger* contract
    features. Raw attributes go next, then derived candidates, each strongest
    first, and each is checked against every contract feature (the contract
    carries it whether or not it passed a screen) and every feature already kept.
    Raw before derived is deliberate: a derived candidate has to add something the
    raw record does not already carry, so a trend or a log of a column can never
    displace the column itself by being a hair stronger.
    """
    names, x, m = block
    if not names:
        return
    position = {name: i for i, name in enumerate(names)}
    tau = rules["max_abs_correlation"]
    contract = [it for it in items if it.in_contract]
    others = [it for it in items if not it.in_contract]

    def key(it: Item) -> tuple:
        return (it.kind != "raw", -it.signal if has_target else 0.0, it.name)

    contract.sort(key=key)
    eligible = sorted([it for it in others if not it.failed], key=key)
    contract_idx = [position[it.name] for it in contract]

    # Overlap with the contract for everything: the number the slide's table shows.
    corr_contract = np.abs(_masked_corr(x, m, x[:, contract_idx], m[:, contract_idx])) \
        if contract_idx else np.empty((len(names), 0))
    for it in items:
        row = corr_contract[position[it.name]].copy() if contract_idx else np.array([])
        if it.in_contract:
            row[contract_idx.index(position[it.name])] = np.nan
        if row.size and np.isfinite(row).any():
            j = int(np.nanargmax(row))
            it.tests["overlap"] = _round(row[j])
            it.tests["overlap_with"] = contract[j].name
        else:
            it.tests["overlap"], it.tests["overlap_with"] = None, None

    # Contract features: against stronger contract features only.
    for rank, it in enumerate(contract):
        stronger = contract_idx[:rank]
        if not stronger:
            continue
        row = corr_contract[position[it.name]][:rank]
        if np.isfinite(row).any() and np.nanmax(row) > tau:
            j = int(np.nanargmax(row))
            it.tests["redundant_with"] = contract[j].name
            it.failed.append(discovery.REDUNDANT)

    # Everything that failed another screen: against the contract only.
    for it in others:
        if it.failed and (it.tests.get("overlap") or 0.0) > tau:
            it.tests["redundant_with"] = it.tests["overlap_with"]
            it.failed.append(discovery.REDUNDANT)

    # Everything still standing: against the contract, then against the kept set.
    eligible_idx = [position[it.name] for it in eligible]
    corr_eligible = np.abs(_masked_corr(x[:, eligible_idx], m[:, eligible_idx],
                                        x[:, eligible_idx], m[:, eligible_idx])) \
        if eligible_idx else np.empty((0, 0))
    kept: list[int] = []
    for i, it in enumerate(eligible):
        if (it.tests.get("overlap") or 0.0) > tau:
            it.tests["redundant_with"] = it.tests["overlap_with"]
            it.failed.append(discovery.REDUNDANT)
            continue
        if kept:
            row = corr_eligible[i, kept]
            if np.isfinite(row).any() and np.nanmax(row) > tau:
                j = kept[int(np.nanargmax(row))]
                it.tests["redundant_with"] = eligible[j].name
                it.tests["redundant_correlation"] = _round(np.nanmax(row))
                it.failed.append(discovery.REDUNDANT)
                continue
        kept.append(i)


def _categorical_redundancy(items: list[Item], sample: pd.DataFrame,
                            rules: dict[str, float], has_target: bool) -> None:
    """The same greedy pass for categoricals, on bias-corrected Cramer's V."""
    tau = rules["max_abs_correlation"]
    cache: dict[tuple[str, str], float | None] = {}

    def v(a: str, b: str) -> float | None:
        k = tuple(sorted((a, b)))
        if k not in cache:
            cache[k] = profiling.cramers_v(sample[a], sample[b].astype(str), bias_corrected=True)
        return cache[k]

    def key(it: Item) -> tuple:
        return (-it.signal if has_target else 0.0, it.name)

    contract = sorted([it for it in items if it.in_contract], key=key)
    others = sorted([it for it in items if not it.in_contract], key=key)
    for it in items:
        pairs = [(c.name, v(it.name, c.name)) for c in contract if c.name != it.name]
        pairs = [(n, s) for n, s in pairs if s is not None]
        best = max(pairs, key=lambda p: p[1]) if pairs else (None, None)
        it.tests["overlap"], it.tests["overlap_with"] = _round(best[1]), best[0]

    for rank, it in enumerate(contract):
        pairs = [(c.name, v(it.name, c.name)) for c in contract[:rank]]
        hits = [(n, s) for n, s in pairs if s is not None and s > tau]
        if hits:
            it.tests["redundant_with"] = max(hits, key=lambda p: p[1])[0]
            it.failed.append(discovery.REDUNDANT)

    kept: list[Item] = []
    for it in others:
        if (it.tests.get("overlap") or 0.0) > tau:
            it.tests["redundant_with"] = it.tests["overlap_with"]
            it.failed.append(discovery.REDUNDANT)
            continue
        if it.failed:
            continue
        hits = [(k.name, v(it.name, k.name)) for k in kept]
        hits = [(n, s) for n, s in hits if s is not None and s > tau]
        if hits:
            it.tests["redundant_with"] = max(hits, key=lambda p: p[1])[0]
            it.failed.append(discovery.REDUNDANT)
            continue
        kept.append(it)


def _order_failures(item: Item) -> None:
    item.failed = [code for code in SCREEN_ORDER if code in item.failed]


# --------------------------------------------------------- 5. the latent audit


def _latent_audit(sample: pd.DataFrame, raw: list[dict], latent: list[str]) -> list[dict]:
    """How closely each raw attribute tracks a latent column, on the sample.

    The latent columns *evaluate* here and nothing else: the result changes no
    disposition and no screen. It exists because excluding ``true_pd`` is only
    half of the leakage question - an attribute that is a near-copy of it carries
    it in anyway, and a reader of the exclusion list deserves to know.
    """
    if not latent:
        return []
    out = []
    numeric = [r["column"] for r in raw if r["dtype"] == "numeric"]
    if numeric:
        names, x, m = _rank_block({c: pd.to_numeric(sample[c], errors="coerce").to_numpy(float)
                                   for c in numeric + latent})
        idx = {n: i for i, n in enumerate(names)}
        a = [idx[c] for c in numeric]
        b = [idx[c] for c in latent]
        corr = np.abs(_masked_corr(x[:, a], m[:, a], x[:, b], m[:, b]))
        for i, column in enumerate(numeric):
            if np.isfinite(corr[i]).any():
                j = int(np.nanargmax(corr[i]))
                out.append({"column": column, "latent": latent[j],
                            "association": _round(corr[i, j]), "measure": "|spearman|"})
    for r in raw:
        if r["dtype"] != "categorical":
            continue
        best = (None, -1.0)
        groups = sample[r["column"]].astype(str)
        for column in latent:
            y = pd.to_numeric(sample[column], errors="coerce")
            ok = y.notna()
            if ok.sum() < 50 or y[ok].var() == 0:
                continue
            means = y[ok].groupby(groups[ok]).transform("mean")
            centre = y[ok].mean()
            eta = float(np.sqrt(((means - centre) ** 2).sum() / ((y[ok] - centre) ** 2).sum()))
            if eta > best[1]:
                best = (column, eta)
        if best[0]:
            out.append({"column": r["column"], "latent": best[0], "association": _round(best[1]),
                        "measure": "correlation ratio (eta)"})
    out.sort(key=lambda d: -(d["association"] or 0.0))
    return out


# ------------------------------------------------------------- governance
# The funnel reads approvals; it does not reach into the governance package for
# them. ``aavas.features`` is declared standalone (``agents/packages.py``) and the
# boundary test holds it to that, so whoever composes the two - the API router,
# the script - reads the store and hands the evidence in as a plain dict:
#     {"admitted_proposals": {name: proposal_id}, "approvals": {name: approval_id}}


def _approval_evidence(approvals: dict[str, Any] | None) -> dict[str, Any]:
    if approvals is None:
        return {"consulted": False, "available": False, "admitted_proposals": {},
                "approvals": {},
                "note": ("no approvals source was supplied, so the governance store was not "
                         "read; the review counts authored reason labels only")}
    out = {"consulted": True, "available": bool(approvals.get("available", True)),
           "admitted_proposals": dict(approvals.get("admitted_proposals") or {}),
           "approvals": dict(approvals.get("approvals") or {})}
    if approvals.get("error"):
        out["error"] = approvals["error"]
    return out


# -------------------------------------------------------------- the run


def run(model: str = "stage2", frame: pd.DataFrame | None = None,
        sample: int = DEFAULT_SAMPLE, seed: int | None = None,
        families: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        approvals: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure the funnel for one model contract. Admits nothing, writes nothing.

    ``approvals`` is the governance evidence for this model (see above); without
    it the review stage still counts authored labels and says the store was not
    read.
    """
    started = time.perf_counter()
    contract = spec(model)                                       # KeyError if unknown
    seed = int(model_config().get("seed", 42) if seed is None else seed)
    families = list(families or FAMILIES)
    rules = {**discovery.DEFAULTS, **EXTRA_THRESHOLDS, **(thresholds or {})}
    target = resolve_target(model)
    stamp = fingerprint(model, sample=sample, seed=seed, families=families, thresholds=rules,
                        approvals=approvals)
    seconds: dict[str, float] = {}

    # ------------------------------------------------------ 1. raw attributes
    t0 = time.perf_counter()
    source = "caller-supplied frame"
    if frame is None:
        from ..ingestion import loader

        # Rebuilt from the raw file rather than read from data/processed, which
        # can lag the raw data, and not saved, because other processes read it.
        frame = loader.build_training_table(save=False)
        source = "loader.build_training_table(save=False)"
    frame = model_view(frame, model)
    time_column = model_config()["data"].get("time_column", "application_date")
    if time_column in frame.columns:
        frame = frame.assign(**{time_column: pd.to_datetime(frame[time_column], errors="coerce")})
        # Stability compares older with newer halves, so row order is time order.
        frame = frame.sort_values(time_column, kind="stable")
    frame = frame.reset_index(drop=True)
    frame, sentinel_notes = _clean_sentinels(frame)
    if target is not None and target not in frame.columns:
        raise ValueError(f"target {target!r} for {model!r} is not in the training table")

    raw, excluded = classify_columns(frame, model)
    rows, population_rule = _population(frame, target)
    population = frame[rows]
    target_series = (pd.to_numeric(population[target].astype(float), errors="coerce")
                     if target is not None else None)
    target_bins = _target_bins(target_series) if target_series is not None else None
    numeric_raw = [r["column"] for r in raw if r["dtype"] == "numeric"]
    categorical_raw = [r["column"] for r in raw if r["dtype"] == "categorical"]
    by_reason: dict[str, int] = {}
    for e in excluded:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    seconds["raw_attributes"] = time.perf_counter() - t0

    # --------------------------------------------------- 2. derived candidates
    t0 = time.perf_counter()
    time_series = frame[time_column] if time_column in frame.columns else None
    candidates, generation = derive(frame, numeric_raw, families, time_series)
    by_family = {f: 0 for f in families}
    for c in candidates:
        by_family[c.generator] = by_family.get(c.generator, 0) + 1
    seconds["derived_candidates"] = time.perf_counter() - t0

    # ---------------------------------------------------------- 3. screening
    t0 = time.perf_counter()
    has_target = target_series is not None
    n_sample = min(int(sample), len(population))
    rng = np.random.default_rng(seed)
    sample_pos = np.sort(rng.choice(len(population), size=n_sample, replace=False))
    contract_cols = set(contract.columns)

    items: list[Item] = []
    sample_columns: dict[str, np.ndarray] = {}
    for r in raw:
        column = r["column"]
        item = Item(name=column, kind="raw", dtype=r["dtype"], family=r["group"],
                    recipe=column, inputs=[column], in_contract=column in contract_cols)
        if r["dtype"] == "numeric":
            values = pd.to_numeric(population[column], errors="coerce")
            item.tests = _numeric_tests(values, target_series)
            sample_columns[column] = values.to_numpy(float)[sample_pos]
        else:
            item.tests = _categorical_tests(population[column], target_bins)
        items.append(item)
    for c in candidates:
        values = pd.to_numeric(c.values[rows.to_numpy()], errors="coerce") \
            .replace([np.inf, -np.inf], np.nan)
        item = Item(name=c.name, kind="derived", dtype="numeric", family=c.generator,
                    recipe=c.recipe, inputs=list(c.inputs))
        item.tests = _numeric_tests(values, target_series)
        sample_columns[c.name] = values.to_numpy(float)[sample_pos]
        items.append(item)
        c.values = None                                          # free it as we go
    for item in items:
        item.failed = _first_screens(item, rules, has_target)

    numeric_items = [it for it in items if it.dtype == "numeric"]
    _redundancy(numeric_items, _rank_block(sample_columns), rules, has_target)
    del sample_columns
    categorical_items = [it for it in items if it.dtype == "categorical"]
    sample_frame = population.iloc[sample_pos]
    _categorical_redundancy(categorical_items, sample_frame, rules, has_target)
    for item in items:
        _order_failures(item)
        item.labelled = discovery.has_reason_label(item.name)

    passed = [it for it in items if it.screened]
    any_counts = {code: {"raw": 0, "derived": 0} for code in SCREEN_ORDER}
    first_counts = {code: {"raw": 0, "derived": 0} for code in SCREEN_ORDER}
    for it in items:
        for code in it.failed:
            any_counts[code][it.kind] += 1
        if it.failed:
            first_counts[it.failed[0]][it.kind] += 1
    seconds["screening"] = time.perf_counter() - t0

    # ------------------------------------------------------- 4. the review
    t0 = time.perf_counter()
    governance = _approval_evidence(approvals)
    approved = [it for it in passed if it.labelled]
    governed = set(governance["admitted_proposals"]) | set(governance["approvals"])
    rejected = sorted([it for it in passed if it.kind == "derived" and not it.labelled],
                      key=lambda it: (-it.signal, it.name))
    unlabelled_raw = sorted([it for it in passed if it.kind == "raw" and not it.labelled],
                            key=lambda it: (-it.signal, it.name))
    seconds["review"] = time.perf_counter() - t0

    # ----------------------------------------------------- 5. in the model
    t0 = time.perf_counter()
    by_name = {it.name: it for it in items}
    excluded_by_name = {e["column"]: e for e in excluded}
    reconciliation = []
    for column in contract.columns:
        declared = "numeric" if column in contract.numeric else "categorical"
        row: dict[str, Any] = {"feature": column, "declared": declared,
                               "reason_label": discovery.has_reason_label(column)}
        if column in by_name:
            it = by_name[column]
            row.update({"dtype": it.dtype, "group": it.family, "tests": it.tests,
                        "failed": it.failed, "passed_screening": it.screened,
                        "approved": it.screened and it.labelled})
            if it.screened and it.labelled:
                row["verdict"] = "screened_and_approved"
            elif it.screened:
                row["verdict"] = "screened_not_labelled"
            else:
                row["verdict"] = "failed_screening"
            if discovery.THIN in it.failed:
                row["missing_explained_by"] = _missing_explained_by(population, column,
                                                                    categorical_raw)
        elif column in excluded_by_name:
            e = excluded_by_name[column]
            row.update({"verdict": "excluded_at_raw", "passed_screening": False, "approved": False,
                        "exclusion": {k: e[k] for k in ("reason", "why", "detail")}})
        else:
            row.update({"verdict": "not_in_data", "passed_screening": False, "approved": False})
        row["governance"] = {"proposal": governance["admitted_proposals"].get(column),
                             "approval": governance["approvals"].get(column)}
        reconciliation.append(row)
    verdicts: dict[str, int] = {}
    for row in reconciliation:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    in_model_approved = sum(1 for r in reconciliation if r["approved"])
    in_model_screened = sum(1 for r in reconciliation if r["passed_screening"])
    seconds["in_model"] = time.perf_counter() - t0

    latent = [c for c in schemas.LATENT_COLUMNS if c in frame.columns]
    audit = _latent_audit(sample_frame, raw, latent)

    raw_passed = sum(1 for it in passed if it.kind == "raw")
    derived_passed = len(passed) - raw_passed
    raw_approved = sum(1 for it in approved if it.kind == "raw")
    seconds["total"] = time.perf_counter() - started
    seconds = {k: round(v, 2) for k, v in seconds.items()}

    funnel = [
        {"stage": "raw_attributes", "label": "Raw attributes", "count": len(raw),
         "detail": {"numeric": len(numeric_raw), "categorical": len(categorical_raw),
                    "excluded": len(excluded)}},
        {"stage": "derived_candidates", "label": "Derived candidates", "count": len(candidates),
         "detail": by_family},
        {"stage": "passed_screening", "label": "Passed screening", "count": len(passed),
         "detail": {"raw": raw_passed, "derived": derived_passed,
                    "tested": len(items)}},
        {"stage": "approved", "label": "Approved in review", "count": len(approved),
         "detail": {"raw": raw_approved, "derived": len(approved) - raw_approved}},
        {"stage": "in_model", "label": f"In the {DISPLAY_NAMES.get(model, model)} model",
         "count": len(contract.columns),
         "detail": {"screened_and_approved": in_model_approved,
                    "passed_screening": in_model_screened,
                    "failed_or_excluded": len(contract.columns) - in_model_screened}},
    ]

    return {
        "model": model, "display_name": DISPLAY_NAMES.get(model, model),
        "target": target,
        "target_kind": (None if target_series is None else
                        "binary" if target_series.nunique() <= 2 else "continuous"),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"dataset": int(len(frame)), "population": int(len(population)),
                 "population_rule": population_rule, "correlation_sample": n_sample,
                 "seed": seed, "source": source},
        "thresholds": rules,
        "seconds": seconds,
        "funnel": funnel,
        "stages": {
            "raw_attributes": {
                "count": len(raw), "numeric": numeric_raw, "categorical": categorical_raw,
                "attributes": raw, "exclusions": excluded,
                "excluded_by_reason": {k: {"count": v, "why": EXCLUSIONS[k]}
                                       for k, v in sorted(by_reason.items())},
                "sentinels": sentinel_notes, "seconds": seconds["raw_attributes"]},
            "derived_candidates": {
                "count": len(candidates), "by_family": by_family,
                "generation": generation, "inputs": len(numeric_raw),
                "seconds": seconds["derived_candidates"]},
            "screening": {
                "tested": len(items), "passed": len(passed),
                "passed_raw": raw_passed, "passed_derived": derived_passed,
                "refusals_any": any_counts, "refusals_first": first_counts,
                "screens": {c: {"label": SCREENS[c], "why": SCREEN_WHY[c]} for c in SCREEN_ORDER},
                "seconds": seconds["screening"]},
            "review": {
                "approved": len(approved), "approved_raw": raw_approved,
                "approved_derived": len(approved) - raw_approved,
                "approved_names": [it.name for it in approved],
                "approved_not_in_model": [it.name for it in approved if not it.in_contract],
                "refused_no_reason_label": len(passed) - len(approved),
                "unlabelled_raw_attributes": [_row(it) for it in unlabelled_raw],
                "governance": {**governance,
                               "approved_without_label": sorted(
                                   n for n in governed if not discovery.has_reason_label(n))},
                "rule": ("approved = passed every screen AND an authored entry in "
                         "feature_config.reason_code_labels. A governance approval is shown "
                         "as evidence; on its own, without the label, it does not count."),
                "seconds": seconds["review"]},
            "in_model": {
                "count": len(contract.columns), "screened_and_approved": in_model_approved,
                "by_verdict": verdicts, "reconciliation": reconciliation,
                "seconds": seconds["in_model"]},
        },
        "rejected_unexplainable": [_row(it) for it in rejected],
        "latent_audit": {
            "rows": audit,
            "note": ("Latent columns used to evaluate only - this changes no disposition and no "
                     "screen. An association near 1 means the attribute is close to a copy of "
                     "the synthetic truth, which excluding the latent column does not remove.")},
        "method": _method(n_sample, seed, len(population), target, rules),
        "fingerprint": stamp,
        "candidates": [it.as_dict() for it in items],
    }


def _row(it: Item) -> dict[str, Any]:
    return {"name": it.name, "recipe": it.recipe, "family": it.family, "inputs": it.inputs,
            "signal": it.tests.get("signal"), "signal_test": it.tests.get("signal_test"),
            "overlap": it.tests.get("overlap"), "overlap_with": it.tests.get("overlap_with"),
            "psi": it.tests.get("psi"), "coverage": it.tests.get("coverage")}


def _missing_explained_by(population: pd.DataFrame, column: str,
                          categorical: list[str]) -> str | None:
    """A flag whose True rows are exactly this column's missing rows, if one exists.

    Missingness that a flag in the same record explains is structural - a
    new-to-credit file has no bureau score - and a reader should be told so next
    to the coverage failure rather than left to guess."""
    missing = population[column].isna()
    for flag in categorical:
        values = population[flag]
        if not pd.api.types.is_bool_dtype(values):
            continue
        if bool((values.astype(bool) == missing).mean() >= 0.999):
            return flag
    return None


def _method(sample: int, seed: int, population: int, target: str | None,
            rules: dict[str, float]) -> dict[str, str]:
    return {
        "signal": (f"Spearman with {target} over all {population:,} population rows (exact, "
                   f"pairwise-complete). Categoricals: bias-corrected Cramer's V against the "
                   f"target, continuous targets binned into deciles."
                   if target else "no target for this contract, so no signal screen"),
        "redundancy": (f"Spearman on a {sample:,}-row sample (seed {seed}): ranks per column, "
                       f"Pearson over pairwise-complete rows. Contract features are checked "
                       f"against stronger contract features; everything else against every "
                       f"contract feature and every stronger feature already kept, so the "
                       f"stronger of a correlated pair survives. Categoricals: bias-corrected "
                       f"Cramer's V against other categoricals. Categorical-to-numeric "
                       f"redundancy is not measured."),
        "stability": ("PSI between the older and newer halves of the population by application "
                      "date: ten quantile buckets for numerics (observed levels when ten or "
                      "fewer), level shares for categoricals. Too few rows reads as not "
                      "measurable, which does not fail the screen."),
        "coverage": f"share of population rows with a finite value; floor {rules['min_coverage']}",
        "trend": (f"x - median(x) over applications in the prior {discovery.TREND_WINDOW.lower()} "
                  f"(strictly earlier dates, at least {discovery.TREND_MIN_PERIODS} of them), "
                  f"computed over the whole application stream - never over an outcome. There "
                  f"is no per-loan history in the data, so this is the market's trend, not the "
                  f"customer's."),
        "sentinels": ("; ".join(f"{column}: {rule}" for column, rule in sentinel_rules()[0].items())
                      + f" - from {sentinel_rules()[1]}; read as missing before anything else"),
        "univariate": ("Every screen is univariate. A feature that matters only in combination "
                       "with others can fail the signal screen and still earn its place in a "
                       "fitted model; the in-model reconciliation is where that shows, and the "
                       "model's own attributions are the multivariate check."),
    }


# ------------------------------------------------------- saving and freshness

_DIGESTS: dict[tuple[str, int, int], str] = {}
FINGERPRINT_SOURCES = ["features/funnel.py", "features/discovery.py", "features/profiling.py",
                       "features/stage1_features.py", "features/stage2_features.py",
                       "features/sales_features.py", "kernel/statistics.py",
                       "ingestion/loader.py", "ingestion/validator.py", "ingestion/schemas.py"]


def _file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return _DIGESTS[key]


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def fingerprint(model: str, sample: int = DEFAULT_SAMPLE, seed: int | None = None,
                families: list[str] | None = None,
                thresholds: dict[str, float] | None = None,
                approvals: dict[str, Any] | None = None) -> dict[str, str | None]:
    """Everything the funnel's numbers depend on, as short digests.

    Read from disk rather than from the cached config, so a long-running process
    notices a change it has not reloaded.
    """
    package = Path(__file__).resolve().parents[1]
    data_cfg = model_config()["data"]
    with open(CONFIG_DIR / "feature_config.yaml", "r", encoding="utf-8") as fh:
        features = yaml.safe_load(fh) or {}
    evidence = _approval_evidence(approvals)
    return {
        "data": _file_digest(resolve(data_cfg["raw_path"])),
        "contract": _digest({"block": features.get(model), "sentinels": features.get("sentinels")}),
        "reason_labels": _digest(features.get("reason_code_labels")),
        "approvals": _digest({k: evidence[k] for k in ("consulted", "admitted_proposals",
                                                        "approvals")}),
        "code": _digest([_file_digest(package / p) for p in FINGERPRINT_SOURCES]),
        "params": _digest({"sample": int(sample),
                           "seed": int(model_config().get("seed", 42) if seed is None else seed),
                           "families": list(families or FAMILIES),
                           "thresholds": {**discovery.DEFAULTS, **EXTRA_THRESHOLDS,
                                          **(thresholds or {})},
                           "time_column": data_cfg.get("time_column")}),
    }


FINGERPRINT_LABELS = {
    "data": "the training data changed",
    "contract": "the model's contract or the sentinel rules changed",
    "reason_labels": "reason labels were added or edited",
    "approvals": "the feature approvals on record changed",
    "code": "the funnel, builder or loader code changed",
    "params": "sample, seed, families or thresholds differ",
}


def report_path(model: str, directory: Path | None = None) -> Path:
    return (directory or REPORT_DIR) / f"funnel_{model}.json"


def save(report: dict[str, Any], directory: Path | None = None) -> Path:
    """Indented, except the per-item records: one compact line each, so a file
    that keeps every tested candidate stays small and still diffs line by line."""
    path = report_path(report["model"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in report.items() if k != "candidates"}
    text = json.dumps(body, indent=1, default=str)
    if "candidates" in report:
        rows = ",\n  ".join(json.dumps(c, separators=(",", ":"), default=str)
                            for c in report["candidates"])
        text = text[: text.rfind("}")].rstrip() + f',\n "candidates": [\n  {rows}\n ]\n}}'
    path.write_text(text + "\n")
    return path


def load_saved(model: str, directory: Path | None = None) -> dict[str, Any] | None:
    path = report_path(model, directory)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def _saved_params(saved: dict[str, Any]) -> dict[str, Any]:
    rows = saved.get("rows") or {}
    families = list((saved.get("stages", {}).get("derived_candidates", {})
                     .get("by_family") or {}).keys())
    return {"sample": rows.get("correlation_sample", DEFAULT_SAMPLE), "seed": rows.get("seed"),
            "families": families or None, "thresholds": saved.get("thresholds")}


def freshness(saved: dict[str, Any], approvals: dict[str, Any] | None = None) -> dict[str, Any]:
    """A saved report is fresh when every input it was computed from is unchanged:
    the raw data file's bytes, the model's contract block and the sentinel rules,
    the reason labels, the feature approvals on record for this model, the funnel,
    builder and loader source, and the run parameters. Age alone is not a
    criterion - a week-old report over unchanged inputs is exact, a minute-old one
    over a changed contract is wrong."""
    current = fingerprint(saved["model"], approvals=approvals, **_saved_params(saved))
    recorded = saved.get("fingerprint") or {}
    changed = [k for k in current if recorded.get(k) != current[k]]
    return {"fresh": not changed, "changed": changed,
            "because": [FINGERPRINT_LABELS.get(k, k) for k in changed]}


FRESHNESS_RULE = " ".join((freshness.__doc__ or "").split())
_MEMO: dict[tuple[str, str], dict[str, Any]] = {}


def serve(model: str, refresh: bool = False, directory: Path | None = None,
          approvals_source: Callable[[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    """The saved report when it is fresh; otherwise computed now (and memoised in
    this process against its fingerprint). Never writes the report file - the
    script does that, so a GET has no side effects on disk.

    ``approvals_source(model)`` supplies the governance evidence; the API passes
    one that reads the store.
    """
    spec(model)                                                  # KeyError if unknown
    approvals = approvals_source(model) if approvals_source else None
    saved = load_saved(model, directory)
    params: dict[str, Any] = {"sample": DEFAULT_SAMPLE, "seed": None}
    status: dict[str, Any] = {"path": str(report_path(model, directory)), "rule": FRESHNESS_RULE}
    if saved is not None:
        saved_params = _saved_params(saved)
        params = {"sample": saved_params["sample"], "seed": saved_params["seed"]}
        check = freshness(saved, approvals)
        if check["fresh"] and not refresh:
            return {**saved, "served": {**status, "from": "saved_report", "fresh": True}}
        status["stale_because"] = check["because"]
    else:
        status["stale_because"] = ["no saved report"]
    if refresh:
        status["refresh_requested"] = True

    key = (model, _digest(fingerprint(model, approvals=approvals, **params)))
    if key not in _MEMO or refresh:
        _MEMO[key] = run(model, approvals=approvals, **params)
    return {**_MEMO[key], "served": {**status, "from": "computed", "fresh": True}}


def text_summary(report: dict[str, Any], top: int = 12) -> str:
    """The funnel and the rejected table, as the slide reads them."""
    lines = [f"{report['display_name']} ({report['model']}) - target {report['target']}, "
             f"{report['rows']['population']:,} rows ({report['rows']['population_rule']}), "
             f"{report['seconds']['total']}s"]
    for step in report["funnel"]:
        d = step["detail"]
        if step["stage"] == "raw_attributes":
            note = (f"{d['numeric']} numeric, {d['categorical']} categorical; "
                    f"{d['excluded']} columns excluded")
        elif step["stage"] == "derived_candidates":
            note = ", ".join(f"{k} {v:,}" for k, v in d.items())
        elif step["stage"] == "passed_screening":
            note = f"of {d['tested']:,} tested: {d['raw']} raw, {d['derived']} derived"
        elif step["stage"] == "approved":
            note = f"{d['raw']} raw, {d['derived']} derived - reason label authored"
        else:
            note = (f"{d['screened_and_approved']} screened and approved; "
                    f"{d['failed_or_excluded']} failed a screen or were excluded")
        lines.append(f"  {step['label']:<38}{step['count']:>7,}   ({note})")

    in_model = report["stages"]["in_model"]["reconciliation"]
    failing = [r for r in in_model if r["verdict"] != "screened_and_approved"]
    if failing:
        lines.append("  Contract features that did not clear the funnel:")
        for r in failing:
            why = (", ".join(r.get("failed") or []) or
                   (r.get("exclusion") or {}).get("reason") or r["verdict"])
            extra = f" (missing exactly when {r['missing_explained_by']})" \
                if r.get("missing_explained_by") else ""
            lines.append(f"    {r['feature']:<34}{r['verdict']:<24}{why}{extra}")

    rejected = report["rejected_unexplainable"]
    lines.append(f"  Rejected: strong, stable and unexplainable - top {min(top, len(rejected))} "
                 f"of {len(rejected)}")
    lines.append(f"    {'candidate':<56}{'signal':>8}{'overlap':>9}{'PSI':>8}")
    for row in rejected[:top]:
        lines.append(f"    {row['name'][:55]:<56}{_fmt(row['signal']):>8}"
                     f"{_fmt(row['overlap']):>9}{_fmt(row['psi']):>8}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
