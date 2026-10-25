"""Agentic feature engineering: proposes, tests, and refuses to admit.

The split this module is built around, stated once because everything else
follows from it:

    **Feature discovery can be automated. Feature admission cannot.**

A discovered feature is a hypothesis. Admitting it into the contract changes what
every future model is fitted on, what appears in a customer-facing explanation,
and what a regulator will ask about - so it is a decision a human takes through
the approval path, not a threshold a search clears.

What the agent does:

* **Generates** candidates only from columns already inside the stage's contract.
  A ratio of two Stage-1 fields is a Stage-1 candidate; a ratio touching a
  post-assessment field cannot even be constructed for Stage 1. Leakage is
  prevented by construction rather than caught by a test afterwards.
* **Tests** each candidate for signal, redundancy, stability and coverage, and
  records every number rather than a verdict.
* **Refuses** outright on four grounds, which are not scores: leakage, redundancy
  with a feature already in the contract, instability across time, and - the one
  that is not statistical - **no human-authored reason label**.

That last refusal is the point. `feature_config.reason_code_labels` is what turns
an attribution into a sentence a branch can read out to a customer. A feature with
no label cannot be explained, so it cannot be admitted however strong it is. The
agent will not write the label itself: a machine-generated explanation of a
machine-generated feature is a circle with a customer at the end of it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import feature_config, model_config
from ..ingestion import schemas
from ..kernel.statistics import psi
from ..logging_utils import get_logger
from .feature_pipeline import spec

log = get_logger(__name__)

# Rejections. Each is a refusal, not a low score - a candidate rejected for any of
# these cannot be promoted by scoring better elsewhere.
LEAKAGE = "leakage"
REDUNDANT = "redundant"
UNSTABLE = "unstable"
THIN = "thin_coverage"
NO_LABEL = "no_reason_label"
DEGENERATE = "degenerate"

REFUSALS = {
    LEAKAGE: "built from a column the stage may not see",
    REDUNDANT: "carries what a contract feature already carries",
    UNSTABLE: "its distribution moves between time windows",
    THIN: "too few rows carry a usable value",
    DEGENERATE: "almost no variation, or undefined for most rows",
    NO_LABEL: "no human-authored reason label, so it cannot be explained to a customer",
}

DEFAULTS = {
    "max_abs_correlation": 0.92,   # with any single contract feature
    "max_psi": 0.25,               # between the older and newer halves
    "min_coverage": 0.90,          # share of rows with a finite value
    "min_std": 1e-9,
    "min_abs_target_corr": 0.02,   # below this it is noise, not signal
}


# --------------------------------------------------------------- the generators
# Each takes the source frame plus the columns the stage is allowed to use, and
# yields (name, series, recipe). The recipe is what a human reads when deciding
# whether the feature means anything - `income_to_emi` is a claim about the
# world; `f_0417` is not.

@dataclass
class Candidate:
    name: str
    recipe: str
    inputs: list[str]
    generator: str
    values: pd.Series = field(repr=False, default=None)
    tests: dict[str, Any] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)

    @property
    def admissible(self) -> bool:
        return not self.refusals

    def as_dict(self, include_values: bool = False) -> dict[str, Any]:
        out = {"name": self.name, "recipe": self.recipe, "inputs": self.inputs,
               "generator": self.generator, "tests": self.tests,
               "refusals": [{"code": r, "why": REFUSALS[r]} for r in self.refusals],
               "admissible": self.admissible}
        if include_values and self.values is not None:
            out["sample"] = [float(v) for v in self.values.dropna().head(5)]
        return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def gen_ratios(df: pd.DataFrame, columns: list[str]) -> list[Candidate]:
    """Ratios between pairs. The most productive family in credit, because most
    credit judgements are about proportion rather than magnitude."""
    out = []
    for a, b in itertools.permutations(columns, 2):
        if df[a].le(0).all() or df[b].le(0).all():
            continue
        out.append(Candidate(
            name=f"{a}_per_{b}", recipe=f"{a} / {b}", inputs=[a, b], generator="ratio",
            values=_safe_ratio(df[a], df[b])))
    return out


def gen_gaps(df: pd.DataFrame, columns: list[str]) -> list[Candidate]:
    """Differences between pairs that are on the same scale."""
    out = []
    for a, b in itertools.combinations(columns, 2):
        if not _comparable(df[a], df[b]):
            continue
        out.append(Candidate(
            name=f"{a}_minus_{b}", recipe=f"{a} - {b}", inputs=[a, b], generator="gap",
            values=df[a] - df[b]))
    return out


def _comparable(left: pd.Series, right: pd.Series) -> bool:
    """Same order of magnitude. Subtracting a rupee amount from a percentage is
    arithmetic that produces a number and means nothing."""
    scales = []
    for series in (left, right):
        finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty or finite.abs().max() <= 0:
            return False
        scales.append(np.log10(finite.abs().max()))
    return abs(scales[0] - scales[1]) <= 1.0


def gen_log(df: pd.DataFrame, columns: list[str]) -> list[Candidate]:
    """Log of strictly positive, heavily skewed amounts."""
    out = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        positive = values.dropna()
        if positive.empty or positive.le(0).any() or abs(positive.skew()) < 1.0:
            continue
        out.append(Candidate(
            name=f"log_{column}", recipe=f"log1p({column})", inputs=[column],
            generator="log", values=np.log1p(values)))
    return out


def gen_interactions(df: pd.DataFrame, columns: list[str]) -> list[Candidate]:
    """Products. Kept last and kept few: an interaction is the hardest family to
    explain to a customer, so the bar for admitting one is correspondingly high."""
    out = []
    for a, b in itertools.combinations(columns, 2):
        out.append(Candidate(
            name=f"{a}_x_{b}", recipe=f"{a} * {b}", inputs=[a, b], generator="interaction",
            values=pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")))
    return out


TREND_WINDOW = "90D"
TREND_MIN_PERIODS = 30


def gen_trend(df: pd.DataFrame, columns: list[str], time: pd.Series | None = None,
              window: str = TREND_WINDOW, min_periods: int = TREND_MIN_PERIODS) -> list[Candidate]:
    """Each value against the recent book: ``x - median(x)`` over the applications
    received in the ``window`` before this one.

    The data has an application date and no per-loan time series, so a "trend"
    here is the *market's* trend, not the customer's: is this competitor quote,
    this ticket, this valuation high or low for what the book has been seeing?

    Leakage-safe by construction, for two reasons. The window is ``closed="left"``,
    so it holds only applications from strictly earlier dates - nothing filed the
    same day, nothing later. And it is computed over the attribute itself, never
    over a target or an outcome, so no row learns how an earlier loan turned out.
    Low-cardinality columns (flags, small counts) are skipped: the trailing median
    of a flag is almost always 0 and the "trend" would be the flag again.
    """
    if time is None:
        return []
    stamps = pd.to_datetime(pd.Series(time, index=df.index), errors="coerce")
    if stamps.isna().any() or len(stamps) < min_periods:
        return []
    order = np.argsort(stamps.to_numpy(), kind="stable")
    sorted_index = pd.DatetimeIndex(stamps.to_numpy()[order])
    label = window.lower()

    out = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.nunique() <= 10:
            continue
        ordered = pd.Series(values.to_numpy(float)[order], index=sorted_index)
        trailing = ordered.rolling(window, closed="left", min_periods=min_periods).median()
        baseline = np.empty(len(values))
        baseline[order] = trailing.to_numpy()
        out.append(Candidate(
            name=f"{column}_vs_trailing_{label}",
            recipe=f"{column} - median({column} over applications in the prior {label})",
            inputs=[column], generator="trend",
            values=values - pd.Series(baseline, index=values.index)))
    return out


GENERATORS: dict[str, Callable[..., list[Candidate]]] = {
    "ratio": gen_ratios,
    "gap": gen_gaps,
    "log": gen_log,
    "interaction": gen_interactions,
    # Needs the application date, so it is only run when a caller passes one -
    # see ``discover`` and ``features.funnel``. Called without it, it builds nothing.
    "trend": gen_trend,
}


# ------------------------------------------------------------------- the tests


def _psi_halves(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 200:
        return float("nan")
    cut = len(clean) // 2
    return psi(clean.iloc[:cut].to_numpy(), clean.iloc[cut:].to_numpy())


def _target_correlation(values: pd.Series, target: pd.Series) -> float:
    frame = pd.concat([pd.to_numeric(values, errors="coerce"),
                       pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    if len(frame) < 50 or frame.iloc[:, 0].std() == 0:
        return float("nan")
    return float(frame.corr(method="spearman").iloc[0, 1])


def _max_contract_correlation(values: pd.Series, contract: pd.DataFrame) -> tuple[float, str]:
    best, against = 0.0, ""
    clean = pd.to_numeric(values, errors="coerce")
    for column in contract.columns:
        frame = pd.concat([clean, pd.to_numeric(contract[column], errors="coerce")],
                          axis=1).dropna()
        if len(frame) < 50 or frame.iloc[:, 0].std() == 0 or frame.iloc[:, 1].std() == 0:
            continue
        value = abs(float(frame.corr(method="spearman").iloc[0, 1]))
        if value > best:
            best, against = value, column
    return best, against


def has_reason_label(name: str) -> bool:
    """Authored, not defaulted. ``feature_pipeline.reason_label`` falls back to a
    title-cased column name, which is a placeholder and must not read as a label.
    """
    return name in (feature_config().get("reason_code_labels") or {})


def evaluate(candidate: Candidate, contract: pd.DataFrame, target: pd.Series,
             thresholds: dict[str, float] | None = None) -> Candidate:
    rules = {**DEFAULTS, **(thresholds or {})}
    values = pd.to_numeric(candidate.values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    coverage = float(values.notna().mean())
    std = float(values.std(skipna=True) or 0.0)
    psi_value = _psi_halves(values)
    target_corr = _target_correlation(values, target)
    redundancy, against = _max_contract_correlation(values, contract)

    candidate.tests = {
        "coverage": round(coverage, 4),
        "std": None if not np.isfinite(std) else round(std, 6),
        "psi_between_halves": None if not np.isfinite(psi_value) else round(psi_value, 4),
        "target_spearman": None if not np.isfinite(target_corr) else round(target_corr, 4),
        "max_correlation_with_contract": round(redundancy, 4),
        "most_correlated_with": against,
        "has_reason_label": has_reason_label(candidate.name),
    }

    refusals = []
    if coverage < rules["min_coverage"]:
        refusals.append(THIN)
    if not np.isfinite(std) or std < rules["min_std"]:
        refusals.append(DEGENERATE)
    if np.isfinite(psi_value) and psi_value > rules["max_psi"]:
        refusals.append(UNSTABLE)
    if redundancy > rules["max_abs_correlation"]:
        refusals.append(REDUNDANT)
    if not candidate.tests["has_reason_label"]:
        refusals.append(NO_LABEL)
    candidate.refusals = refusals
    return candidate


# ------------------------------------------------------------------- the agent


@dataclass
class DiscoveryReport:
    stage: str
    target: str
    rows: int
    considered: int
    candidates: list[Candidate]
    thresholds: dict[str, float]

    def ranked(self) -> list[Candidate]:
        def key(c: Candidate) -> float:
            value = (c.tests.get("target_spearman") or 0.0)
            return -abs(value)
        return sorted(self.candidates, key=key)

    def shortlist(self, limit: int = 10) -> list[Candidate]:
        """Candidates that failed only for want of a label.

        The most useful output of the whole exercise: strong, stable, independent
        and unexplainable. Each one is a question for a human - *is there an honest
        sentence to put next to this?* - and the answer is often no.
        """
        out = [c for c in self.ranked()
               if c.refusals == [NO_LABEL]
               and abs(c.tests.get("target_spearman") or 0.0) >= DEFAULTS["min_abs_target_corr"]]
        return out[:limit]

    def admissible(self) -> list[Candidate]:
        return [c for c in self.ranked() if c.admissible]

    def as_dict(self) -> dict[str, Any]:
        by_refusal: dict[str, int] = {}
        for candidate in self.candidates:
            for refusal in candidate.refusals:
                by_refusal[refusal] = by_refusal.get(refusal, 0) + 1
        return {
            "stage": self.stage, "target": self.target, "rows": self.rows,
            "generated": self.considered, "evaluated": len(self.candidates),
            "thresholds": self.thresholds,
            "refusals": {k: {"count": v, "why": REFUSALS[k]} for k, v in sorted(by_refusal.items())},
            "admissible": [c.as_dict() for c in self.admissible()],
            "awaiting_a_reason_label": [c.as_dict() for c in self.shortlist()],
            "top_by_signal": [c.as_dict() for c in self.ranked()[:15]],
            "note": ("Nothing here has been admitted. A candidate that clears every test still "
                     "needs a human-authored reason label and an approved change to "
                     "feature_config.yaml - the agent proposes, it does not widen the "
                     "contract."),
        }


def allowed_columns(stage: str) -> list[str]:
    """The numeric columns this stage may build from.

    Drawn from the contract itself, so a candidate touching a forbidden column
    cannot be constructed rather than being generated and then rejected.
    """
    contract = spec(stage)
    forbidden = set(schemas.OUTCOME_COLUMNS) | set(schemas.LATENT_COLUMNS) | \
        set(schemas.DECISION_COLUMNS)
    if stage == "stage1":
        forbidden |= set(schemas.POST_ASSESSMENT_COLUMNS)
    return [c for c in contract.numeric if c not in forbidden]


def discover(df: pd.DataFrame, stage: str = "stage2", target: str | None = None,
             generators: list[str] | None = None,
             thresholds: dict[str, float] | None = None,
             max_candidates: int = 400) -> DiscoveryReport:
    """Generate, test and refuse. Admits nothing."""
    target = target or schemas.TARGETS.get(stage, "sanctioned_rate")
    if target not in df.columns:
        raise ValueError(f"target {target!r} is not in the frame")

    columns = [c for c in allowed_columns(stage) if c in df.columns]
    if len(columns) < 2:
        raise ValueError(f"stage {stage!r} has fewer than two usable numeric columns")

    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    chosen = generators or ["ratio", "gap", "log", "interaction"]

    candidates: list[Candidate] = []
    for name in chosen:
        generator = GENERATORS.get(name)
        if generator is None:
            log.warning("unknown generator %r, skipped", name)
            continue
        if name == "trend":
            time_column = model_config()["data"].get("time_column", "application_date")
            candidates.extend(gen_trend(numeric, columns, time=df.get(time_column)))
            continue
        candidates.extend(generator(numeric, columns))

    considered = len(candidates)
    if considered > max_candidates:
        log.info("generated %s candidates, evaluating the first %s", considered, max_candidates)
        candidates = candidates[:max_candidates]

    # Drop anything that duplicates a contract feature by name before testing.
    existing = set(spec(stage).columns)
    candidates = [c for c in candidates if c.name not in existing]

    target_series = df[target]
    evaluated = [evaluate(c, numeric, target_series, thresholds) for c in candidates]

    return DiscoveryReport(stage=stage, target=target, rows=len(df), considered=considered,
                           candidates=evaluated, thresholds={**DEFAULTS, **(thresholds or {})})


class FeatureDiscoveryAgent:
    """The agent wrapper, so the control plane can show it beside the others.

    It is deliberately **design-time**: nothing it produces reaches a priced
    decision. It generates candidates, tests them, and returns refusals. It does
    not record proposals either - a human does that through the API, under their
    own identity, because a proposal is a claim that someone thought a candidate
    was worth looking at.
    """

    name = "feature_discovery"

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or {}

    def assess(self, df: pd.DataFrame, stage: str = "stage2",
               generators: list[str] | None = None) -> DiscoveryReport:
        return discover(df, stage=stage, generators=generators, thresholds=self.thresholds)


def summary() -> dict[str, Any]:
    """What the agent is, and what it is not allowed to do."""
    return {
        "generators": {
            "ratio": "a / b for positive pairs - most credit judgements are proportions",
            "gap": "a - b where the two are on the same scale",
            "log": "log1p of strictly positive, skewed amounts",
            "interaction": "a * b, the hardest family to explain and therefore the highest bar",
            "trend": (f"x - median(x) over applications in the prior {TREND_WINDOW.lower()}: "
                      "the market's trend, from strictly earlier applications only"),
        },
        "refusals": REFUSALS,
        "thresholds": DEFAULTS,
        "admission": ("The agent cannot admit a feature. Admission is an approved change to "
                      "config/feature_config.yaml, requiring a human-authored reason label "
                      "first - see the 'feature' artefact type in config/governance.yaml."),
        "leakage": ("Candidates are built only from columns already inside the stage's "
                    "contract, so a Stage-1 candidate cannot touch a post-assessment field. "
                    "Prevented by construction, not caught afterwards."),
    }


# ------------------------------------------------------- when research is needed
# The Research Agent is conditional, and the condition is the valuable half. Most
# of what a feature engineer does is a known method on a known problem; routing
# that through an LLM is slower, dearer and less predictable than executing it.
# These triggers decide when a question is genuinely outside the library - and
# they are all computed from the evidence, with no model call.

RESEARCH_TRIGGERS = {
    "unusual_distribution": "the shape is not one the proven transformations handle",
    "non_monotonic_relationship": "the WoE moves up and down across bins, so no simple "
                                  "transformation captures the relationship",
    "methods_disagree": "correlation says one thing and mutual information says another, "
                        "which means the relationship is not linear",
    "unexpected_strength": "predictive power high enough that the question is whether it "
                           "leaks, not whether to use it",
    "high_cardinality": "too many categories for one-hot and too few rows per category for "
                        "target encoding to be safe",
    "unstable": "the distribution moves between windows and the cause is not obvious",
}


def research_needed(examination: dict[str, Any]) -> dict[str, Any]:
    """Decide whether this feature is worth a research call. Usually it is not."""
    triggers: list[str] = []
    profile = examination.get("profile") or {}
    iv = examination.get("information_value") or {}
    stability = examination.get("stability") or {}

    if profile.get("kind") == "numeric":
        skew = profile.get("skew")
        kurtosis = profile.get("kurtosis")
        if skew is not None and abs(skew) > 3:
            triggers.append("unusual_distribution")
        elif kurtosis is not None and kurtosis > 20:
            triggers.append("unusual_distribution")
        if iv.get("iv") is not None and iv.get("monotonic_woe") is False and iv["iv"] > 0.05:
            triggers.append("non_monotonic_relationship")
        if iv.get("iv") is not None and iv["iv"] > 0.5:
            triggers.append("unexpected_strength")
        mi = examination.get("mutual_information")
        auc = (examination.get("separation") or {}).get("effect_size_auc")
        if mi is not None and auc is not None and mi > 0.05 and abs(auc - 0.5) < 0.02:
            triggers.append("methods_disagree")
    elif profile.get("high_cardinality"):
        triggers.append("high_cardinality")

    psi = stability.get("psi")
    if psi is not None and psi > 0.25:
        triggers.append("unstable")

    return {
        "research_needed": bool(triggers),
        "triggers": [{"code": t, "why": RESEARCH_TRIGGERS[t]} for t in dict.fromkeys(triggers)],
        "note": ("No trigger means the library already covers this: a known method on a known "
                 "problem executes directly. Routing that through a model would be slower, "
                 "dearer and less predictable, for an answer that is already known."
                 if not triggers else
                 "A trigger is a question for a human or a research call, not an instruction "
                 "to generate code. See docs on why generated code is not executed here."),
    }


# ------------------------------------------------------- the function registry
# The proven transformations, declared with what each is for and when it applies.
# A registry is what makes "is a function already available?" answerable before
# anything is generated - and it is the thing a code-generation step would extend.

FUNCTION_REGISTRY: dict[str, dict[str, Any]] = {
    "ratio": {"applies_to": "two positive numeric columns on any scale",
              "produces": "a / b", "family": "proportion",
              "why": "most credit judgements are about proportion, not magnitude"},
    "gap": {"applies_to": "two numeric columns within one order of magnitude",
            "produces": "a - b", "family": "difference",
            "why": "declared minus verified is a different claim from their ratio"},
    "log": {"applies_to": "a strictly positive column with |skew| > 1",
            "produces": "log1p(x)", "family": "transformation",
            "why": "compresses a right tail so a linear term is meaningful"},
    "interaction": {"applies_to": "any two numeric columns",
                    "produces": "a * b", "family": "interaction",
                    "why": "captures a joint effect - and is the hardest family to explain, "
                           "so it carries the highest admission bar"},
    "trend": {"applies_to": "a numeric column with more than ten distinct values, plus the "
                            "application date",
              "produces": f"x - median(x over the prior {TREND_WINDOW.lower()})",
              "family": "temporal",
              "why": "there is no per-loan history, so the trend that exists is the market's - "
                     "a quote or a valuation is read against what the book saw just before"},
}


def registry_covers(generator: str) -> bool:
    return generator in FUNCTION_REGISTRY
