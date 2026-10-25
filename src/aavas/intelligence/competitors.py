"""Competitor analysis from what the notes say, joined to what then happened.

Two products, both descriptive history for a human, neither a pricing input:

* **Market view** (``CompetitorAnalysis.summary`` / ``profile``) - per canonical
  lender: how many of our files mention them, where, at what quoted rates, how
  far under our sanctioned rate and the card rate, how much of it is
  documented, our win rate when they are present and how it falls as they
  undercut us by more, how often and how deep we override when they are cited,
  and the monthly trend. Lenders that appear only in the text - the structured
  ``competitor_name`` never had a slot for them - get a "who is new in our
  market" view; lender-shaped names the registry does not know are listed for
  review.
* **Per-loan context** (``context_for``) - for the file being priced: each
  lender the notes (or the structured record) name, that lender's historical
  quotes in the same city (narrowed to segment / grade when the sample allows),
  a plausibility check against the city's 5th percentile, our historical win
  rate against them at a similar gap, and comparable past negotiations.

Joining outcomes is legitimate here because this is analysis of closed files,
not a feature: nothing in this module is read by a model or by the optimiser.
Every number carries its ``n``; a cell with too few observations is suppressed
(value ``None``, ``suppressed: true``) rather than shown as noise. Win rates are
observational - the price we offered already responded to the competitor - so
they describe the book, they do not estimate a causal effect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .note_extraction import DOCUMENTED, UNDOCUMENTED, UNKNOWN, NoteExtractor, extract_current
from .registry import UNRECOGNISED, CompetitorRegistry, default_registry, load_config

log = get_logger(__name__)

LOAN_COLUMNS = ["loan_id", "city", "state", "branch_id", "segment", "risk_grade", "sanctioned_rate",
                "policy_card_rate", "disbursed", "was_overridden", "override_bps", "application_month",
                "competitor_name"]

PRICING_EFFECT = ("none - competitor context informs the reviewer; it does not enter the price "
                  "(see agents/sales_context_agent.py)")
CAVEATS = [
    "Quotes are what the sales notes say; the evidence mix shows how much of it is documented.",
    "Win and override rates are observational: the price we offered already responded to the "
    "competitor, so they describe the book rather than estimate a causal effect.",
    "Gap = competitor quote minus our sanctioned rate, in bps; negative means they undercut us.",
]


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else round(number, digits)


class CompetitorAnalysis:
    """Historical competitor tables built once from the extraction and the loan book."""

    def __init__(self, mentions: pd.DataFrame, loans: pd.DataFrame,
                 registry: CompetitorRegistry | None = None, settings: dict[str, Any] | None = None):
        self.registry = registry or default_registry()
        cfg = settings if settings is not None else (load_config().get("analysis") or {})
        self.min_n_cell = int(cfg.get("min_n_cell", 20))
        self.min_n_rate = int(cfg.get("min_n_rate", 30))
        self.min_n_benchmark = int(cfg.get("min_n_benchmark", 15))
        self.gap_edges = [float(e) for e in cfg.get("gap_buckets_bps", [-75, -50, -25, 0, 25])]
        self.plausibility_pct = float(cfg.get("plausibility_percentile", 5))
        self.k_comparable = int(cfg.get("comparable_negotiations", 3))

        keep = [c for c in LOAN_COLUMNS if c in loans.columns]
        self.loans = loans[keep].drop_duplicates("loan_id").copy()
        for col in ("disbursed", "was_overridden"):
            if col in self.loans:
                self.loans[col] = self.loans[col].astype(bool)
        if "application_month" not in self.loans and "application_date" in loans:
            self.loans["application_month"] = pd.to_datetime(loans["application_date"]).dt.to_period("M").astype(str)
        self.total_loans = int(len(self.loans))
        self.structured_names = {self.registry.resolve(n) or n
                                 for n in self.loans.get("competitor_name", pd.Series(dtype=str)).dropna().unique()}

        m = mentions.copy()
        m["competitor"] = m["competitor"].fillna("").astype(str)
        m["note_date"] = pd.to_datetime(m["note_date"], errors="coerce")
        self.unrecognised_mentions = m[(m["match"] == UNRECOGNISED)].copy()
        self.mentions = m[(m["competitor"] != "") & (m["match"] != UNRECOGNISED)].copy()
        self.notes_total = None
        self.lc = self._loan_competitor_table()

    # -------------------------------------------------------------- building
    def _loan_competitor_table(self) -> pd.DataFrame:
        """One row per (loan, lender): the latest quote in the notes, the
        evidence on that quote, and the loan's price and outcome."""
        m = self.mentions.sort_values(["loan_id", "note_date", "note_id", "mention_index"])
        if m.empty:
            return pd.DataFrame(columns=["loan_id", "competitor", "quote", "evidence", "notes",
                                         "first_seen", "gap_ours_bps", "gap_card_bps", "gap_bucket"])
        grouped = m.groupby(["loan_id", "competitor"], sort=False)
        base = grouped.agg(notes=("note_id", "nunique"), first_seen=("note_date", "min"),
                           last_seen=("note_date", "max")).reset_index()
        quoted = m[m["rate"].notna()].groupby(["loan_id", "competitor"], sort=False).tail(1)
        quoted = quoted[["loan_id", "competitor", "rate", "evidence"]].rename(columns={"rate": "quote"})
        documented = (m.assign(doc=m["evidence"] == DOCUMENTED).groupby(["loan_id", "competitor"])["doc"]
                      .any().rename("any_documented").reset_index())
        out = base.merge(quoted, on=["loan_id", "competitor"], how="left") \
                  .merge(documented, on=["loan_id", "competitor"], how="left") \
                  .merge(self.loans, on="loan_id", how="inner")
        out["evidence"] = out["evidence"].fillna("")
        out["gap_ours_bps"] = ((out["quote"] - out["sanctioned_rate"]) * 100).round(1)
        out["gap_card_bps"] = ((out["quote"] - out["policy_card_rate"]) * 100).round(1)
        out["gap_bucket"] = self._bucket(out["gap_ours_bps"])
        return out

    def _bucket(self, gaps: pd.Series) -> pd.Series:
        edges = [-np.inf, *self.gap_edges, np.inf]
        return pd.cut(gaps, bins=edges, labels=self.bucket_labels(), right=True).astype(object)

    def bucket_labels(self) -> list[str]:
        e = self.gap_edges
        labels = [f"≤ {e[0]:+.0f}"]
        labels += [f"{a:+.0f} to {b:+.0f}" for a, b in zip(e, e[1:])]
        labels.append(f"> {e[-1]:+.0f}")
        return labels

    def bucket_for(self, gap_bps: float | None) -> str | None:
        if gap_bps is None:
            return None
        value = self._bucket(pd.Series([gap_bps])).iloc[0]
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

    # -------------------------------------------------------------- stats
    def _dist(self, values: pd.Series, min_n: int | None = None, p5: bool = False) -> dict[str, Any]:
        values = pd.to_numeric(values, errors="coerce").dropna()
        n, floor = int(len(values)), (self.min_n_cell if min_n is None else min_n)
        if n < floor or n == 0:
            out = {"n": n, "median": None, "p10": None, "p90": None, "suppressed": n > 0}
            if p5:
                out["p5"] = None
            return out
        out = {"n": n, "median": _round(values.median()), "p10": _round(values.quantile(0.10)),
               "p90": _round(values.quantile(0.90)), "suppressed": False}
        if p5:
            out["p5"] = _round(values.quantile(self.plausibility_pct / 100))
        return out

    def _rate(self, flags: pd.Series, base: float | None = None) -> dict[str, Any]:
        flags = flags.dropna().astype(bool)
        n = int(len(flags))
        if n < self.min_n_rate:
            return {"n": n, "rate": None, "suppressed": n > 0, "base_rate": _round(base, 4)}
        rate = float(flags.mean())
        out = {"n": n, "rate": _round(rate, 4), "suppressed": False, "base_rate": _round(base, 4)}
        if base is not None:
            out["lift_pp"] = _round((rate - base) * 100, 1)
        return out

    @property
    def base_win(self) -> float | None:
        return float(self.loans["disbursed"].mean()) if "disbursed" in self.loans and len(self.loans) else None

    @property
    def base_override(self) -> float | None:
        return float(self.loans["was_overridden"].mean()) if "was_overridden" in self.loans and len(self.loans) else None

    # ---------------------------------------------------------- per lender
    def _row(self, name: str, lc: pd.DataFrame | None = None) -> dict[str, Any]:
        lc = self.lc[self.lc["competitor"] == name] if lc is None else lc
        quoted = lc[lc["quote"].notna()]
        n_loans = int(lc["loan_id"].nunique())
        overridden = lc[lc["was_overridden"]] if "was_overridden" in lc else lc.iloc[0:0]
        evidence = quoted["evidence"].replace("", UNKNOWN)
        counts = {k: int((evidence == k).sum()) for k in (DOCUMENTED, UNDOCUMENTED, UNKNOWN)}
        n_q = int(len(quoted))
        return {
            "competitor": name,
            "lender_type": self.registry.lender_type(name),
            "in_structured_data": name in self.structured_names,
            "loans_mentioned": n_loans,
            "share_of_loans": _round(n_loans / self.total_loans, 4) if self.total_loans else None,
            "notes_mentioning": int(self.mentions.loc[self.mentions["competitor"] == name, "note_id"].nunique()),
            "quotes": self._dist(quoted["quote"]),
            "gap_to_our_rate_bps": self._dist(quoted["gap_ours_bps"]),
            "gap_to_card_bps": self._dist(quoted["gap_card_bps"]),
            "evidence": {**counts, "n_quoted": n_q,
                         "documented_share": _round(counts[DOCUMENTED] / n_q, 4) if n_q >= self.min_n_cell else None},
            "win_rate": self._rate(lc["disbursed"], self.base_win),
            "override_when_cited": {
                **self._rate(lc["was_overridden"], self.base_override),
                "median_depth_bps": self._dist(overridden["override_bps"])["median"],
                "n_overridden": int(len(overridden)),
            },
            "most_aggressive_city": (self._aggressive(quoted, "city", top=1) or [None])[0],
            "first_seen": _date(lc["first_seen"].min()),
            "last_seen": _date(lc["last_seen"].max()),
        }

    def _aggressive(self, quoted: pd.DataFrame, by: str, top: int = 3) -> list[dict[str, Any]]:
        """Where a lender undercuts our sanctioned rate the most (median gap),
        among places with enough quotes to say so."""
        rows = []
        for key, group in quoted.groupby(by):
            if len(group) < self.min_n_cell:
                continue
            rows.append({by: key, "median_gap_bps": _round(group["gap_ours_bps"].median(), 1),
                         "median_quote": _round(group["quote"].median()), "n": int(len(group))})
        return sorted(rows, key=lambda r: r["median_gap_bps"])[:top]

    def _breakdown(self, lc: pd.DataFrame, by: str) -> list[dict[str, Any]]:
        if by not in lc or by not in self.loans:
            return []
        denominators = self.loans.groupby(by)["loan_id"].nunique()
        rows = []
        for key, group in lc.groupby(by):
            n = int(group["loan_id"].nunique())
            quoted = group[group["quote"].notna()]
            dist = self._dist(quoted["quote"])
            gap = self._dist(quoted["gap_ours_bps"])
            base = int(denominators.get(key, 0))
            rows.append({by: key, "loans_mentioned": n,
                         "share_of_loans": _round(n / base, 4) if base >= self.min_n_cell else None,
                         "loans_in_" + by: base, "median_quote": dist["median"],
                         "median_gap_bps": gap["median"], "n_quotes": dist["n"],
                         "suppressed": dist["suppressed"]})
        return sorted(rows, key=lambda r: -r["loans_mentioned"])

    def gap_curve(self, lc: pd.DataFrame) -> list[dict[str, Any]]:
        """Win rate by how far the quote sits under (or over) our sanctioned rate."""
        quoted = lc[lc["quote"].notna()]
        out = []
        for label in self.bucket_labels():
            group = quoted[quoted["gap_bucket"] == label]
            out.append({"bucket": label, **self._rate(group["disbursed"]),
                        "median_gap_bps": _round(group["gap_ours_bps"].median(), 1) if len(group) else None})
        return out

    def _monthly(self, lc: pd.DataFrame) -> list[dict[str, Any]]:
        if "application_month" not in lc:
            return []
        denominators = self.loans.groupby("application_month")["loan_id"].nunique()
        rows = []
        for month in sorted(denominators.index):
            group = lc[lc["application_month"] == month]
            quotes = group["quote"].dropna()
            rows.append({"month": month, "loans_mentioned": int(group["loan_id"].nunique()),
                         "share_of_loans": _round(group["loan_id"].nunique() / denominators[month], 4),
                         "median_quote": _round(quotes.median()) if len(quotes) >= self.min_n_cell else None,
                         "n_quotes": int(len(quotes))})
        return rows

    # ------------------------------------------------------------- products
    def league(self) -> list[dict[str, Any]]:
        names = self.lc["competitor"].unique().tolist()
        return sorted((self._row(n) for n in names), key=lambda r: -r["loans_mentioned"])

    def emerging(self) -> list[dict[str, Any]]:
        """Lenders that appear only in the text: first seen, monthly mentions,
        footprint. The structured field never recorded them at all."""
        out = []
        for name in self.lc["competitor"].unique():
            if name in self.structured_names:
                continue
            lc = self.lc[self.lc["competitor"] == name]
            months = lc.groupby("application_month")["loan_id"].nunique()
            out.append({
                "competitor": name, "lender_type": self.registry.lender_type(name),
                "first_seen": _date(lc["first_seen"].min()), "last_seen": _date(lc["last_seen"].max()),
                "loans_mentioned": int(lc["loan_id"].nunique()),
                "quotes": self._dist(lc["quote"]),
                "cities": int(lc["city"].nunique()), "states": sorted(lc["state"].dropna().unique().tolist()),
                "top_cities": [{"city": c, "loans": int(v)} for c, v in
                               lc.groupby("city")["loan_id"].nunique().sort_values(ascending=False).head(3).items()],
                "monthly_loans": [{"month": k, "loans": int(v)} for k, v in months.sort_index().items()],
                "win_rate": self._rate(lc["disbursed"], self.base_win),
            })
        return sorted(out, key=lambda r: r["first_seen"] or "")

    def unrecognised(self) -> list[dict[str, Any]]:
        u = self.unrecognised_mentions
        if u.empty:
            return []
        rows = []
        for alias, group in u.groupby(u["alias"].str.strip()):
            rows.append({"name": alias, "notes": int(group["note_id"].nunique()),
                         "loans": int(group["loan_id"].nunique()),
                         "first_seen": _date(group["note_date"].min()),
                         "cities": sorted(group["city"].dropna().unique().tolist())[:8],
                         "median_quote": _round(group["rate"].median()) if group["rate"].notna().any() else None})
        return sorted(rows, key=lambda r: -r["notes"])

    def summary(self) -> dict[str, Any]:
        lenders_with_quotes = self.lc[self.lc["quote"].notna()]
        loans_with = set(self.lc["loan_id"])
        no_comp = self.loans[~self.loans["loan_id"].isin(loans_with)]
        return {
            "coverage": {
                "loans": self.total_loans,
                "loans_with_competitor_mentions": len(loans_with),
                "loan_competitor_pairs": int(len(self.lc)),
                "quoted_pairs": int(len(lenders_with_quotes)),
                "mentions": int(len(self.mentions)),
                "competitors_seen": int(self.lc["competitor"].nunique()),
            },
            "base_rates": {
                "win_rate": _round(self.base_win, 4), "override_rate": _round(self.base_override, 4),
                "win_rate_no_competitor_in_notes": self._rate(no_comp["disbursed"])["rate"],
                "n_no_competitor_in_notes": int(len(no_comp)),
            },
            "league": self.league(),
            "gap_win_curve_all": self.gap_curve(self.lc),
            "emerging": self.emerging(),
            "unrecognised": self.unrecognised(),
            "settings": {"min_n_cell": self.min_n_cell, "min_n_rate": self.min_n_rate,
                         "min_n_benchmark": self.min_n_benchmark, "gap_buckets_bps": self.gap_edges,
                         "plausibility_percentile": self.plausibility_pct},
            "caveats": CAVEATS,
            "pricing_effect": PRICING_EFFECT,
        }

    def profile(self, name: str) -> dict[str, Any] | None:
        canonical = self.registry.resolve(name) or (name if name in set(self.lc["competitor"]) else None)
        if canonical is None:
            return None
        lc = self.lc[self.lc["competitor"] == canonical]
        quoted = lc[lc["quote"].notna()]
        hist = []
        if len(quoted):
            lo, hi = np.floor(quoted["quote"].min() * 4) / 4, np.ceil(quoted["quote"].max() * 4) / 4 + 0.25
            counts, edges = np.histogram(quoted["quote"], bins=np.arange(lo, hi + 1e-9, 0.25))
            hist = [{"from": _round(a), "to": _round(b), "n": int(c)} for a, b, c in zip(edges, edges[1:], counts)]
        return {
            **self._row(canonical, lc),
            "known_aliases": next((l["aliases"] + l["ambiguous"] for l in self.registry.listing()
                                   if l["name"] == canonical), []),
            "aliases_seen": (self.mentions.loc[self.mentions["competitor"] == canonical, "alias"]
                             .str.strip().value_counts().head(12).to_dict()),
            "by_state": self._breakdown(lc, "state"),
            "by_city": self._breakdown(lc, "city"),
            "by_branch": self._breakdown(lc, "branch_id"),
            "by_segment": self._breakdown(lc, "segment"),
            "by_grade": self._breakdown(lc, "risk_grade"),
            "gap_win_curve": self.gap_curve(lc),
            "monthly": self._monthly(lc),
            "quote_histogram": hist,
            "most_aggressive_cities": self._aggressive(quoted, "city", top=3),
            "caveats": CAVEATS,
        }

    # ------------------------------------------------------ per-loan context
    def benchmark(self, competitor: str, city: str | None = None, state: str | None = None,
                  segment: str | None = None, grade: str | None = None,
                  exclude_loan: str | None = None) -> dict[str, Any]:
        """That lender's historical quotes, as specific as the sample allows:
        city + segment + grade, then city + segment, city, state, everywhere."""
        quoted = self.lc[(self.lc["competitor"] == competitor) & self.lc["quote"].notna()]
        if exclude_loan:
            quoted = quoted[quoted["loan_id"] != exclude_loan]
        levels = []
        if city and segment and grade:
            levels.append(("city+segment+grade", {"city": city, "segment": segment, "risk_grade": grade}))
        if city and segment:
            levels.append(("city+segment", {"city": city, "segment": segment}))
        if city:
            levels.append(("city", {"city": city}))
        if state:
            levels.append(("state", {"state": state}))
        levels.append(("all", {}))
        tried = []
        for label, filters in levels:
            subset = quoted
            for column, value in filters.items():
                subset = subset[subset[column] == value]
            stats = self._dist(subset["quote"], min_n=self.min_n_benchmark, p5=True)
            tried.append({"level": label, "n": stats["n"]})
            if not stats["suppressed"] and stats["n"]:
                return {"level": label, "filters": filters, **stats, "levels_tried": tried}
        return {"level": None, "filters": {}, "n": 0, "median": None, "p5": None, "p10": None,
                "p90": None, "suppressed": True, "levels_tried": tried}

    def win_rate_at_gap(self, competitor: str, gap_bps: float | None, city: str | None = None,
                        exclude_loan: str | None = None) -> dict[str, Any]:
        if gap_bps is None:
            return {"available": False, "reason": "no rate under consideration was supplied, so the gap is unknown"}
        bucket = self.bucket_for(gap_bps)
        history = self.lc[self.lc["competitor"] == competitor]
        if exclude_loan:
            history = history[history["loan_id"] != exclude_loan]
        lc = history[history["gap_bucket"] == bucket]
        overall = self._rate(lc["disbursed"], self.base_win)
        in_city = self._rate(lc[lc["city"] == city]["disbursed"], self.base_win) if city else None
        out = {"available": True, "gap_bps": _round(gap_bps, 1), "bucket": bucket,
               "all_cities": overall, "city": in_city, "beyond_history": False}
        # The end buckets are open. A gap no past file came near is not "similar"
        # to anything; say so rather than lend it the edge bucket's rate.
        seen = history["gap_ours_bps"].dropna()
        if len(seen) and not (seen.min() <= gap_bps <= seen.max()):
            out["beyond_history"] = True
            out["observed_range_bps"] = [_round(seen.min(), 0), _round(seen.max(), 0)]
            out["note"] = (f"no past {competitor} file had a gap this large (seen "
                           f"{seen.min():+.0f} to {seen.max():+.0f} bps); the open-ended "
                           f"'{bucket}' bucket is shown for reference only")
        return out

    def context_for(self, application: dict[str, Any], notes: list[str] | None = None, *,
                    extractor: NoteExtractor | None = None, retriever=None) -> dict[str, Any]:
        """Competitor context for one file. Informs the reviewer; never the price."""
        app = dict(application or {})
        sales = dict(app.get("sales_context") or {})
        for key in ("competitor_name", "competitor_rate", "competitor_evidence_verified"):
            if key in app and key not in sales:
                sales[key] = app[key]
        loan_id = app.get("loan_id")
        city, state = app.get("city"), app.get("state")
        if not state and city and "state" in self.loans:
            known = self.loans.loc[self.loans["city"] == city, "state"]
            state = known.iloc[0] if len(known) else None
        segment, grade = app.get("segment"), app.get("risk_grade")
        our_rate, our_source = None, None
        for key in ("our_rate", "offered_rate", "recommended_rate", "policy_card_rate"):
            if app.get(key) is not None:
                our_rate, our_source = float(app[key]), key
                break

        extraction = extract_current(notes or [], sales, extractor=extractor)
        claims: dict[str, dict[str, Any]] = {}
        for item in extraction["competitors"]:
            claims[item["competitor"]] = {
                "competitor": item["competitor"], "lender_type": item["lender_type"],
                "quote": item["rate"], "quote_source": "notes" if item["rate"] is not None else None,
                "quote_text": item.get("quote"), "evidence": item["evidence"],
                "evidence_cue": item["evidence_cue"], "in_notes": True}
        structured = self.registry.resolve(sales.get("competitor_name"))
        s_rate = sales.get("competitor_rate")
        if structured:
            entry = claims.setdefault(structured, {
                "competitor": structured, "lender_type": self.registry.lender_type(structured),
                "quote": None, "quote_source": None, "quote_text": None, "evidence": UNKNOWN,
                "evidence_cue": None, "in_notes": False})
            entry["structured_quote"] = s_rate
            entry["structured_verified"] = sales.get("competitor_evidence_verified")
            if entry["quote"] is None and s_rate is not None:
                entry["quote"], entry["quote_source"] = float(s_rate), "structured"
            # The notes win on evidence when they say something; when they are
            # silent, the file's flag is the only statement there is - labelled so.
            if entry["evidence"] == UNKNOWN and sales.get("competitor_evidence_verified") is not None:
                entry["evidence"] = DOCUMENTED if sales["competitor_evidence_verified"] else UNDOCUMENTED
                entry["evidence_cue"] = "structured evidence flag"

        assessed = []
        for name, entry in claims.items():
            bench = self.benchmark(name, city=city, state=state, segment=segment, grade=grade,
                                   exclude_loan=loan_id)
            city_bench = (bench if bench["level"] == "city" else
                          self.benchmark(name, city=city, state=state, exclude_loan=loan_id))
            quote = entry["quote"]
            gap = (quote - our_rate) * 100 if quote is not None and our_rate is not None else None
            assessed.append({
                **entry,
                "benchmark": bench,
                "plausibility": self._plausibility(quote, city_bench, entry["evidence"]),
                "gap_to_our_rate_bps": _round(gap, 1),
                "win_rate_at_gap": self.win_rate_at_gap(name, gap, city=city, exclude_loan=loan_id),
                "market": self._market_line(name),
            })
        assessed.sort(key=lambda r: (r["quote"] is None, r["quote"] or 0))

        return {
            "loan_id": loan_id, "city": city, "state": state, "segment": segment, "risk_grade": grade,
            "our_rate": our_rate, "our_rate_source": our_source,
            "competitors": assessed,
            "conflicts": extraction["conflicts"],
            "extraction": extraction,
            "comparable_negotiations": self._comparables(retriever, assessed, city, loan_id),
            "pricing_effect": PRICING_EFFECT,
            "caveats": CAVEATS,
        }

    def _plausibility(self, quote: float | None, bench: dict[str, Any], evidence: str) -> dict[str, Any]:
        if quote is None:
            return {"checked": False, "reason": "no quoted rate"}
        p5 = bench.get("p5")
        if p5 is None:
            return {"checked": False, "reason": "too few historical quotes from this lender to judge"}
        pct = int(self.plausibility_pct)
        if quote < p5:
            message = ("unusually low - verify the sanction letter on file" if evidence == DOCUMENTED
                       else "unusually low - request the sanction letter")
            return {"checked": True, "flag": True, "message": message, "p5": p5, "level": bench["level"],
                    "n": bench["n"], "percentile": pct}
        return {"checked": True, "flag": False, "message": f"within this lender's historical range (p{pct} {p5:.2f}%)",
                "p5": p5, "level": bench["level"], "n": bench["n"], "percentile": pct}

    def _market_line(self, name: str) -> dict[str, Any]:
        lc = self.lc[self.lc["competitor"] == name]
        return {"loans_mentioned": int(lc["loan_id"].nunique()),
                "share_of_loans": _round(lc["loan_id"].nunique() / self.total_loans, 4) if self.total_loans else None,
                "win_rate": self._rate(lc["disbursed"], self.base_win)}

    def _comparables(self, retriever, assessed: list[dict[str, Any]], city: str | None,
                     loan_id: str | None) -> list[dict[str, Any]]:
        """Past negotiations most like this one: same lender, nearest quote, same
        city where possible. Lexical retrieval cannot see that 8.48% is close to
        8.50% - TF-IDF drops the digits - so it fetches candidates and the
        extracted quote does the ranking."""
        if retriever is None or not hasattr(retriever, "search_notes") or not assessed:
            return []
        target = next((a for a in assessed if a["quote"] is not None), assessed[0])
        name, quote = target["competitor"], target["quote"]
        evidence_words = {DOCUMENTED: "sanction letter offer letter",
                          UNDOCUMENTED: "verbal quote claims no document"}.get(target["evidence"], "quote")
        query = f"{name} {evidence_words}"
        try:
            pool = retriever.search_notes(query, k=120, filters={"city": city} if city else None,
                                          exclude_loan=loan_id)
            if city and sum(1 for h in pool if _claim(h, name)) < self.k_comparable:
                pool += retriever.search_notes(query, k=120, exclude_loan=loan_id)
        except Exception as exc:                                     # noqa: BLE001 - context is optional
            log.warning("comparable negotiations unavailable: %s", exc)
            return []

        ranked, seen = [], set()
        for hit in pool:
            if hit["doc_id"] in seen:
                continue
            seen.add(hit["doc_id"])
            claim = _claim(hit, name)
            if claim is None:
                continue
            distance = abs(claim["rate"] - quote) if claim.get("rate") is not None and quote is not None else 99.0
            same_city = hit["metadata"].get("city") == city
            ranked.append(((distance, not same_city, -hit["score"]), hit, claim))
        ranked.sort(key=lambda item: item[0])
        out = []
        for _, hit, claim in ranked[: self.k_comparable]:
            why = [f"{name} quoted" + (f" at {claim['rate']:.2f}%" if claim.get("rate") is not None else "")]
            if hit["metadata"].get("city") == city:
                why.append(f"same city ({city})")
            why.append(f"evidence {claim['evidence']}")
            out.append({**hit, "why_comparable": ", ".join(why)})
        return out


def _claim(hit: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((c for c in hit.get("competitors") or [] if c["competitor"] == name), None)


def _date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def build(mentions: pd.DataFrame | None = None, loans: pd.DataFrame | None = None) -> CompetitorAnalysis:
    """The analysis over the cached historical extraction and the training table."""
    if mentions is None:
        from .note_extraction import load_history
        _, mentions = load_history()
    if loans is None:
        from ..ingestion import loader
        loans = loader.load_training_table()
    return CompetitorAnalysis(mentions, loans)
