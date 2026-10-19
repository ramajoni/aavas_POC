"""Index the pricing policy so a recommendation can cite the rule it applied.

"Why can't I give 70 bps?" should be answerable with the clause, not with a
stack trace.
"""

from __future__ import annotations

from ..config import policy
from ..logging_utils import get_logger
from .vector_store import Document, VectorStore, get_store

log = get_logger(__name__)


def build_documents() -> list[Document]:
    pol = policy()
    version = pol["policy_version"]
    documents: list[Document] = []

    def add(doc_id: str, text: str, **metadata) -> None:
        documents.append(Document(doc_id=f"{version}:{doc_id}", text=text,
                                  metadata={"policy_version": version, "source": "pricing_policy", **metadata}))

    ref = pol["reference"]
    add("reference_rates",
        f"The product floor rate is {ref['card_rate_floor']}% and the ceiling is {ref['card_rate_ceiling']}%. "
        f"No home loan may be priced outside this range under policy {version}.",
        clause="reference")

    for segment, bps in pol["base_spread_bps"].items():
        add(f"base_spread_{segment}",
            f"The base spread for the {segment.replace('_', ' ')} segment is {bps} basis points over the "
            f"floor rate of {ref['card_rate_floor']}%.", clause="base_spread", segment=segment)

    for grade, bps in pol["risk_grade_loading_bps"].items():
        add(f"risk_loading_{grade}", f"Risk grade {grade} carries a loading of {bps} basis points.",
            clause="risk_loading", risk_grade=grade)

    for band in pol["ltv_loading_bps"]:
        add(f"ltv_loading_{band['max_ltv']}",
            f"Loans with LTV up to {band['max_ltv']:.0%} carry an LTV loading of {band['bps']} basis points.",
            clause="ltv_loading")

    con = pol["constraints"]
    add("hard_constraints",
        f"Maximum LTV is {con['max_ltv']:.0%}, maximum FOIR is {con['max_foir']:.0%}, maximum tenure is "
        f"{con['max_tenure_years']} years and the maximum loan amount is Rs {con['max_loan_amount']:,}. "
        f"The maximum total discount from the card rate is {con['absolute_discount_floor_bps']} basis points.",
        clause="constraints")
    add("new_to_credit",
        f"New-to-credit applicants are {'permitted' if con['new_to_credit_allowed'] else 'not permitted'} and "
        f"carry a minimum additional spread of {con['new_to_credit_min_spread_bps']} basis points.",
        clause="new_to_credit")

    for level in pol["approval_authority"]:
        add(f"authority_{level['level']}",
            f"A {level['level'].replace('_', ' ')} may approve a discount of up to "
            f"{level['max_discount_bps']} basis points and a premium of up to {level['max_premium_bps']} "
            "basis points from the card rate.", clause="approval_authority", level=level["level"])

    ovr = pol["override"]
    add("override_reasons",
        "Every override requires a reason. Permitted reasons are: "
        + ", ".join(r.replace("_", " ") for r in ovr["allowed_reasons"]) + ".", clause="override")
    add("override_evidence",
        "Reasons requiring documentary evidence are: "
        + ", ".join(r.replace("_", " ") for r in ovr["evidence_required_reasons"])
        + f". Without evidence, the maximum discount attributable to such a claim is "
          f"{ovr['max_unevidenced_discount_bps']} basis points. Unverified competitor claims are recorded "
          "as context and must not automatically determine the customer's price.",
        clause="override_evidence")

    econ = pol["economics"]
    add("economics",
        f"Pricing economics assume a funding cost of {econ['funding_cost_pct']}%, servicing cost of "
        f"{econ['servicing_cost_pct']}%, capital charge of {econ['capital_charge_pct']}%, acquisition cost of "
        f"{econ['acquisition_cost_pct']}% and processing fee income of {econ['processing_fee_pct']}% of the "
        f"disbursed amount.", clause="economics")

    opt = pol["optimizer"]
    add("optimizer_guardrails",
        f"The pricing engine will not recommend a price whose conversion probability is below "
        f"{opt['min_conversion_probability']:.0%} or whose balance-transfer probability exceeds "
        f"{opt['max_bt_out_probability']:.0%}.", clause="optimizer")

    return documents


def _register(store: VectorStore) -> str | None:
    """Register the built index, degrading rather than failing.

    A retriever that refused to start because the governance store was unwritable
    would take the policy endpoint down for the sake of an audit row.
    """
    try:
        from ..governance import knowledge_registry

        documents = [doc.text for doc in build_documents()]
        return knowledge_registry.register(
            "policy", documents, backend=type(store).__name__,
            corpus="config/pricing_policy.yaml")
    except Exception as exc:                                    # noqa: BLE001
        log.warning("policy index not registered: %s", exc)
        return None


def build_index(store: VectorStore | None = None) -> VectorStore:
    store = store or get_store()
    documents = build_documents()
    store.add(documents)
    log.info("indexed %s policy clauses (%s)", len(documents), policy()["policy_version"])
    return store


class PolicyRetriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or build_index()
        # Record what this index was built from. The index is rebuilt at every
        # start, so without this a cited clause cannot be traced to the corpus it
        # was chosen against.
        self.index_id = _register(self.store)

    @property
    def provenance(self) -> dict:
        from ..governance import knowledge_registry

        return knowledge_registry.latest("policy") or {}

    def cite(self, question: str, k: int = 3) -> list[dict]:
        return [hit.as_dict() for hit in self.store.search(question, k=k)]

    def explain_limit(self, discount_bps: float) -> list[dict]:
        return self.cite(f"maximum discount {discount_bps:.0f} basis points approval authority", k=3)
