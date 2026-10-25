"""The four integration endpoints, exercised as Salesforce/LOS would call them."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aavas.api.main import app


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(app)


@pytest.fixture
def stage1_payload() -> dict:
    return {
        "loan_id": "API-TEST-001", "segment": "salaried_formal", "city": "Jaipur",
        "channel": "branch_walkin", "property_type": "resale_flat",
        "declared_monthly_income": 55000, "requested_loan_amount": 1500000,
        "requested_tenure_years": 20, "applicant_age": 36, "declared_obligations": 4000,
        "estimated_property_value": 2100000, "bureau_score": 742,
        "sales_context": {"competitor_rate": 10.4, "competitor_name": "HDFC",
                          "competitor_evidence_verified": True, "customer_negotiating": True},
    }


@pytest.fixture
def stage2_payload(stage1_payload) -> dict:
    return {**stage1_payload, "loan_id": "API-TEST-002",
            "verified_monthly_income": 53000, "appraised_property_value": 2050000,
            "sanctioned_loan_amount": 1450000, "tenure_years": 20,
            "bureau_vintage_months": 96, "delinquency_count_24m": 0,
            "banking_behaviour_score": 71.0, "deviation_count": 1,
            "legal_flag": 0, "technical_flag": 0}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) >= {"stage1", "stage2", "conversion"}


def test_pre_assessment_returns_a_band(client, stage1_payload):
    body = client.post("/pricing/pre-assessment", json=stage1_payload).json()
    assert body["band"]["min_rate"] < body["band"]["max_rate"]
    assert 0 < body["confidence"] <= 1
    assert body["eligible_amount"]["max"] > 0
    assert body["disclaimer"]


def test_pre_assessment_rejects_an_impossible_bureau_score(client, stage1_payload):
    response = client.post("/pricing/pre-assessment", json={**stage1_payload, "bureau_score": 1200})
    assert response.status_code == 422


def test_final_assessment_returns_a_defensible_recommendation(client, stage2_payload):
    body = client.post("/pricing/final-assessment", json=stage2_payload).json()
    recommendation = body["recommendation"]
    assert body["envelope"]["min_rate"] <= recommendation["recommended_rate"] <= body["envelope"]["max_rate"]
    assert body["policy"]["required_approval"]
    assert body["explanation"]["reason_codes"]
    assert body["risk"]["risk_grade"] in list("ABCDE")
    assert body["human_decision_required"] is True
    assert "scenario_table" not in recommendation, "the heavy table is opt-in"


def test_scenarios_are_available_on_request(client, stage2_payload):
    body = client.post("/pricing/final-assessment?include_scenarios=true", json=stage2_payload).json()
    assert len(body["recommendation"]["scenario_table"]) > 1


def test_decision_is_retrievable(client, stage2_payload):
    decision_id = client.post("/pricing/final-assessment", json=stage2_payload).json()["decision_id"]
    stored = client.get(f"/pricing/decision/{decision_id}").json()
    assert stored["decision_id"] == decision_id
    assert stored["policy_version"]


def test_unknown_decision_is_404(client):
    assert client.get("/pricing/decision/DEC-NOPE").status_code == 404


def test_override_requires_an_approved_reason(client, stage2_payload):
    client.post("/pricing/final-assessment", json=stage2_payload)
    response = client.post("/pricing/override", json={
        "loan_id": stage2_payload["loan_id"], "human_rate": 10.9, "reason": "manager_said_so"})
    assert response.status_code == 422
    assert "allowed_reasons" in response.json()["detail"]


def test_override_is_captured_with_authority(client, stage2_payload):
    """Authority is assessed against the *card* rate, not the recommendation, so
    the test discounts from the card rate to make the expected level unambiguous."""
    recommendation = client.post("/pricing/final-assessment", json=stage2_payload).json()
    card_rate = recommendation["policy"]["card_rate"]

    body = client.post("/pricing/override", json={
        "loan_id": stage2_payload["loan_id"], "human_rate": round(card_rate - 0.30, 2),
        "reason": "competitor_lower_rate", "approved_by": "s.verma:sales_officer",
        "evidence_verified": True}).json()

    assert body["ai_rate"] == recommendation["recommendation"]["recommended_rate"], \
        "the AI recommendation must be preserved alongside the human decision"
    assert body["required_approval"] in {"regional_manager", "credit_head", "pricing_committee"}
    # 30 bps from card is beyond a sales officer's authority.
    assert body["within_authority"] is False
    assert "warning" in body


def test_override_within_authority_is_not_flagged(client, stage2_payload):
    recommendation = client.post("/pricing/final-assessment", json=stage2_payload).json()
    card_rate = recommendation["policy"]["card_rate"]
    body = client.post("/pricing/override", json={
        "loan_id": stage2_payload["loan_id"], "human_rate": round(card_rate - 0.05, 2),
        "reason": "strong_negotiation", "approved_by": "s.verma:sales_officer"}).json()
    assert body["required_approval"] == "sales_officer"
    assert body["within_authority"] is True
    assert "warning" not in body


def test_override_on_an_unknown_loan_is_404(client):
    response = client.post("/pricing/override", json={
        "loan_id": "NOT-A-LOAN", "human_rate": 11.0, "reason": "strong_negotiation"})
    assert response.status_code == 404


def test_outcome_closes_the_loop(client, stage2_payload):
    client.post("/pricing/final-assessment", json=stage2_payload)
    body = client.post("/pricing/outcome", json={
        "loan_id": stage2_payload["loan_id"], "disbursed": True, "customer_response": "accepted",
        "sanctioned_rate": 10.95, "months_to_exit": 54}).json()
    assert body["disbursed"] is True
    assert body["vs_expected"]["decision_id"]


def test_audit_trail_is_complete(client, stage2_payload):
    body = client.get(f"/pricing/audit/{stage2_payload['loan_id']}").json()
    actions = {entry["action"] for entry in body["audit_entries"]}
    assert "post_assessment_decision" in actions
    assert "capture_override" in actions
    assert "capture_outcome" in actions


def test_policy_can_be_cited(client):
    body = client.get("/pricing/policy/explain", params={"question": "branch manager discount limit"}).json()
    assert body["clauses"]
    assert "basis points" in body["clauses"][0]["text"]


def test_model_cards_expose_limitations(client):
    cards = client.get("/model-cards").json()
    assert cards["conversion"]["limitations"], "a model card without limitations is marketing"


@pytest.mark.parametrize('name', ['Stage1Request', 'Stage2Request', 'OverrideRequest', 'OutcomeRequest'])
def test_swagger_examples_are_valid(name):
    from aavas.api import contracts

    schema = app.openapi()['components']['schemas'][name]
    for example in schema['examples']:
        getattr(contracts, name).model_validate(example)


def test_swagger_examples_complete_flow(client):
    schemas = app.openapi()['components']['schemas']

    def payload(name):
        return {**schemas[name]['examples'][0], 'loan_id': 'SWAGGER-FLOW-TEST'}

    for path, name in [('/pricing/pre-assessment', 'Stage1Request'),
                       ('/pricing/final-assessment', 'Stage2Request'),
                       ('/pricing/override', 'OverrideRequest'),
                       ('/pricing/outcome', 'OutcomeRequest')]:
        response = client.post(path, json=payload(name))
        assert response.status_code == 200, response.text
        if name == 'OverrideRequest':
            assert response.json()['policy_check']['allowed']
            assert response.json()['within_authority']
    history = client.get('/pricing/audit/SWAGGER-FLOW-TEST').json()
    actions = {entry['action'] for entry in history['audit_entries']}
    assert {'post_assessment_decision', 'capture_override', 'capture_outcome'} <= actions


# ------------------------------------------------- explainability (admin view)
def test_xai_index_lists_a_report_per_model(client):
    body = client.get("/xai").json()
    assert body["available"] is True
    names = {m["model"] for m in body["models"]}
    assert {"stage1", "stage2", "conversion", "risk"} <= names


def test_xai_index_translates_rank_agreement_into_a_reading(client):
    rows = {m["model"]: m for m in client.get("/xai").json()["models"]}
    # risk scores ~0.39: the two methods disagree, and the payload has to say so
    assert rows["risk"]["trust"]["level"] == "weak"
    assert "not lean" in rows["risk"]["trust"]["reading"]
    assert rows["stage2"]["trust"]["level"] == "strong"


def test_xai_detail_carries_importance_limitations_and_captures(client):
    body = client.get("/xai/stage2").json()
    assert body["global_importance"] and body["global_importance"][0]["rank"] == 1
    assert body["limitations"], "a model card without written limitations is not a card"
    assert body["counts"]["n_test"] > 0
    assert all("url" in c for c in body["captures"])


def test_xai_detail_trims_instances_to_the_requested_count(client):
    assert len(client.get("/xai/stage2?instances=1").json()["instances"]) == 1


def test_xai_unknown_model_says_how_to_generate_the_report(client):
    response = client.get("/xai/not_a_model")
    assert response.status_code == 404
    assert "make xai" in response.json()["detail"]


def test_xai_report_returns_the_written_brief(client):
    response = client.get("/xai/stage2/report")
    assert response.status_code == 200
    assert len(response.text) > 200


def test_xai_capture_serves_a_png_and_refuses_traversal(client):
    ok = client.get("/xai/capture/stage2-global.png")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert client.get("/xai/capture/../../../etc/passwd").status_code in (400, 404)
    assert client.get("/xai/capture/notes.txt").status_code == 400


def test_xai_credit_score_band_reports_both_ends(client):
    body = client.get("/xai/stage2/credit-score").json()
    assert body["available"] is True
    observed = body["observed"]
    assert observed["score_min"] >= 300, "the new-to-credit sentinel is not a credit score"
    assert observed["floor_cohort"]["files"] and observed["ceiling_cohort"]["files"]
    assert observed["gap"] is not None
    assert body["reading"]


def test_xai_credit_score_keeps_thin_files_out_of_the_band(client):
    """-1 means "no bureau history". Folded into the band it becomes the worst
    credit in the book, which is a different population, not a worse score."""
    body = client.get("/xai/risk/credit-score").json()
    assert body["sentinel"] == -1
    assert body["observed"]["new_to_credit"]["files"] > 0
    assert body["curve"]["sentinel_points_dropped"] == [-1.0]
    assert body["curve"]["at_min"]["value"] >= 300


def test_xai_detail_carries_the_band_and_the_files_at_its_ends(client):
    body = client.get("/xai/risk").json()
    band = body["credit_score"]
    assert set(band["instances"]) == {"lowest credit score", "highest credit score"}
    explained = {i["loan_id"] for i in body["instances"]}
    for end in band["instances"].values():
        assert end["loan_id"] in explained, "the band names a file the payload does not explain"
    assert all("credit_score" in i for i in body["instances"])


def test_xai_credit_score_is_absent_politely(client, monkeypatch):
    """A report written before the band existed must read as "regenerate", not
    as a missing model."""
    from aavas.api import xai as xai_api

    monkeypatch.setattr(xai_api, "_load", lambda model: {"generated_at": "2026-01-01T00:00:00+00:00"})
    body = client.get("/xai/stage2/credit-score").json()
    assert body["available"] is False
    assert "make xai-reports" in body["reason"]


# --------------------------------------------------- tracking and dashboards


def test_tracking_index_summarises_the_store_without_importing_mlflow(client, monkeypatch):
    """The API must read the run history off the file store. mlflow is optional
    everywhere else in this repo, and a page that needs it would be a page that
    breaks when nobody started a tracking server."""
    import builtins

    real_import = builtins.__import__

    def refuse_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow must not be needed to read the store")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_mlflow)
    body = client.get("/tracking").json()
    assert body["mlflow"]["store_kind"] == "file"
    assert body["mlflow"]["runs"] >= body["mlflow"]["searches"]
    assert body["mlflow"]["searches"] + body["mlflow"]["candidates"] == body["mlflow"]["runs"]


def test_tracking_runs_nest_candidates_under_their_search(client):
    body = client.get("/tracking/runs").json()
    if not body["available"]:
        pytest.skip(body["reason"])
    assert body["searches"], "a store with runs must produce at least one search"
    parent = max(body["searches"], key=lambda s: s["child_count"])
    assert parent["child_count"] == len(parent["children"])
    for child in parent["children"]:
        assert child["parent_run_id"] == parent["run_id"]
        assert "auc" in child["metrics"] or "ece" in child["metrics"]


def test_tracking_runs_can_be_returned_flat(client):
    nested = client.get("/tracking/runs").json()
    if not nested["available"]:
        pytest.skip(nested["reason"])
    flat = client.get("/tracking/runs?flat=true").json()
    assert len(flat["searches"]) == flat["runs"] >= len(nested["searches"])


def test_tracking_services_report_reachability_and_how_to_start(client):
    body = client.get("/tracking/services").json()
    assert body["checked_from"] == "the API host"
    for service in body["services"]:
        assert service["url"].startswith("http")
        assert isinstance(service["reachable"], bool)
        assert service["start"], "a service that is down must say how to start it"


def test_tracking_names_each_omnixai_dashboard_from_the_manifest(client, tmp_path, monkeypatch):
    """A Dash process does not say which model it holds, so run_omnixai writes a
    manifest. Without it the admin view can only report a bare port."""
    import json

    from aavas.api import tracking as tracking_api

    manifest = tmp_path / "dashboards.json"
    manifest.write_text(json.dumps({"dashboards": [
        {"model": "risk", "port": 8765, "url": "http://127.0.0.1:8765"},
        {"model": "stage2", "port": 8766, "url": "http://127.0.0.1:8766"}]}), encoding="utf-8")
    monkeypatch.setattr(tracking_api, "DASHBOARD_MANIFEST", manifest)
    monkeypatch.setattr(tracking_api, "_listening", lambda url, timeout=0.25: url.endswith("8765"))

    services = {s["name"]: s for s in client.get("/tracking/services").json()["services"]}
    assert services["OmniXAI · risk"]["reachable"] is True
    # A named dashboard that is down is still listed: "stage2's is not running"
    # is the answer to the question, not an empty row.
    assert services["OmniXAI · stage2"]["reachable"] is False


def test_tracking_reports_a_remote_store_instead_of_guessing(client, monkeypatch):
    from aavas.api import tracking as tracking_api

    monkeypatch.setattr(tracking_api, "_uri", lambda: "http://mlflow.internal:5000")
    body = client.get("/tracking/runs").json()
    assert body["available"] is False and "remote" in body["reason"]
    assert client.get("/tracking").json()["mlflow"]["store_kind"] == "remote"


def test_tracking_hides_runs_deleted_in_mlflow(client, tmp_path, monkeypatch):
    """mlflow soft-deletes a run by flipping lifecycle_stage. A run someone
    removed in the UI must not reappear through this API."""
    from aavas.api import tracking as tracking_api

    store = tmp_path / "mlruns"
    experiment = store / "1"
    (experiment).mkdir(parents=True)
    (experiment / "meta.yaml").write_text(
        "experiment_id: '1'\nname: t\nlifecycle_stage: active\n", encoding="utf-8")
    for name, stage in (("alive", "active"), ("gone", "deleted")):
        run = experiment / name
        (run / "tags").mkdir(parents=True)
        (run / "metrics").mkdir()
        (run / "params").mkdir()
        (run / "meta.yaml").write_text(
            f"run_id: {name}\nrun_name: {name}\nstatus: 3\nstart_time: 1789979716251\n"
            f"end_time: 1789979716451\nlifecycle_stage: {stage}\n", encoding="utf-8")
    monkeypatch.setattr(tracking_api, "_store_dir", lambda: store)

    names = {r["name"] for r in client.get("/tracking/runs?flat=true").json()["searches"]}
    assert names == {"alive"}


def test_xai_credit_score_dashboard_covers_every_model(client):
    """The cross-model view. One model answers what the score does to that
    prediction; six answer what it does to the customer."""
    body = client.get("/xai/credit-score").json()
    assert body["available"] is True
    names = {m["model"] for m in body["models"]}
    assert {"stage1", "stage2", "risk", "conversion", "prepayment", "override"} <= names
    for model in body["models"]:
        assert model["observed"]["score_min"] >= 300
        assert model["units"] in body["units"]
        assert model["trust"]["level"] in {"strong", "moderate", "weak", "unknown"}


def test_xai_credit_score_dashboard_route_is_not_shadowed_by_the_model_route(client):
    """`/xai/credit-score` and `/xai/{model}` are both one segment. Registered
    the wrong way round, the dashboard becomes a 404 for a model named
    'credit-score'."""
    body = client.get("/xai/credit-score").json()
    assert "models" in body and "model" not in body
    assert client.get("/xai/risk/credit-score").json()["model"] == "risk"


def test_xai_credit_score_dashboard_keeps_units_separate(client):
    """A probability and a rate share no axis; the payload has to say which is
    which or the dashboard will plot them together."""
    body = client.get("/xai/credit-score").json()
    assert set(body["units"]) == {"probability", "% p.a."}
    by_units = {m["model"]: m["units"] for m in body["models"]}
    assert by_units["risk"] == "probability" and by_units["stage2"] == "% p.a."


def test_agents_index_separates_wired_from_dormant(client):
    body = client.get("/agents").json()
    by_name = {a["name"]: a for a in body["agents"]}
    assert by_name["pricing"]["wired"] is True and by_name["pricing"]["pipeline_positions"]
    # Three agents exist and nothing calls them. A roster that hid that would be
    # describing a pipeline other than the one running.
    for name in ("extraction", "evidence", "acceptance"):
        assert by_name[name]["status"] == "dormant"
        assert by_name[name]["dormant_because"]
    assert body["summary"]["wired"] + body["summary"]["dormant"] == body["summary"]["agents"]


def test_a_priced_decision_carries_its_step_trace(client):
    from aavas.api import examples

    priced = client.post("/pricing/final-assessment", json=examples.FINAL_ASSESSMENT).json()
    trace = priced["trace"]
    assert [s["agent"] for s in trace["steps"]][:3] == ["underwriting", "risk", "policy"]
    assert all(s["summary"] for s in trace["steps"]), "every step must say what it produced"
    assert trace["failed"] == []

    fetched = client.get(f"/agents/trace/{priced['decision_id']}").json()
    assert fetched["decision_id"] == priced["decision_id"]
    assert fetched["recommended_rate"] == priced["recommendation"]["recommended_rate"]


def test_agent_trace_latest_is_not_read_as_a_decision_id(client):
    """`/agents/trace/latest` and `/agents/trace/{decision_id}` are both one
    segment; registered the wrong way round, 'latest' 404s as a missing id."""
    from aavas.api import examples

    client.post("/pricing/final-assessment", json=examples.FINAL_ASSESSMENT)
    body = client.get("/agents/trace/latest").json()
    assert body["steps"] and body["decision_id"].startswith("DEC-")


def test_a_decision_without_a_trace_says_so_rather_than_404ing_blankly(client, monkeypatch):
    from aavas.api import agents as agents_api

    monkeypatch.setattr(agents_api.decision_logger, "get_decision",
                        lambda decision_id: {"decision_id": decision_id, "payload": {}})
    response = client.get("/agents/trace/DEC-OLD")
    assert response.status_code == 404
    assert "before the step trace existed" in response.json()["detail"]


# ------------------------------------------------------------- configuration


def test_config_index_covers_every_file_the_loader_knows(client):
    """A config file the engine reads and this endpoint does not list is a
    setting nobody can see from the admin view."""
    from aavas.config import _FILES

    body = client.get("/config").json()
    assert {f["name"] for f in body["files"]} == set(_FILES)
    assert body["settings"] == sum(f["settings"] for f in body["files"]) > 0
    for f in body["files"]:
        assert f["version"], f"{f['file']} declares no version"
        assert f["sections"] and f["description"]


def test_config_is_read_only_without_a_signed_in_administrator(client):
    """Editing these values changes the priced path. With authentication disabled
    nobody is identified, so nobody may write; the payload has to say so.
    test_config_editing.py covers the administrator's write path."""
    body = client.get("/config").json()
    assert body["editable"] is False
    assert "administrator" in body["note"]
    assert client.put("/config/policy", json={"raw": "x: 1"}).status_code == 403
    for verb in ("PATCH", "DELETE"):
        assert client.request(verb, "/config/policy").status_code == 405


def test_config_never_serves_a_secret(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-appear")
    body = client.get("/config").json()
    key = next(e for e in body["environment"] if e["name"] == "ANTHROPIC_API_KEY")
    assert key["secret"] is True and key["set"] is True
    assert key["value"] is None
    assert "sk-ant-should-never-appear" not in client.get("/config").text


def test_config_detail_keeps_the_comments(client):
    """The comment is often the control: "never learn that more expensive
    converts better" is not expressible in YAML, so a parsed-only view loses
    the reason a constraint exists."""
    body = client.get("/config/features").json()
    assert "#" in body["raw"] and "monotone" in body["raw"]
    paths = {row["path"] for row in body["flat"]}
    assert "conversion.monotone.offered_rate" in paths
    assert body["content"]["conversion"]["monotone"]["offered_rate"] == -1


def test_config_search_spans_every_file(client):
    body = client.get("/config/search?q=monotone").json()
    assert body["matches"], "the monotone constraints must be findable"
    assert {m["config"] for m in body["matches"]} == {"features"}
    wide = client.get("/config/search?q=rate").json()
    assert len({m["config"] for m in wide["matches"]}) > 1, "search must cross files"


def test_config_search_is_not_read_as_a_config_name(client):
    """'/config/search' and '/config/{name}' are both one segment."""
    assert "matches" in client.get("/config/search?q=auc").json()
    assert client.get("/config/not_a_config").status_code == 404


def test_config_unknown_file_lists_what_exists(client):
    detail = client.get("/config/nope").json()["detail"]
    assert "policy" in detail and "features" in detail


def test_connectors_endpoint_reports_which_gate_holds_each_system(client):
    body = client.get("/connectors").json()
    names = {c["name"] for c in body["connectors"]}
    assert names == {"whatsapp", "gmail", "outlook", "salesforce_lms", "oracle_fusion"}
    assert body["ready"] == [] and len(body["blocked"]) == 5
    for connector in body["connectors"]:
        assert connector["blocked_reason"], "a blocked connector must say which gate"
        assert connector["transport"]["kind"] == "mcp"


def test_connectors_keep_communication_apart_from_systems_of_record(client):
    layers = client.get("/connectors").json()["by_layer"]
    assert set(layers["acceptance_signals"]) == {"whatsapp", "gmail", "outlook"}
    assert layers["application_record"] == ["salesforce_lms"]
    assert layers["economics"] == ["oracle_fusion"]


def test_connector_detail_and_unknown_name(client):
    body = client.get("/connectors/whatsapp").json()
    assert body["consent_required"] is True and body["personal_data"] is True
    assert client.get("/connectors/nope").status_code == 404
