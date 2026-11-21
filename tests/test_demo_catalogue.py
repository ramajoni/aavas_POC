"""The area-manager case catalogue (scripts/demo_catalogue.py): the rules it classifies by, its guard,
the catalogue in the repo, and the report built from it. No API is needed."""

from __future__ import annotations

import json
import sys

import pytest

from aavas.config import policy

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import demo_catalogue as dc  # noqa: E402

pytestmark = pytest.mark.skipif(not dc.CATALOGUE.exists(), reason="scripts/demo_catalogue.json not generated")


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(dc.CATALOGUE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize("years,band", [(3, "36-60"), (5, "36-60"), (6, "61-84"), (7, "61-84"), (8, "85-120"), (10, "85-120")])
def test_tenure_band_boundaries(years, band):
    assert dc.band_of(years) == band


@pytest.mark.parametrize("rate,pos", [(13.39, "below_floor"), (13.396, "at_floor"), (13.40, "at_floor"),
                                       (13.403, "at_floor"), (13.41, "within_room")])
def test_position_boundaries(rate, pos):
    assert dc.position_of(rate, 13.40) == pos


def test_evidence_classes_are_read_against_the_files_own_floor_and_recommendation():
    floor, rec = 13.40, 13.90
    assert dc.evidence_class(None, floor, rec) == "none"
    assert dc.evidence_class((12.9, "SBI", False), floor, rec) == "claimed"        # not seen, whatever the rate
    assert dc.evidence_class((13.0, "SBI", True), floor, rec) == "below_floor"
    assert dc.evidence_class((13.40, "SBI", True), floor, rec) == "between"          # at the floor is not below it
    assert dc.evidence_class((13.89, "SBI", True), floor, rec) == "between"
    assert dc.evidence_class((13.90, "SBI", True), floor, rec) == "at_or_above_rec"
    assert dc.evidence_class((14.50, "SBI", True), floor, rec) == "at_or_above_rec"


def test_asks_follow_the_rm_25_bps_rule():
    # recommendation 13.98 over a floor of 13.40: all three positions exist
    assert dc.ask_for("within_room", 13.98, 13.40) == 13.68
    assert dc.ask_for("at_floor", 13.98, 13.40) == 13.40
    assert dc.ask_for("below_floor", 13.98, 13.40) == 13.15
    # recommendation within 25 bps of the floor: nothing at or above the floor can reach the area manager
    assert dc.ask_for("at_floor", 13.60, 13.40) is None
    assert dc.ask_for("within_room", 13.60, 13.40) is None
    assert dc.ask_for("within_room", 13.69, 13.40) is None
    assert dc.ask_for("within_room", 13.70, 13.40) is None      # that ask would be the floor itself
    assert dc.ask_for("within_room", 13.71, 13.40) == 13.41
    assert dc.available_positions(13.40, 13.40) == ["below_floor"]
    # a file whose floor is the 13.00% minimum cannot carry a below-floor ask
    assert dc.ask_for("below_floor", 13.40, 13.00) is None
    assert dc.available_positions(13.00, 13.00) == []


def test_static_unreachable_cells_follow_the_matrix():
    assert "same as the policy floor" in dc.static_unreachable("A+", "36-60", "within_room", "none")
    assert dc.static_unreachable("A+", "61-84", "below_floor", "none")
    assert dc.static_unreachable("A+", "61-84", "within_room", "none") is None
    assert dc.static_unreachable("A", "36-60", "below_floor", "none") is None
    assert dc.static_unreachable("C", "85-120", "below_floor", "none") is None


def test_floor_is_card_minus_60_bps_never_below_13():
    assert dc.floor_of(14.0) == 13.4
    assert dc.floor_of(13.5) == 13.0
    assert dc.floor_of(13.0) == 13.0
    assert dc.floor_of(16.75) == 16.15


def test_uplift_stays_between_the_recommendation_and_the_ceiling():
    obs = {"rec": 13.60, "card": 14.0, "ceiling": 14.0, "quoted_max": None}
    assert dc.uplift_rate(obs) == 13.75
    assert dc.uplift_rate({**obs, "quoted_max": 13.62}) is None          # the quoted range caps it
    assert dc.uplift_rate({**obs, "rec": 14.0}) is None                   # already at the matrix rate


# ---------------------------------------------------------------- the guard
@pytest.mark.parametrize("rate", [12.75, 12.99, 17.01, 20.0])
def test_guard_stops_a_rate_outside_the_band(rate):
    with pytest.raises(SystemExit) as exc:
        dc.check_band(rate, "case X-1", 13.0, 17.0)
    assert "case X-1" in str(exc.value) and f"{rate:.2f}" in str(exc.value) and "13.00-17.00" in str(exc.value)


@pytest.mark.parametrize("rate", [13.0, 15.5, 17.0])
def test_guard_lets_a_rate_inside_the_band_through(rate):
    assert dc.check_band(rate, "case X-1", 13.0, 17.0) == rate


def test_band_comes_from_the_policy():
    ref = policy()["reference"]
    assert dc.policy_band() == (float(ref["card_rate_floor"]), float(ref["card_rate_ceiling"]))


def test_probe_refuses_without_scratch_and_on_the_live_port(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo_catalogue.py", "probe", "--base", "http://127.0.0.1:8011"])
    with pytest.raises(SystemExit) as exc:
        dc.main()
    assert "--scratch" in str(exc.value)
    monkeypatch.setattr(sys, "argv", ["demo_catalogue.py", "probe", "--base", "http://127.0.0.1:8000", "--scratch"])
    with pytest.raises(SystemExit) as exc:
        dc.main()
    assert "8000" in str(exc.value)


# ---------------------------------------------------------------- the catalogue in the repo
def test_every_primary_cell_is_present_once(data):
    keys = [c["key"] for c in data["cells"]]
    assert len(keys) == len(set(keys)) == len(dc.GRADES) * len(dc.BANDS) * len(dc.POSITIONS) * len(dc.EVIDENCE)
    assert set(keys) == {dc.cell_key(*c) for c in dc.all_cells()}


def test_every_cell_is_filled_or_explained(data):
    cases = {c["case_id"]: c for c in data["cases"]}
    for cell in data["cells"]:
        if cell["status"] == "filled":
            case = cases[cell["case_id"]]
            assert case["kind"] == "matrix"
            assert dc.cell_key(case["cell"]["grade"], case["cell"]["band"], case["cell"]["position"],
                               case["cell"]["evidence"]) == cell["key"]
        else:
            assert cell["status"] in ("impossible", "not_found") and len(cell["reason"]) > 20, cell["key"]
            if cell["status"] == "not_found":
                assert cell["reason"].startswith("Not found"), cell["key"]
    s = data["summary"]
    assert s["cells_filled"] + s["cells_impossible"] + s["cells_not_found"] == s["cells_total"]
    assert s["cells_filled"] == sum(c["status"] == "filled" for c in data["cells"]) == s["matrix_cases"]


def test_cells_the_matrix_rules_out_are_impossible(data):
    by = {c["key"]: c for c in data["cells"]}
    for pos in dc.POSITIONS:
        for ev in dc.EVIDENCE:
            assert by[dc.cell_key("A+", "36-60", pos, ev)]["status"] == "impossible"
    for ev in dc.EVIDENCE:
        assert by[dc.cell_key("A+", "61-84", "below_floor", ev)]["status"] == "impossible"
    items = " ".join(x["item"] for x in data["other_unreachable"])
    for word in ("Grade D", "LAP", "25 bps", "100 bps"):
        assert word in items


def test_ids_names_mobiles_and_cities_are_clean(data):
    cases = data["cases"]
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert len({c["profile"]["name"] for c in cases}) == len(cases)
    mobiles = [c["profile"]["mobile"] for c in cases]
    assert len(set(mobiles)) == len(mobiles) and all(m.startswith("9820013") and len(m) == 10 for m in mobiles)
    assert {c["profile"]["city"] for c in cases} <= set(dc.CITIES)
    assert {c["profile"]["property_type"] for c in cases} <= set(dc.PROPERTIES)
    assert {c["profile"]["channel"] for c in cases} <= set(dc.CHANNELS)


def test_every_rate_is_inside_the_policy_band_and_the_ask_is_valid(data):
    lo, hi = dc.policy_band()
    for c in data["cases"]:
        o = c["observed"]
        assert c["ask"] is not None and lo <= c["ask"] <= hi, c["case_id"]
        assert lo <= o["floor"] <= o["rec"] <= hi, c["case_id"]
        assert o["rec"] - c["ask"] > 0.25, c["case_id"]                     # beyond the RM's own 25 bps
        assert dc.position_of(c["ask"], o["floor"]) == c["cell"]["position"], c["case_id"]
        assert dc.ask_for(c["cell"]["position"], o["rec"], o["floor"], lo, hi) == c["ask"], c["case_id"]
        assert o["grade"] in dc.GRADES and not o["infeasible"] and not o["no_offer"]
        if c["kind"] == "history" and c["decision"].get("rate") is not None:
            assert lo <= c["decision"]["rate"] <= hi, c["case_id"]


def test_override_reasons_are_allowed_and_all_are_used(data):
    allowed = set(policy()["override"]["allowed_reasons"])
    used = {c["reason"] for c in data["cases"]}
    assert used <= allowed
    assert used == allowed, f"unused reasons: {allowed - used}"
    uplift = set(policy()["override"]["uplift"]["reasons"])
    for c in data["cases"]:
        if c["kind"] == "history" and c["outcome"] == "uplift":
            assert c["decision"]["reason"] in uplift


def test_applicants_meet_the_intake_rules(data):
    for c in data["cases"]:
        p, v = c["profile"], c["profile"]["verified"]
        assert p["score"] == -1 or p["score"] >= 675, c["case_id"]
        assert 3 <= p["years"] <= 10 and p["loan"] <= 7_500_000
        assert v["loan"] / v["appraised"] <= 0.90 + 1e-9 and p["obligations"] < p["income"]
        limit = 60 if p["segment"].startswith("salaried") else 70
        assert p["age"] + p["years"] <= limit, c["case_id"]


def test_secondary_values_each_appear(data):
    sec = data["secondary"]
    assert set(sec["segment"]) == set(dc.SEGMENTS)
    assert set(sec["channel"]) == set(dc.CHANNELS)
    assert set(sec["property_type"]) == set(dc.PROPERTIES)
    assert set(sec["existing"]) == {"True", "False"} and set(sec["negotiating"]) == {"True", "False"}


def test_history_covers_the_outcomes_the_manager_can_take(data):
    outcomes = {c["outcome"] for c in data["cases"] if c["kind"] == "history"}
    listed = {x["item"] for x in data["other_unreachable"]}
    for o in dc.HISTORY_OUTCOMES:
        assert o in outcomes or f"History outcome: {o}" in listed, o
    for c in data["cases"]:
        if c["kind"] != "history":
            continue
        d, o = c["decision"], c["observed"]
        if c["outcome"] in ("approve_as_asked", "approve_at_floor", "counter"):
            assert d["approve"] and o["floor"] - 1e-9 <= d["rate"] < o["rec"], c["case_id"]
        if c["outcome"] == "approve_as_asked":
            assert d["rate"] == c["ask"] >= o["floor"]
        if c["outcome"] == "counter":
            assert c["ask"] < o["floor"] and d["rate"] == o["floor"]
        if c["outcome"] == "decline":
            assert d["approve"] is False
        if c["outcome"] == "uplift":
            assert d["uplift"] and o["rec"] < d["rate"] <= dc.ceiling_of(o) + 1e-9


def test_special_situations_are_each_present_or_explained(data):
    done = {c["situation"] for c in data["cases"] if c["kind"] == "special"}
    listed = {x["item"] for x in data["other_unreachable"]}
    for kind, label in dc.SPECIALS.items():
        assert kind in done or label in listed, kind


# ---------------------------------------------------------------- the report
def test_report_lists_every_cell_and_how_to_run(data):
    text = dc.render_report(data)
    for g in dc.GRADES:
        assert f"### Grade {g}" in text
    for c in data["cases"]:
        assert c["case_id"] in text
    assert "## How to run" in text and "--decide-history" in text and "--scratch" in text
    s = data["summary"]
    assert text.count("x (") >= s["cells_impossible"] // len(dc.EVIDENCE)
    assert text.count("? (") >= s["cells_not_found"] // len(dc.EVIDENCE)
    assert "Grade D" in text and "LAP" in text
    assert "\r" not in text


def test_report_writes_lf_file(data, tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "REPORT", tmp_path / "docs" / "demo" / "out.md")
    monkeypatch.setattr(dc, "load_catalogue", lambda *_: data)
    dc.run_report()
    raw = (tmp_path / "docs" / "demo" / "out.md").read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"# Area manager cases")


def test_report_in_the_repo_matches_the_catalogue(data):
    if not dc.REPORT.exists():
        pytest.skip("docs/demo/area_manager_cases.md not generated")
    assert dc.REPORT.read_text(encoding="utf-8") == dc.render_report(data)


def test_history_approvals_rest_on_a_verified_letter(data):
    for c in data["cases"]:
        if c["kind"] == "history" and c["outcome"] in ("approve_as_asked", "approve_at_floor", "counter"):
            comp = c["profile"]["comp"]
            assert comp and comp[2], c["case_id"]          # the policy needs evidence behind a cut above 15 bps


def test_a_cell_with_room_to_spare_is_not_found_not_impossible():
    stats = {"A|61-84": {"profiles": 41, "max_room_bps": 40, "min_gap_to_breakeven": 4.4}}
    status, why = dc.explain_unreachable(("A", "61-84", "within_room", "below_floor"), stats, 13.0)
    assert status == "not_found" and "possible under the policy" in why and "40 bps" in why
    status, why = dc.explain_unreachable(("B", "85-120", "at_floor", "none"),
                                         {"B|85-120": {"profiles": 66, "max_room_bps": 0}}, 13.0)
    assert status == "not_found" and "never rose more than 0 bps" in why and "more than 25 bps" in why
    status, why = dc.explain_unreachable(("C", "36-60", "below_floor", "between"),
                                         {"C|36-60": {"profiles": 74, "max_room_bps": 0}}, 13.0)
    assert status == "not_found" and "more than 0 bps" in why
    status, _ = dc.explain_unreachable(("A+", "36-60", "below_floor", "none"), {}, 13.0)
    assert status == "impossible"


# ---------------------------------------------------------------- trim (a temp sqlite store)
import argparse  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402

REASONS = list(policy()["override"]["allowed_reasons"])


def _seed(tmp_path, monkeypatch, files: int = 16):
    """A temp store with `files` loan files that each have a rate request (every fourth decided), one
    file with no rate request, upload folders for all, and some API request-log and audit rows."""
    monkeypatch.setenv("AAVAS_STORE_URL", f"sqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("AAVAS_UPLOAD_DIR", str(tmp_path / "uploads"))
    from aavas.governance import audit, store
    loans = []
    for i in range(files + 1):
        loan = f"APP{i:013d}"
        has_request = i < files
        grade, position, evidence = dc.GRADES[i % 4], dc.POSITIONS[i % 3], dc.EVIDENCE[i % 5]
        years = (4, 7, 10)[i % 3]
        card = dc.MATRIX[grade][i % 3]
        floor = dc.floor_of(card)
        ask = {"within_room": card - 0.30, "at_floor": floor, "below_floor": floor - 0.25}[position]
        comp = {"below_floor": (floor - 0.3, True), "between": (floor + 0.2, True), "at_or_above_rec": (card + 0.2, True),
                "claimed": (floor + 0.1, False), "none": None}[evidence]
        sales = ({"competitor_rate": comp[0], "competitor_name": "SBI", "competitor_evidence_verified": comp[1]}
                 if comp else {})
        mobile = ("9820011" if i % 5 == 0 else "9820013") + f"{i:03d}"
        applicant = {"name": f"Customer {i}", "mobile": mobile, "segment": dc.SEGMENTS[i % 4], "tenure_years": years,
                     "sanctioned_loan_amount": 7_200_000 if i == 3 else 2_000_000, "appraised_property_value": 12_000_000,
                     "verified_monthly_income": 300_000, "declared_obligations": 0, "applicant_age": 35,
                     "bureau_score": -1 if i == 5 else 760, "sales_context": {**sales, "customer_negotiating": True}}
        stamp = "2026-11-01T00:00:00+00:00"
        store.insert("applications", {"loan_id": loan, "owner": mobile, "applicant_name": f"Customer {i}",
                                      "applicant": applicant, "created_at": stamp, "updated_at": stamp})
        store.insert("decisions", {"decision_id": f"DEC-{i}", "loan_id": loan, "stage": "post_assessment", "created_at": stamp,
                                   "card_rate": card, "min_rate": floor, "recommended_rate": card,
                                   "payload": {"risk": {"risk_grade": grade}}})
        store.insert("application_documents", {"document_id": f"DOC-{i}", "loan_id": loan, "document_type": "income_proof",
                                               "filename": "a.pdf", "content_type": "application/pdf", "size_bytes": 5,
                                               "sha256": "x", "uploaded_by": "rohit", "uploaded_at": stamp})
        folder = tmp_path / "uploads" / loan
        folder.mkdir(parents=True)
        (folder / f"DOC-{i}").write_bytes(b"%PDF")
        store.insert("customer_logins", {"username": mobile, "issued_by": "rohit", "loan_id": loan, "issued_at": stamp})
        store.insert("borrower_accounts", {"username": mobile, "name": f"Customer {i}", "mobile": mobile, "password": "x",
                                           "created_at": stamp})
        if has_request:
            decided = i % 4 == 3
            store.insert("rate_requests", {"request_id": f"RRQ-{i:03d}", "loan_id": loan, "decision_id": f"DEC-{i}",
                                           "requested_rate": ask, "recommended_rate": card,
                                           "reason": REASONS[i % len(REASONS)], "requested_by": "rohit",
                                           "requested_at": f"2026-11-{1 + i:02d}T00:00:00+00:00",
                                           "status": "approved" if decided else "pending"})
            store.insert("offers", {"offer_id": f"OFR-{i}", "loan_id": loan, "decision_id": f"DEC-{i}", "rate": ask,
                                    "recommended_rate": card, "route": "manager_approved", "sent_by": "meera",
                                    "sent_at": stamp, "status": "sent", "request_id": f"RRQ-{i:03d}"})
            store.insert("decision_memory", {"episode_id": f"MEM-{i}", "request_id": f"RRQ-{i:03d}", "loan_id": loan,
                                             "manager": "meera", "decided_at": stamp, "action": "approve",
                                             "requested_rate": ask, "recommended_rate": card, "case_features": {},
                                             "format": "MEM-1", "recorded_at": stamp})
            store.insert("requests", {"request_id": f"REQ-{i}", "route": "/x", "method": "POST", "started_at": "t",
                                      "loan_id": loan})
        loans.append(loan)
    audit.record("test", "seeded", "store", "x", {})
    return loans


def _counts(path) -> dict:
    conn = sqlite3.connect(path)
    names = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in names}
    conn.close()
    return out


def _features(tmp_path, monkeypatch):
    from aavas.governance import store
    _seed(tmp_path, monkeypatch)
    with store.connect() as conn:
        return dc.load_request_features(conn, policy()["eligibility_limits"], 13.0)


def test_trim_selection_is_deterministic_and_covers_every_value(tmp_path, monkeypatch):
    feats = _features(tmp_path, monkeypatch)
    first = [f["request_id"] for f in dc.select_scenes(feats, 10)]
    shuffled = list(feats)
    random.Random(1).shuffle(shuffled)
    assert [f["request_id"] for f in dc.select_scenes(shuffled, 10)] == first
    assert len(first) == 10
    kept = [f for f in feats if f["request_id"] in first]
    for dim, have, missed in dc.coverage_table(feats, kept, REASONS):
        assert not missed, (dim, missed)


def test_trim_keeps_only_pending_and_prefers_hand_written_on_ties(tmp_path, monkeypatch):
    feats = _features(tmp_path, monkeypatch)
    assert any(f["status"] != "pending" for f in feats)
    assert all(f["status"] == "pending" for f in dc.select_scenes(feats, 10))
    assert len(dc.select_scenes(feats, 500)) == sum(f["status"] == "pending" for f in feats)   # fewer pending than asked
    twins = [{**feats[0], "request_id": "RRQ-T1", "status": "pending", "mobile": "9820013999", "requested_at": "a"},
             {**feats[0], "request_id": "RRQ-T2", "status": "pending", "mobile": "9820011999", "requested_at": "z"}]
    assert dc.select_scenes(twins, 1)[0]["request_id"] == "RRQ-T2"                              # hand-written beats older


def test_swaps_recover_a_single_value_greedy_spent_on_pairs():
    def scene(rid, elements, mobile="9820013000"):
        return {"request_id": rid, "status": "pending", "requested_at": rid, "mobile": mobile, "elements": set(elements)}
    # Greedy takes A (most weight: two pairs), then B; with two slots it never reaches C,
    # the only request with new to credit. Swapping A for C keeps every single value and adds one.
    a = scene("RRQ-A", {("grade", "A"), ("pair", "p1"), ("pair", "p2"), ("pair", "p3"), ("pair", "p4"), ("pair", "p5")})
    b = scene("RRQ-B", {("grade", "A"), ("grade", "B")})
    c = scene("RRQ-C", {("grade", "A"), ("special", "new_to_credit")})
    kept = dc.select_scenes([a, b, c], 2)
    singles = set().union(*(f["elements"] for f in kept))
    assert ("special", "new_to_credit") in singles and ("grade", "B") in singles
    assert {f["request_id"] for f in kept} == {"RRQ-B", "RRQ-C"}
    assert [f["request_id"] for f in dc.select_scenes([c, b, a], 2)] == [f["request_id"] for f in kept]


def test_trim_labels_the_scenes(tmp_path, monkeypatch):
    feats = _features(tmp_path, monkeypatch)
    for f in dc.select_scenes(feats, 10):
        label = dc.scene_label(f)
        assert label.startswith(f["grade"]) and f"{f['years']}y" in label and ": " in label
    assert any("new to credit" in dc.scene_label(f) for f in feats)
    assert any("Rs 70 lakh" in dc.scene_label(f) for f in feats)


def test_trim_dry_run_writes_nothing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    before = _counts(tmp_path / "store.db")
    folders = sorted(p.name for p in (tmp_path / "uploads").iterdir())
    dc.run_trim(argparse.Namespace(keep=10, apply=False))
    assert _counts(tmp_path / "store.db") == before
    assert sorted(p.name for p in (tmp_path / "uploads").iterdir()) == folders
    assert not list(tmp_path.glob("*before-trim*"))


def test_trim_apply_removes_the_others_from_every_loan_table_but_requests(tmp_path, monkeypatch):
    loans = _seed(tmp_path, monkeypatch)
    before = _counts(tmp_path / "store.db")
    dc.run_trim(argparse.Namespace(keep=10, apply=True))
    after = _counts(tmp_path / "store.db")
    conn = sqlite3.connect(tmp_path / "store.db")
    kept = {r[0] for r in conn.execute("SELECT loan_id FROM rate_requests")}
    assert len(kept) == 10 and all(r[0] == "pending" for r in conn.execute("SELECT status FROM rate_requests"))
    removed = [loan for loan in loans[:-1] if loan not in kept]
    assert len(removed) == 6
    for table in ("applications", "application_documents", "rate_requests", "decisions", "offers", "decision_memory",
                  "customer_logins"):
        for loan in removed:
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE loan_id = ?", (loan,)).fetchone()[0] == 0, (table, loan)
    assert after["borrower_accounts"] == before["borrower_accounts"] - 6
    assert after["requests"] == before["requests"]                                  # the API request log keeps every row
    assert after["audit_log"] >= before["audit_log"]                                # nothing removed from the audit log
    assert after["decision_memory"] == before["decision_memory"] - 6
    conn.close()


def test_trim_leaves_files_without_a_rate_request_alone(tmp_path, monkeypatch):
    loans = _seed(tmp_path, monkeypatch)
    bystander = loans[-1]
    dc.run_trim(argparse.Namespace(keep=3, apply=True))
    conn = sqlite3.connect(tmp_path / "store.db")
    owner = conn.execute("SELECT owner FROM applications WHERE loan_id = ?", (bystander,)).fetchone()[0]
    for table, col, value in (("applications", "loan_id", bystander), ("decisions", "loan_id", bystander),
                              ("application_documents", "loan_id", bystander), ("customer_logins", "loan_id", bystander),
                              ("borrower_accounts", "username", owner)):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (value,)).fetchone()[0] == 1, table
    assert (tmp_path / "uploads" / bystander).exists()
    conn.close()


def test_trim_deletes_upload_folders_of_removed_files_only(tmp_path, monkeypatch):
    loans = _seed(tmp_path, monkeypatch)
    dc.run_trim(argparse.Namespace(keep=10, apply=True))
    conn = sqlite3.connect(tmp_path / "store.db")
    kept = {r[0] for r in conn.execute("SELECT loan_id FROM rate_requests")}
    conn.close()
    assert {p.name for p in (tmp_path / "uploads").iterdir()} == kept | {loans[-1]}


def test_trim_backs_up_before_removing_anything(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    seen = {}
    real = shutil.rmtree

    def spy(path, *a, **k):
        seen.setdefault("backups", list(tmp_path.glob("store.before-trim-*.db")))
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", spy)
    dc.run_trim(argparse.Namespace(keep=10, apply=True))
    assert len(seen["backups"]) == 1                                                  # present when the first folder went
    backup = _counts(seen["backups"][0])
    assert backup["rate_requests"] == 16 and backup["applications"] == 17            # the backup holds everything
    assert _counts(tmp_path / "store.db")["rate_requests"] == 10


def test_trim_never_resolves_an_upload_folder_outside_the_upload_root(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    from aavas.intake import applications
    root = applications.upload_dir().resolve()
    assert dc.upload_folder("APP1234567890123") == root / "APP1234567890123"
    for bad in ("../outside", "..", "a/b", "..\\x", "", None, "x" * 41, "/etc"):
        assert dc.upload_folder(bad) is None, bad
