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
