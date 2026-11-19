"""The policy kit: schema, template, README table and the validate command.

The live policy must pass the schema and every engine check. The blank template must fail,
and say which values are "not filled in". A policy with a broken invariant must fail on the
engine check even though the schema accepts it. `build` must reproduce the committed files.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from aavas.governance import policy_checks

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "policy_kit"
LIVE = ROOT / "config" / "pricing_policy.yaml"
MODEL_CONFIG = ROOT / "config" / "model_config.yaml"

_spec = importlib.util.spec_from_file_location("policy_kit_cli", ROOT / "scripts" / "policy_kit.py")
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)


@pytest.fixture(scope="module")
def live() -> dict:
    return yaml.safe_load(LIVE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((KIT / pk.SCHEMA_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cutoffs() -> dict:
    return yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))["risk"]["grade_cutoffs"]


def _write(tmp_path: Path, policy: dict) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return path


def _validate(path: Path, capsys) -> tuple[int, str]:
    code = pk.main(["validate", str(path), "--cutoffs", str(MODEL_CONFIG)])
    return code, capsys.readouterr().out


def test_live_policy_passes_schema_and_every_check(live, schema, cutoffs, capsys):
    holes, errors = pk.schema_problems(pk.normalise(live), schema)
    assert holes == [] and errors == []
    results = policy_checks.run(live, cutoffs)
    assert policy_checks.summary(results)["ok"]
    code, out = _validate(LIVE, capsys)
    assert code == 0, out
    assert "RESULT: PASS" in out


def test_sections_cover_every_policy_key_and_every_part(live):
    meta = yaml.safe_load((KIT / "sections.yaml").read_text(encoding="utf-8"))
    assert list(live) and set(meta["sections"]) == set(live)
    assert {i["part"] for i in meta["sections"].values()} == set(meta["parts"]) == set("ABCDEF")
    assert {i["requirement"] for i in meta["sections"].values()} == set(pk.REQUIREMENTS)
    owners = {"risk", "credit policy", "pricing", "compliance", "technology"}
    assert {i["owner"] for i in meta["sections"].values()} <= owners


def test_blank_template_fails_and_says_not_filled_in(capsys):
    code, out = _validate(KIT / pk.TEMPLATE_NAME, capsys)
    assert code == 1
    assert "not filled in" in out
    assert "RESULT: FAIL" in out
    template = yaml.safe_load((KIT / pk.TEMPLATE_NAME).read_text(encoding="utf-8"))
    meta = yaml.safe_load((KIT / "sections.yaml").read_text(encoding="utf-8"))
    assert set(template) == set(meta["sections"])               # same structure as the live policy


def test_template_has_a_header_for_every_section():
    text = (KIT / pk.TEMPLATE_NAME).read_text(encoding="utf-8")
    meta = yaml.safe_load((KIT / "sections.yaml").read_text(encoding="utf-8"))
    for name, info in meta["sections"].items():
        assert f"# {info['title']}\n" in text
        assert f"#   Requirement: {info['requirement']}\n" in text
        assert f"#   Owner:       {info['owner']}\n" in text
        assert f"\n{name}:" in text


def test_company_names_are_accepted_in_vocabulary_sections(live, schema):
    policy = copy.deepcopy(live)
    policy["risk_grade_loading_bps"] = {"G1": 0, "G2": 100, "G3": None}
    policy["base_spread_bps"] = {"retail": 10}
    holes, errors = pk.schema_problems(pk.normalise(policy), schema)
    assert not holes
    assert errors == []                                          # the engine checks will object, the schema does not


def test_broken_invariant_passes_schema_but_fails_engine_checks(live, schema, tmp_path, capsys):
    broken = copy.deepcopy(live)
    broken["reference"]["card_rate_ceiling"] = broken["reference"]["card_rate_floor"] - 1
    holes, errors = pk.schema_problems(pk.normalise(broken), schema)
    assert holes == [] and errors == []
    code, out = _validate(_write(tmp_path, broken), capsys)
    assert code == 1
    assert "INV-05" in out and "RESULT: FAIL" in out


def test_a_grade_priced_below_a_better_grade_fails(live, tmp_path, capsys):
    broken = copy.deepcopy(live)
    broken["pricing"]["grade_base_roi"]["B"] = broken["pricing"]["grade_base_roi"]["A"] - 1
    code, out = _validate(_write(tmp_path, broken), capsys)
    assert code == 1
    assert "INV-02" in out and "INV-03" in out


def test_schema_errors_name_the_path_the_expectation_and_what_was_found(live, schema):
    bad = copy.deepcopy(live)
    bad["constraints"]["max_ltv"] = 90                           # a percent where a fraction is expected
    bad["constraints"]["max_tenure_months"] = "ten years"
    bad["pricing"]["matrix"]["A"][0] = 140.0                     # a rate above 100
    del bad["reference"]["card_rate_floor"]
    del bad["fair_pricing"]
    holes, errors = pk.schema_problems(pk.normalise(bad), schema)
    text = "\n".join(errors)
    assert not holes
    assert "constraints.max_ltv: expected a number <= 1, found 90" in text
    assert "constraints.max_tenure_months: expected integer, found \"ten years\"" in text
    assert "pricing.matrix.A[0]: expected a number <= 100, found 140.0" in text
    assert "reference: missing required key 'card_rate_floor'" in text
    assert ": missing required key 'fair_pricing'" in text


def test_null_is_accepted_where_the_policy_uses_it(live, schema):
    policy = copy.deepcopy(live)
    policy["supersedes"] = None
    assert policy["pricing"]["grade_base_roi"]["D"] is None
    holes, errors = pk.schema_problems(pk.normalise(policy), schema)
    assert holes == [] and errors == []


def test_unquoted_date_is_read_as_text():
    assert pk.normalise(yaml.safe_load("effective_from: 2026-12-01")) == {"effective_from": "2026-12-01"}


def test_build_reproduces_the_committed_files_exactly():
    files = pk.generate()
    assert set(files) == {pk.SCHEMA_NAME, pk.TEMPLATE_NAME, pk.README_NAME}
    for name, text in files.items():
        committed = (KIT / name).read_bytes().decode("utf-8")
        assert "\r" not in committed, f"{name} has CRLF line endings"
        assert committed == text, f"{name} has drifted: run python scripts/policy_kit.py build"


def test_build_writes_to_an_out_directory(tmp_path):
    assert pk.main(["build", "--out", str(tmp_path)]) == 0
    for name in (pk.SCHEMA_NAME, pk.TEMPLATE_NAME, pk.README_NAME):
        assert (tmp_path / name).read_bytes() == (KIT / name).read_bytes()


def test_readme_table_lists_every_section():
    readme = (KIT / pk.README_NAME).read_text(encoding="utf-8")
    meta = yaml.safe_load((KIT / "sections.yaml").read_text(encoding="utf-8"))
    for name in meta["sections"]:
        assert f"| `{name}` |" in readme
    for letter in meta["parts"]:
        assert f"#### Part {letter}:" in readme


def test_no_company_name_in_the_kit():
    files = [p for p in KIT.rglob("*") if p.is_file()]
    assert len(files) >= 4
    for path in files:
        assert "aavas" not in path.read_text(encoding="utf-8").lower(), path.name
        assert "aavas" not in path.name.lower()
