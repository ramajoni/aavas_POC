"""The connector registry's gates.

Five external systems, three layers, three gates. What these tests protect is
not that a connector fetches - none of them can here - but that the engine
refuses to read one it may not, and says which gate refused it.
"""

from __future__ import annotations

import pytest

from aavas.connectors import mcp_client
from aavas.connectors.registry import (all_connectors, consent_backend, for_layer,
                                        status, usable)


def test_all_five_systems_are_declared():
    names = set(all_connectors())
    assert names == {"whatsapp", "gmail", "outlook", "salesforce_lms", "oracle_fusion"}


def test_communication_and_systems_of_record_land_in_different_layers():
    """A WhatsApp thread is a customer's own words; a Salesforce record is a
    typed field. Collapsing the two is how the second becomes an LLM input."""
    layers = status()["by_layer"]
    assert set(layers["acceptance_signals"]) == {"whatsapp", "gmail", "outlook"}
    assert layers["application_record"] == ["salesforce_lms"]
    assert layers["economics"] == ["oracle_fusion"]


def test_nothing_is_enabled_by_default():
    assert all(not c.enabled for c in all_connectors().values())
    assert status()["ready"] == []


def test_consent_blocks_a_communication_connector_even_when_switched_on():
    """The gate that matters. A pricing engine that read a customer's messages
    because a YAML flag was flipped has a consent problem, not a config one."""
    whatsapp = all_connectors()["whatsapp"]
    whatsapp.enabled = True
    assert "consent" in whatsapp.blocked_reason("none")
    assert whatsapp.blocked_reason("vault") is None, "with a consent store it may run"


def test_a_system_of_record_needs_no_consent_to_be_readable():
    salesforce = all_connectors()["salesforce_lms"]
    salesforce.enabled = True
    assert salesforce.blocked_reason("none") is None
    assert salesforce.consent_required is False


def test_an_unreachable_server_is_a_different_finding_from_a_missing_switch():
    connector = all_connectors()["oracle_fusion"]
    assert "switched off" in connector.blocked_reason("none")
    connector.enabled = True
    assert connector.blocked_reason("none", reachable=False).startswith("MCP server")
    assert connector.blocked_reason("none", reachable=True) is None


def test_the_salesforce_field_list_stays_inside_the_feature_contract():
    """A connector must not be able to widen what a model may see."""
    from aavas.config import feature_config

    declared = set(all_connectors()["salesforce_lms"].fields)
    known = {c for block in feature_config().values() if isinstance(block, dict)
             for key in ("numeric", "categorical") for c in (block.get(key) or [])}
    known |= {"branch_id"}          # an operating unit, not a model feature
    assert declared <= known, f"not in any feature contract: {declared - known}"


def test_usable_refuses_a_connector_from_another_layer():
    with pytest.raises(ValueError):
        for_layer("not_a_layer")
    assert usable("acceptance_signals") == []


def test_an_unconfigured_endpoint_is_not_the_same_as_an_unreachable_one():
    assert mcp_client.reachable(None) is None
    assert mcp_client.reachable("") is None
    assert mcp_client.reachable("http://127.0.0.1:9") is False


def test_a_call_that_reads_nothing_says_why():
    """Silence would let an acceptance score claim it saw a source it never
    read; the result has to distinguish "nothing there" from "never asked"."""
    result = mcp_client.call(all_connectors()["whatsapp"])
    assert result.available is False
    assert result.reason and result.documents == []
    assert result.as_dict()["documents"] == 0


def test_the_acceptance_source_refuses_a_non_signal_connector(monkeypatch):
    from aavas.acceptance.sources import McpSource

    source = McpSource("salesforce_lms", {"reliability": "verified"}, "salesforce_lms")
    monkeypatch.setattr("aavas.connectors.registry.consent_backend", lambda: "vault")
    connector = all_connectors()["salesforce_lms"]
    connector.enabled = True
    monkeypatch.setattr("aavas.connectors.registry.all_connectors",
                        lambda: {"salesforce_lms": connector})
    assert source.fetch({"loan_id": "X"}, {}) == []


def test_status_reports_the_consent_backend_it_actually_has():
    state = status()
    assert state["consent"]["configured"] is False
    assert consent_backend() == "none"
    assert state["version"]
