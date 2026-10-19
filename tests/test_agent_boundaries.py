"""Each reusable agent package is independent, and this is what makes that true.

The claim "every agent is completely independent" is worth exactly as much as the
check behind it. These tests read the actual import graph and compare it with what
``agents/packages.py`` declares, in both directions - so a package that quietly
grows a dependency fails here rather than being discovered when somebody tries to
lift it out.

The kernel rule is the load-bearing one. ``aavas.kernel`` may import nothing from
aavas except logging; the moment it needs config, a store or a model it stops
being a primitive and starts being a package, and every agent that depends on it
inherits that.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from aavas.agents import packages

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
AAVAS = ROOT / "aavas"


def _top_level_imports(package_dir: pathlib.Path) -> set[str]:
    """The aavas sub-packages this directory imports, however deeply nested."""
    found: set[str] = set()
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        here = list(path.relative_to(AAVAS).parent.parts)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                module = node.module or ""
                if module.startswith("aavas."):
                    found.add(module.split(".")[1])
                continue
            base = here[: len(here) - (node.level - 1)] if node.level > 1 else here
            resolved = ".".join([*base, node.module]) if node.module else ".".join(base)
            head = resolved.split(".")[0] if resolved else ""
            if head:
                found.add(head)
    return {f for f in found if f}


def _package_dir(declared: dict) -> pathlib.Path:
    return AAVAS / declared["package"].split(".", 1)[1].replace(".", "/")


IDS = [p["name"] for p in packages.PACKAGES]


# ------------------------------------------------------------------ the kernel

def test_the_kernel_imports_nothing_but_logging() -> None:
    """A primitive that needs config is not a primitive. Every agent depends on
    the kernel, so anything the kernel drags in is dragged in four times over."""
    imports = _top_level_imports(AAVAS / "kernel")
    allowed = {"kernel", "logging_utils"}
    assert imports <= allowed, f"the kernel reaches outside itself: {sorted(imports - allowed)}"


def test_the_kernel_holds_the_primitives_that_used_to_couple_the_agents() -> None:
    from aavas.kernel import boolean_column, fairness_evaluate, numeric_column, psi

    assert all(callable(f) for f in (numeric_column, boolean_column, psi, fairness_evaluate))


# ------------------------------------------------------- declaration vs reality

@pytest.mark.parametrize("declared", packages.PACKAGES, ids=IDS)
def test_a_package_imports_only_what_it_declares(declared: dict) -> None:
    """Drift direction one: a dependency picked up and not declared."""
    own = declared["package"].split(".")[-1]
    actual = _top_level_imports(_package_dir(declared)) - {own}
    undeclared = actual - set(declared["depends_on"]) - packages.SHARED_KERNEL
    assert not undeclared, (
        f"{declared['name']} imports {sorted(undeclared)}, which it does not declare. "
        f"Either declare it or remove the edge - an undeclared dependency is the thing that "
        f"breaks when somebody lifts this package out.")


@pytest.mark.parametrize("declared", packages.PACKAGES, ids=IDS)
def test_a_package_does_not_declare_a_dependency_it_dropped(declared: dict) -> None:
    """Drift direction two: a dependency removed and still advertised. A manifest
    that overstates coupling makes a package look harder to reuse than it is."""
    own = declared["package"].split(".")[-1]
    actual = _top_level_imports(_package_dir(declared)) - {own}
    stale = [d for d in declared["depends_on"]
             if d not in actual and d not in packages.SHARED_KERNEL]
    assert not stale, f"{declared['name']} declares {stale} but no longer imports them"


@pytest.mark.parametrize("declared", packages.PACKAGES, ids=IDS)
def test_every_public_symbol_exists(declared: dict) -> None:
    """The public surface is the reuse contract. A name in it that does not
    resolve is a promise to a future consumer that will not be kept."""
    for reference in declared["public"]:
        module_name, _, symbol = reference.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), f"{reference} does not exist"


@pytest.mark.parametrize("declared", packages.PACKAGES, ids=IDS)
def test_a_package_owns_the_config_it_claims(declared: dict) -> None:
    for relative in declared["owns_config"]:
        assert (ROOT.parent / relative).exists(), f"{declared['name']}: {relative} is missing"


# ------------------------------------------------------------ peer independence

def test_no_reusable_package_reaches_into_another() -> None:
    """The point of the whole exercise.

    Before the kernel existed, the pricing optimiser imported the feature package
    for two column helpers, and feature discovery imported the learning package
    for one statistic. Neither was a real dependency - they were primitives that
    happened to live somewhere.
    """
    owners = {p["package"].split(".")[-1]: p["name"] for p in packages.PACKAGES}
    offences = []
    for declared in packages.PACKAGES:
        own = declared["package"].split(".")[-1]
        for imported in _top_level_imports(_package_dir(declared)) - {own}:
            if imported in owners:
                offences.append(f"{declared['name']} -> {owners[imported]}")
    assert not offences, f"peer coupling between reusable packages: {offences}"


def test_a_package_that_is_not_standalone_says_why(  ) -> None:
    """Acceptance is coupled on purpose - it checks a permission before reading a
    customer's own words. Recording the reason is the difference between a known
    constraint and an accident."""
    for declared in packages.PACKAGES:
        if declared["standalone"]:
            assert "not_standalone_because" not in declared, declared["name"]
            continue
        reason = declared.get("not_standalone_because", "")
        assert len(reason) > 60, f"{declared['name']} is coupled and does not say why"


@pytest.mark.parametrize("declared", packages.PACKAGES, ids=IDS)
def test_a_package_states_how_it_would_be_reused(declared: dict) -> None:
    assert len(declared["reuse"]) > 60, f"{declared['name']} has no reuse note"
    assert declared["emits"], f"{declared['name']} does not say what it produces"


def test_the_four_packages_cover_the_subsystems_that_are_worth_lifting() -> None:
    assert set(packages.BY_NAME) == {"feature_engineering", "automl", "acceptance", "pricing"}
