"""Every pricing-policy version a decision may have been priced under, readable.

A decision records the policy version it was priced under (``HL-POLICY-2026.09``,
``HL-POLICY-2026.10``, ...). The live file is ``config/pricing_policy.yaml``; each
earlier version is kept as written under ``config/policy_archive/<version>.yaml``,
and any version replaced through the configuration screen is also in the change
history. So a reviewer looking at an older decision can read the policy it was
priced under, and what changed from the version before.
"""

from __future__ import annotations

from typing import Any

import yaml

from ..config import config_path
from ..paths import CONFIG_DIR
from . import config_changes, store

ARCHIVE = CONFIG_DIR / "policy_archive"


def _raws() -> dict[str, str]:
    """version -> raw YAML, from the archive, the change history and the live file."""
    found: dict[str, str] = {}
    if ARCHIVE.exists():
        for path in sorted(ARCHIVE.glob("*.yaml")):
            raw = path.read_text(encoding="utf-8")
            version = (yaml.safe_load(raw) or {}).get("policy_version") or path.stem
            found[version] = raw
    try:
        rows = store.query("SELECT before_raw, after_raw FROM config_changes WHERE name = 'policy' "
                           "ORDER BY changed_at")
    except Exception:                                              # noqa: BLE001 - no history table yet
        rows = []
    for row in rows:
        for raw in (row.get("before_raw"), row.get("after_raw")):
            if raw:
                version = (yaml.safe_load(raw) or {}).get("policy_version")
                if version:
                    found.setdefault(version, raw)
    live = config_path("policy").read_text(encoding="utf-8")
    found[(yaml.safe_load(live) or {}).get("policy_version")] = live
    return found


def _order(version: str) -> tuple:
    """HL-POLICY-2026.9 before HL-POLICY-2026.10: numbers compare as numbers."""
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in str(version).replace("-", ".").split("."))


def versions() -> list[dict[str, Any]]:
    raws = _raws()
    live = (yaml.safe_load(config_path("policy").read_text(encoding="utf-8")) or {}).get("policy_version")
    out = []
    for version in sorted(raws, key=_order):
        parsed = yaml.safe_load(raws[version]) or {}
        out.append({"version": version, "effective_from": parsed.get("effective_from"), "current": version == live})
    return out


def get(version: str) -> dict[str, Any] | None:
    raws = _raws()
    if version not in raws:
        return None
    ordered = sorted(raws, key=_order)
    index = ordered.index(version)
    policy = yaml.safe_load(raws[version]) or {}
    previous = ordered[index - 1] if index > 0 else None
    following = ordered[index + 1] if index + 1 < len(ordered) else None
    live = versions()[-1]["version"] if ordered else None
    return {
        "version": version, "effective_from": policy.get("effective_from"),
        "current": next((v["current"] for v in versions() if v["version"] == version), False),
        "previous": previous, "next": following, "latest": live,
        "policy": policy, "raw": raws[version],
        "changes_from_previous": config_changes.diff(yaml.safe_load(raws[previous]) or {}, policy) if previous else [],
    }
