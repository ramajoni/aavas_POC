"""OpenObserve: ship drift findings, and provision the alerts and dashboard that watch them.

Connection comes from the environment, never from config: AAVAS_O2_URL, AAVAS_O2_ORG,
AAVAS_O2_USER, AAVAS_O2_PASSWORD. Locally ``.env.openobserve`` (git-ignored) holds them
and ``load_env_file`` reads it. API used (OpenObserve v1.1):

    POST /api/{org}/{stream}/_json               ingest a JSON array into a log stream
    POST /api/{org}/_search                      SQL search
    POST /api/{org}/alerts/templates             alert message template
    POST /api/{org}/alerts/destinations          where a firing alert is sent (our webhook)
    GET/POST/PUT /api/v2/{org}/alerts            the drift alerts
    GET/POST/PUT /api/{org}/dashboards           the drift dashboard

Provisioning is idempotent: run it again and existing objects are updated, not duplicated.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config import load
from ..paths import ROOT


class OpenObserveError(Exception):
    pass


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env.openobserve into the environment (existing values win)."""
    path = path or ROOT / ".env.openobserve"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


class Client:
    def __init__(self, conf: dict[str, Any] | None = None):
        load_env_file()
        self.conf = conf or load("monitoring")["openobserve"]
        self.url = os.environ.get("AAVAS_O2_URL", self.conf["url_default"]).rstrip("/")
        self.org = os.environ.get("AAVAS_O2_ORG", self.conf["org_default"])
        user, password = os.environ.get("AAVAS_O2_USER"), os.environ.get("AAVAS_O2_PASSWORD")
        if not user or not password:
            raise OpenObserveError("AAVAS_O2_USER / AAVAS_O2_PASSWORD are not set (see .env.openobserve)")
        self._auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        self.timeout = int(self.conf.get("timeout_seconds", 15))

    # ------------------------------------------------------------- transport
    def request(self, method: str, path: str, body: Any = None, ok_missing: bool = False) -> Any:
        data = None if body is None else json.dumps(body, default=str).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": self._auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if ok_missing and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OpenObserveError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OpenObserveError(f"OpenObserve at {self.url} is unreachable: {exc.reason}") from exc
        return json.loads(raw) if raw else None

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=5) as resp:
                return resp.status == 200
        except Exception:                                            # noqa: BLE001
            return False

    # ------------------------------------------------------------- data
    def ingest(self, stream: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"successful": 0}
        out = self.request("POST", f"/api/{self.org}/{stream}/_json", records)
        status = (out or {}).get("status", [{}])[0]
        return {"stream": stream, "successful": status.get("successful", 0), "failed": status.get("failed", 0)}

    def search(self, sql: str, minutes: int = 24 * 60, size: int = 100) -> list[dict[str, Any]]:
        end = int(time.time() * 1_000_000)
        body = {"query": {"sql": sql, "start_time": end - minutes * 60 * 1_000_000, "end_time": end,
                          "from": 0, "size": size}}
        return (self.request("POST", f"/api/{self.org}/_search", body) or {}).get("hits", [])

    # ------------------------------------------------------------- provisioning
    def provision(self) -> dict[str, Any]:
        done = {"template": self._template(), "destination": self._destination()}
        done["alerts"] = [self._alert(a) for a in self.conf["alerts"]]
        done["dashboard"] = self._dashboard()
        return done

    def _template(self) -> str:
        name = "aavas_drift_alert"
        body = {"name": name, "type": "http", "title": "", "body": json.dumps({
            "source": "openobserve", "alert": "{alert_name}", "status": "{alert_status}",
            "stream": "{stream_name}", "rows": "{rows}", "triggered_at": "{alert_trigger_time_str}"})}
        existing = self.request("GET", f"/api/{self.org}/alerts/templates/{name}", ok_missing=True)
        self.request("PUT" if existing else "POST",
                     f"/api/{self.org}/alerts/templates" + (f"/{name}" if existing else ""), body)
        return name

    def _destination(self) -> str:
        dest = self.conf["alert_destination"]
        token = os.environ.get("AAVAS_O2_WEBHOOK_TOKEN", "")
        headers = {"X-Aavas-Webhook-Token": token} if token else {}      # the API refuses alerts without it
        body = {"name": dest["name"], "type": "http", "url": dest["url"], "method": "post",
                "skip_tls_verify": False, "template": "aavas_drift_alert", "headers": headers, "emails": []}
        existing = self.request("GET", f"/api/{self.org}/alerts/destinations/{dest['name']}", ok_missing=True)
        self.request("PUT" if existing else "POST",
                     f"/api/{self.org}/alerts/destinations" + (f"/{dest['name']}" if existing else ""), body)
        return dest["name"]

    def _alert(self, spec: dict[str, Any]) -> dict[str, Any]:
        stream = self.conf["streams"]["findings"]
        body = {
            "name": spec["name"], "description": spec["description"], "enabled": True,
            "stream_type": "logs", "stream_name": stream, "is_real_time": False,
            "query_condition": {"type": "sql", "sql": (
                f"SELECT name, status, psi, ks_stat, current_value, reference_value FROM \"{stream}\" "
                f"WHERE kind = '{spec['kind']}' AND status = 'alert'")},
            "trigger_condition": {"period": int(self.conf["alert_period_minutes"]), "operator": ">=",
                                  "threshold": 1, "frequency": int(self.conf["alert_frequency_minutes"]),
                                  "frequency_type": "minutes", "silence": int(self.conf["alert_silence_minutes"])},
            "destinations": [self.conf["alert_destination"]["name"]],
            "creates_incident": True, "tags": ["aavas", "drift", spec["kind"]],
        }
        listing = self.request("GET", f"/api/v2/{self.org}/alerts") or {}
        found = next((a for a in listing.get("list", listing.get("alerts", [])) if a.get("name") == spec["name"]), None)
        if found:
            alert_id = found.get("alert_id") or found.get("id")
            self.request("PUT", f"/api/v2/{self.org}/alerts/{alert_id}", {**body, "id": alert_id})
            return {"name": spec["name"], "id": alert_id, "action": "updated"}
        created = self.request("POST", f"/api/v2/{self.org}/alerts", body) or {}
        return {"name": spec["name"], "id": created.get("id") or created.get("alert_id"), "action": "created"}

    def _dashboard(self) -> dict[str, Any]:
        s = self.conf["streams"]
        title = "Aavas model drift"

        def panel(i, kind, ptitle, sql, stream, x, ys, layout):
            axis = lambda a: {"label": a, "alias": a, "column": a, "color": None, "functionName": None}  # noqa: E731
            return {"id": f"panel_{i}", "type": kind, "title": ptitle, "description": "",
                    "config": {"show_legends": True, "legends_position": None, "decimals": 3},
                    "queryType": "sql", "layout": {"x": layout[0], "y": layout[1], "w": layout[2], "h": layout[3], "i": i},
                    "queries": [{"query": sql, "customQuery": True, "vrlFunctionQuery": "",
                                 "fields": {"stream": stream, "stream_type": "logs", "x": [axis(x)],
                                            "y": [axis(y) for y in ys], "z": [], "breakdown": [],
                                            "filter": {"filterType": "group", "logicalOperator": "AND", "conditions": []}},
                                 "config": {"promql_legend": "", "layer_type": "scatter", "weight_fixed": 1,
                                            "limit": 0, "min": 0, "max": 100}}]}

        panels = [
            panel(1, "table", "Drift runs: status by kind",
                  f"SELECT run_id, scenario, overall_status, data_status, model_status, concept_status, current_from, "
                  f"current_to FROM \"{s['summary']}\" ORDER BY _timestamp DESC", s["summary"], "run_id",
                  ["scenario", "overall_status", "data_status", "model_status", "concept_status"], (0, 0, 96, 10)),
            panel(2, "table", "Findings at warn or alert",
                  f"SELECT run_id, kind, name, status, psi, ks_stat, reference_value, current_value, change_pct, "
                  f"change_detected_at FROM \"{s['findings']}\" WHERE status != 'ok' ORDER BY _timestamp DESC",
                  s["findings"], "name", ["kind", "status", "psi", "ks_stat", "current_value"], (0, 10, 96, 12)),
            panel(3, "bar", "Data drift: PSI by feature (all runs)",
                  f"SELECT name AS x, MAX(psi) AS y FROM \"{s['findings']}\" WHERE kind = 'data' GROUP BY name "
                  f"ORDER BY y DESC LIMIT 15", s["findings"], "x", ["y"], (0, 22, 48, 12)),
            panel(4, "bar", "Model drift: PSI by prediction (all runs)",
                  f"SELECT name AS x, MAX(psi) AS y FROM \"{s['findings']}\" WHERE kind = 'model' GROUP BY name "
                  f"ORDER BY y DESC", s["findings"], "x", ["y"], (48, 22, 48, 12)),
            panel(5, "line", "Concept: Stage-2 rate error by month (bps)",
                  f"SELECT month AS x, AVG(stage2_mae_bps) AS y FROM \"{s['monthly']}\" GROUP BY month ORDER BY month",
                  s["monthly"], "x", ["y"], (0, 34, 48, 12)),
            panel(6, "line", "Concept: conversion observed vs predicted by month (%)",
                  f"SELECT month AS x, AVG(observed_conversion_pct) AS observed, AVG(predicted_conversion_pct) AS "
                  f"predicted FROM \"{s['monthly']}\" GROUP BY month ORDER BY month",
                  s["monthly"], "x", ["observed", "predicted"], (48, 34, 48, 12)),
        ]
        body = {"version": 5, "title": title, "description": "Data, model and concept drift of the pricing models",
                "variables": {"list": []},                 # created / owner / role: set by the server
                "defaultDatetimeDuration": {"type": "relative", "relativeTimePeriod": "7d", "startTime": 0, "endTime": 0},
                "tabs": [{"tabId": "default", "name": "Drift", "panels": panels}]}
        listing = self.request("GET", f"/api/{self.org}/dashboards") or {}
        items = listing.get("dashboards", listing.get("list", []))
        found = next((d for d in items if (d.get("title") or (d.get("v5") or {}).get("title")) == title), None)
        if found:
            dash_id = found.get("dashboard_id") or found.get("dashboardId") or (found.get("v5") or {}).get("dashboardId")
            hash_ = found.get("hash")
            path = f"/api/{self.org}/dashboards/{dash_id}" + (f"?hash={hash_}" if hash_ else "")
            self.request("PUT", path, {**body, "dashboardId": dash_id})
            return {"title": title, "id": dash_id, "action": "updated"}
        created = self.request("POST", f"/api/{self.org}/dashboards", body) or {}
        dash_id = created.get("dashboardId") or (created.get("v5") or {}).get("dashboardId") or created.get("dashboard_id")
        return {"title": title, "id": dash_id, "action": "created"}


# ------------------------------------------------------------------ report -> streams


def ship(report: dict[str, Any], client: Client | None = None) -> dict[str, Any]:
    """Send one drift report to its three streams."""
    client = client or Client()
    s = client.conf["streams"]
    base = {"run_id": report["run_id"], "scenario": report["scenario"], "config_version": report["config_version"]}
    summary = {**base, "overall_status": report["status"]["overall"], "data_status": report["status"]["data"],
               "model_status": report["status"]["model"], "concept_status": report["status"]["concept"],
               "reference_from": report["windows"]["reference"]["from"], "reference_to": report["windows"]["reference"]["to"],
               "reference_n": report["windows"]["reference"]["n"], "current_from": report["windows"]["current"]["from"],
               "current_to": report["windows"]["current"]["to"], "current_n": report["windows"]["current"]["n"],
               **{f"{k}_{st}": n for k, counts in report["counts"].items() for st, n in counts.items()}}
    findings = [{**base, **{k: v for k, v in f.items() if v is not None}} for kind in ("data", "model", "concept")
                for f in report[kind]]
    # The month is a field, not the event time: OpenObserve refuses events older than its
    # ingest window (ZO_INGEST_ALLOWED_UPTO, 5 hours by default), and these months are history.
    monthly = [{**base, **{k: v for k, v in m.items() if v is not None}} for m in report["monthly"]]
    return {"summary": client.ingest(s["summary"], [summary]), "findings": client.ingest(s["findings"], findings),
            "monthly": client.ingest(s["monthly"], monthly)}
