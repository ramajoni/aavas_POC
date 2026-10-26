"""FastAPI application.

    POST /pricing/pre-assessment    indicative band, eligibility, confidence
    POST /pricing/final-assessment  risk-adjusted recommendation + reason codes
    POST /pricing/override          capture the human decision
    POST /pricing/outcome           capture what actually happened

Everything else is supporting surface: policy citation, audit retrieval and the
override-learning review.
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from ..config import versions
from ..governance import model_registry, request_log, store
from ..logging_utils import get_logger
from . import (access, agents, applications, auth, competitors, customers, learning, market_check, offers, configuration, connectors, control_tower,
               explain, features, governance, outcome, override, portfolio, security, stage1,
               system_of_record, stage2, tracking, xai)
from .contracts import HealthResponse

log = get_logger(__name__)

app = FastAPI(
    title="Aavas Home Loan Pricing Intelligence",
    version=versions()["app_version"],
    description=(
        "Two-stage home-loan pricing with risk-adjusted yield optimization, explainable "
        "recommendations, human override capture and outcome-driven learning.\n\n"
        "The engine recommends and explains. It does not approve: every recommendation is "
        "routed to a human authority and every override is recorded with its reason.\n\n"
        "### Try one complete demo flow\n"
        "Use the prefilled examples in this order (all use `DEMO-FLOW-001`):\n"
        "1. `GET /health`: confirm all models are loaded.\n"
        "2. `POST /pricing/pre-assessment`: get the indicative band.\n"
        "3. `POST /pricing/final-assessment`: get the recommendation; save `decision_id`. "
        "Set `include_scenarios=true` to inspect pricing alternatives.\n"
        "4. `GET /pricing/override/reasons`: inspect reasons and authority limits.\n"
        "5. `POST /pricing/override` (optional): use a negotiated `human_rate` inside "
        "the returned envelope and an appropriate approver. Inspect `policy_check.allowed` "
        "and `within_authority`; recording an override is not approval. The sample approver "
        "is fictional demo data.\n"
        "6. `POST /pricing/outcome`: set `sanctioned_rate` to the actual agreed rate "
        "(the override rate if used); leave unobserved future outcomes null.\n"
        "7. `GET /pricing/audit/{loan_id}`: use `DEMO-FLOW-001`; "
        "`GET /pricing/decision/{decision_id}`: paste the ID from step 3.\n"
        "8. Explore `/model-cards`, `/pricing/policy/explain`, and `/pricing/learning/*`. "
        "Learning reports use the training dataset, not just this demo loan.\n\n"
        "Bureau scores must be 300–900, or -1 for new to credit. "
        "Examples are synthetic; reuse the same loan ID throughout a flow, "
        "and choose a new ID for a separate application."
    ),
)

def route_path(request: Request) -> str:
    """The path the router dispatches on.

    Not ``request.url.path``: Starlette builds that URL partly from the Host
    header, so ``Host: x/health?`` would make any route read as /health to a
    check that used it, while the router still ran the real handler.
    """
    path, root = request.scope["path"], request.scope.get("root_path", "")
    return path[len(root):] if root and path.startswith(root) else path


@app.middleware("http")
async def log_priced_requests(request: Request, call_next):
    """Open a request row before the pipeline runs, close it however it ends.

    Only the four priced endpoints are tracked; logging dashboard polling would
    bury the signal. An unhandled exception is recorded and re-raised - swallowing
    it here would turn a 500 into a silent success.
    """
    if not (request.method == "POST" and request_log.tracked(route_path(request))):
        return await call_next(request)

    started = time.perf_counter()
    try:
        request_id = request_log.open_request(route_path(request), request.method)
    except Exception as exc:                                    # noqa: BLE001
        # A store that cannot be written must not take the pricing path down; the
        # row is the audit's, not the caller's.
        log.warning("request log unavailable: %s", exc)
        return await call_next(request)

    try:
        response = await call_next(request)
    except Exception as exc:                                    # noqa: BLE001
        request_log.close_request(
            request_id, status_code=500, elapsed_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__, error_detail=str(exc))
        raise

    elapsed = (time.perf_counter() - started) * 1000
    decision_id = loan_id = None
    error_type = error_detail = None
    body = b""
    if response.status_code < 400 or response.status_code >= 500:
        # Read the body to recover the decision id, then hand back an identical
        # response - an iterator consumed here would otherwise arrive empty.
        async for chunk in response.body_iterator:
            body += chunk
        response = Response(content=body, status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type)
    if body:
        try:
            parsed = json.loads(body)
            decision_id = parsed.get("decision_id")
            loan_id = parsed.get("loan_id")
        except Exception:                                       # noqa: BLE001
            pass
    if response.status_code >= 400:
        error_type = "http_error"
        error_detail = f"status {response.status_code}"

    request_log.close_request(request_id, status_code=response.status_code,
                              elapsed_ms=elapsed, decision_id=decision_id, loan_id=loan_id,
                              error_type=error_type, error_detail=error_detail)
    return response


@app.middleware("http")
async def enforce_access(request: Request, call_next):
    """Sign-in and the role rules in config/access.yaml, before any handler runs.

    Registered after the request log, so it runs first: a caller who is not signed
    in, or whose role does not reach this route, never touches the pricing path.
    Enforced in session mode only; the other modes keep their own checks.
    """
    if security.mode() != security.SESSION or request.method == "OPTIONS":
        return await call_next(request)

    method, path = request.method, route_path(request)
    try:
        public = access.is_public(method, path)
        principal = None if public else security.session_principal(
            security.bearer_token(request))
        decision = None if public or principal is None else access.decide(
            method, path, principal.roles)
    except Exception as exc:                                    # noqa: BLE001
        # A missing or malformed access file must close the service, not open it.
        log.error("access control unavailable: %s", exc)
        return JSONResponse({"detail": "access control is unavailable"}, status_code=503)

    if public:
        return await call_next(request)
    if principal is None:
        return JSONResponse({"detail": "sign in to continue"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})
    if principal.must_change_password and not (path == "/auth" or path.startswith("/auth/")):
        # A temporary password opens one door: setting your own.
        return JSONResponse({"detail": "Set your own password before continuing."}, status_code=403)
    if not decision.allowed:
        label = access.role_config(principal.roles[0]).get("label", principal.roles[0])
        return JSONResponse(
            {"detail": f"Your role ({label}) does not have access to this part of the service."},
            status_code=403)
    request.state.principal = principal
    return await call_next(request)


app.include_router(auth.router)
app.include_router(applications.router)
app.include_router(customers.router)
app.include_router(offers.router)
app.include_router(stage1.router)
app.include_router(stage2.router)
app.include_router(explain.router)
app.include_router(override.router)
app.include_router(outcome.router)
app.include_router(agents.router)
app.include_router(configuration.router)
app.include_router(connectors.router)
app.include_router(control_tower.router)
app.include_router(governance.router)
app.include_router(features.router)
app.include_router(system_of_record.router)
app.include_router(tracking.router)
app.include_router(xai.router)
app.include_router(portfolio.router)
# Before competitors: its GET /competitors/{name} would otherwise take /competitors/lenders.
app.include_router(learning.router)
app.include_router(market_check.router)
app.include_router(competitors.router)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    bundle = model_registry.load_bundle()
    try:
        decisions = store.count("decisions")
    except Exception:                               # noqa: BLE001
        decisions = -1
    return HealthResponse(
        status="ok" if bundle.ready else "degraded",
        models_loaded=bundle.versions(),
        models_missing=bundle.missing or [],
        versions=versions(),
        decisions_recorded=decisions,
    )


@app.get("/model-cards", tags=["operations"], summary="Model cards, including known limitations")
def model_cards() -> dict:
    return model_registry.load_bundle().cards()


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "aavas-pricing", "docs": "/docs", "health": "/health"}
