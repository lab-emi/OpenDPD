"""Authenticated local Studio sweep endpoints."""

from fastapi import APIRouter, Depends, Request

from opendpd.schemas.common import Slug
from opendpd.schemas.conditions import ConditionSet, ConditionAudit
from opendpd.schemas.sweep import SweepDraft, SweepPreview, SweepRecord, SweepStart, SweepReport
from opendpd.server.routes import require_csrf, require_session
from opendpd.services.sweeps import audit_conditions, preview, report

router = APIRouter(tags=["sweeps"])


@router.post("/sweeps/conditions/validate", response_model=list[ConditionAudit], dependencies=[Depends(require_session)])
def conditions_validate(body: ConditionSet, request: Request):
    return audit_conditions(request.app.state.ws, body)


@router.post("/sweeps/preview", response_model=SweepPreview, dependencies=[Depends(require_session)])
def sweep_preview(body: SweepDraft, request: Request):
    return preview(request.app.state.ws, body)


@router.get("/sweeps", response_model=list[SweepRecord], dependencies=[Depends(require_session)])
def sweeps_list(request: Request):
    return request.app.state.sweeps.list()


@router.post("/sweeps", response_model=SweepRecord, dependencies=[Depends(require_csrf)])
def sweeps_create(body: SweepDraft, request: Request):
    return request.app.state.sweeps.create(body)


@router.get("/sweeps/{sweep_id}", response_model=SweepRecord, dependencies=[Depends(require_session)])
def sweeps_get(sweep_id: Slug, request: Request):
    return request.app.state.sweeps.get(sweep_id)


@router.get("/sweeps/{sweep_id}/report", response_model=SweepReport, dependencies=[Depends(require_session)])
def sweeps_report(sweep_id: Slug, request: Request):
    return report(request.app.state.ws, request.app.state.sweeps.get(sweep_id))


@router.post("/sweeps/{sweep_id}/start", response_model=SweepRecord, dependencies=[Depends(require_csrf)])
def sweeps_start(sweep_id: Slug, body: SweepStart, request: Request):
    return request.app.state.sweeps.start_board(sweep_id, resume_failed=body.resume_failed)


@router.post("/sweeps/{sweep_id}/cancel", response_model=SweepRecord, dependencies=[Depends(require_csrf)])
def sweeps_cancel(sweep_id: Slug, request: Request):
    return request.app.state.sweeps.cancel(sweep_id)
