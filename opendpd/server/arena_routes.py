"""Authenticated Arena rankings and protocol-bound local evaluation."""

from fastapi import APIRouter, Depends, Request

from opendpd.schemas.arena import (ArenaCatalog, ArenaLeaderboard,
    ArenaSubmission, ArenaSubmissionRequest)
from opendpd.server.routes import require_csrf, require_session

router = APIRouter(prefix="/arena", tags=["DPD Arena"])


@router.get("", response_model=ArenaCatalog, dependencies=[Depends(require_session)])
def catalog(request: Request):
    return request.app.state.arena.catalog()


@router.get("/boards/{board_id}", response_model=ArenaLeaderboard, dependencies=[Depends(require_session)])
def leaderboard(board_id: str, request: Request):
    return request.app.state.arena.leaderboard(board_id)


@router.get("/submissions", response_model=list[ArenaSubmission], dependencies=[Depends(require_session)])
def submissions(request: Request):
    return request.app.state.arena.list()


@router.post("/submissions", response_model=ArenaSubmission, status_code=202,
             dependencies=[Depends(require_csrf)])
def submit(body: ArenaSubmissionRequest, request: Request):
    return request.app.state.arena.submit(body)


@router.get("/submissions/{submission_id}", response_model=ArenaSubmission,
            dependencies=[Depends(require_session)])
def submission(submission_id: str, request: Request):
    return request.app.state.arena.get(submission_id)
