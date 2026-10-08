"""Authenticated local MATLINK queue; MATLAB retains control of its workspace."""
from fastapi import APIRouter, Depends, Request

from opendpd.schemas.matlink import (
    MatlinkCompletion, MatlinkConnect, MatlinkConnection, MatlinkDisconnected,
    MatlinkHeartbeat, MatlinkPoll, MatlinkRequest, MatlinkState, MatlinkTransfer,
)
from opendpd.server.errors import api_error
from opendpd.server.routes import require_csrf, require_session

router = APIRouter(prefix="/matlink", tags=["MATLINK"])
BRIDGE_HEADER = "x-opendpd-matlink"


def _secret(request):
    values = request.headers.getlist(BRIDGE_HEADER)
    if len(values) != 1:
        raise api_error(403, "matlink_bridge_required", "Provide the connected MATLAB bridge token exactly once.")
    return values[0]


@router.get("", response_model=MatlinkState, dependencies=[Depends(require_session)])
def state(request: Request):
    return request.app.state.matlink.snapshot()


@router.post("/connect", response_model=MatlinkConnection, status_code=201, dependencies=[Depends(require_csrf)])
def connect(body: MatlinkConnect, request: Request):
    return request.app.state.matlink.connect(body)


@router.post("/requests", response_model=MatlinkTransfer, status_code=201, dependencies=[Depends(require_csrf)])
def transfer(body: MatlinkRequest, request: Request):
    return request.app.state.matlink.request(body)


@router.post("/{client_id}/heartbeat", response_model=MatlinkPoll, dependencies=[Depends(require_csrf)])
def heartbeat(client_id: str, body: MatlinkHeartbeat, request: Request):
    return request.app.state.matlink.heartbeat(client_id, _secret(request), body)


@router.post("/{client_id}/requests/{request_id}/complete", response_model=MatlinkTransfer,
             dependencies=[Depends(require_csrf)])
def complete(client_id: str, request_id: str, body: MatlinkCompletion, request: Request):
    return request.app.state.matlink.complete(client_id, _secret(request), request_id, body)


@router.post("/{client_id}/disconnect", response_model=MatlinkDisconnected, dependencies=[Depends(require_csrf)])
def disconnect(client_id: str, request: Request):
    request.app.state.matlink.disconnect(client_id, _secret(request))
    return MatlinkDisconnected()
