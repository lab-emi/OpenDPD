"""Authenticated virtual PA simulation: input-only data becomes paired data explicitly."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from opendpd.core.virtual_pa import catalog
from opendpd.schemas.virtual_pa import VirtualPAModel, VirtualPARequest, VirtualPASimulation, PairedDatasetRequest
from opendpd.schemas.signal_generator import GeneratorDatasetResponse
from opendpd.server.routes import require_csrf, require_session
from opendpd.services import virtual_pa

router = APIRouter(prefix="/pa-library", tags=["virtual PA library"])


@router.get("/models", response_model=list[VirtualPAModel], dependencies=[Depends(require_session)])
def models():
    return catalog()


@router.post("/simulations", response_model=VirtualPASimulation, status_code=201, dependencies=[Depends(require_csrf)])
def simulate(body: VirtualPARequest, request: Request):
    return virtual_pa.preview(request.app.state.ws, body)


@router.get("/simulations/{simulation_id}", response_model=VirtualPASimulation, dependencies=[Depends(require_session)])
def simulation(simulation_id: str, request: Request):
    return virtual_pa.read_simulation(request.app.state.ws, simulation_id)


@router.get("/simulations/{simulation_id}/{filename}", dependencies=[Depends(require_session)])
def download(simulation_id: str, filename: str, request: Request):
    from opendpd.server.routes import _error
    kinds = {"output.csv": "output", "paired.csv": "paired", "metadata.json": "metadata"}
    if filename not in kinds:
        raise _error(404, "export_not_found", "Unknown PA simulation export.")
    return FileResponse(virtual_pa.export(request.app.state.ws, simulation_id, kinds[filename]),
        media_type="application/json" if filename.endswith(".json") else "text/csv",
        filename=f"{simulation_id[:18]}-{filename}")


@router.post("/simulations/{simulation_id}/dataset", response_model=GeneratorDatasetResponse, status_code=201,
             dependencies=[Depends(require_csrf)])
def dataset(simulation_id: str, body: PairedDatasetRequest, request: Request):
    return virtual_pa.create_dataset(request.app.state.ws, simulation_id, body)
