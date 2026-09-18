"""Authenticated local signal generation; exported data remains private."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from opendpd.core.waveforms.generator_presets import presets
from opendpd.core.waveforms.generator import allocation
from opendpd.schemas.signal_generator import (DatasetSampleCounts, GeneratedSignal, GeneratorConfig,
    GeneratorDatasetRequest, GeneratorDatasetResponse, GeneratorPreset, GeneratorBatchRequest)
from opendpd.server.routes import require_csrf, require_session
from opendpd.server.errors import api_error as _error
from opendpd.services import signal_generator as service
from opendpd.schemas.virtual_pa import PAInputDataset
from opendpd.schemas.signal_dataset import SignalDataset

router = APIRouter(tags=["signal generator"])


@router.get("/signal-generator/datasets/{dataset_id}", response_model=SignalDataset, dependencies=[Depends(require_session)])
def signal_dataset(dataset_id: str, request: Request):
    from opendpd.services.signal_datasets import read_dataset
    return read_dataset(request.app.state.ws, dataset_id)


@router.post("/signal-generator/datasets/{dataset_id}/archive", response_model=SignalDataset, dependencies=[Depends(require_csrf)])
def archive_signal_dataset(dataset_id: str, request: Request):
    from opendpd.services.signal_datasets import archive_dataset
    return archive_dataset(request.app.state.ws, dataset_id)


@router.post("/signal-generator/datasets/{dataset_id}/restore", response_model=SignalDataset, dependencies=[Depends(require_csrf)])
def restore_signal_dataset(dataset_id: str, request: Request):
    from opendpd.services.signal_datasets import archive_dataset
    return archive_dataset(request.app.state.ws, dataset_id, restore=True)


@router.get("/signal-generator/datasets/{dataset_id}/download", dependencies=[Depends(require_session)])
def signal_dataset_download(dataset_id: str, request: Request):
    import shutil
    from starlette.background import BackgroundTask
    from opendpd.services.signal_datasets import export_dataset
    path, temporary = export_dataset(request.app.state.ws, dataset_id)
    return FileResponse(path, filename=path.name, media_type="application/zip",
        background=BackgroundTask(shutil.rmtree, temporary, ignore_errors=True))


@router.get("/signal-generator/signals", response_model=list[PAInputDataset], dependencies=[Depends(require_session)])
def inputs(request: Request):
    return service.list_inputs(request.app.state.ws)


@router.get("/signal-generator/signals/{signal_id}/input.csv", dependencies=[Depends(require_session)])
def input_csv(signal_id: str, request: Request):
    return FileResponse(service.export_input(request.app.state.ws, signal_id, "csv"),
        media_type="text/csv", filename=f"{signal_id[:19]}-pa-input.csv")


@router.get("/signal-generator/signals/{signal_id}/metadata.json", dependencies=[Depends(require_session)])
def metadata(signal_id: str, request: Request):
    return FileResponse(service.export_input(request.app.state.ws, signal_id, "metadata"),
        media_type="application/json", filename=f"{signal_id[:19]}-pa-input-metadata.json")


@router.post("/signal-generator/validate", response_model=GeneratorConfig, dependencies=[Depends(require_csrf)])
def validate(body: GeneratorConfig):
    try:
        if body.waveform == "ofdm":
            allocation(body)
    except ValueError as exc:
        raise _error(422, "invalid_signal", str(exc)) from exc
    return body


@router.get("/signal-generator/presets", response_model=list[GeneratorPreset], dependencies=[Depends(require_session)])
def list_presets():
    return presets()


@router.post("/signal-generator/signals", response_model=GeneratedSignal, status_code=201, dependencies=[Depends(require_csrf)])
def generate(body: GeneratorConfig, request: Request):
    return service.generate(request.app.state.ws, body)


@router.get("/signal-generator/signals/{signal_id}", response_model=GeneratedSignal, dependencies=[Depends(require_session)])
def get(signal_id: str, request: Request):
    return service.read_signal(request.app.state.ws, signal_id)


@router.get("/signal-generator/signals/{signal_id}/download", dependencies=[Depends(require_session)])
def download(signal_id: str, request: Request):
    return FileResponse(service.export_signal(request.app.state.ws, signal_id), media_type="application/zip", filename=f"{signal_id[:19]}-waveform.zip")


@router.post("/signal-generator/signals/{signal_id}/dataset", response_model=GeneratorDatasetResponse, deprecated=True,
             status_code=201, dependencies=[Depends(require_csrf)])
def dataset(signal_id: str, body: GeneratorDatasetRequest, request: Request):
    return service.create_dataset(request.app.state.ws, signal_id, body)


@router.get("/datasets/{dataset_id}/sample-counts", response_model=DatasetSampleCounts, dependencies=[Depends(require_session)])
def counts(dataset_id: str, request: Request, version: str = "raw-v1"):
    return service.sample_counts(request.app.state.ws, dataset_id, version)


@router.post('/signal-generator/signals/{signal_id}/archive', dependencies=[Depends(require_csrf)])
def archive(signal_id: str, request: Request):
    return service.archive_input(request.app.state.ws, signal_id)


@router.post('/signal-generator/signals/{signal_id}/restore', response_model=PAInputDataset, dependencies=[Depends(require_csrf)])
def restore(signal_id: str, request: Request):
    return service.archive_input(request.app.state.ws, signal_id, restore=True)


@router.post("/signal-generator/batches", response_model=list[PAInputDataset], status_code=201,
             dependencies=[Depends(require_csrf)])
def generate_batch(body: GeneratorBatchRequest, request: Request):
    return service.generate_batch(request.app.state.ws, body)


@router.get("/datasets/{dataset_id}/download", dependencies=[Depends(require_session)])
def dataset_download(dataset_id: str, request: Request, version: str = "raw-v1", collection: bool = True):
    import shutil
    from starlette.background import BackgroundTask
    from opendpd.services.dataset_downloads import export_dataset
    path, temporary = export_dataset(request.app.state.ws, dataset_id, version, collection)
    return FileResponse(path, filename=path.name,
        media_type="application/zip" if path.suffix == ".zip" else "text/csv",
        background=BackgroundTask(shutil.rmtree, temporary, ignore_errors=True))
