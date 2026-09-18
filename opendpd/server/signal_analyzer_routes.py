"""Authenticated signal inspection and quarantined numeric CSV import."""
from opendpd.server.uploads import consume_upload
from fastapi import APIRouter, Depends, Request, UploadFile
from opendpd.schemas.signal_analyzer import AnalyzerRequest, AnalyzerSourceInfo, SignalAnalysis
from opendpd.schemas.signal_dataset import AnalyzerDataset
from opendpd.server.routes import require_csrf, require_session
from opendpd.server.errors import api_error as _error
from opendpd.services import signal_analyzer as service
from opendpd.services.csv_upload import MAX_UPLOAD_BYTES, CsvUploadRejected, check_filename, quarantine_path

router = APIRouter(tags=["signal analyzer"])


@router.get("/signal-analyzer/datasets", response_model=list[AnalyzerDataset], dependencies=[Depends(require_session)])
def datasets(request: Request):
    return service.list_datasets(request.app.state.ws)


@router.get("/signal-analyzer/sources", response_model=list[AnalyzerSourceInfo], dependencies=[Depends(require_session)])
def sources(request: Request):
    return service.list_sources(request.app.state.ws)


@router.post("/signal-analyzer/analyze", response_model=SignalAnalysis, dependencies=[Depends(require_csrf)])
def analyze(body: AnalyzerRequest, request: Request):
    return service.analyze(request.app.state.ws, body)


@router.post("/signal-analyzer/upload", response_model=AnalyzerSourceInfo, status_code=201, dependencies=[Depends(require_csrf)])
async def upload(request: Request, file: UploadFile):
    def receive(chunks):
        path = None
        try:
            check_filename(file.filename or "")
            path = quarantine_path(request.app.state.ws)
            size = 0
            with path.open("xb") as output:
                for chunk in chunks:
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise _error(413, "payload_too_large", "CSV files must be at most 25 MiB.")
                    output.write(chunk)
            return service.admit_signal_upload(request.app.state.ws, path)
        except CsvUploadRejected as exc:
            raise _error(422, "csv_rejected", str(exc)) from exc
        finally:
            if path:
                path.unlink(missing_ok=True)
    return await consume_upload(file, receive)
