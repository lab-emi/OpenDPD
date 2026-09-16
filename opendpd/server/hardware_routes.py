"""Qualified implementation cost evidence; local Studio only."""
from opendpd.server.uploads import consume_upload
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import FileResponse

from opendpd.schemas.hardware import CostAttachment, HardwareCostDraft, HardwareCostEntry, HardwareCostReport
from opendpd.server.routes import require_csrf, require_session
from opendpd.services import hardware_costs
from opendpd.services.workspace import InvalidInput, NotFound, Conflict, sha256_file

router = APIRouter(tags=['hardware costs'])


@router.get('/hardware/costs', response_model=HardwareCostReport, dependencies=[Depends(require_session)])
def costs(request: Request, runs: str, profile: str = 'general-spectral-v1'):
    from pydantic import TypeAdapter
    from opendpd.schemas.common import Slug
    try:
        identifiers = [TypeAdapter(Slug).validate_python(s) for s in runs.split(',')]
        profile = TypeAdapter(Slug).validate_python(profile)
    except ValueError:
        raise InvalidInput('Choose run and metric profile identifiers, not paths.') from None
    return hardware_costs.report(request.app.state.ws, identifiers, profile)


@router.post('/hardware/reports/upload', response_model=CostAttachment, dependencies=[Depends(require_csrf)])
async def upload(request: Request, file: UploadFile):
    def receive(chunks):
        data = bytearray()
        for chunk in chunks:
            data.extend(chunk)
            if len(data) > 5 * 1024 * 1024:
                raise InvalidInput('Hardware reports must be at most 5 MiB.')
        return hardware_costs.attachment(request.app.state.ws, file.filename or '', bytes(data))
    return await consume_upload(file, receive)


@router.post('/hardware/costs', response_model=HardwareCostEntry, status_code=201, dependencies=[Depends(require_csrf)])
def record(request: Request, body: HardwareCostDraft):
    return hardware_costs.reported(request.app.state.ws, body)


@router.get('/hardware/reports/{digest}/download', dependencies=[Depends(require_session)])
def download(request: Request, digest: str):
    import re
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise InvalidInput('Invalid hardware report hash.')
    sources = (hardware_costs.directory(request.app.state.ws) / 'attachments').glob(f'{digest}.*')
    path = next((p for p in sources if p.is_file() and not p.is_symlink()), None)
    if path is None:
        raise NotFound('The hardware report is missing.')
    if sha256_file(path) != digest:
        raise Conflict('The hardware report has changed.')
    return FileResponse(path, media_type='application/octet-stream', filename=path.name)
