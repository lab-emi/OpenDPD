"""Qualified implementation cost evidence; local Studio only."""
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import FileResponse

from opendpd.schemas.hardware import CostAttachment, HardwareCostDraft, HardwareCostEntry, HardwareCostReport
from opendpd.server.routes import require_csrf, require_session
from opendpd.services import hardware_costs
from opendpd.services.workspace import WorkspaceError, sha256_file

router = APIRouter(tags=['hardware costs'])


@router.get('/hardware/costs', response_model=HardwareCostReport, dependencies=[Depends(require_session)])
def costs(request: Request, runs: str, profile: str = 'general-spectral-v1'):
    from pydantic import TypeAdapter
    from opendpd.schemas.common import Slug
    try:
        identifiers = [TypeAdapter(Slug).validate_python(s) for s in runs.split(',')]
        profile = TypeAdapter(Slug).validate_python(profile)
    except ValueError:
        raise WorkspaceError('Choose run and metric profile identifiers, not paths.') from None
    return hardware_costs.report(request.app.state.ws, identifiers, profile)


@router.post('/hardware/reports/upload', response_model=CostAttachment, dependencies=[Depends(require_csrf)])
async def upload(request: Request, file: UploadFile):
    try:
        data = await file.read(5 * 1024 * 1024 + 1)
        return hardware_costs.attachment(request.app.state.ws, file.filename or '', data)
    finally:
        await file.close()


@router.post('/hardware/costs', response_model=HardwareCostEntry, status_code=201, dependencies=[Depends(require_csrf)])
def record(request: Request, body: HardwareCostDraft):
    return hardware_costs.reported(request.app.state.ws, body)


@router.get('/hardware/reports/{digest}/download', dependencies=[Depends(require_session)])
def download(request: Request, digest: str):
    import re
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise WorkspaceError('Invalid hardware report hash.')
    sources = (hardware_costs.directory(request.app.state.ws) / 'attachments').glob(f'{digest}.*')
    path = next((p for p in sources if p.is_file() and not p.is_symlink() and sha256_file(p) == digest), None)
    if path is None:
        raise WorkspaceError('The hardware report is missing or changed.')
    return FileResponse(path, media_type='application/octet-stream', filename=path.name)
