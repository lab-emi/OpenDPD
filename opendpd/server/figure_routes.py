"""Publication authoring and private, combined metric/view reproduction."""
from fastapi import APIRouter, Depends, Request

from opendpd.schemas.review import FigurePreview, FigureSources, FigureSourcesRequest, FigureSpec
from opendpd.server.routes import require_csrf, require_session
from opendpd.services import figures

router = APIRouter(tags=['publication figures'])


@router.post('/figure-sources', response_model=FigureSources, dependencies=[Depends(require_csrf)])
def sources(request: Request, body: FigureSourcesRequest):
    return figures.sources(request.app.state.ws, body.profiles)


@router.post('/figure-preview', response_model=FigurePreview, dependencies=[Depends(require_csrf)])
def preview(request: Request, body: FigureSpec):
    return figures.preview_figure(request.app.state.ws, body)


@router.get('/figures/{figure_id}/reproduction', dependencies=[Depends(require_session)])
def reproduction(request: Request, figure_id: str):
    import shutil
    import tempfile
    from pathlib import Path
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask
    from opendpd.services.figure_reproduction import export_reproduction
    folder = Path(tempfile.mkdtemp(prefix='opendpd-private-figure-'))
    try:
        out = export_reproduction(request.app.state.ws, figure_id, folder / 'reproduction.zip')
        return FileResponse(out, media_type='application/zip', filename=f'{figure_id}-private-reproduction.zip',
                            background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
