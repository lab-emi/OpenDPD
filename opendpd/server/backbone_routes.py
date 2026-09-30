"""Authenticated data-only backbone upload and explicit public submission."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from opendpd.core.backbone_template import TEMPLATE
from opendpd.schemas.user_backbones import (BackboneCapability, BackboneCatalog, BackboneConsent,
    BackboneUpload, BackboneUploadRequest)
from opendpd.server.errors import api_error
from opendpd.server.routes import require_csrf, require_session
from opendpd.services.user_backbones import verify_package

router = APIRouter(prefix="/backbones", tags=["user backbones"])


@router.get("/template", dependencies=[Depends(require_session)])
def template():
    return Response(TEMPLATE, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="opendpd_backbone_template.py"'})


@router.get("/capability", response_model=BackboneCapability, dependencies=[Depends(require_session)])
def capability(request: Request):
    if not request.app.state.allow_backbone_publications:
        return BackboneCapability(reason="Public backbone contributions are disabled on this Studio host. Private template uploads and training are available.")
    return request.app.state.user_backbones.publisher.capability()


@router.get("/uploads", response_model=list[BackboneUpload], dependencies=[Depends(require_session)])
def uploads(request: Request):
    return request.app.state.user_backbones.list()


@router.get("/uploads/{publication_id}/source", dependencies=[Depends(require_session)])
def source(publication_id: str, request: Request):
    controller = request.app.state.user_backbones
    record = controller.get(publication_id)
    package = controller.directory(publication_id) / "package"
    verify_package(record, package)
    return Response((package / "backbone.py").read_bytes(), media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="backbone.py"'})


@router.post("/uploads", response_model=BackboneUpload, status_code=201, dependencies=[Depends(require_csrf)])
def upload(body: BackboneUploadRequest, request: Request):
    try:
        source = body.source.encode("utf-8")
    except UnicodeEncodeError:
        raise api_error(422, "invalid_backbone", "Backbone source must be valid UTF-8 text.") from None
    return request.app.state.user_backbones.upload(body.filename, source)


@router.post("/uploads/{publication_id}/submit", response_model=BackboneUpload, dependencies=[Depends(require_csrf)])
def submit(publication_id: str, body: BackboneConsent, request: Request):
    if not request.app.state.allow_backbone_publications:
        raise api_error(403, "publication_unavailable", "Public backbone contributions are disabled on this Studio host.")
    return request.app.state.user_backbones.start(publication_id, body)


@router.get("/catalog", response_model=BackboneCatalog, dependencies=[Depends(require_session)])
def catalog(request: Request):
    return request.app.state.user_backbones.catalog()


@router.post("/catalog/refresh", response_model=BackboneCatalog, dependencies=[Depends(require_csrf)])
def refresh(request: Request):
    return request.app.state.user_backbones.refresh_catalog()
