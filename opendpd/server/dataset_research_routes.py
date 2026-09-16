"""Local dataset generation and explicit public contribution workflow."""
from __future__ import annotations

import zipfile
from typing import List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from opendpd.schemas.dataset_catalog import (DatasetPublication, DatasetPublicationDraft,
    PublicationCapability, PublicationConsent, SyntheticSuite, SyntheticSuiteRequest)
from opendpd.server.routes import require_csrf, require_session
from opendpd.server.errors import api_error as _error
from opendpd.services.dataset_catalog import checked_file
from opendpd.services.synthetic_datasets import generate_suite

router = APIRouter(tags=["dataset research"])


@router.post("/datasets/synthetic", response_model=SyntheticSuite, status_code=201, dependencies=[Depends(require_csrf)])
def synthetic(body: SyntheticSuiteRequest, request: Request):
    return generate_suite(request.app.state.ws, body)


@router.get("/dataset-publications/capability", response_model=PublicationCapability, dependencies=[Depends(require_session)])
def capability(request: Request):
    if not request.app.state.allow_dataset_publications:
        return PublicationCapability(available=False, reason="Public dataset submissions are disabled on this Studio host. The host operator can enable the dataset review service.")
    return request.app.state.dataset_publications.publisher.capability()


@router.get("/dataset-publications", response_model=List[DatasetPublication], dependencies=[Depends(require_session)])
def publications(request: Request, dataset_id: str | None = None):
    return request.app.state.dataset_publications.list(dataset_id)


@router.post("/dataset-publications/prepare", response_model=DatasetPublication, status_code=201, dependencies=[Depends(require_csrf)])
def prepare(body: DatasetPublicationDraft, request: Request):
    return request.app.state.dataset_publications.prepare(body)


@router.get("/dataset-publications/{publication_id}", response_model=DatasetPublication, dependencies=[Depends(require_session)])
def publication(publication_id: str, request: Request):
    return request.app.state.dataset_publications.get(publication_id)


@router.post("/dataset-publications/{publication_id}/submit", response_model=DatasetPublication, dependencies=[Depends(require_csrf)])
def submit(publication_id: str, body: PublicationConsent, request: Request):
    if not request.app.state.allow_dataset_publications:
        raise _error(403, "publication_unavailable", "Public dataset submissions are disabled on this Studio host.")
    return request.app.state.dataset_publications.start(publication_id, body)


@router.get("/dataset-publications/{publication_id}/download", dependencies=[Depends(require_session)])
def download(publication_id: str, request: Request):
    controller = request.app.state.dataset_publications
    record = controller.get(publication_id)
    package = controller.directory(publication_id) / "package"
    target = controller.directory(publication_id) / "preview.zip"
    with controller.lock:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for ref in record.files:
                archive.write(checked_file(package, ref), arcname=ref.path)
    return FileResponse(target, media_type="application/zip", filename=f"{publication_id}.zip")
