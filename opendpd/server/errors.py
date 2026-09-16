"""Public error helpers for API routers and middleware."""
from fastapi import HTTPException
from opendpd.schemas.common import error_payload


def api_error(status: int, code: str, message: str, hint=None, details=None):
    return HTTPException(status, error_payload(code, message, details, hint))
