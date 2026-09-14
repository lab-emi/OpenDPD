"""The public allowlist must publish only routes somebody listed on purpose.

`policy.ROUTES` matches request paths with regexes. A pattern that stands in for
an *id* (``SLUG``/``FILE_ID``) also matches any static sibling route under the
same prefix, and FastAPI then dispatches that request to the static endpoint
rather than to the by-id endpoint the pattern was written for. The route is
published without appearing in anyone's list.

This test walks the real application and fails when a static route is reachable
*only* through an id pattern, so a new local route cannot become public by
accident. It is the standing check behind `policy.NEVER_PUBLIC`.
"""

import re
from pathlib import Path

from fastapi.routing import APIRoute

from opendpd.server.app import create_app
from opendpd.web.policy import FILE_ID, NEVER_PUBLIC, ROUTES, SLUG, allowed

API_PREFIX = "/api/v1"
ID_PLACEHOLDERS = (SLUG, FILE_ID)


def _static_api_routes(workspace: Path):
    app = create_app(workspace, bootstrap_token="surface")
    for route in app.routes:
        if not isinstance(route, APIRoute) or "{" in route.path:
            continue                                  # only static routes can be shadowed
        if not route.path.startswith(API_PREFIX):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            yield method, route.path[len(API_PREFIX):], route.endpoint.__name__


def test_no_static_route_is_published_only_by_an_id_pattern(tmp_path):
    accidental = []
    for method, path, endpoint in _static_api_routes(tmp_path):
        if not allowed(method, path):
            continue
        matching = [p for p in ROUTES.get(method, []) if re.fullmatch(p, path)]
        deliberate = [p for p in matching if not any(ph in p for ph in ID_PLACEHOLDERS)]
        if not deliberate:
            accidental.append(f"{method} {path} -> {endpoint}() published only by {matching}")
    assert accidental == [], (
        "these local routes are reachable from the public app without being listed in "
        "policy.ROUTES; add them to policy.NEVER_PUBLIC or list them deliberately:\n"
        + "\n".join(accidental))


def test_import_roots_is_never_published():
    """It answers with absolute host paths, which the public app hides elsewhere."""
    assert "/datasets/import-roots" in NEVER_PUBLIC
    assert not allowed("GET", "/datasets/import-roots")
    assert not allowed("GET", "/datasets/import-roots/data/files")


def test_never_public_does_not_shadow_a_real_dataset_id():
    """The guard matches whole segments, not prefixes of a longer id."""
    assert allowed("GET", "/datasets/import-roots-2024")
    assert allowed("GET", "/datasets/dpa-200mhz")
