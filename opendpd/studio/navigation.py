"""Same-origin Studio destinations shared by native clients and bootstrap."""

import re

PAGES = {
    "home": "/",
    "datasets": "/datasets",
    "experiments": "/experiments",
    "new-experiment": "/experiments/new",
    "results": "/results",
    "matlink": "/matlink",
}
_DETAIL = re.compile(r"/(?:datasets|runs|results)/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def valid_destination(path: str) -> bool:
    return path in PAGES.values() or _DETAIL.fullmatch(path) is not None


def destination(page="home", run_id=None):
    if page in PAGES and run_id is None:
        return PAGES[page]
    if page in ("run", "result") and isinstance(run_id, str):
        path = f"/{'runs' if page == 'run' else 'results'}/{run_id}"
        if valid_destination(path):
            return path
    raise ValueError("Choose a Studio page; run/result pages require a valid run ID")
