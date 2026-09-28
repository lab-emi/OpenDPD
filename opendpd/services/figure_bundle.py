"""Read figure bundles as data using installed code, never their Python modules."""
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from contextlib import contextmanager

from opendpd.safe_paths import contained_path
from opendpd.services.packages import MAX_MANIFEST_BYTES, MAX_PACKAGE_MEMBERS, MAX_PACKAGE_UNPACKED_BYTES, _safe_member


def verify_directory(root):
    root = Path(root)
    manifest_path = contained_path(root, "manifest.json")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("bundle manifest exceeds size limit")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    if not isinstance(files, dict) or len(files) > MAX_PACKAGE_MEMBERS:
        raise ValueError("invalid bundle member list")
    total = 0
    for name, expected in files.items():
        path = contained_path(root, name)
        size = path.stat().st_size
        if name.endswith('.json') and size > (64 * 1024 * 1024 if name == 'plot-data.json' else MAX_MANIFEST_BYTES):
            raise ValueError("bundle JSON exceeds size limit")
        total += size
        if total > MAX_PACKAGE_UNPACKED_BYTES:
            raise ValueError("bundle exceeds size limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"bundle hash mismatch: {name}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle links are forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("bundle contains unlisted or missing files")
    return manifest


@contextmanager
def read_bundle(path):
    path = Path(path)
    if path.is_dir():
        verify_directory(path)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="opendpd-figure-input-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_PACKAGE_MEMBERS or sum(i.file_size for i in members) > MAX_PACKAGE_UNPACKED_BYTES:
                raise ValueError("bundle exceeds size limit")
            seen = set()
            for item in members:
                _safe_member(item.filename, item)
                key = item.filename.casefold()
                if key in seen or item.is_dir():
                    raise ValueError("ambiguous bundle member")
                seen.add(key)
            for item in members:
                target = contained_path(root, item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        verify_directory(root)
        yield root


def replay_bundle(path, output):
    from opendpd.services.figure_render import render_figure
    from opendpd.schemas.review import SavedFigure
    with read_bundle(path) as root:
        figure = SavedFigure.model_validate_json((root / "figure.json").read_text())
        plots = json.loads((root / "plot-data.json").read_text())
        from opendpd.schemas.plot_input import validate_plot
        for plot in plots.values():
            validate_plot(plot)
        render_figure(figure.model_dump(mode="json"), plots, output)


def reproduce_bundle(path, workspace):
    from opendpd.services.reproduce_figure import reproduce
    with read_bundle(path) as root:
        return reproduce(root, workspace, use_bundled_source=False, output=Path(workspace) / 'reproduction')
