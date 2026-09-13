"""Private bundles that can rebuild RF metrics and a saved figure together."""
import hashlib
import io
import json
import importlib.metadata
import tempfile
import zipfile
from pathlib import Path

from opendpd.services import figures, packages, reproduce_figure
from opendpd.services.workspace import PACKAGE_ROOT, sha256_file, software_provenance


def implementation_files():
    paths = [p for folder in ('opendpd', 'backbones', 'modules', 'steps', 'quant', 'utils') for p in (PACKAGE_ROOT / folder).rglob('*.py')]
    paths += [PACKAGE_ROOT / f for f in ('arguments.py', 'project.py', 'models.py', 'main.py')]
    paths += list((PACKAGE_ROOT / 'opendpd' / 'studio' / 'locales').glob('*.json'))
    return {p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in paths if p.is_file()}


def export_reproduction(ws, figure_id, out):
    figure = figures.load_figure(ws, figure_id)
    # Source validation precedes private data collection and is repeated at the end.
    content = figures.export_figure(ws, figure_id)
    out = Path(out)
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='opendpd-reproduce-runs-') as tmp, zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name, value):
            archive.writestr(name, value)
            hashes[name] = hashlib.sha256(value).hexdigest()
        with zipfile.ZipFile(io.BytesIO(content)) as original:
            for name in original.namelist():
                if name != 'manifest.json':
                    add(name, original.read(name))
        for run in figure.spec.profiles:
            path = Path(tmp) / f'{run}.zip'
            packages.export_run(ws, run, path, kind='full', include_builtin_data=True)
            name = f'run-packages/{run}.zip'
            archive.write(path, name)
            hashes[name] = sha256_file(path)
        add('reproduce.py', Path(reproduce_figure.__file__).read_bytes())
        implementation = implementation_files()
        add('implementation.json', json.dumps(implementation, indent=2, sort_keys=True).encode())
        for name in implementation:
            add(f'runtime-source/{name}', (PACKAGE_ROOT / name).read_bytes())
        versions = {}
        for name in ('torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pydantic'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        add('reproduction-environment.json', json.dumps({'studio': software_provenance().model_dump(mode='json'), 'packages': versions}, indent=2).encode())
        add('runtime-requirements.txt', ('\n'.join(f'{name}=={version}' for name, version in versions.items() if version) + '\n').encode())
        add('FULL-REPRODUCTION.md', (
            '# Private metric and view reproduction\n\n'
            'This bundle includes raw datasets (including built-ins), weights and private run metadata.\n'
            'Extract it into an empty directory. With the matching OpenDPD source installed, run:\n\n'
            '`python reproduce.py . --workspace /path/to/a/new/empty-workspace`\n\n'
            'Alternatively, use the exact included Python source (including uncommitted changes):\n\n'
            '`python reproduce.py . --workspace /path/to/a/new/empty-workspace --use-bundled-source`\n\n'
            'This option executes the included OpenDPD source; use it only for a bundle you trust. '
            'runtime-requirements.txt pins the numerical dependencies and can be installed in a separate environment.\n\n'
            'The script verifies hashes and exact Python source identity, imports each run into a separate workspace, '
            're-evaluates its saved profile, regenerates display arrays, and renders the saved layout with regenerated coordinates. '
            'It never retrains, sends data, publishes or changes the source workspace.\n\n'
            'Inspect reproduction-check.json, reproduced-figures/ and replayed/. CPU float32 metric checks use the proposed '
            'repository tolerance rel=1e-4 / abs=1e-5; this is not an independently approved scientific threshold. '
            'Other devices require their separately recorded tolerances. The same numeric check is explicitly reported for display coordinates. '
            'A mismatch fails the command and remains visible in the report.\n\n'
            'implementation.json identifies the required source, including uncommitted changes; version number alone is insufficient. '
            'reproduction-environment.json records the exporter dependencies and device. Install dependencies in a separate environment if needed. '
            'The ordinary `python replay.py .` only renders saved arrays and needs Matplotlib.\n'
        ).encode())
        figures.validate_sources(ws, figure)
        add('manifest.json', json.dumps({'version': 'figure-reproduction-v1', 'files': dict(hashes)}, indent=2).encode())
    return out
