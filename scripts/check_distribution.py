"""Check the built release contains runtime assets, without operational or planning files."""
from pathlib import Path
import tarfile
import zipfile


def check(dist):
    wheel = next(dist.glob('*.whl'))
    source = next(dist.glob('*.tar.gz'))
    blocked = {'.github', '.vscode', 'deployment', 'papers', 'docs', 'frontend', 'tests', 'output', 'benchmark'}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'opendpd/studio/static/index.html' in names
        assert 'opendpd/studio/static/build-info.json' in names
    with tarfile.open(source) as archive:
        names = [name.split('/', 1)[-1] for name in archive.getnames()]
        assert 'opendpd/studio/static/index.html' in names
        assert not [name for name in names if name.split('/')[0] in blocked]
        assert not [name for name in names if name.endswith('CODE_REVIEW_REPORT.md') or 'Development_Plan' in name]
    assert source.stat().st_size < 60 * 1024 * 1024
    print('Wheel assets and source-distribution boundaries verified.')


if __name__ == '__main__':
    import sys
    check(Path(sys.argv[1] if len(sys.argv) > 1 else 'dist'))
