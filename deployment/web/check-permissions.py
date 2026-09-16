"""Refuse writable service code or exposed service credentials before startup."""
import os
from pathlib import Path
import stat
import sys


def check(path, *, secret=False):
    path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & (0o077 if secret else 0o022):
        raise SystemExit(f'Unsafe owner or permissions: {path.name}')


if __name__ == '__main__':
    if sys.argv[1] == 'gpu':
        root = Path('/opt/opendpd-gpu')
        check(root)
        for path in root.rglob('*'):
            check(path)
        check('/etc/opendpd-web/gpu-agent.env', secret=True)
        check(os.environ['OPENDPD_GPU_TOKEN_FILE'], secret=True)
    else:
        check('/etc/opendpd-web/admin_ed25519', secret=True)
        check('/etc/opendpd-web/gpu-agent.env', secret=True)
