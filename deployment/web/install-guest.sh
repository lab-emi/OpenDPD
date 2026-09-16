#!/usr/bin/env bash
# Run inside the dedicated VM after copying the source to /opt/opendpd and
# installing the runtime to /opt/opendpd-venv. The guest never needs Internet.
set -euo pipefail
test "$(id -u)" -eq 0
test "$(id -u worker)" -eq 1001
test "$(id -g worker)" -eq 1001
test -x /opt/opendpd-venv/bin/python
test -f /etc/opendpd-web.env
test -f /opt/opendpd/opendpd/web/app.py
/opt/opendpd-venv/bin/python -c 'import torch, fastapi, uvicorn, psutil, python_multipart'
# The offline environment must match the reviewed lock before source activation.
/opt/opendpd-venv/bin/python - <<'PY_LOCK'
import importlib.metadata, re
from pathlib import Path
for line in Path('/opt/opendpd/deployment/web/requirements-vm.lock').read_text().splitlines():
    match = re.match(r'([A-Za-z0-9_.-]+)==([^ ]+)', line)
    if match:
        name, expected = match.groups()
        if importlib.metadata.version(name) != expected:
            raise SystemExit(f'Runtime lock mismatch: {name}; install the reviewed wheelhouse first')
PY_LOCK
# Install this source revision, including runtime subpackages. Merely setting
# WorkingDirectory is insufficient: training workers use their own run directory.
/usr/local/bin/uv pip install --python /opt/opendpd-venv/bin/python --no-index --no-deps --no-build-isolation -e /opt/opendpd
# Move the bridge credential out of inherited process environments.
/opt/opendpd-venv/bin/python - <<'PY_TOKEN'
from pathlib import Path
import grp, os, shlex
path = Path('/etc/opendpd-web.env')
lines = path.read_text().splitlines()
for i, line in enumerate(lines):
    if line.startswith('OPENDPD_GPU_TOKEN='):
        token = shlex.split(line.split('=', 1)[1])[0]
        target = Path('/etc/opendpd-web-gpu.token')
        target.write_text(token + '\n')
        os.chown(target, 0, grp.getgrnam('worker').gr_gid)
        target.chmod(0o640)
        lines[i] = 'OPENDPD_GPU_TOKEN_FILE=' + str(target)
path.write_text('\n'.join(lines) + '\n')
PY_TOKEN
chmod 0600 /etc/opendpd-web.env
chown -R root:root /opt/opendpd
chmod -R go-w /opt/opendpd
# The web runtime and source are read-only to worker; only ephemeral tmpfs is writable.
chown -R root:root /opt/opendpd-venv/
chmod -R go-w /opt/opendpd-venv/
install -m 0644 /opt/opendpd/deployment/web/opendpd-web.service /etc/systemd/system/
install -m 0644 /opt/opendpd/deployment/web/opendpd-web-reset.service /etc/systemd/system/
install -m 0644 /opt/opendpd/deployment/web/opendpd-web-reset.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now opendpd-web-reset.timer
systemctl enable --now opendpd-web.service
systemctl is-active opendpd-web.service opendpd-web-reset.timer
