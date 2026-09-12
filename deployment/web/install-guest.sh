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
# Install this source revision, including runtime subpackages. Merely setting
# WorkingDirectory is insufficient: training workers use their own run directory.
/usr/local/bin/uv pip install --python /opt/opendpd-venv/bin/python --no-index --no-deps --no-build-isolation -e /opt/opendpd
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
