#!/usr/bin/env bash
# Run as root on the existing OpenDPD sandbox host. Creates a separate CPU VM;
# never changes the validation images or binds/unbinds a host GPU.
set -euo pipefail
test "$(id -u)" -eq 0
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
state=/var/lib/opendpd-sandbox
admin=/etc/opendpd-web
test -f "$state/guest.qcow2"
if systemctl is-active --quiet opendpd-vm-validation.service; then
  echo 'Stop the CPU validation VM before taking the production baseline copy.' >&2
  exit 1
fi
if test -e "$state/web.qcow2"; then
  echo 'A web VM already exists; refusing to overwrite its disk.' >&2
  exit 1
fi
install -d -m 0700 "$admin"
if ! test -e "$admin/admin_ed25519"; then
  ssh-keygen -q -t ed25519 -N '' -C opendpd-web-admin -f "$admin/admin_ed25519"
fi
# A frozen, independent base avoids backing-chain corruption if the original
# validation VM is used again. Reflinks share blocks where supported.
cp --reflink=auto "$state/guest.qcow2" "$state/web-base.qcow2"
chown root:opendpd-sandbox "$state/web-base.qcow2"
chmod 0440 "$state/web-base.qcow2"
qemu-img create -q -f qcow2 -F qcow2 -b "$state/web-base.qcow2" "$state/web.qcow2"
install -d -m 0700 "$admin/seed"
python3 - "$admin" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
key = (root / 'admin_ed25519.pub').read_text().strip()
# JSON is a YAML subset. Only the public key is placed in the seed ISO.
data = {'users': [{'name': 'vmadmin', 'groups': ['sudo'], 'shell': '/bin/bash',
                  'sudo': ['ALL=(ALL) NOPASSWD:ALL'], 'lock_passwd': True, 'ssh_authorized_keys': [key]}],
        'ssh_pwauth': False, 'disable_root': True}
(root / 'seed/user-data').write_text('#cloud-config\n' + json.dumps(data))
(root / 'seed/meta-data').write_text('instance-id: opendpd-web-v1\nlocal-hostname: opendpd-web\n')
PY
xorriso -as mkisofs -quiet -output "$state/web-seed.iso" -volid cidata -joliet -rock "$admin/seed"
chown opendpd-sandbox:opendpd-sandbox "$state/web.qcow2" "$state/web-seed.iso"
chmod 0600 "$state/web.qcow2"
chmod 0400 "$state/web-seed.iso"
install -m 0644 "$source_dir/poweroff.py" "$admin/poweroff.py"
# Only this non-secret helper is readable by the VM account.
chmod 0711 "$admin"
install -m 0644 "$source_dir/opendpd-web-vm.service" /etc/systemd/system/
systemctl daemon-reload
systemctl start opendpd-web-vm.service
echo 'VM started. SSH: sudo ssh -i /etc/opendpd-web/admin_ed25519 -p 22224 vmadmin@127.0.0.1'
