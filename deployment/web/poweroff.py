"""Request ACPI shutdown through this VM's private QMP socket and wait for it."""
import json
import socket
import time

# systemd terminates whatever is left the moment ExecStop= returns, which would
# cut the guest's power mid-shutdown. Stay until QEMU closes the socket (guest
# powered off). Shorter than the unit's TimeoutStopSec=90.
WAIT_SECONDS = 75

try:
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(5)
        sock.connect("/run/opendpd-web-vm/qmp.sock")
        stream = sock.makefile("rwb", buffering=0)
        stream.readline()
        for command in ["qmp_capabilities", "system_powerdown"]:
            stream.write(json.dumps({"execute": command}).encode() + b"\n")
            while True:
                data = stream.readline()
                if not data:
                    break
                reply = json.loads(data)
                if "return" in reply or "error" in reply:
                    break
        deadline = time.monotonic() + WAIT_SECONDS
        while (remaining := deadline - time.monotonic()) > 0:
            sock.settimeout(remaining)
            if not stream.readline():  # EOF: QEMU exited after guest power-off.
                break
except (OSError, ValueError):
    pass  # The service stop timeout remains the final shutdown bound.
