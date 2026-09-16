"""Request ACPI shutdown through this VM's private QMP socket."""
import json
import socket

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
except (OSError, ValueError):
    pass  # The service stop timeout remains the final shutdown bound.
