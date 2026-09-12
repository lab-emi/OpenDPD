"""Request ACPI shutdown through this VM's private QMP socket."""
import json
import socket

with socket.socket(socket.AF_UNIX) as sock:
    sock.settimeout(5)
    sock.connect("/run/opendpd-web-vm/qmp.sock")
    stream = sock.makefile("rwb", buffering=0)
    stream.readline()
    for command in ["qmp_capabilities", "system_powerdown"]:
        stream.write(json.dumps({"execute": command}).encode() + b"\n")
        while True:
            reply = json.loads(stream.readline())
            if "return" in reply or "error" in reply:
                break
