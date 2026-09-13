"""A failed browser opener must leave the printed loopback URL reachable."""
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import httpx
import pytest

pytestmark = pytest.mark.integration


def test_browser_failure_keeps_real_service_alive(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    code = ('from pathlib import Path; import sys; from opendpd.studio.launcher import launch; '
            'sys.exit(launch(Path(sys.argv[1]),port=int(sys.argv[2]),mode="browser",opener=lambda url:False))')
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]),'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
         'SSH_CONNECTION':'192.0.2.1 1234 192.0.2.2 22'}
    log=tmp_path/'server.log'
    with log.open('w') as output:
        process=subprocess.Popen([sys.executable,'-u','-c',code,str(tmp_path/'workspace'),str(port)],env=env,stdout=output,stderr=output)
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                text=log.read_text()
                if 'Studio is still running' in text:
                    break
                assert process.poll() is None, 'GUI process exited before the browser fallback became ready'
                time.sleep(.1)
            else:
                pytest.fail('Browser fallback did not become ready')
            assert 'SSH session detected' in text and f'-L {port}:127.0.0.1:{port}' in text
            url=re.search(r'OpenDPD Studio: (http://\S+)',text).group(1)
            with httpx.Client(trust_env=False,follow_redirects=True) as client:
                assert client.get(url).status_code==200
                for _ in range(3):
                    assert process.poll() is None
                    assert client.get(f'http://127.0.0.1:{port}/healthz').status_code==200
                    time.sleep(.2)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
