"""Measure hosted admission over loopback HTTP without using production sessions.

Run with the source on PYTHONPATH; writes only aggregate, credential-free data.
The fixed daytime clock makes this repeatable during scheduled cleanup windows.
"""
import argparse
import json
import socket
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil
import uvicorn

from opendpd import __version__
from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='opendpd-admission-') as directory:
        config = WebConfig(Path(directory) / 'web', 'https://opendpd.com', 'api.opendpd.com', 'a' * 48 + '.internal')
        app = create_web_app(config, now=lambda: DAY * 20000 + 3600)
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, access_log=False, log_level='error', proxy_headers=False))
        runner = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        runner.start()
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started
        process = psutil.Process()
        initial_rss = process.memory_info().rss
        headers = {'Host': config.tunnel_host, 'Origin': config.origin, 'X-Forwarded-Proto': 'https'}
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', headers=headers, timeout=30, trust_env=False)
        try:
            sessions, latencies = [], []
            for index in range(config.max_sessions):
                ip = f'198.51.100.{1 + index // 32}'
                started = time.perf_counter()
                response = client.post('/api/v1/web/sessions', headers={'CF-Connecting-IP': ip}, json={})
                assert response.status_code == 201, response.text
                latencies.append((time.perf_counter() - started) * 1000)
                sessions.append({'Authorization': 'Bearer ' + response.json()['access_token'], 'CF-Connecting-IP': ip})
            occupied_rss = process.memory_info().rss
            for index in range(config.max_waiting):
                ip = f'203.0.113.{1 + index // config.waiting_per_ip}'
                response = client.post('/api/v1/web/sessions', headers={'CF-Connecting-IP': ip}, json={})
                assert response.status_code == 202 and response.json()['queue_position'] == index + 1
            assert len(app.state.manager.tenants) == config.max_sessions
            assert app.state.manager.waiting_count() == config.max_waiting
            rejected = client.post('/api/v1/web/sessions', headers={'CF-Connecting-IP': '192.0.2.1'}, json={})
            assert rejected.status_code == 429 and rejected.json()['error']['code'] == 'queue_full'
            def status(index):
                started = time.perf_counter()
                result = client.get('/api/v1/system/status', headers=sessions[index])
                assert result.status_code == 200, result.text
                data = result.json()
                assert data['workspaces'] == 256 and data['waiting_sessions'] == 1024
                return (time.perf_counter() - started) * 1000
            # Warm the shared status cache, then model 100 independent viewers.
            status(0)
            with ThreadPoolExecutor(max_workers=8) as pool:
                status_ms = list(pool.map(status, range(100)))
            cpu_before = sum(process.cpu_times()[:2])
            idle_start = time.perf_counter()
            time.sleep(2)
            idle_cpu = (sum(process.cpu_times()[:2]) - cpu_before) / (time.perf_counter() - idle_start) * 100
            def timings(values):
                return {'median_ms': round(statistics.median(values), 3), 'p95_ms': round(sorted(values)[int(.95 * (len(values)-1))], 3), 'max_ms': round(max(values), 3)}
            report = {'version': __version__, 'transport': 'loopback HTTP', 'workspaces': config.max_sessions,
                'waiting_tickets': config.max_waiting, 'overflow_status': rejected.status_code,
                'creation': timings(latencies), 'status_100_requests_8_clients': timings(status_ms),
                'rss_initial_mib': round(initial_rss / 2**20, 2), 'rss_256_workspaces_mib': round(occupied_rss / 2**20, 2),
                'rss_with_waiting_mib': round(process.memory_info().rss / 2**20, 2),
                'background_threads': [t.name for t in threading.enumerate() if t.name.startswith('opendpd-')],
                'idle_process_cpu_percent_one_core': round(idle_cpu, 2),
                'scope': 'Empty workspace admission and aggregate status; not 256 simultaneous training jobs or a distributed-load SLA.'}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            client.close()
            server.should_exit = True
            runner.join(30)
            sock.close()
            assert not runner.is_alive()


if __name__ == '__main__':
    main()
