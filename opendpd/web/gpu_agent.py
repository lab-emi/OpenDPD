"""Host-only GPU agent. Pulls from the VM; never listens on a network socket."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from opendpd.web.gpu_archive import MAX_BYTES, pack, unpack, read_regular

LABEL = "opendpd.gpu-worker=true"
stopping = False


def podman(*args, **kwargs):
    return subprocess.run(["/usr/bin/podman", *args], timeout=30, check=True, **kwargs)


def cleanup_containers():
    result = podman("ps", "-aq", "--filter", "label=" + LABEL, capture_output=True, text=True)
    for identifier in result.stdout.split():
        podman("rm", "-f", identifier, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def container_command(image, name, root, run_id, seconds):
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("GPU image must be pinned by local image ID")
    return ["/usr/bin/podman", "run", "--rm", "--pull=never", "--log-driver=none", "--name", name, "--label", LABEL,
            "--network=none", "--read-only", "--user=65532:65532", "--cap-drop=all",
            "--security-opt=no-new-privileges", "--device=nvidia.com/gpu=0",
            "--memory=6g", "--memory-swap=6g", "--cpus=4", "--pids-limit=128",
            "--ulimit=core=0:0", "--ulimit=fsize=67108864:67108864", "--shm-size=256m",
            "--timeout", str(seconds), "--stop-timeout=3",
            "--tmpfs=/tmp:rw,size=256m,nodev,nosuid,noexec,mode=1777",
            "--env=HOME=/tmp", "--env=MPLCONFIGDIR=/tmp/matplotlib", "--env=XDG_CACHE_HOME=/tmp/cache",
            "--env=OMP_NUM_THREADS=4", "--env=OPENBLAS_NUM_THREADS=4", "--env=MKL_NUM_THREADS=4",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONUNBUFFERED=1",
            "--volume", f"{root}:/workspace:rw,nosuid,nodev,noexec", "--workdir=/workspace",
            "--entrypoint=python", image, "-m", "opendpd.web.gpu_container", "--workspace", "/workspace", "--run-id", run_id]


class Agent:
    def __init__(self, root, token, image, port=18765):
        self.root, self.token, self.image = root, token, image
        if not 1024 <= port <= 65535:
            raise ValueError("invalid private VM port")
        self.base = f"http://127.0.0.1:{port}/_gpu"
        # No environment proxy, redirects or arbitrary URLs from server responses.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, body=None, job=None, code=None):
        headers = {"X-OpenDPD-GPU": self.token, "Content-Type": "application/json"}
        if job:
            headers["X-OpenDPD-Lease"] = job["lease"]
        if code is not None:
            headers["X-OpenDPD-Exit"] = str(code)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=body, headers=headers)
        with self.http.open(request, timeout=20) as response:
            data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("GPU response exceeds transfer limit")
            return json.loads(data) if response.headers.get_content_type() == "application/json" else data

    def update(self, job, root, offsets):
        payload = dict(offsets)
        for key, name in [("log", "logs/worker.log"), ("events", "events.jsonl")]:
            data = read_regular(root, name, offsets.get(key + "_offset", 0), 256 * 1024)
            payload[key] = base64.b64encode(data).decode()
        live = read_regular(root, "live.json", limit=2 * 1024 * 1024)
        if live:
            payload["live"] = base64.b64encode(live).decode()
        result = self.request(f"/jobs/{job['id']}/update", payload, job)
        offsets.update({k: result[k] for k in ("log_offset", "events_offset") if k in result})
        return result["continue"]

    def run(self, job):
        if (not re.fullmatch(r"[a-f0-9]{32}", job["id"])
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job["run_id"])
                or not math.isfinite(job["expires_at"])):
            raise ValueError("invalid GPU job")
        seconds = int(min(1800, job["expires_at"] - time.time()))
        if seconds <= 0:
            return
        name = "opendpd-gpu-" + job["id"]
        root = self.root / job["id"]
        root.mkdir(mode=0o700)
        process = None
        try:
            unpack(self.request(f"/jobs/{job['id']}/input", job=job), root)
            run = root / "runs" / job["run_id"]
            (run / "logs").mkdir(exist_ok=True)
            for path in [root, *root.rglob("*")]:
                os.chown(path, 65532, 65532)
            offsets = {"log_offset": 0, "events_offset": 0}
            if not self.update(job, run, offsets):
                return
            with (run / "logs" / "worker.log").open("wb") as log:
                command = container_command(self.image, name, root, job["run_id"], seconds)
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                cancelled = False
                while process.poll() is None:
                    if stopping or time.time() >= job["expires_at"] or not self.update(job, run, offsets):
                        cancelled = True
                        podman("kill", name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        break
                    size = 0
                    for count, path in enumerate(root.rglob("*")):
                        if path.is_symlink():
                            raise RuntimeError("GPU workspace links are forbidden")
                        size += path.stat(follow_symlinks=False).st_size
                        if count > 4096 or size > MAX_BYTES:
                            raise RuntimeError("GPU workspace storage limit exceeded")
                    time.sleep(2)
                code = process.wait(timeout=15)
            # Full final logs/events are an atomic extension of their streamed prefix.
            data = pack(run, run.rglob("*"))
            self.request(f"/jobs/{job['id']}/result", data, job, 3 if cancelled else 0 if code == 0 else 1)
        finally:
            subprocess.run(["/usr/bin/podman", "rm", "-f", name], timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if process:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            shutil.rmtree(root)

    def loop(self):
        cleanup_containers()
        # Advertise CUDA only after this exact isolated image performs a real operation.
        podman("run", "--rm", "--pull=never", "--log-driver=none", "--label", LABEL,
               "--network=none", "--read-only", "--user=65532:65532", "--cap-drop=all",
               "--security-opt=no-new-privileges", "--device=nvidia.com/gpu=0", "--timeout=20",
               "--entrypoint=python", self.image, "-c",
               "import torch; assert torch.cuda.is_available(); assert torch.ones(1,device='cuda').item()==1",
               stdout=subprocess.DEVNULL)
        for path in self.root.iterdir():
            if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name):
                shutil.rmtree(path)
        while not stopping:
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits", "--id=0"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
                name, free = gpu.rsplit(",", 1)
                if int(free.strip()) >= 4096:
                    job = self.request("/poll", {"name": name})["job"]
                    if job:
                        self.run(job)
            except Exception as error:
                # Do not log payloads, bearer capabilities, paths or user data.
                print("GPU agent operation failed:", type(error).__name__, flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup_containers()
        return
    root = Path(os.environ.get("OPENDPD_GPU_ROOT", "/run/opendpd-gpu"))
    if root.is_symlink() or not root.is_absolute():
        raise ValueError("GPU root must be a dedicated absolute tmpfs")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    filesystem = subprocess.run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)], check=True, capture_output=True, text=True).stdout.strip()
    if filesystem != "tmpfs":
        raise ValueError("GPU workspace must be on tmpfs; user data must not enter persistent storage")
    agent = Agent(root, Path(os.environ["OPENDPD_GPU_TOKEN_FILE"]).read_text().strip(), os.environ["OPENDPD_GPU_IMAGE"],
                  int(os.environ.get("OPENDPD_GPU_PORT", "18765")))
    def stop(*_):
        global stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    agent.loop()


if __name__ == "__main__":
    main()
