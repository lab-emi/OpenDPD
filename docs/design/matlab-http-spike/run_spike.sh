#!/bin/bash
# Re-run the feasibility spike of docs/design/matlab-http-transport.md.
#
#   bash docs/design/matlab-http-spike/run_spike.sh a    # the handshake with a real local Studio service
#   bash docs/design/matlab-http-spike/run_spike.sh b    # binary bodies against a stand-in server (Linux: reads /proc)
#
# PYTHON must import this checkout's opendpd (default python3; PYTHONPATH is set to the checkout); MATLAB_BIN is the matlab
# executable (default matlab on PATH). Everything runs in a temporary directory with its own MATLAB preferences folder and a
# scratch Studio workspace on a loopback port; nothing else is read or changed, and the services are stopped on exit.
set -u
part="${1:-}"
case "$part" in a|b) ;; *) echo "usage: $0 a|b" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=2
PYTHON="${PYTHON:-python3}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
WORK="$(mktemp -d)"
mkdir -p "$WORK/ws" "$WORK/prefs"
pids=()
cleanup() {
    touch "$WORK/spike-stop" "$WORK/standin-stop"
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    rm -rf "$WORK"
}
trap cleanup EXIT

PORT="$("$PYTHON" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")"
if [ "$part" = a ]; then
    "$PYTHON" "$HERE/real_service.py" "$WORK/ws" > "$WORK/real.log" 2>&1 &
    pids+=($!)
    for _ in $(seq 1 120); do [ -f "$WORK/spike-ready" ] && break; sleep 0.5; done
    if [ ! -f "$WORK/spike-ready" ]; then echo "the service did not start:"; cat "$WORK/real.log"; exit 1; fi
else
    "$PYTHON" "$HERE/standin_server.py" "$PORT" "$WORK" > "$WORK/standin.log" 2>&1 &
    pids+=($!)
    sleep 2
fi
export SPIKE_WORKSPACE="$WORK/ws" SPIKE_STANDIN_PORT="$PORT" MATLAB_PREFDIR="$WORK/prefs"
"$MATLAB_BIN" -batch "run(fullfile(getenv('REPO_ROOT'), 'docs', 'design', 'matlab-http-spike', 'spike_http_${part}.m'))" 2>&1 | grep -v '^$'
