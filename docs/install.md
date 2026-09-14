# Installation

[Use the hosted Studio](https://opendpd.com/studio/) without installation, or run OpenDPD **2.2.6** locally. Python 3.12 is the recommended starting point; the compute suite also covers 3.10–3.13.

## Install uv

macOS / Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal so uv is on PATH. See the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/) for package-manager alternatives.

## Create your project environment

These commands work on macOS, Linux and Windows; activation is unnecessary because the environment is named explicitly.

```sh
mkdir opendpd-lab
cd opendpd-lab
uv venv --python 3.12
uv pip install --python .venv "opendpd==2.2.6" --torch-backend=auto
uv run --no-project --python .venv opendpd doctor
uv run --no-project --python .venv opendpd gui
```

The package installs PyTorch, the API server, the prebuilt Studio frontend, and pywebview. The old `[gui]` and `[desktop]` extras remain compatible aliases. A released wheel does not require Node.js. Local datasets and runs persist in `~/opendpd-workspace`; change this with `opendpd gui --workspace PATH`.

## Platform and accelerator selection

| Platform | Native window | Compute |
| --- | --- | --- |
| Apple Silicon macOS | pywebview installs PyObjC and uses WKWebView | PyTorch's macOS wheel includes MPS; Studio selects it when available |
| Intel macOS | WKWebView on a supported macOS/Python combination | CPU; availability of current PyTorch wheels depends on architecture/version |
| Linux | pywebview installs Qt, PyQt6 and Qt WebEngine | uv selects a compatible detected GPU backend; Studio supports CUDA/ROCm through `cuda`, otherwise CPU |
| Windows | pywebview installs pythonnet and uses WebView2 | uv selects CUDA with a compatible NVIDIA driver, otherwise CPU |

`--torch-backend=auto` is uv's hardware-aware installation option; it is not a pip feature. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) and the [PyTorch installation selector](https://pytorch.org/get-started/locally/). A Python package cannot install an operating-system GPU driver, manufacture accelerator support on incompatible hardware, or guarantee every model supports every device. Intel XPU is not an OpenDPD execution target. `opendpd doctor` and Studio's device selector report what is actually available.

If PyTorch was already installed with an unsuitable backend, repair only that environment:

```sh
uv pip install --python .venv --reinstall torch --torch-backend=auto
```

Linux still needs a graphical session and the system libraries Qt uses (typically `libgl1`, `libegl1`, `libxcb-cursor0` and X11/XCB libraries on Debian/Ubuntu). Minimal distributions and containers may lack them. Windows needs the Microsoft WebView2 Runtime. See [pywebview platform prerequisites](https://pywebview.flowrl.com/guide/installation). OpenDPD does not silently install system packages or disable the browser sandbox. On a headless machine, use `--no-browser`; `--browser` bypasses the native window and `--window` reports a missing native backend directly.

## SSH or another computer

`127.0.0.1` always refers to the computer running the browser. If OpenDPD runs on a Linux server and you browse from a Mac, establish a tunnel **in a terminal on the Mac**:

```sh
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Replace `USER@SERVER` with your server login. Keep both this tunnel and the terminal running `opendpd gui --no-browser --port 8765` open. Open the **bootstrap URL printed by that running server** in the Mac browser. If Studio chose a different port, use that port on both sides of `-L`. You may choose another local port if 8765 is occupied; then change only the browser URL's port to match it. Do not publish the bootstrap token.

A browser-open failure is nonfatal: Studio continues serving its URL. The launcher now explains local addresses and prints forwarding guidance when SSH is detected. Its loopback health checks bypass system HTTP proxies. The service stays bound to loopback; opening it on a LAN is unnecessary.

## Troubleshooting connection refused

1. Keep the launcher running. Closing its native window, pressing Ctrl+C, closing the terminal, or disconnecting SSH stops it.
2. Use the current printed port and URL. Another machine requires the tunnel above.
3. On the server, check `curl http://127.0.0.1:8765/healthz` (PowerShell: `Invoke-RestMethod http://127.0.0.1:8765/healthz`). If this fails, inspect the launcher's terminal for a startup error and run `opendpd doctor` in the same environment.
4. If the health check works locally but the browser cannot connect, check your SSH forward or browser proxy's localhost bypass. A session-expired response is different from connection refused; reopen the current bootstrap URL to renew the session.

## pip and source installations

Standard pip installs all Python dependencies too:

```sh
python -m pip install "opendpd==2.2.6"
opendpd gui
```

pip cannot detect your GPU driver to choose a custom PyTorch index. For acceleration, prefer the uv command above, or follow the official PyTorch selector before installing OpenDPD. Running from a source checkout also needs Git and Node.js 22.22+:

--8<-- "README.md:source-install"

After source changes, rerun the editable install and frontend build. See [Contributing](https://github.com/lab-emi/OpenDPD/blob/main/CONTRIBUTING.md) and the [platform support matrix](releases/support-matrix.md).
