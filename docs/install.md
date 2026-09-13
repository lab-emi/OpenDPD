# Installation

Try **[OpenDPD Studio on the web](https://opendpd.com/studio/)** without installing anything, or install **2.2.0** locally.

| Installation | Includes | Requirements |
| --- | --- | --- |
| PyPI, `opendpd[gui]` | Packaged Studio, workspace CLI and Python API | Python 3.10–3.13; no Node.js |
| PyPI, `opendpd[desktop]` | Studio in a native window | Above, plus a working system web view |
| Source, `.[gui]` | Editable Studio and research scripts | Python 3.10–3.13, Git, Node.js 22.22+ |

```bash
python -m pip install "opendpd[gui]==2.2.0"
opendpd gui
```

Local workspaces stay on your computer. The hosted app uses temporary server workspaces and deletes their files within 24 hours.

## Install Studio from source

Install Python, Git and Node.js first. The following creates an isolated Python environment, installs the browser dependencies and builds the frontend. Run it in a terminal on macOS or Linux; use `python3` if that is your Python command.

[Source quick-start commands on GitHub](https://github.com/lab-emi/OpenDPD#get-started-with-studio) (also expanded below on the documentation site).

--8<-- "README.md:source-install"

On **Windows PowerShell**, replace the activation line with:

```powershell
.venv\Scripts\Activate.ps1
```

The remaining commands are the same. Platform verification is recorded separately in the [support matrix](releases/support-matrix.md).

Studio starts a service on `127.0.0.1` and opens your browser. The default workspace is `~/opendpd-workspace`; choose another with `opendpd gui --workspace PATH`. Datasets, runs and results stay in that workspace. Follow the [Studio walkthrough](tutorials/gui-quickstart.md) once the Home page appears.

### Native desktop window

From the same checkout and activated environment:

```bash
python -m pip install -e ".[desktop]"
opendpd gui --window
```

The desktop window uses the same local UI and workspace. `--window` reports an error if the native backend is unavailable; `--browser` always uses the system browser. Run `opendpd doctor` to inspect the installed backend. Linux may also need WebKit2GTK system packages or a Qt backend; see the [launcher guide](tutorials/gui-quickstart.md#launch-options).

### CPU and GPU

CPU is enough for a first smoke experiment. For NVIDIA GPUs, install the PyTorch build matching your system using the [official PyTorch selector](https://pytorch.org/get-started/locally/) **before** installing OpenDPD. Select `cuda`, `mps` or `cpu` only when available on your machine. See the [support matrix](releases/support-matrix.md) for what has been verified.

### Update a checkout

After pulling new source changes, refresh the Python installation and rebuild the frontend:

```bash
git pull --ff-only
python -m pip install -e ".[gui]"
npm --prefix frontend ci
npm --prefix frontend run build
opendpd gui
```

Node.js is needed to build the frontend from source. Running a wheel that already contains the built frontend does not require Node.js.

## Install the released core from PyPI

In an activated Python environment:

```bash
python -m pip install opendpd
opendpd-cli --help
```

This provides `opendpd.train_pa`, `train_dpd`, `run_dpd`, `plot_dpd`, `load_dataset` and `create_dataset`. Start with [Python examples](examples.md) or the [API reference](api.md). `opendpd-cli` mirrors the repository's `python main.py`; it is distinct from the newer workspace command `opendpd run`.

For the current core from source without the GUI, use `python -m pip install -e .` in the checkout. Research scripts run from the repository root; see [training](training.md) and [reproduction](reproducing.md).

## Troubleshooting

- **`opendpd` is missing:** check that the virtual environment is active and that you installed the source preview, not only the PyPI 2.1.0 release.
- **Frontend assets are missing:** run both frontend build commands above from the repository root, then restart Studio.
- **An old UI appears:** stop Studio, rebuild the frontend and relaunch. `opendpd doctor` checks asset availability and version.
- **The session expired:** reopen the URL printed by the current launcher. A link from an earlier server instance will not work.

Developer setup and the repository layout are in [Contributing](https://github.com/lab-emi/OpenDPD/blob/main/CONTRIBUTING.md) and [Testing](testing.md).
