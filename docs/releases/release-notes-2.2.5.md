# OpenDPD 2.2.5

Studio adds interactive LaTeX formulas, configurable ILC/ILA experiments, clearer dataset controls and a simpler default installation.

## Changes

- Virtual PA equations render locally with KaTeX. Clicking a coefficient, slider or numeric input highlights the matching parameter and every occurrence in the equation.
- Signal Generator creates an input only after **Generate & preview**. PA Input Datasets can be removed from the picker and restored with Undo; existing simulations retain their source bytes.
- Custom OFDMA uses shared channel settings by default. Uncheck the option for independent subcarrier counts, modulation and power; checking it again copies channel 1 to all channels.
- PA Training/Testing now includes **ILC linearization** after forward PA identification. DPD includes **ILC-DPD / Ideal benchmark**, with bounded feedback, backtracking, peak limits and configurable hyperparameters.
- ILC learns training waveforms and ILA fits a transferable memory-polynomial DPD. The separately labelled Ideal baseline optimizes the test waveform with feedback. Results retain convergence, stop reasons, sample counts, plant identity and downloadable waveforms. Ideal input/output PSDs appear at their respective signal-chain positions.
- Metric definitions render formulas in LaTeX. **Metric calculation** replaces the unexplained profile label; the result selector lives inside Metric definitions. Existing calculation protocols are unchanged.
- Next-step controls sit above configuration and carry an arrow. About no longer requests or shows GitHub activity.
- Default installation includes PyTorch, pywebview and the Studio server. Platform markers install Qt bindings on Linux and native bindings on macOS/Windows. The recommended uv commands select an appropriate PyTorch backend automatically.
- Launcher messages explain localhost and SSH port forwarding. Failed browser opening leaves the server running; local health probes bypass HTTP proxies.
- Updated README, installation and workflow guides, support records and actual GUI screenshots. CI checks default installation on Linux, macOS and Windows.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
mkdir opendpd-lab
cd opendpd-lab
uv venv --python 3.12
uv pip install --python .venv "opendpd==2.2.5" --torch-backend=auto
uv run --no-project --python .venv opendpd gui
```

[Installation and SSH access](../install.md) · [Hosted Studio](https://opendpd.com/studio/) · [ILC guide](../guides/ilc-dpd.md)

## Evidence and scope

[Validation record](../performance/studio-2.2.5.md). ILC is experimental waveform control through a trained surrogate. The Ideal reference is specific to the tested waveform, not proof of a global optimum or physical PA performance. The reported defaults are starting values; target convergence is not guaranteed. Polynomial fitting remains CPU complex128 while PA replay uses the selected device.

Virtual PAs and generated pairs remain synthetic. Public sessions retain isolation, quotas and expiry. OS GUI libraries and GPU drivers remain platform prerequisites; installation checks do not establish native-window or accelerator support on every device. Human dataset review and independent RF/EVM acceptance requirements remain unchanged.
