# OpenDPD Studio 2.2.12

Fixes [issue #50](https://github.com/lab-emi/OpenDPD/issues/50): PA training could fail with `Failed to find C compiler` when DeltaGRU first initialized its optional Triton CUDA kernels.

- Local DeltaGRU training and inference now fall back to PyTorch on the same device when Triton cannot find a C compiler. The layer warns once and continues with the same weights, states and gradients. Unrelated CUDA and kernel failures still surface normally.
- The hosted GPU image includes GCC and the Python development headers needed by Triton. Generated shared libraries use a private, size-limited temporary cache; uploaded workspaces and the ordinary temporary directory remain mounted `noexec`.
- The GPU worker now verifies cold-cache DeltaGRU forward and backward execution under the production container restrictions before accepting jobs. This catches missing compilers, headers and unusable caches before a user starts training.

Install with `uv pip install "opendpd==2.2.12" --torch-backend=auto`, or open [Studio on the web](https://opendpd.com/studio/?v=2.2.12).
